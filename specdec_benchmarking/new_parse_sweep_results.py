#!/usr/bin/env python3
"""
Parses a hierarchical vllm bench sweep into a single summary CSV.

Expected layout (produced by benchmark_random.sh / benchmark.sh):
    <result_root>/<tag>/r<REQUEST_RATE>/result_*.json      (vllm bench --save-result)
    <result_root>/<tag>/r<REQUEST_RATE>/<role>_metrics.csv (/metrics poller)
    <result_root>/<tag>/r<REQUEST_RATE>/gpu_metrics.txt    (dcgmi/nvidia-smi, optional)

    Legacy concurrency sweeps also accepted:
    <result_root>/<tag>/c<CONCURRENCY>/...
nadal

Role CSVs on disk (from the bench poller):
    server   - client-facing serve / proxy
    draft    - Disagg-DFlash draft :9101
    prefill  - P/D prefill engine
    decode   - P/D decode engine

These are remapped into two fixed server slots for the summary CSV:

    topology=colocated   (only server, or nothing else useful)
        S1.* <- server     S2.* empty

    topology=pd_disagg   (prefill + decode present; P/D ± colocated SD)
        S1.* <- prefill    S2.* <- decode

    topology=sd_disagg   (draft present; verify + remote draft)
        S1.* <- server     S2.* <- draft
        (S1 = verify / client-facing engine)

One row per run = flattened JSON keys, plus:
    gpu, tag, request_rate   (from r<RATE>/ dirs; legacy c<N>/ still accepted)
    topology, S1.role, S2.role
    S1.* / S2.*  -> KV %, block size, KV GiB (blocks×block_size), queues, MFU, ...
    dcgm.*       -> best-effort means from gpu_metrics.txt
    extra --col key=value columns on every row

KV GiB comes from the poller as:
    total = (num_gpu_blocks - 1) * kv_cache_block_size_bytes
    used  = usage_perc * total
and is also re-derived in summarize_role_metrics as a fallback.

Column schema is fixed (full S1.* + S2.* + common bench/dcgm keys) so
CSVs from colocated / P/D / SD-disagg merge cleanly; missing values are blank.

MFU / memory-bandwidth require ``--enable-mfu-metrics`` on the engines.
Rates are (last - first) / wall-time over the poll window.

Usage:
    python3 new_parse_sweep_results.py [result_root] [out_csv] --gpu A100
    python3 new_parse_sweep_results.py bench_results sweep.csv --gpu A100 \\
        --col cluster=a100x8-vm1 --col vllm=0.11.0
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

MAX_LIST_LEN_TO_KEEP = 20

# Metric field names always emitted under S1.* / S2.* (fixed schema for merges).
# Empty when that slot/role has no poller CSV or the gauge was absent.
SERVER_METRIC_FIELDS = (
    "kv_cache_median",
    "kv_cache_mean",
    "kv_cache_p95",
    "kv_cache_max",
    "kv_block_size_bytes",
    "kv_num_blocks",
    "kv_usage_gib_mean",
    "kv_usage_gib_max",
    "kv_total_gib",
    "kv_total_gib_mean",
    "kv_total_gib_max",
    "kv_usage_bytes_mean",
    "kv_usage_bytes_max",
    "kv_total_bytes",
    "kv_total_bytes_mean",
    "kv_total_bytes_max",
    "mfu_tflops_per_gpu",
    "mem_bw_gbps_per_gpu",
    "mem_read_gbps_per_gpu",
    "mem_write_gbps_per_gpu",
    "running_mean",
    "running_max",
    "waiting_mean",
    "waiting_max",
    "preemptions_delta",
    "prompt_tokens_delta",
    "generation_tokens_delta",
    "gen_tok_per_s",
    "prefix_cache_hit_rate",
    "samples",
)

# Always-present DCGM columns (empty if gpu_metrics.txt missing / unparsed).
DCGM_FIELDS = (
    "dcgm.tensor_active_mean",
    "dcgm.dram_active_mean",
    "dcgm.sm_active_mean",
    "dcgm.graphics_active_mean",
)

# Core vllm-bench JSON keys always present (empty if absent in a run).
BENCH_JSON_FIELDS = (
    "max_concurrency",
    "num_prompts",
    "completed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "std_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "std_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "std_itl_ms",
    "backend",
    "burstiness",
    "date",
    "endpoint_type",
    "failed",
    "label",
    "max_concurrent_requests",
    "max_output_tokens_per_s",
    "model",
    "model_id",
    "request_goodput",
    "rtfx",
    "source_file",
    "tokenizer_id",
    "total_input_tokens",
    "total_output_tokens",
    "spec_decode_acceptance_length",
    "spec_decode_acceptance_rate",
    "spec_decode_accepted_tokens",
    "spec_decode_draft_tokens",
    "spec_decode_num_drafts",
    "spec_decode_per_position_acceptance_rates",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def flatten(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        elif isinstance(v, list):
            if len(v) > MAX_LIST_LEN_TO_KEEP:
                continue  # raw per-request arrays -> drop
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))]


def col_floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def rate_from_counters(ts, values):
    """Return (last-first)/(ts_last-ts_first) when both series are usable."""
    if len(ts) < 2 or len(values) < 2:
        return None
    n = min(len(ts), len(values))
    dt = ts[n - 1] - ts[0]
    if dt <= 0:
        return None
    return (values[n - 1] - values[0]) / dt


# --------------------------------------------------------------------------- #
# <role>_metrics.csv -> <prefix>.* aggregates
# --------------------------------------------------------------------------- #
def summarize_role_metrics(path, prefix):
    """Aggregate one poller CSV into ``{prefix}.*`` columns."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    out = {f"{prefix}.samples": len(rows)}

    kv = col_floats(rows, "kv_cache_usage_perc")
    if kv:
        out.update({
            f"{prefix}.kv_cache_mean": round(statistics.mean(kv), 4),
            f"{prefix}.kv_cache_median": round(statistics.median(kv), 4),
            f"{prefix}.kv_cache_p95": round(pctl(kv, 95), 4),
            f"{prefix}.kv_cache_max": round(max(kv), 4),
        })

    # Block size / usable block count are constant for a run. Drop zeros from
    # early polls before the block-size gauge / cache_config_info is ready.
    bs = [x for x in col_floats(rows, "kv_cache_block_size_bytes") if x > 0]
    if bs:
        out[f"{prefix}.kv_block_size_bytes"] = round(statistics.median(bs), 0)
    nblocks = [x for x in col_floats(rows, "kv_cache_num_blocks") if x > 0]
    if nblocks:
        out[f"{prefix}.kv_num_blocks"] = round(statistics.median(nblocks), 0)

    # Usage varies over the run → mean/max.
    for key, name in (
        ("kv_cache_usage_gib", "kv_usage_gib"),
        ("kv_cache_usage_bytes", "kv_usage_bytes"),
    ):
        xs = [x for x in col_floats(rows, key) if x > 0] or col_floats(rows, key)
        if xs:
            out[f"{prefix}.{name}_mean"] = round(statistics.mean(xs), 4)
            out[f"{prefix}.{name}_max"] = round(max(xs), 4)

    # Total KV capacity is fixed; early zero samples must not drag the mean.
    for key, name in (
        ("kv_cache_total_gib", "kv_total_gib"),
        ("kv_cache_total_bytes", "kv_total_bytes"),
    ):
        xs = [x for x in col_floats(rows, key) if x > 0]
        if xs:
            total = statistics.median(xs)
            out[f"{prefix}.{name}"] = round(total, 4)
            out[f"{prefix}.{name}_mean"] = round(total, 4)
            out[f"{prefix}.{name}_max"] = round(total, 4)

    # Fallback: derive GiB/bytes from perc × blocks × block_size if poller
    # left gib/bytes empty (e.g. older CSV without computed columns).
    if f"{prefix}.kv_total_gib" not in out and bs and nblocks:
        block_size = statistics.median(bs)
        num_blocks = statistics.median(nblocks)
        total_bytes = num_blocks * block_size
        gib = 1024.0 ** 3
        out[f"{prefix}.kv_total_bytes"] = round(total_bytes, 0)
        out[f"{prefix}.kv_total_bytes_mean"] = round(total_bytes, 0)
        out[f"{prefix}.kv_total_bytes_max"] = round(total_bytes, 0)
        out[f"{prefix}.kv_total_gib"] = round(total_bytes / gib, 4)
        out[f"{prefix}.kv_total_gib_mean"] = round(total_bytes / gib, 4)
        out[f"{prefix}.kv_total_gib_max"] = round(total_bytes / gib, 4)
        if kv:
            usage = [u * total_bytes for u in kv]
            out[f"{prefix}.kv_usage_bytes_mean"] = round(statistics.mean(usage), 0)
            out[f"{prefix}.kv_usage_bytes_max"] = round(max(usage), 0)
            out[f"{prefix}.kv_usage_gib_mean"] = round(
                statistics.mean(usage) / gib, 4
            )
            out[f"{prefix}.kv_usage_gib_max"] = round(max(usage) / gib, 4)

    for key, name in (("num_requests_running", "running"),
                      ("num_requests_waiting", "waiting")):
        xs = col_floats(rows, key)
        if xs:
            out[f"{prefix}.{name}_mean"] = round(statistics.mean(xs), 2)
            out[f"{prefix}.{name}_max"] = max(xs)

    # cumulative counters -> delta over the run
    for key, name in (("preemptions_total", "preemptions"),
                      ("prompt_tokens_total", "prompt_tokens"),
                      ("generation_tokens_total", "generation_tokens")):
        xs = col_floats(rows, key)
        if len(xs) >= 2:
            out[f"{prefix}.{name}_delta"] = xs[-1] - xs[0]

    # server-side generation throughput (cross-check for the client number)
    ts = col_floats(rows, "unix_ts")
    gt = col_floats(rows, "generation_tokens_total")
    if len(ts) >= 2 and len(gt) >= 2 and ts[-1] > ts[0]:
        out[f"{prefix}.gen_tok_per_s"] = round(
            (gt[-1] - gt[0]) / (ts[-1] - ts[0]), 1
        )

    # prefix cache hit rate over the run
    q = col_floats(rows, "prefix_cache_queries_total")
    h = col_floats(rows, "prefix_cache_hits_total")
    if len(q) >= 2 and len(h) >= 2 and (q[-1] - q[0]) > 0:
        out[f"{prefix}.prefix_cache_hit_rate"] = round(
            (h[-1] - h[0]) / (q[-1] - q[0]), 4
        )

    # MFU (TF/s/GPU) + estimated memory bandwidth (GB/s/GPU) from counters
    flops = col_floats(rows, "estimated_flops_per_gpu_total")
    rbytes = col_floats(rows, "estimated_read_bytes_per_gpu_total")
    wbytes = col_floats(rows, "estimated_write_bytes_per_gpu_total")
    flops_rate = rate_from_counters(ts, flops)
    if flops_rate is not None:
        out[f"{prefix}.mfu_tflops_per_gpu"] = round(flops_rate / 1e12, 3)
    if len(ts) >= 2 and len(rbytes) >= 2 and len(wbytes) >= 2:
        n = min(len(ts), len(rbytes), len(wbytes))
        dt = ts[n - 1] - ts[0]
        if dt > 0:
            bw = (
                (rbytes[n - 1] - rbytes[0]) + (wbytes[n - 1] - wbytes[0])
            ) / dt
            out[f"{prefix}.mem_bw_gbps_per_gpu"] = round(bw / 1e9, 3)
            out[f"{prefix}.mem_read_gbps_per_gpu"] = round(
                (rbytes[n - 1] - rbytes[0]) / dt / 1e9, 3
            )
            out[f"{prefix}.mem_write_gbps_per_gpu"] = round(
                (wbytes[n - 1] - wbytes[0]) / dt / 1e9, 3
            )

    return out


def discover_role_metrics(run_dir):
    """Return {role: path} for every ``<role>_metrics.csv`` under run_dir."""
    found = {}
    for path in glob.glob(os.path.join(run_dir, "*_metrics.csv")):
        base = os.path.basename(path)
        role = base[: -len("_metrics.csv")]
        if role:
            found[role] = path
    return found


def assign_s1_s2(role_csvs):
    """Map on-disk role CSVs -> (topology, s1_role, s1_path, s2_role, s2_path).

    Rules:
      - prefill + decode present  -> pd_disagg:  S1=prefill, S2=decode
      - draft present             -> sd_disagg:  S1=server (verify), S2=draft
      - else                      -> colocated:  S1=server, S2 empty
    """
    has_prefill = "prefill" in role_csvs
    has_decode = "decode" in role_csvs
    has_draft = "draft" in role_csvs

    if has_prefill and has_decode:
        return (
            "pd_disagg",
            "prefill",
            role_csvs["prefill"],
            "decode",
            role_csvs["decode"],
        )
    if has_draft:
        s1_path = role_csvs.get("server")
        return (
            "sd_disagg",
            "server" if s1_path else "",
            s1_path,
            "draft",
            role_csvs["draft"],
        )
    if "server" in role_csvs:
        return ("colocated", "server", role_csvs["server"], "", None)
    return ("colocated", "", None, "", None)


# --------------------------------------------------------------------------- #
# gpu_metrics.txt (dcgmi dmon) -> dcgm.* means, best effort
# --------------------------------------------------------------------------- #
def summarize_gpu_metrics(path):
    """Parses `dcgmi dmon -e 1002,1004,1005` output. Returns mean of each
    numeric column across all samples/GPUs. Silently returns {} on any
    unfamiliar format (e.g. nvidia-smi fallback)."""
    if not os.path.exists(path):
        return {}
    try:
        header, cols = None, {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    parts = line.lstrip("#").split()
                    if len(parts) >= 2 and header is None:
                        header = parts  # e.g. ['Entity', 'SMACT', 'TENSO', 'DRAMA']
                    continue
                parts = line.split()
                if len(parts) < 3 or parts[0].upper() != "GPU":
                    continue
                vals = parts[2:]  # skip 'GPU', '<id>'
                names = (header[1:] if header and len(header) - 1 == len(vals)
                         else [f"col{i}" for i in range(len(vals))])
                for name, v in zip(names, vals):
                    try:
                        cols.setdefault(name, []).append(float(v))
                    except ValueError:
                        pass
        rename = {"SMACT": "sm_active", "TENSO": "tensor_active",
                  "DRAMA": "dram_active", "GRACT": "graphics_active"}
        return {f"dcgm.{rename.get(k, k.lower())}_mean": round(statistics.mean(v), 4)
                for k, v in cols.items() if v}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_root", nargs="?", default="./bench_results")
    ap.add_argument("out_csv", nargs="?", default="sweep_summary.csv")
    ap.add_argument("--gpu", required=True, help="GPU label added to every row, e.g. A100")
    ap.add_argument("--col", action="append", default=[], metavar="KEY=VALUE",
                    help="extra constant column(s), repeatable")
    args = ap.parse_args()

    extra_cols = {"gpu": args.gpu}
    for kv in args.col:
        if "=" not in kv:
            ap.error(f"--col expects KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        extra_cols[k] = v

    files = sorted(glob.glob(os.path.join(args.result_root, "**", "result_*.json"),
                             recursive=True))
    if not files:
        print(f"No result_*.json files found under {args.result_root}")
        sys.exit(1)

    rows, all_keys = [], set()
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        flat = flatten(data)

        run_dir = os.path.dirname(fp)                       # .../<tag>/r<N> or c<N>
        cdir = os.path.basename(run_dir)                    # r<N> or c<N>
        tag = os.path.basename(os.path.dirname(run_dir))    # <tag>

        # Rate-based sweeps: r128 or r128p5 (dot -> p). Legacy: c128.
        m = re.fullmatch(r"r(\d+(?:p\d+)?)", cdir)
        if m:
            request_rate = float(m.group(1).replace("p", "."))
        else:
            m = re.fullmatch(r"c(\d+)", cdir)
            if m:
                # Legacy max-concurrency sweep dirs.
                request_rate = int(m.group(1))
            else:  # legacy flat layout: fall back to the filename
                m2 = re.search(r"_r(\d+(?:p\d+)?)\.json$", os.path.basename(fp))
                if m2:
                    request_rate = float(m2.group(1).replace("p", "."))
                else:
                    m3 = re.search(r"_c(\d+)\.json$", os.path.basename(fp))
                    request_rate = int(m3.group(1)) if m3 else ""
                tag = os.path.relpath(run_dir, args.result_root)

        flat["tag"] = tag
        # Prefer directory-derived rate; JSON may already have request_rate.
        if request_rate != "":
            flat["request_rate"] = request_rate
        flat.update(extra_cols)

        role_csvs = discover_role_metrics(run_dir)
        topology, s1_role, s1_path, s2_role, s2_path = assign_s1_s2(role_csvs)
        flat["topology"] = topology
        flat["S1.role"] = s1_role
        flat["S2.role"] = s2_role
        if s1_path:
            flat.update(summarize_role_metrics(s1_path, "S1"))
        if s2_path:
            flat.update(summarize_role_metrics(s2_path, "S2"))

        flat.update(summarize_gpu_metrics(os.path.join(run_dir, "gpu_metrics.txt")))
        flat["source_file"] = os.path.relpath(fp, args.result_root)

        rows.append(flat)
        all_keys.update(flat.keys())

    # Fixed schema: always emit S1.* and S2.* (and other known cols) even when
    # empty, so CSVs from colocated / P/D / SD-disagg merge cleanly.
    slot_cols = []
    for slot in ("S1", "S2"):
        slot_cols.extend(f"{slot}.{field}" for field in SERVER_METRIC_FIELDS)

    fieldnames = (
        ["gpu", "tag", "request_rate", "topology", "S1.role", "S2.role"]
        + sorted(k for k in extra_cols if k != "gpu")
        + list(BENCH_JSON_FIELDS)
        + slot_cols
        + list(DCGM_FIELDS)
    )
    # Any unexpected keys from a particular JSON still append (stable order).
    remaining = sorted(k for k in all_keys if k not in fieldnames)
    fieldnames = fieldnames + remaining

    def _rate_key(r):
        v = r.get("request_rate")
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=lambda r: (str(r.get("tag", "")), _rate_key(r)))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Tags: {sorted(set(str(r['tag']) for r in rows))}")
    print(f"Columns ({len(fieldnames)}): {fieldnames}")


if __name__ == "__main__":
    main()
