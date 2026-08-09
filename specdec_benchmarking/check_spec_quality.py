#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quality + speculative-acceptance sanity checks (no dual-run).

Modes:
  run       Drive vllm bench serve over quality_20 (or any custom JSONL)
            at one or more temperatures; save detailed outputs + accept stats.
  compare   Diff two result JSONs: per-position acceptance rates and
            basic garbage / answer heuristics.

Examples:
  # Server must use real rejection sampling (--no-synthetic on launch).
  python3 check_spec_quality.py run --tag quality_remote \\
      --temperatures 0,0.7,1.0 --port 8000

  python3 check_spec_quality.py compare \\
      --baseline bench_results/quality_colocated/t0/result_*.json \\
      --candidate bench_results/quality_remote/t0/result_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "bench_data" / "quality_20.jsonl"


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_one(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {pattern!r}")
    if len(matches) > 1:
        # Prefer the newest.
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


def _load_prompts(dataset: Path) -> list[dict[str, Any]]:
    rows = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_answer(text: str) -> str:
    text = text.strip().replace(",", "")
    # Prefer #### style, else last integer/decimal token.
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return re.sub(r"[^\d\.\-]+", "", m.group(1)) or m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else text.strip().lower()


def _is_garbage(text: str | None) -> tuple[bool, str]:
    if text is None:
        return True, "missing_text"
    t = text.strip()
    if len(t) < 8:
        return True, "too_short"
    # Heavy character repetition (e.g. aaaaa...)
    if re.search(r"(.)\1{40,}", t):
        return True, "char_repeat"
    # Same short token repeated many times
    toks = t.split()
    if len(toks) >= 20 and len(set(toks)) <= 3:
        return True, "token_collapse"
    # Mostly non-printable / replacement chars
    bad = sum(1 for c in t if ord(c) < 9 or c == "\ufffd")
    if bad > 0.05 * len(t):
        return True, "binary_noise"
    return False, "ok"


def _answer_match(text: str, gold: str | None) -> bool | None:
    if gold is None:
        return None
    pred = _normalize_answer(text)
    gold_n = _normalize_answer(str(gold))
    if not pred or not gold_n:
        return False
    if pred.lower() == gold_n.lower():
        return True
    # Numeric tolerance
    try:
        return abs(float(pred) - float(gold_n)) < 1e-6
    except ValueError:
        return gold_n.lower() in text.lower()


def analyze_result(
    result: dict[str, Any],
    prompts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    texts = result.get("generated_texts") or []
    accept_rates = result.get("spec_decode_per_position_acceptance_rates")
    summary: dict[str, Any] = {
        "output_throughput": result.get("output_throughput"),
        "spec_decode_acceptance_rate": result.get("spec_decode_acceptance_rate"),
        "spec_decode_acceptance_length": result.get(
            "spec_decode_acceptance_length"
        ),
        "spec_decode_per_position_acceptance_rates": accept_rates,
        "num_prompts": result.get("num_prompts") or len(texts),
        "temperature": result.get("temperature"),
        "garbage": [],
        "answer_correct": 0,
        "answer_total": 0,
        "answer_wrong": [],
    }

    golds = [None] * len(texts)
    if prompts:
        # Bench order follows dataset order when shuffle is disabled; we pass
        # --disable-shuffle via env/metadata when possible. Fall back to index.
        for i in range(min(len(texts), len(prompts))):
            golds[i] = prompts[i].get("answer")

    for i, text in enumerate(texts):
        bad, reason = _is_garbage(text if isinstance(text, str) else None)
        if bad:
            summary["garbage"].append({"idx": i, "reason": reason, "preview": (text or "")[:160]})
        matched = _answer_match(text or "", golds[i] if i < len(golds) else None)
        if matched is not None:
            summary["answer_total"] += 1
            if matched:
                summary["answer_correct"] += 1
            else:
                summary["answer_wrong"].append(
                    {
                        "idx": i,
                        "gold": golds[i],
                        "pred": _normalize_answer(text or ""),
                        "preview": (text or "")[:200],
                    }
                )
    return summary


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_rate_delta: float,
    max_len_delta: float,
) -> int:
    b_rates = baseline.get("spec_decode_per_position_acceptance_rates") or []
    c_rates = candidate.get("spec_decode_per_position_acceptance_rates") or []
    print("=== Acceptance rates ===")
    print(
        f"baseline acc_rate={baseline.get('spec_decode_acceptance_rate')} "
        f"acc_len={baseline.get('spec_decode_acceptance_length')}"
    )
    print(
        f"candidate acc_rate={candidate.get('spec_decode_acceptance_rate')} "
        f"acc_len={candidate.get('spec_decode_acceptance_length')}"
    )
    if not b_rates or not c_rates:
        print("ERROR: missing per-position acceptance rates in one or both results")
        return 2
    if len(b_rates) != len(c_rates):
        print(
            f"ERROR: position-rate length mismatch "
            f"{len(b_rates)} vs {len(c_rates)}"
        )
        return 2

    ok = True
    print("pos  baseline  candidate  delta")
    for i, (b, c) in enumerate(zip(b_rates, c_rates)):
        d = abs(float(c) - float(b))
        flag = "OK" if d <= max_rate_delta else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"{i:>3}  {b:8.4f}  {c:8.4f}  {d:6.4f}  {flag}")

    b_len = float(baseline.get("spec_decode_acceptance_length") or 0)
    c_len = float(candidate.get("spec_decode_acceptance_length") or 0)
    len_delta = abs(c_len - b_len)
    print(
        f"\nacceptance_length delta={len_delta:.4f} "
        f"(limit {max_len_delta}) "
        f"{'OK' if len_delta <= max_len_delta else 'FAIL'}"
    )
    if len_delta > max_len_delta:
        ok = False

    # Optional garbage / answer stats if present in side-car summaries
    for label, res in ("baseline", baseline), ("candidate", candidate):
        g = res.get("_quality_summary")
        if g:
            print(
                f"{label}: garbage={len(g.get('garbage', []))} "
                f"answer={g.get('answer_correct')}/{g.get('answer_total')}"
            )

    return 0 if ok else 1


def _find_tokenizer(model: str) -> str:
    model_cache = f"models--{model.replace('/', '--')}"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snap = hub / "hub" / model_cache / "snapshots"
    if snap.is_dir():
        kids = [p for p in snap.iterdir() if p.is_dir()]
        if kids:
            return str(kids[0])
    return model


def run_bench_for_temp(
    *,
    tag: str,
    temperature: float,
    dataset: Path,
    model: str,
    port: int,
    output_len: int,
    num_prompts: int,
    request_rate: float,
    seed: int,
) -> Path:
    t_tag = f"t{str(temperature).replace('.', 'p')}"
    run_dir = SCRIPT_DIR / "bench_results" / tag / t_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = f"result_{tag}_{t_tag}.json"
    tokenizer = _find_tokenizer(model)
    base_url = f"http://localhost:{port}"

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset),
        "--custom-output-len",
        str(output_len),
        "--seed",
        str(seed),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--burstiness",
        "1",
        "--temperature",
        str(temperature),
        "--disable-shuffle",
        "--num-warmups",
        "0",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        out_file,
        "--metadata",
        f"tag={tag}",
        f"temperature={temperature}",
        f"dataset={dataset.name}",
    ]
    print("Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))
    result_path = run_dir / out_file
    if not result_path.exists():
        # vllm may prefix/date; pick newest json
        jsons = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not jsons:
            raise FileNotFoundError(f"No result JSON in {run_dir}")
        result_path = jsons[-1]
    return result_path


def cmd_run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    if not dataset.exists():
        # Auto-create quality_20 if missing.
        prep = SCRIPT_DIR / "prepare_math_datasets.py"
        subprocess.run(
            [sys.executable, str(prep), "--skip-download", "--out-dir", str(dataset.parent)],
            check=True,
        )
    prompts = _load_prompts(dataset)
    n = args.num_prompts if args.num_prompts > 0 else len(prompts)
    temps = [float(x) for x in args.temperatures.split(",") if x.strip()]

    summaries = []
    for temp in temps:
        result_path = run_bench_for_temp(
            tag=args.tag,
            temperature=temp,
            dataset=dataset,
            model=args.model,
            port=args.port,
            output_len=args.output_len,
            num_prompts=n,
            request_rate=args.request_rate,
            seed=args.seed,
        )
        result = _load_json(result_path)
        # Attach temperature if missing
        result.setdefault("temperature", temp)
        q = analyze_result(result, prompts[:n])
        result["_quality_summary"] = q
        side = result_path.with_name(result_path.stem + "_quality.json")
        with side.open("w", encoding="utf-8") as f:
            json.dump({"result_path": str(result_path), **q}, f, indent=2)
        # Rewrite result with summary for easier compare
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\n=== temp={temp} -> {result_path} ===")
        print(
            f"acc_rate={q['spec_decode_acceptance_rate']} "
            f"acc_len={q['spec_decode_acceptance_length']}"
        )
        print(f"per_pos={q['spec_decode_per_position_acceptance_rates']}")
        print(f"garbage={len(q['garbage'])}/{q['num_prompts']}")
        if q["answer_total"]:
            print(f"answer_match={q['answer_correct']}/{q['answer_total']}")
        if q["garbage"]:
            print("garbage samples:", q["garbage"][:3])
        summaries.append((temp, q, result_path))

    # Cross-temp acceptance should stay in the same ballpark (not identical).
    if len(summaries) >= 2:
        print("\n=== Cross-temperature acceptance length ===")
        for temp, q, _ in summaries:
            print(f"  t={temp}: acc_len={q['spec_decode_acceptance_length']}")

    # Fail if any temp produced mostly garbage.
    hard_fail = any(len(q["garbage"]) > max(1, q["num_prompts"] // 5) for _, q, _ in summaries)
    return 1 if hard_fail else 0


def cmd_compare(args: argparse.Namespace) -> int:
    b_path = _resolve_one(args.baseline)
    c_path = _resolve_one(args.candidate)
    baseline = _load_json(b_path)
    candidate = _load_json(c_path)
    print(f"baseline:  {b_path}")
    print(f"candidate: {c_path}")
    return compare_results(
        baseline,
        candidate,
        max_rate_delta=args.max_rate_delta,
        max_len_delta=args.max_len_delta,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run quality prompts at one or more temperatures")
    r.add_argument("--tag", required=True)
    r.add_argument("--model", default="openai/gpt-oss-20b")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    r.add_argument("--temperatures", default="0,0.7,1.0")
    r.add_argument("--output-len", type=int, default=512)
    r.add_argument("--num-prompts", type=int, default=0, help="0 = all in dataset")
    r.add_argument("--request-rate", type=float, default=4.0)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Compare acceptance rates of two result JSONs")
    c.add_argument("--baseline", required=True, help="Path or glob")
    c.add_argument("--candidate", required=True, help="Path or glob")
    c.add_argument(
        "--max-rate-delta",
        type=float,
        default=0.05,
        help="Max abs delta per position acceptance rate",
    )
    c.add_argument(
        "--max-len-delta",
        type=float,
        default=0.25,
        help="Max abs delta for mean acceptance length",
    )
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
