#!/usr/bin/env bash
# Launch vLLM servers for PD / PD+SD / P/D-disagg bench topologies.
#
# Examples:
#   ./launch_bench_server.sh --list
#   ./launch_bench_server.sh PD1_b8192 --devices 0 --port 8000
#   ./launch_bench_server.sh PD2S1 --devices 0,1 --port 8000 --batched-tokens 8192
#   ./launch_bench_server.sh P2_D2S1 \
#       --prefill-devices 0,1 --decode-devices 2,3 \
#       --prefill-port 8100 --decode-port 8200 --batched-tokens 16384
#   ./launch_bench_server.sh PD4S4_b4096 --print-only
#   ./launch_bench_server.sh PD1 --devices 0 --port 8000 -- --max-model-len 8192
#   # Milestone-2 HS NIXL remote-only draft (verify GPU0; sink draft on GPU1):
#   ./launch_bench_server.sh PD1S1_b8192 --devices 0 --draft-devices 1 --hs-nixl-sink --nsys
#
# Case grammar:
#   PD{tp}[S{draft_tp}][_b{batched}]     colocated prefill+decode
#   P{ptp}_D{dtp}[S{draft_tp}][_b{batched}]   PD disagg (SD on decode)
#
# Defaults match recent gpt-oss-20b / DFlash / Nixl runs on this machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-openai/gpt-oss-20b}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/gpt-oss-20b-DFlash}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32786}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5600}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5601}"
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
PROXY_PORT="${PROXY_PORT:-8000}"
# Disagg-DFlash (PV*S*): remote draft server bind + verify connect addr.
DRAFT_BIND="${DRAFT_BIND:-tcp://0.0.0.0:50051}"
DRAFT_ADDR="${DRAFT_ADDR:-tcp://127.0.0.1:50051}"
# Prometheus /metrics for HS NIXL sink draft KV (benchmark_random --draft-metrics-url).
DRAFT_METRICS_PORT="${DRAFT_METRICS_PORT:-9101}"
DISAGG_DFLASH_TRANSPORT="${DISAGG_DFLASH_TRANSPORT:-nixl}"
DISAGG_ASYNC=1
# Milestone-2: colocated PD*S* + NIXL HS sink on --draft-devices (remote-only draft).
HS_NIXL_SINK=0
# Legacy Milestone-1 dual-run compare (requires remote_only=false; unused by default).
DUAL_RUN_CHECK=0
REMOTE_ONLY=1
# Milestone-4: non-blocking remote propose + real spec_token_ids ready gating.
ASYNC_VERIFY=0
# Off by default: SD timing / Disagg profile use CUDA synchronize and skew TPOT.
ENABLE_DISAGG_PROFILE=0
# Default: synthetic rejection (paper A/B latency). --no-synthetic → real
# draft↔target reject sampling (standard) for quality / correctness checks.
USE_SYNTHETIC=1
# Live terminal + file under startup_logs/<tag>_<role>_<timestamp>.log
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/startup_logs}"
# Nsight Systems (opt-in): wrap GPU roles with nsys profile.
# Default dir (after TAG is known): bench_results/<tag>/nsys/ — next to
# benchmark_random.sh outputs so profiles are not overwritten across tags.
NSYS=0
# Preserve NSYS_DIR from the environment if the user exported it.
NSYS_DIR_ENV="${NSYS_DIR:-}"
NSYS_DIR=""
NSYS_DIR_SET=0
NSYS_BIN="${NSYS_BIN:-nsys}"
# Timed capture for roles without cudaProfilerApi (draft). kill=none keeps
# the process alive after the window so benches can finish.
NSYS_DELAY="${NSYS_DELAY:-90}"
NSYS_DURATION="${NSYS_DURATION:-30}"
# Off by default: needs CAP_SYS_ADMIN / nvidia counter perms (ERR_NVGPUCTRPERM).
# Kernel + CUDA-graph traces still work without it.
NSYS_GPU_METRICS=0
NSYS_NVTX=1
NSYS_RUN_TS=""
# PV*S* Disagg-DFlash: one nsys profile over a parent that starts draft+verify
# → single combined .nsys-rep (nsys start/launch multi-process is flaky on 2025.1).
NSYS_COMBINED_OUT=""

# Synthetic acceptance rates used in prior PD*S* sweeps.
SYNTH_RATES='[0.9184485330681254,0.8179331856606844,0.7422584874101532,0.6783825324352425,0.6191175805795398,0.5591293341169025,0.49742326296279554]'

CASE=""
BATCHED=""
DEVICES=""
PORT=""
PREFILL_DEVICES=""
DECODE_DEVICES=""
DRAFT_DEVICES=""
PREFILL_PORT="8100"
DECODE_PORT="8200"
PRINT_ONLY=0
DO_PROXY=0
LIST_ONLY=0
ENABLE_LOGGING_ITERATION_DETAILS=0
EXTRA_ARGS=()

CASES=(
  PD1 PD2 PD4
  PD1S1 PD2S2 PD4S4
  PD2S1 PD4S1
  P1_D1 P2_D2
  P1_D1S1 P2_D2S2 P2_D2S1
  PV1S1 PV2S1
)
BATCHED_SWEEP=(4096 8192 16384)

usage() {
  cat <<'EOF'
Usage:
  launch_bench_server.sh <CASE[_bBATCHED]> [options] [-- extra vllm args...]
  launch_bench_server.sh --list

Options:
  --batched-tokens N       max-num-batched-tokens for vllm serve / verify
                           (alias: --max-num-batched-tokens; or use _bN in CASE).
                           Not passed to Disagg-DFlash draft server.
  --max-num-seqs N         max-num-seqs (default: 600, or MAX_NUM_SEQS env)
  --devices IDS            CUDA_VISIBLE_DEVICES for colocated / verify (e.g. 0,1)
  --port PORT              HTTP port for colocated / verify server (default 8000)
  --prefill-devices IDS    GPUs for prefill (P/D disagg)
  --decode-devices IDS     GPUs for decode (P/D disagg)
  --draft-devices IDS      GPUs for Disagg-DFlash draft server (PV*S*)
  --prefill-port PORT      Prefill HTTP port (default 8100)
  --decode-port PORT       Decode HTTP port (default 8200)
  --draft-bind ADDR        Draft server bind (default tcp://0.0.0.0:50051)
  --draft-addr ADDR        Verify→draft/sink connect addr (default tcp://127.0.0.1:50051)
  --hs-nixl-sink           Milestone-2: start HS NIXL sink on --draft-devices and
                           set speculative_config.disagg_dflash_address=$DRAFT_ADDR
                           + disagg_dflash_remote_only=true (colocated PD*S* only;
                           verify skips local draft; serves sink draft tokens)
  --no-remote-only         With --hs-nixl-sink: Milestone-1 dual-run (local draft
                           still serves; remote draft for compare/profiling only)
  --dual-run-check         With --hs-nixl-sink --no-remote-only: compare remote vs
                           local draft tokens (adds GPU sync)
  --async-verify           With --hs-nixl-sink remote-only: Milestone-4 async
                           propose (disagg_dflash_async_verify=true; no [-1]*K
                           placeholders; schedule SD only when draft ids ready)
  --no-disagg-async        disagg_dflash_async_complete=false (sync propose)
  --no-synthetic           real rejection sampling (rejection_sample_method=
                           standard). Default is synthetic rates for paper A/B.
                           Use for curl quality checks on PV*S* / PD*S*.
  --enable-disagg-profile  Opt-in SD timing + Disagg/DFlash profile logs
                           (CUDA sync — skews latency; off by default for benches)
  --proxy                  Also launch toy_proxy_server.py on --proxy-port
  --proxy-port PORT        Client-facing proxy port (default 8000)
  --proxy-script PATH      Override path to toy_proxy_server.py
  --log-dir DIR            Where to write startup logs (default: ./startup_logs)
  --enable-logging-iteration-details
                           Opt-in: pass to vllm serve and (for PV*S*) draft
                           server. Off by default.
  --nsys                   Wrap GPU servers with Nsight Systems.
                           All modes use cudaProfilerApi capture range —
                           pair with benchmark_random.sh --nsys so the bench
                           window calls /start_profile…/stop_profile.
                           PV*S* and PD*S*+--hs-nixl-sink: one combined report
                           (draft/sink + verify on both GPUs);
                           (bench_results/<tag>/nsys/<tag>_combined_<ts>.nsys-rep).
  --nsys-dir DIR           override nsys output dir
                           (default: <script>/bench_results/<tag>/nsys)
  --nsys-delay SEC         Legacy timed-capture roles only (default 90).
                           Combined modes use cudaProfilerApi like PD.
  --nsys-duration SEC      Legacy timed-capture roles only (default 30; 0=until exit).
                           Combined modes use cudaProfilerApi like PD.
  --nsys-gpu-metrics       Add --gpu-metrics-devices=all (needs nvidia counter
                           privileges; off by default — ERR_NVGPUCTRPERM)
  --nsys-no-nvtx           Do not set VLLM_NVTX_SCOPES_FOR_PROFILING=1
  --print-only             Print commands, do not exec
  --list                   List supported cases / full sweep matrix
  -h, --help               Show this help

Cases:
  PD{tp}[S{draft_tp}][_bN]           colocated prefill+decode (+ optional SD)
  P{ptp}_D{dtp}[S{draft_tp}][_bN]    P/D KV disagg (SD on decode)
  PV{vtp}S{draft_gpus}[_bN]           Disagg-DFlash: verify + remote draft server
                                      e.g. PV1S1_b8192  (verify GPU0, draft GPU1)

Logs:
  stdout+stderr are teed live to the terminal and to
  <log-dir>/<tag>_<role>_YYYYMMDD_HHMMSS.log

Env overrides:
  MODEL DRAFT_MODEL NUM_SPEC_TOKENS GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN
  BLOCK_SIZE NIXL_PREFILL_PORT NIXL_DECODE_PORT LOG_DIR PROXY_WAIT_TIMEOUT
  DRAFT_BIND DRAFT_ADDR DISAGG_DFLASH_TRANSPORT
  NSYS_DIR NSYS_BIN NSYS_DELAY NSYS_DURATION

Nsight tip:
  ./launch_bench_server.sh PD1_b8192 --devices 0 --nsys
  # after ready:
  ./benchmark_random.sh --tag PD1_b8192 --nsys --request-rates 4 \\
      --num-prompts 8 --no-warmup --duration-sec 20

  # Disagg-DFlash overlap (one report, draft+verify GPUs; cudaProfilerApi):
  ./launch_bench_server.sh PV1S1_b8192 --devices 0 --draft-devices 1 --nsys
  ./benchmark_random.sh --tag PV1S1_b8192 --nsys --request-rates 4 \\
      --num-prompts 8 --no-warmup --duration-sec 20
  # Ctrl+C / exit the launcher finalizes
  #   bench_results/PV1S1_b8192/nsys/PV1S1_b8192_combined_<ts>.nsys-rep

  # Milestone-4 async remote-only overlap (HS NIXL sink + non-blocking propose):
  ./launch_bench_server.sh PD1S1_b8192 --devices 0 --draft-devices 1 \\
      --hs-nixl-sink --async-verify --nsys
  ./benchmark_random.sh --tag PD1S1_b8192 --nsys --request-rates 4 \\
      --num-prompts 32 --no-warmup --duration-sec 20
  # In the combined .nsys-rep, look for:
  #   sync baseline: long dflash_hs_nixl_remote_wait on GPU0 while GPU1
  #     runs dflash_sink_generate (serialized)
  #   async: short dflash_hs_nixl_async_kick, then GPU0 target kernels
  #     overlap GPU1 dflash_sink_generate; remote_wait mostly gone
  #     (async_catchup_wait should be rare)
EOF
}

list_matrix() {
  echo "Supported base cases:"
  printf '  %s\n' "${CASES[@]}"
  echo
  echo "Batched-token sweep values: ${BATCHED_SWEEP[*]}"
  echo "Full tag form: <CASE>_b<BATCHED>  e.g. PD2S1_b8192  or  PV1S1_b8192"
  echo
  echo "Disagg-DFlash: PV{verify_tp}S{draft_gpus}_bN  (remote draft via NIXL)"
  echo "  e.g. ./launch_bench_server.sh PV1S1_b8192 --devices 0 --draft-devices 1"
  echo
  echo "=== Full sweep (case × batched) ==="
  for c in "${CASES[@]}"; do
    for b in "${BATCHED_SWEEP[@]}"; do
      echo "  ${c}_b${b}"
    done
  done
}

die() { echo "error: $*" >&2; exit 1; }

# Parse CASE -> MODE / TP / DRAFT_TP / optional embedded batched
parse_case() {
  local raw="$1"
  local base="$raw"
  BATCHED_FROM_CASE=""

  if [[ "$raw" =~ ^(.*)_b([0-9]+)$ ]]; then
    base="${BASH_REMATCH[1]}"
    BATCHED_FROM_CASE="${BASH_REMATCH[2]}"
  fi

  MODE=""
  TP=""
  PREFILL_TP=""
  DECODE_TP=""
  DRAFT_TP=""
  VERIFY_TP=""
  DRAFT_GPUS=""

  if [[ "$base" =~ ^PV([0-9]+)S([0-9]+)$ ]]; then
    MODE=sd_disagg
    VERIFY_TP="${BASH_REMATCH[1]}"
    DRAFT_GPUS="${BASH_REMATCH[2]}"
  elif [[ "$base" =~ ^PD([0-9]+)S([0-9]+)$ ]]; then
    MODE=colocated_sd
    TP="${BASH_REMATCH[1]}"
    DRAFT_TP="${BASH_REMATCH[2]}"
  elif [[ "$base" =~ ^PD([0-9]+)$ ]]; then
    MODE=colocated
    TP="${BASH_REMATCH[1]}"
  elif [[ "$base" =~ ^P([0-9]+)_D([0-9]+)S([0-9]+)$ ]]; then
    MODE=disagg_sd
    PREFILL_TP="${BASH_REMATCH[1]}"
    DECODE_TP="${BASH_REMATCH[2]}"
    DRAFT_TP="${BASH_REMATCH[3]}"
  elif [[ "$base" =~ ^P([0-9]+)_D([0-9]+)$ ]]; then
    MODE=disagg
    PREFILL_TP="${BASH_REMATCH[1]}"
    DECODE_TP="${BASH_REMATCH[2]}"
  else
    die "unrecognized case '$raw' (try --list)"
  fi

  CASE_BASE="$base"
}

default_devices_for_tp() {
  local n="$1"
  case "$n" in
    1) echo "0" ;;
    2) echo "0,1" ;;
    4) echo "0,1,2,3" ;;
    *)
      local ids=()
      local i
      for ((i = 0; i < n; i++)); do ids+=("$i"); done
      (IFS=,; echo "${ids[*]}")
      ;;
  esac
}

default_disagg_devices() {
  # Prefer packing P then D on consecutive GPUs.
  local ptp="$1" dtp="$2"
  local p_ids=() d_ids=()
  local i
  for ((i = 0; i < ptp; i++)); do p_ids+=("$i"); done
  for ((i = 0; i < dtp; i++)); do d_ids+=("$((ptp + i))"); done
  PREFILL_DEVICES_DEFAULT=$(IFS=,; echo "${p_ids[*]}")
  DECODE_DEVICES_DEFAULT=$(IFS=,; echo "${d_ids[*]}")
}

default_sd_disagg_devices() {
  # Verify on first VERIFY_TP GPUs; draft on the next DRAFT_GPUS GPUs.
  local vtp="$1" dgpus="$2"
  local v_ids=() d_ids=()
  local i
  for ((i = 0; i < vtp; i++)); do v_ids+=("$i"); done
  for ((i = 0; i < dgpus; i++)); do d_ids+=("$((vtp + i))"); done
  VERIFY_DEVICES_DEFAULT=$(IFS=,; echo "${v_ids[*]}")
  DRAFT_DEVICES_DEFAULT=$(IFS=,; echo "${d_ids[*]}")
}

count_csv() {
  local s="$1"
  if [[ -z "$s" ]]; then echo 0; return; fi
  awk -F, '{print NF}' <<<"$s"
}

reject_sample_json_fields() {
  # Emits rejection_sample_method (+ synthetic rates when enabled) as JSON
  # fragments inserted into speculative-config objects.
  if [[ "$USE_SYNTHETIC" -eq 1 ]]; then
    printf '"rejection_sample_method":"synthetic","synthetic_acceptance_rates":%s' \
      "$SYNTH_RATES"
  else
    printf '"rejection_sample_method":"standard"'
  fi
}

spec_json() {
  local draft_tp="$1"
  # Colocated SD: set draft_tensor_parallel_size.
  # Optional Milestone-2 HS NIXL remote-only (or M1 dual-run via --no-remote-only).
  if [[ "$HS_NIXL_SINK" -eq 1 ]]; then
    local dual_check="false"
    local remote_only="true"
    local async_verify="false"
    [[ "${DUAL_RUN_CHECK:-0}" -eq 1 ]] && dual_check="true"
    [[ "${REMOTE_ONLY:-1}" -eq 1 ]] || remote_only="false"
    [[ "${ASYNC_VERIFY:-0}" -eq 1 ]] && async_verify="true"
    # dual-run check only meaningful when local draft still runs.
    if [[ "$remote_only" == "true" ]]; then
      dual_check="false"
    else
      # async verify requires remote-only.
      async_verify="false"
    fi
    printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s,%s,"disagg_dflash_address":"%s","disagg_dflash_remote_only":%s,"disagg_dflash_dual_run_check":%s,"disagg_dflash_async_verify":%s}' \
      "$DRAFT_MODEL" "$NUM_SPEC_TOKENS" "$draft_tp" "$(reject_sample_json_fields)" \
      "$DRAFT_ADDR" "$remote_only" "$dual_check" "$async_verify"
  else
    printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s,%s}' \
      "$DRAFT_MODEL" "$NUM_SPEC_TOKENS" "$draft_tp" "$(reject_sample_json_fields)"
  fi
}

spec_json_sd_disagg() {
  # Remote DFlash: verify does not load draft weights; RPCs to draft server.
  local async_c="true"
  local profile="false"
  [[ "$DISAGG_ASYNC" -eq 1 ]] || async_c="false"
  [[ "$ENABLE_DISAGG_PROFILE" -eq 1 ]] && profile="true"
  # Default synthetic acceptance matches colocated PD*S* for paper A/B
  # (latency/overlap, not draft quality). --no-synthetic → standard reject.
  printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,%s,"disagg_dflash_address":"%s","disagg_dflash_transport":"%s","disagg_dflash_cross_step":true,"disagg_dflash_async_complete":%s,"disagg_dflash_profile":%s,"attention_backend":"FLASH_ATTN"}' \
    "$DRAFT_MODEL" "$NUM_SPEC_TOKENS" "$(reject_sample_json_fields)" \
    "$DRAFT_ADDR" "$DISAGG_DFLASH_TRANSPORT" \
    "$async_c" "$profile"
}

kv_json() {
  local role="$1" # kv_producer | kv_consumer
  printf '{"kv_connector":"NixlConnector","kv_role":"%s","kv_buffer_device":"cuda"}' "$role"
}

common_flags() {
  # Shared flags for all roles.
  cat <<EOF
--served-model-name ${MODEL}
--dtype auto
--gpu-memory-utilization ${GPU_MEM_UTIL}
--max-num-seqs ${MAX_NUM_SEQS}
--max-model-len ${MAX_MODEL_LEN}
--max-num-batched-tokens ${BATCHED}
--block-size ${BLOCK_SIZE}
--no-enable-prefix-caching
--enable-mfu-metrics
--enable-auto-tool-choice
--tool-call-parser openai
--reasoning-parser openai_gptoss
--enable-prompt-tokens-details
EOF
}

find_proxy_script() {
  if [[ -n "$PROXY_SCRIPT" && -f "$PROXY_SCRIPT" ]]; then
    echo "$PROXY_SCRIPT"
    return
  fi
  # Nixl PD toy proxy (preferred) — correctly threads kv_transfer_params P→D.
  local candidates=(
    "${SCRIPT_DIR}/../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
    "${HOME}/vllm_2307/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
    "${HOME}/vllm-sd-disagg/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      # Resolve to absolute path for logs / tips.
      (cd "$(dirname "$c")" && echo "$(pwd)/$(basename "$c")")
      return
    fi
  done
  return 1
}

# Wait until HTTP GET url returns 200 (engines ready before proxy).
wait_http_ready() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-${PROXY_WAIT_TIMEOUT:-600}}"
  local start now code
  start=$(date +%s)
  echo "# waiting for ${name} at ${url} (timeout ${timeout_s}s)..."
  while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "$url" || true)
    if [[ "$code" == "200" ]]; then
      echo "# ${name} ready (HTTP 200)"
      return 0
    fi
    now=$(date +%s)
    if [[ $((now - start)) -ge "$timeout_s" ]]; then
      die "${name} not ready after ${timeout_s}s (last HTTP ${code:-000}) -- check its log"
    fi
    sleep 2
  done
}

# Wait until the newest log for role contains a ready needle (draft load).
wait_log_ready() {
  local role="$1"
  local needle="$2"
  local name="$3"
  local timeout_s="${4:-${PROXY_WAIT_TIMEOUT:-600}}"
  local start now logfile
  start=$(date +%s)
  echo "# waiting for ${name} log needle '${needle}' (timeout ${timeout_s}s)..."
  while true; do
    logfile=$(ls -t "${LOG_DIR}/${TAG}_${role}_"*.log 2>/dev/null | head -1 || true)
    if [[ -n "$logfile" ]] && grep -q "$needle" "$logfile" 2>/dev/null; then
      echo "# ${name} ready (log=${logfile})"
      return 0
    fi
    now=$(date +%s)
    if [[ $((now - start)) -ge "$timeout_s" ]]; then
      die "${name} not ready after ${timeout_s}s -- check ${logfile:-${LOG_DIR}/${TAG}_${role}_*.log}"
    fi
    sleep 2
  done
}

shell_join() {
  # Human-readable join; quote only tokens that need it.
  local out="" t
  for t in "$@"; do
    if [[ "$t" =~ [[:space:]|{}\"\'\\] ]]; then
      out+=" $(printf '%q' "$t")"
    else
      out+=" $t"
    fi
  done
  echo "${out# }"
}

# Build nsys profile argv prefix for a role.
# mode=api  → cudaProfilerApi (vllm serve + --profiler-config cuda)
# mode=timed → delay/duration (draft / processes without profiler API)
nsys_wrap_prefix() {
  local role="$1"
  local mode="${2:-api}"
  # Timestamped basename so re-launches of the same tag do not clobber.
  local out="${NSYS_DIR}/${TAG}_${role}_${NSYS_RUN_TS}"
  local -a wrap=(
    "$NSYS_BIN" profile
    -o "$out"
    --trace=cuda,nvtx,osrt,cudnn,cublas
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --cuda-event-trace=false
    --force-overwrite=true
    --kill=none
  )
  if [[ "$NSYS_GPU_METRICS" -eq 1 ]]; then
    wrap+=(--gpu-metrics-devices=all)
  fi
  if [[ "$mode" == api ]]; then
    wrap+=(--capture-range=cudaProfilerApi --capture-range-end=repeat)
  else
    if [[ "${NSYS_DELAY}" -gt 0 ]]; then
      wrap+=(--delay="${NSYS_DELAY}")
    fi
    if [[ "${NSYS_DURATION}" -gt 0 ]]; then
      wrap+=(--duration="${NSYS_DURATION}")
    fi
  fi
  printf '%s\n' "${wrap[@]}"
}

# Resolve argv0 to an absolute path. nsys does not use the full user PATH
# for the application binary (bare "env" / "vllm" → "Not executable").
nsys_resolve_exe() {
  local exe="$1"
  local resolved
  if [[ "$exe" == /* ]]; then
    echo "$exe"
    return 0
  fi
  resolved="$(command -v "$exe" 2>/dev/null || true)"
  if [[ -n "$resolved" && -x "$resolved" ]]; then
    echo "$resolved"
    return 0
  fi
  return 1
}

# Under nsys, /usr/local/gib/lib64's libnccl.so.2 (2.27.x) can win over the
# venv's nvidia-nccl (2.28.x). TP>1 then dies during NCCL init with a silent
# WorkerProc failure. Prefer the active venv's NCCL and drop gib from the path.
nsys_ld_library_path() {
  local venv_nccl="" py site p out=""
  py="$(command -v python3 2>/dev/null || true)"
  if [[ -n "$py" ]]; then
    site="$("$py" - <<'PY' 2>/dev/null || true
import os, sys
try:
    import nvidia.nccl as n
    print(os.path.join(os.path.dirname(n.__file__), "lib"))
except Exception:
    # Fallback: site-packages/nvidia/nccl/lib next to this interpreter.
    for sp in sys.path:
        cand = os.path.join(sp, "nvidia", "nccl", "lib")
        if os.path.exists(os.path.join(cand, "libnccl.so.2")):
            print(cand)
            break
PY
)"
    if [[ -n "$site" && -d "$site" ]]; then
      venv_nccl="$site"
    fi
  fi
  out="$venv_nccl"
  local IFS=':'
  # shellcheck disable=SC2206
  local -a parts=(${LD_LIBRARY_PATH:-})
  for p in "${parts[@]}"; do
    [[ -z "$p" ]] && continue
    [[ "$p" == /usr/local/gib/lib64 ]] && continue
    [[ -n "$venv_nccl" && "$p" == "$venv_nccl" ]] && continue
    if [[ -n "$out" ]]; then
      out+=":$p"
    else
      out="$p"
    fi
  done
  echo "$out"
}

nsys_require_bin() {
  if command -v "$NSYS_BIN" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$PRINT_ONLY" -eq 1 ]]; then
    echo "# warning: '${NSYS_BIN}' not found; printing wrapped command anyway" >&2
    return 0
  fi
  die "nsys not found (NSYS_BIN=${NSYS_BIN}). Install Nsight Systems CLI or set NSYS_BIN."
}

# Mutate cmd array named by $1 into: /usr/bin/env VLLM_... /abs/app args...
# $2=api|timed|session  (api adds --profiler-config cuda)
# $3=role (for error messages)
nsys_finalize_app_cmd() {
  local __nsys_cmd_name="$1"
  local -n _app_cmd="$1"
  local mode="${2:-api}"
  local role="${3:-}"

  local -a env_extra=(VLLM_WORKER_MULTIPROC_METHOD=spawn)
  if [[ "$NSYS_NVTX" -eq 1 ]]; then
    env_extra+=(VLLM_NVTX_SCOPES_FOR_PROFILING=1)
  fi
  local nsys_ld
  nsys_ld="$(nsys_ld_library_path)"
  if [[ -n "$nsys_ld" ]]; then
    env_extra+=("LD_LIBRARY_PATH=${nsys_ld}")
    echo "# nsys LD_LIBRARY_PATH: prefer venv NCCL, drop /usr/local/gib/lib64"
  fi

  local -a body=("${_app_cmd[@]}")
  local -a env_vars=("${env_extra[@]}")
  if [[ "${body[0]}" == env || "${body[0]}" == */env ]]; then
    body=("${body[@]:1}")
  fi
  while [[ ${#body[@]} -gt 0 && "${body[0]}" == *=* && "${body[0]}" != -* ]]; do
    if [[ "${body[0]}" == LD_LIBRARY_PATH=* ]]; then
      body=("${body[@]:1}")
      continue
    fi
    env_vars+=("${body[0]}")
    body=("${body[@]:1}")
  done
  [[ ${#body[@]} -gt 0 ]] || die "nsys wrap: no application left in command for role=${role}"

  local app resolved
  app="${body[0]}"
  if resolved="$(nsys_resolve_exe "$app")"; then
    body[0]="$resolved"
  else
    die "nsys wrap: cannot resolve executable '${app}' to an absolute path (activate the venv?)"
  fi

  local env_bin
  env_bin="$(nsys_resolve_exe env || true)"
  [[ -n "$env_bin" ]] || env_bin="/usr/bin/env"

  # vllm serve: cudaProfilerApi (bench --profile /start_profile…/stop_profile).
  if [[ "$mode" == api ]]; then
    body+=(--profiler-config '{"profiler":"cuda"}')
  fi

  _app_cmd=("$env_bin" "${env_vars[@]}" "${body[@]}")
  unset __nsys_cmd_name
}

# Prepend nsys profile to a cmd array named by $1.
# $2=role  $3=api|timed
maybe_wrap_nsys() {
  local __wrap_name="$1"
  local -n _wrap_cmd="$1"
  local role="$2"
  local mode="${3:-api}"
  [[ "$NSYS" -eq 1 ]] || return 0
  nsys_require_bin

  local -a prefix=()
  local line
  while IFS= read -r line; do
    [[ -n "$line" ]] && prefix+=("$line")
  done < <(nsys_wrap_prefix "$role" "$mode")

  # Pass the caller's array name (not _wrap_cmd) to avoid nameref cycles.
  nsys_finalize_app_cmd "$__wrap_name" "$mode" "$role"
  _wrap_cmd=("${prefix[@]}" "${_wrap_cmd[@]}")
  echo "# nsys: ${NSYS_DIR}/${TAG}_${role}_${NSYS_RUN_TS}.nsys-rep  (mode=${mode})"
}

# ---- Combined profile (PV*S*): one nsys profile → draft + verify in one report ----
# Writes executable run scripts + a parent helper, then wraps the helper with
# `nsys profile --trace-fork-before-exec=true` so both process trees land in
# the same .nsys-rep (avoids nsys start/launch "Process launch is not allowed").

nsys_write_run_script() {
  # $1=path  $2=nameref to argv array
  local path="$1"
  local -n _argv="$2"
  {
    echo '#!/usr/bin/env bash'
    echo 'exec \'
    local i
    for ((i = 0; i < ${#_argv[@]}; i++)); do
      printf '  %q' "${_argv[i]}"
      if [[ "$i" -lt $((${#_argv[@]} - 1)) ]]; then
        echo ' \'
      else
        echo
      fi
    done
  } >"$path"
  chmod +x "$path"
}

nsys_build_combined_helper() {
  # Args: draft_log verify_log helper_path draft_run_script verify_run_script
  # Optional $6: grep needle in draft_log before starting verify (e.g. sink listening).
  # Used for both --nsys (under nsys profile) and plain PV*S* launches so
  # Ctrl+C always tears down draft + verify process groups.
  local draft_log="$1" verify_log="$2" helper="$3" draft_run="$4" verify_run="$5"
  local ready_needle="${6:-}"
  cat >"$helper" <<EOF
#!/usr/bin/env bash
# Auto-generated by launch_bench_server.sh — start draft then verify.
# On SIGINT/SIGTERM/EXIT, kill both servers (and their process groups) so
# draft does not outlive the target. Under --nsys this also lets nsys
# finalize the .nsys-rep cleanly.
set -uo pipefail
draft_log=$(printf '%q' "$draft_log")
verify_log=$(printf '%q' "$verify_log")
draft_run=$(printf '%q' "$draft_run")
verify_run=$(printf '%q' "$verify_run")
ready_needle=$(printf '%q' "$ready_needle")

draft_pid=""
verify_pid=""

cleanup() {
  trap - INT TERM EXIT
  echo "# helper: shutting down draft/verify (pids=\${draft_pid:-none}/\${verify_pid:-none})"
  # Kill process groups rooted at each pipeline leader (covers python children).
  for p in \$draft_pid \$verify_pid; do
    [[ -n "\$p" ]] || continue
    kill -INT "\$p" 2>/dev/null || true
    # Also signal the process group if this pid is a group leader.
    kill -INT -- "-\$p" 2>/dev/null || true
  done
  sleep 2
  for p in \$draft_pid \$verify_pid; do
    [[ -n "\$p" ]] || continue
    kill -TERM "\$p" 2>/dev/null || true
    kill -TERM -- "-\$p" 2>/dev/null || true
    kill -KILL "\$p" 2>/dev/null || true
    kill -KILL -- "-\$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

{
  echo "# launched: \$(date -Is)"
  echo "# role=draft (disagg pair helper)"
  echo "# ----"
} >"\$draft_log"
{
  echo "# launched: \$(date -Is)"
  echo "# role=verify (disagg pair helper)"
  echo "# ----"
} >"\$verify_log"

# Each role in its own session/process-group (setsid) so cleanup can
# signal -\$pid and tear down python children, not just the tee.
# IMPORTANT: do not use \$(start_role ...) — that runs in a subshell and
# re-parents the servers so wait/cleanup cannot see them.
start_role() {
  local run=\$1 log=\$2
  if command -v stdbuf >/dev/null 2>&1; then
    setsid bash -c 'stdbuf -oL -eL "\$0" 2>&1 | tee -a "\$1"' "\$run" "\$log" &
  else
    setsid bash -c '"\$0" 2>&1 | tee -a "\$1"' "\$run" "\$log" &
  fi
  _role_pid=\$!
}

start_role "\$draft_run" "\$draft_log"
draft_pid=\$_role_pid
echo "# draft pid=\$draft_pid"
if [[ -n "\$ready_needle" ]]; then
  echo "# waiting for draft ready needle: \$ready_needle"
  _t0=\$(date +%s)
  while ! grep -q "\$ready_needle" "\$draft_log" 2>/dev/null; do
    if [[ \$(( \$(date +%s) - _t0 )) -ge 600 ]]; then
      echo "# ERROR: draft not ready after 600s (needle=\$ready_needle)" >&2
      exit 1
    fi
    sleep 2
  done
  echo "# draft ready"
else
  sleep 2
fi
start_role "\$verify_run" "\$verify_log"
verify_pid=\$_role_pid
echo "# verify pid=\$verify_pid"
wait
EOF
  chmod +x "$helper"
}

# Build combined nsys profile argv into nameref $1 (draft_cmd/verify_cmd already set).
# Mutates draft_cmd/verify_cmd via nsys_finalize_app_cmd; sets NSYS_COMBINED_OUT.
# Optional $2: draft-log ready needle before starting verify.
nsys_prepare_combined_profile_cmd() {
  local -n _out_cmd="$1"
  local ready_needle="${2:-}"
  nsys_require_bin
  mkdir -p "$NSYS_DIR" "$LOG_DIR"
  NSYS_COMBINED_OUT="${NSYS_DIR}/${TAG}_combined_${NSYS_RUN_TS}"

  # session: NVTX env only (no --profiler-config). api: verify gets cuda profiler.
  nsys_finalize_app_cmd draft_cmd session draft
  nsys_finalize_app_cmd verify_cmd api verify

  local draft_log verify_log draft_run verify_run helper
  draft_log="${LOG_DIR}/${TAG}_draft_${NSYS_RUN_TS}.log"
  verify_log="${LOG_DIR}/${TAG}_verify_${NSYS_RUN_TS}.log"
  draft_run="${LOG_DIR}/${TAG}_draft_run_${NSYS_RUN_TS}.sh"
  verify_run="${LOG_DIR}/${TAG}_verify_run_${NSYS_RUN_TS}.sh"
  helper="${LOG_DIR}/${TAG}_nsys_combined_helper_${NSYS_RUN_TS}.sh"

  nsys_write_run_script "$draft_run" draft_cmd
  nsys_write_run_script "$verify_run" verify_cmd
  nsys_build_combined_helper "$draft_log" "$verify_log" "$helper" "$draft_run" "$verify_run" \
    "$ready_needle"

  local bash_bin
  bash_bin="$(nsys_resolve_exe bash || true)"
  [[ -n "$bash_bin" ]] || bash_bin="/bin/bash"

  # --wait=primary: do not hang on re-parented EngineCore/draft children.
  # cudaProfilerApi: only the bench window (benchmark_random.sh --nsys →
  # /start_profile…/stop_profile). Verify mirrors PROFILE over ZMQ so the
  # sink also calls cuda.profiler.start/stop (else GPU1 shows PtoP only).
  _out_cmd=(
    "$NSYS_BIN" profile
    -o "$NSYS_COMBINED_OUT"
    --trace=cuda,nvtx,osrt,cudnn,cublas
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --cuda-event-trace=false
    --force-overwrite=true
    --wait=primary
    --sample=none
    --capture-range=cudaProfilerApi
    --capture-range-end=repeat
  )
  if [[ "$NSYS_GPU_METRICS" -eq 1 ]]; then
    _out_cmd+=(--gpu-metrics-devices=all)
  fi
  _out_cmd+=("$bash_bin" "$helper")

  echo "# nsys combined output: ${NSYS_COMBINED_OUT}.nsys-rep"
  echo "# nsys: cudaProfilerApi — pair bench with --nsys (no startup/idle fluff)"
  echo "# draft log:  ${draft_log}"
  echo "# verify log: ${verify_log}"
  echo "# helper:     ${helper}"
}

# Plain (non-nsys) PV*S*: same draft+verify helper as --nsys so Ctrl+C kills both.
run_sd_disagg_pair_foreground() {
  local ts draft_log verify_log draft_run verify_run helper
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$LOG_DIR"
  draft_log="${LOG_DIR}/${TAG}_draft_${ts}.log"
  verify_log="${LOG_DIR}/${TAG}_verify_${ts}.log"
  draft_run="${LOG_DIR}/${TAG}_draft_run_${ts}.sh"
  verify_run="${LOG_DIR}/${TAG}_verify_run_${ts}.sh"
  helper="${LOG_DIR}/${TAG}_disagg_helper_${ts}.sh"

  nsys_write_run_script "$draft_run" draft_cmd
  nsys_write_run_script "$verify_run" verify_cmd
  nsys_build_combined_helper "$draft_log" "$verify_log" "$helper" "$draft_run" "$verify_run"

  echo
  echo "# ---- ${TAG} DRAFT+VERIFY draft=${DRAFT_DEVICES} verify=${DEVICES} port=${PORT} ----"
  echo "# draft log:  ${draft_log}"
  echo "# verify log: ${verify_log}"
  echo "# helper:     ${helper}"
  echo "# tip: Ctrl+C kills draft and verify (same as --nsys combined helper)"
  shell_join bash "$helper"
  if [[ "$PRINT_ONLY" -eq 1 ]]; then
    return 0
  fi
  bash "$helper"
}

run_cmd() {
  # $1=human description  $2=role slug for logfile  rest=command
  local desc="$1"
  local role="$2"
  shift 2
  local ts logfile
  ts=$(date +%Y%m%d_%H%M%S)
  logfile="${LOG_DIR}/${TAG}_${role}_${ts}.log"

  echo
  echo "# ---- ${desc} ----"
  echo "# log -> ${logfile}"
  shell_join "$@"
  if [[ "$PRINT_ONLY" -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$LOG_DIR"
  if [[ "$NSYS" -eq 1 ]]; then
    mkdir -p "$NSYS_DIR"
  fi
  # Header in the log file for later attribution.
  {
    echo "# launched: $(date -Is)"
    echo "# tag=${TAG} role=${role}"
    echo "# cmd: $(shell_join "$@")"
    echo "# ----"
  } >"$logfile"
  # Live on terminal + append to file (stdout and stderr).
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$@" 2>&1 | tee -a "$logfile" &
  else
    "$@" 2>&1 | tee -a "$logfile" &
  fi
  echo "# pid=$!"
}

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list) LIST_ONLY=1; shift ;;
    --print-only) PRINT_ONLY=1; shift ;;
    --proxy) DO_PROXY=1; shift ;;
    --enable-logging-iteration-details) ENABLE_LOGGING_ITERATION_DETAILS=1; shift ;;
    --batched-tokens|--max-num-batched-tokens) BATCHED="${2:?}"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="${2:?}"; shift 2 ;;
    --devices) DEVICES="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --prefill-devices) PREFILL_DEVICES="${2:?}"; shift 2 ;;
    --decode-devices) DECODE_DEVICES="${2:?}"; shift 2 ;;
    --draft-devices) DRAFT_DEVICES="${2:?}"; shift 2 ;;
    --prefill-port) PREFILL_PORT="${2:?}"; shift 2 ;;
    --decode-port) DECODE_PORT="${2:?}"; shift 2 ;;
    --draft-bind) DRAFT_BIND="${2:?}"; shift 2 ;;
    --draft-addr) DRAFT_ADDR="${2:?}"; shift 2 ;;
    --hs-nixl-sink) HS_NIXL_SINK=1; shift ;;
    --no-remote-only) REMOTE_ONLY=0; shift ;;
    --dual-run-check) DUAL_RUN_CHECK=1; shift ;;
    --async-verify) ASYNC_VERIFY=1; shift ;;
    --no-disagg-async) DISAGG_ASYNC=0; shift ;;
    --no-synthetic) USE_SYNTHETIC=0; shift ;;
    --enable-disagg-profile) ENABLE_DISAGG_PROFILE=1; shift ;;
    --nsys) NSYS=1; shift ;;
    --nsys-dir) NSYS_DIR="${2:?}"; NSYS_DIR_SET=1; shift 2 ;;
    --nsys-delay) NSYS_DELAY="${2:?}"; shift 2 ;;
    --nsys-duration) NSYS_DURATION="${2:?}"; shift 2 ;;
    --nsys-gpu-metrics) NSYS_GPU_METRICS=1; shift ;;
    --nsys-no-gpu-metrics) NSYS_GPU_METRICS=0; shift ;;
    --nsys-no-nvtx) NSYS_NVTX=0; shift ;;
    --proxy-port) PROXY_PORT="${2:?}"; shift 2 ;;
    --proxy-script) PROXY_SCRIPT="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -*)
      die "unknown option: $1 (use -- for extra vllm args)"
      ;;
    *)
      if [[ -n "$CASE" ]]; then
        die "unexpected positional: $1 (case already set to $CASE)"
      fi
      CASE="$1"
      shift
      ;;
  esac
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_matrix
  exit 0
fi

[[ -n "$CASE" ]] || { usage; die "CASE is required"; }

# Resolve batched tokens: flag wins, else _bN, else error.
parse_case "$CASE"
if [[ -z "$BATCHED" ]]; then
  if [[ -n "$BATCHED_FROM_CASE" ]]; then
    BATCHED="$BATCHED_FROM_CASE"
  else
    die "set --batched-tokens N or use ${CASE_BASE}_bN"
  fi
fi
# Recompute tag with resolved batched
TAG="${CASE_BASE}_b${BATCHED}"

echo "# case=${CASE_BASE}  tag=${TAG}  mode=${MODE}  batched=${BATCHED}"
if [[ "$MODE" == sd_disagg || "$MODE" == colocated_sd || "$MODE" == disagg_sd ]]; then
  if [[ "$USE_SYNTHETIC" -eq 1 ]]; then
    echo "# rejection_sample=synthetic (paper A/B). For real quality: --no-synthetic"
  else
    echo "# rejection_sample=standard (real draft↔target reject; curl-friendly)"
  fi
fi
if [[ "$NSYS" -eq 1 ]]; then
  # --nsys-dir > env NSYS_DIR > bench_results/<tag>/nsys
  if [[ "$NSYS_DIR_SET" -eq 0 ]]; then
    if [[ -n "$NSYS_DIR_ENV" ]]; then
      NSYS_DIR="$NSYS_DIR_ENV"
    else
      NSYS_DIR="${SCRIPT_DIR}/bench_results/${TAG}/nsys"
    fi
  fi
  NSYS_RUN_TS="$(date +%Y%m%d_%H%M%S)"
  echo "# nsys=ON  dir=${NSYS_DIR}  run_ts=${NSYS_RUN_TS}"
  if [[ "$MODE" == sd_disagg ]]; then
    echo "# nsys PV*S*: combined cudaProfilerApi → ${TAG}_combined_${NSYS_RUN_TS}.nsys-rep"
    echo "# pair with: ./benchmark_random.sh --tag ${TAG} --nsys ..."
  elif [[ "$MODE" == colocated_sd && "$HS_NIXL_SINK" -eq 1 ]]; then
    echo "# nsys PD*S*+hs-nixl-sink: combined → ${TAG}_combined_${NSYS_RUN_TS}.nsys-rep"
    echo "# pair with: ./benchmark_random.sh --tag ${TAG} --nsys ..."
  else
    echo "# nsys draft_delay=${NSYS_DELAY}s  draft_duration=${NSYS_DURATION}s"
    echo "# pair with: ./benchmark_random.sh --tag ${TAG} --nsys ..."
  fi
fi

# ---------- build + launch ----------
COMMON=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # shellcheck disable=SC2206
  COMMON+=($line)
done < <(common_flags)
COMMON+=("${EXTRA_ARGS[@]}")
if [[ "$ENABLE_LOGGING_ITERATION_DETAILS" -eq 1 ]]; then
  COMMON+=(--enable-logging-iteration-details)
fi

case "$MODE" in
  colocated|colocated_sd)
    PORT="${PORT:-8000}"
    DEVICES="${DEVICES:-$(default_devices_for_tp "$TP")}"
    n_dev=$(count_csv "$DEVICES")
    [[ "$n_dev" -eq "$TP" ]] || die "PD${TP} needs ${TP} devices, got '${DEVICES}' (${n_dev})"

    sink_cmd=()
    if [[ "$MODE" == colocated_sd && "$HS_NIXL_SINK" -eq 1 ]]; then
      # Sink on the first GPU after the verify TP set (PD1S1→1, PD2S1→2).
      # Hardcoding 1 collides with verify TP rank1 when TP>1.
      if [[ -z "${DRAFT_DEVICES:-}" ]]; then
        last_verify_dev=$(awk -F, '{print $NF}' <<<"$DEVICES")
        DRAFT_DEVICES=$((last_verify_dev + 1))
      fi
      echo "# HS NIXL sink: devices=${DRAFT_DEVICES} bind=${DRAFT_BIND} addr=${DRAFT_ADDR}"
      echo "# HS NIXL sink metrics: http://127.0.0.1:${DRAFT_METRICS_PORT}/metrics"
      echo "# pair with: ./benchmark_random.sh ... --draft-metrics-url http://127.0.0.1:${DRAFT_METRICS_PORT}"
      sink_cmd=(
        env
        HF_HUB_OFFLINE=1
        "CUDA_VISIBLE_DEVICES=${DRAFT_DEVICES}"
        python -m vllm.entrypoints.dflash_hs_nixl_sink
        --bind "$DRAFT_BIND"
        --device cuda:0
        --draft-model "$DRAFT_MODEL"
        --target-model "$MODEL"
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$BATCHED"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --block-size "$BLOCK_SIZE"
        --metrics-port "$DRAFT_METRICS_PORT"
      )
    fi

    cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      "CUDA_VISIBLE_DEVICES=${DEVICES}"
      vllm serve "$MODEL"
      --port "$PORT"
      --tensor-parallel-size "$TP"
      "${COMMON[@]}"
    )
    if [[ "$MODE" == colocated_sd ]]; then
      cmd+=(--speculative-config "$(spec_json "$DRAFT_TP")")
    fi

    # PD*S* + --hs-nixl-sink + --nsys: one combined .nsys-rep (GPU0+GPU1), like PV*S*.
    if [[ "$NSYS" -eq 1 && ${#sink_cmd[@]} -gt 0 ]]; then
      draft_cmd=("${sink_cmd[@]}")
      verify_cmd=("${cmd[@]}")
      combined_cmd=()
      nsys_prepare_combined_profile_cmd combined_cmd "DFlash HS NIXL sink listening"
      combined_log="${LOG_DIR}/${TAG}_combined_${NSYS_RUN_TS}.log"
      echo
      echo "# ---- ${TAG} COMBINED nsys sink=${DRAFT_DEVICES} verify=${DEVICES} port=${PORT} ----"
      echo "# log -> ${combined_log}"
      shell_join "${combined_cmd[@]}"
      echo
      echo "# nsys combined: ${NSYS_COMBINED_OUT}.nsys-rep  (GPU0 verify + GPU1 sink)"
      echo "# run: ./benchmark_random.sh --tag ${TAG} --nsys ...  (required for capture)"
      echo "# then Ctrl+C here to finalize the .nsys-rep"
      if [[ "$PRINT_ONLY" -eq 1 ]]; then
        :
      else
        mkdir -p "$LOG_DIR" "$NSYS_DIR"
        {
          echo "# launched: $(date -Is)"
          echo "# tag=${TAG} role=combined"
          echo "# cmd: $(shell_join "${combined_cmd[@]}")"
          echo "# ----"
        } >"$combined_log"
        echo "# tip: one Ctrl+C after the bench — wait for nsys to print Generating..."
        set +e
        if command -v stdbuf >/dev/null 2>&1; then
          stdbuf -oL -eL "${combined_cmd[@]}" > >(tee -a "$combined_log") 2>&1
        else
          "${combined_cmd[@]}" > >(tee -a "$combined_log") 2>&1
        fi
        set -e
        echo "# nsys combined report: ${NSYS_COMBINED_OUT}.nsys-rep"
      fi
    else
      if [[ ${#sink_cmd[@]} -gt 0 ]]; then
        run_cmd "${TAG} hs_nixl_sink devices=${DRAFT_DEVICES}" \
          "hs_nixl_sink" "${sink_cmd[@]}"
        if [[ "$PRINT_ONLY" -eq 0 ]]; then
          wait_log_ready "hs_nixl_sink" "DFlash HS NIXL sink listening" "hs_nixl_sink"
        fi
      fi
      maybe_wrap_nsys cmd colocated api
      run_cmd "${TAG} colocated tp=${TP} devices=${DEVICES} port=${PORT}" \
        "colocated" "${cmd[@]}"
      if [[ "$PRINT_ONLY" -eq 0 ]]; then
        wait
      fi
    fi
    ;;

  disagg|disagg_sd)
    default_disagg_devices "$PREFILL_TP" "$DECODE_TP"
    PREFILL_DEVICES="${PREFILL_DEVICES:-$PREFILL_DEVICES_DEFAULT}"
    DECODE_DEVICES="${DECODE_DEVICES:-$DECODE_DEVICES_DEFAULT}"
    np=$(count_csv "$PREFILL_DEVICES")
    nd=$(count_csv "$DECODE_DEVICES")
    [[ "$np" -eq "$PREFILL_TP" ]] || die "prefill TP=${PREFILL_TP} but --prefill-devices='${PREFILL_DEVICES}'"
    [[ "$nd" -eq "$DECODE_TP" ]] || die "decode TP=${DECODE_TP} but --decode-devices='${DECODE_DEVICES}'"

    p_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PREFILL_PORT}"
      "CUDA_VISIBLE_DEVICES=${PREFILL_DEVICES}"
      vllm serve "$MODEL"
      --port "$PREFILL_PORT"
      --tensor-parallel-size "$PREFILL_TP"
      --kv-transfer-config "$(kv_json kv_producer)"
      "${COMMON[@]}"
    )
    d_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_DECODE_PORT}"
      "CUDA_VISIBLE_DEVICES=${DECODE_DEVICES}"
      vllm serve "$MODEL"
      --port "$DECODE_PORT"
      --tensor-parallel-size "$DECODE_TP"
      --kv-transfer-config "$(kv_json kv_consumer)"
      "${COMMON[@]}"
    )
    if [[ "$MODE" == disagg_sd ]]; then
      d_cmd+=(--speculative-config "$(spec_json "$DRAFT_TP")")
    fi

    maybe_wrap_nsys p_cmd prefill api
    maybe_wrap_nsys d_cmd decode api
    run_cmd "${TAG} PREFILL tp=${PREFILL_TP} devices=${PREFILL_DEVICES} port=${PREFILL_PORT}" \
      "prefill" "${p_cmd[@]}"
    run_cmd "${TAG} DECODE  tp=${DECODE_TP} devices=${DECODE_DEVICES} port=${DECODE_PORT}" \
      "decode" "${d_cmd[@]}"

    if [[ "$DO_PROXY" -eq 1 ]]; then
      pscript=$(find_proxy_script) || die "toy_proxy_server.py not found; set --proxy-script"
      if [[ "$PRINT_ONLY" -eq 0 ]]; then
        # Wait for engines before proxy; toy_proxy does not probe /v1/models itself.
        wait_http_ready "http://localhost:${PREFILL_PORT}/v1/models" "prefill" \
          "${PROXY_WAIT_TIMEOUT:-600}"
        wait_http_ready "http://localhost:${DECODE_PORT}/v1/models" "decode" \
          "${PROXY_WAIT_TIMEOUT:-600}"
      fi
      # CLI: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
      px_cmd=(
        python3 "$pscript"
        --host 127.0.0.1
        --port "$PROXY_PORT"
        --prefiller-hosts localhost
        --prefiller-ports "$PREFILL_PORT"
        --decoder-hosts localhost
        --decoder-ports "$DECODE_PORT"
      )
      run_cmd "${TAG} PROXY port=${PROXY_PORT}" "proxy" "${px_cmd[@]}"
    else
      echo
      echo "# tip: for client traffic, start the Nixl toy proxy, e.g.:"
      echo "#   ./launch_bench_server.sh ${TAG} ... --proxy"
      echo "# or (after engines are up):"
      proxy_path="$(find_proxy_script 2>/dev/null || true)"
      if [[ -n "${proxy_path}" ]]; then
        echo "#   python3 ${proxy_path} --host 127.0.0.1 --port ${PROXY_PORT} \\"
        echo "#       --prefiller-hosts localhost --prefiller-ports ${PREFILL_PORT} \\"
        echo "#       --decoder-hosts localhost --decoder-ports ${DECODE_PORT}"
      fi
    fi

    if [[ "$PRINT_ONLY" -eq 0 ]]; then
      wait
    fi
    ;;

  sd_disagg)
    # Disagg-DFlash: verify (vllm serve) + remote draft_server on separate GPUs.
    PORT="${PORT:-8000}"
    default_sd_disagg_devices "$VERIFY_TP" "$DRAFT_GPUS"
    DEVICES="${DEVICES:-$VERIFY_DEVICES_DEFAULT}"
    DRAFT_DEVICES="${DRAFT_DEVICES:-$DRAFT_DEVICES_DEFAULT}"
    n_v=$(count_csv "$DEVICES")
    n_d=$(count_csv "$DRAFT_DEVICES")
    [[ "$n_v" -eq "$VERIFY_TP" ]] || die "V${VERIFY_TP} needs ${VERIFY_TP} verify devices, got '${DEVICES}'"
    [[ "$n_d" -eq "$DRAFT_GPUS" ]] || die "S${DRAFT_GPUS} needs ${DRAFT_GPUS} draft devices, got '${DRAFT_DEVICES}'"

    draft_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "CUDA_VISIBLE_DEVICES=${DRAFT_DEVICES}"
      python3 -m vllm.entrypoints.dflash_draft_server
      --draft-model "$DRAFT_MODEL"
      --target-model "$MODEL"
      --num-speculative-tokens "$NUM_SPEC_TOKENS"
      --bind "$DRAFT_BIND"
      --transport "$DISAGG_DFLASH_TRANSPORT"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --block-size "$BLOCK_SIZE"
      --attention-backend FLASH_ATTN
    )
    # Same opt-in as verify: only when --enable-logging-iteration-details.
    if [[ "$ENABLE_LOGGING_ITERATION_DETAILS" -eq 1 ]]; then
      draft_cmd+=(--enable-logging-iteration-details)
    fi
    # Optional extra draft-server flags (e.g. --no-enable-cudagraph for dump bisect).
    if [[ -n "${DISAGG_DRAFT_EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      draft_cmd+=(${DISAGG_DRAFT_EXTRA_ARGS})
    fi
    verify_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "CUDA_VISIBLE_DEVICES=${DEVICES}"
      vllm serve "$MODEL"
      --port "$PORT"
      --tensor-parallel-size "$VERIFY_TP"
      --speculative-config "$(spec_json_sd_disagg)"
      "${COMMON[@]}"
    )
    if [[ "$ENABLE_DISAGG_PROFILE" -eq 1 ]]; then
      draft_cmd+=(--enable-sd-timing-model --enable-dflash-draft-profile)
      verify_cmd+=(--enable-sd-timing-model --enable-disagg-dflash-profile)
    fi

    if [[ "$NSYS" -eq 1 ]]; then
      # One nsys profile over a parent helper → both GPUs in one .nsys-rep.
      # Run in the foreground so Ctrl+C hits nsys (background run_cmd left nsys
      # orphaned and the .nsys-rep never finalized).
      combined_cmd=()
      nsys_prepare_combined_profile_cmd combined_cmd
      combined_log="${LOG_DIR}/${TAG}_combined_${NSYS_RUN_TS}.log"
      echo
      echo "# ---- ${TAG} COMBINED nsys draft=${DRAFT_DEVICES} verify=${DEVICES} port=${PORT} ----"
      echo "# log -> ${combined_log}"
      shell_join "${combined_cmd[@]}"
      echo
      echo "# Disagg-DFlash: for overlap debug logs add --enable-disagg-profile (skews latency)"
      echo "# A/B: --no-disagg-async"
      echo "# quality curl: add --no-synthetic (default is synthetic accept)"
      echo "# nsys combined: ${NSYS_COMBINED_OUT}.nsys-rep"
      echo "# run: ./benchmark_random.sh --tag ${TAG} --nsys ...  (required for capture)"
      echo "# then Ctrl+C here to finalize the .nsys-rep"
      if [[ "$PRINT_ONLY" -eq 1 ]]; then
        :
      else
        mkdir -p "$LOG_DIR" "$NSYS_DIR"
        {
          echo "# launched: $(date -Is)"
          echo "# tag=${TAG} role=combined"
          echo "# cmd: $(shell_join "${combined_cmd[@]}")"
          echo "# ----"
        } >"$combined_log"
        # IMPORTANT: do NOT use `nsys | tee` — Ctrl+C kills the pipe and nsys
        # often exits without writing the .nsys-rep. Process substitution keeps
        # nsys as the foreground job so it can finalize on SIGINT.
        echo "# tip: one Ctrl+C after the bench — wait for nsys to print Generating..."
        set +e
        if command -v stdbuf >/dev/null 2>&1; then
          stdbuf -oL -eL "${combined_cmd[@]}" > >(tee -a "$combined_log") 2>&1
        else
          "${combined_cmd[@]}" > >(tee -a "$combined_log") 2>&1
        fi
        nsys_rc=$?
        set -e
        echo "# nsys exit=${nsys_rc}"
        if [[ -f "${NSYS_COMBINED_OUT}.nsys-rep" ]]; then
          echo "# nsys combined report: ${NSYS_COMBINED_OUT}.nsys-rep"
          ls -lh "${NSYS_COMBINED_OUT}.nsys-rep"
        else
          echo "# warning: ${NSYS_COMBINED_OUT}.nsys-rep was NOT written" >&2
          echo "#   (interrupted too hard, or nsys still flushing — check for .qdstrm)" >&2
        fi
      fi
    else
      # Foreground helper with SIGINT cleanup (do not use background run_cmd +
      # wait — that left draft alive after Ctrl+C killed only the verify side).
      echo
      echo "# Disagg-DFlash: for overlap debug logs add --enable-disagg-profile (skews latency)"
      echo "# A/B: --no-disagg-async"
      echo "# quality curl: add --no-synthetic (default is synthetic accept)"
      run_sd_disagg_pair_foreground
    fi
    ;;
esac
