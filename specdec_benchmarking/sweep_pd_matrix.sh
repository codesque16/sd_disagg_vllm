#!/usr/bin/env bash
# Sweep colocated + HS-NIXL disagg PD configs across max-num-seqs × batched-tokens,
# once for numbers and once with nsys traces.
#
# Matrix:
#   Cases:  PD1, PD2, PD1S1 (colocated|disagg), PD2S2 (colocated|disagg)
#   Combos: (max_num_seqs, batched_tokens) =
#             256/8192  256/9728  600/8192  600/11792
#   Modes:  plain (throughput numbers) and --nsys (trace)
#   Benches per cell (same server): random then gsm8k
#
# "disagg" here means PD*S* + --hs-nixl-sink --async-verify (remote draft on
# --draft-devices). Plain PD1/PD2 have no SD / no disagg variant.
#
# Results:
#   ./bench_results/<tag>_random/
#   ./bench_results/<tag>_gsm8k/
#
# Examples:
#   ./sweep_pd_matrix.sh --dry-run
#   ./sweep_pd_matrix.sh --request-rates 16 --continue-on-error
#   ./sweep_pd_matrix.sh --only 'PD1S1.*disagg' --skip-nsys
#   ./sweep_pd_matrix.sh --only-nsys --skip-gsm8k

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LAUNCH="./launch_bench_server.sh"
BENCH_RANDOM="./benchmark_random.sh"
BENCH_DATASET="./benchmark_dataset.sh"

PORT="${PORT:-8000}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
CLEANUP_WAIT_S="${CLEANUP_WAIT_S:-90}"
SLEEP_AFTER_CLEANUP_S="${SLEEP_AFTER_CLEANUP_S:-5}"
TAG_PREFIX="${TAG_PREFIX:-August10}"

REQUEST_RATES="${REQUEST_RATES:-16}"
# Random-bench defaults (match recent PD* random runs).
DURATION_SEC="${DURATION_SEC:-20}"
INPUT_LEN="${INPUT_LEN:-5000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
# GSM8K-bench defaults.
GSM8K_NUM_PROMPTS="${GSM8K_NUM_PROMPTS:-100}"
GSM8K_TEMPERATURE="${GSM8K_TEMPERATURE:-0}"
NO_WARMUP=0

DRY_RUN=0
SKIP_NSYS=0
ONLY_NSYS=0
SKIP_RANDOM=0
SKIP_GSM8K=0
ONLY_RE=""
CONTINUE_ON_ERROR=0

SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-./startup_logs/sweep_pd_matrix}"
mkdir -p "$SWEEP_LOG_DIR"

usage() {
  cat <<'EOF'
Usage: sweep_pd_matrix.sh [options]

Per cell: launch server → benchmark_random → benchmark_dataset(gsm8k) → teardown.

Options:
  --request-rates CSV     both benches (default: 16)
  --duration-sec SEC      random bench arrival window (default: 20)
  --input-len N           random bench (default: 5000)
  --output-len N          random bench (default: 1000)
  --gsm8k-num-prompts N   gsm8k bench (default: 100)
  --gsm8k-temperature T   gsm8k bench (default: 0)
  --no-warmup             pass --no-warmup to both benches
  --skip-random           only run gsm8k
  --skip-gsm8k            only run random
  --port PORT             server port (default 8000)
  --ready-timeout SEC     wait for /v1/models (default 900)
  --only REGEX            only run tags matching REGEX
  --skip-nsys             skip nsys (numbers-only) passes
  --only-nsys             only nsys passes
  --continue-on-error     do not abort the matrix on a failed cell
  --tag-prefix STR        tag prefix (default: August10)
  --dry-run               print planned commands only
  -h, --help              show help

Env:
  PORT READY_TIMEOUT_S CLEANUP_WAIT_S SLEEP_AFTER_CLEANUP_S TAG_PREFIX
  REQUEST_RATES DURATION_SEC INPUT_LEN OUTPUT_LEN
  GSM8K_NUM_PROMPTS GSM8K_TEMPERATURE SWEEP_LOG_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --request-rates) REQUEST_RATES="${2:?}"; shift 2 ;;
    --duration-sec) DURATION_SEC="${2:?}"; shift 2 ;;
    --input-len) INPUT_LEN="${2:?}"; shift 2 ;;
    --output-len) OUTPUT_LEN="${2:?}"; shift 2 ;;
    --gsm8k-num-prompts) GSM8K_NUM_PROMPTS="${2:?}"; shift 2 ;;
    --gsm8k-temperature) GSM8K_TEMPERATURE="${2:?}"; shift 2 ;;
    --no-warmup) NO_WARMUP=1; shift ;;
    --skip-random) SKIP_RANDOM=1; shift ;;
    --skip-gsm8k) SKIP_GSM8K=1; shift ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --ready-timeout) READY_TIMEOUT_S="${2:?}"; shift 2 ;;
    --only) ONLY_RE="${2:?}"; shift 2 ;;
    --skip-nsys) SKIP_NSYS=1; shift ;;
    --only-nsys) ONLY_NSYS=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --tag-prefix) TAG_PREFIX="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$SKIP_NSYS" -eq 1 && "$ONLY_NSYS" -eq 1 ]]; then
  echo "ERROR: --skip-nsys and --only-nsys are mutually exclusive" >&2
  exit 1
fi
if [[ "$SKIP_RANDOM" -eq 1 && "$SKIP_GSM8K" -eq 1 ]]; then
  echo "ERROR: --skip-random and --skip-gsm8k are mutually exclusive" >&2
  exit 1
fi

# (max_num_seqs batched_tokens)
COMBOS=(
  "256 8192"
  "256 9728"
  "600 8192"
  "600 11792"
)

# case_base | variant(colocated|disagg|plain) | devices | draft_devices_or_-
# plain = no SD (PD1/PD2). disagg = hs-nixl-sink + async-verify.
CASES=(
  "PD1|plain|0|-"
  "PD2|plain|0,1|-"
  "PD1S1|colocated|0|-"
  "PD1S1|disagg|0|1"
  "PD2S2|colocated|0,1|-"
  "PD2S2|disagg|0,1|2"
)

LAUNCH_PID=""
LAUNCH_PGID=""

log() { echo "[$(date -Is)] $*"; }

kill_tree() {
  local sig="$1"
  shift
  local pid
  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    kill "-$sig" "$pid" 2>/dev/null || true
    kill "-$sig" -- "-$pid" 2>/dev/null || true
  done
}

nvidia_smi_ok() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

gpu_compute_pids() {
  if ! nvidia_smi_ok; then
    return 0
  fi
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'NF {gsub(/ /,""); print}' | sort -u || true
}

wait_gpus_idle() {
  local timeout_s="${1:-$CLEANUP_WAIT_S}"
  local i n
  if ! nvidia_smi_ok; then
    log "nvidia-smi unavailable; skipping GPU idle wait"
    return 0
  fi
  for ((i = 0; i < timeout_s; i++)); do
    n=$(gpu_compute_pids | wc -l | tr -d ' ')
    if [[ "${n:-0}" -eq 0 ]]; then
      log "GPUs idle"
      return 0
    fi
    sleep 1
  done
  log "WARNING: GPUs still busy after ${timeout_s}s:"
  gpu_compute_pids | while read -r p; do
    ps -p "$p" -o pid,cmd --no-headers 2>/dev/null || echo "$p"
  done || true
  return 1
}

free_port() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v fuser >/dev/null 2>&1; then
    pids=$(fuser "${port}/tcp" 2>/dev/null || true)
  fi
  local p
  for p in $pids; do
    [[ -n "$p" ]] || continue
    log "killing listener pid=${p} on port ${port}"
    kill -9 "$p" 2>/dev/null || true
  done
}

kill_stale_bench_procs() {
  # Do not match this sweep script itself.
  pkill -9 -f '[v]llm serve' 2>/dev/null || true
  pkill -9 -f '[V]LLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'multiprocessing\.spawn' 2>/dev/null || true
  pkill -9 -f '[d]flash_hs_nixl_sink' 2>/dev/null || true
  pkill -9 -f '[d]flash_draft_server' 2>/dev/null || true
  pkill -9 -f '[n]sys profile' 2>/dev/null || true
  pkill -9 -f '/nsys ' 2>/dev/null || true
  pkill -9 -f 'launch_bench_server\.sh' 2>/dev/null || true
  pkill -9 -f 'benchmark_random\.sh' 2>/dev/null || true
  pkill -9 -f 'benchmark_dataset\.sh' 2>/dev/null || true
  pkill -9 -f 'vllm bench serve' 2>/dev/null || true
}

cleanup_servers() {
  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    return 0
  fi
  log "Cleaning up servers / GPU compute processes"
  if [[ -n "${LAUNCH_PGID:-}" ]]; then
    # Prefer INT so nsys combined helpers can finalize .nsys-rep.
    kill_tree INT "$LAUNCH_PGID" "$LAUNCH_PID"
    sleep 3
    kill_tree TERM "$LAUNCH_PGID" "$LAUNCH_PID"
    sleep 2
    kill_tree KILL "$LAUNCH_PGID" "$LAUNCH_PID"
  elif [[ -n "${LAUNCH_PID:-}" ]]; then
    kill_tree INT "$LAUNCH_PID"
    sleep 3
    kill_tree TERM "$LAUNCH_PID"
    sleep 2
    kill_tree KILL "$LAUNCH_PID"
  fi
  LAUNCH_PID=""
  LAUNCH_PGID=""

  kill_stale_bench_procs
  free_port "$PORT"

  local p
  while read -r p; do
    [[ -n "$p" ]] || continue
    log "killing GPU compute pid=${p}"
    kill -9 "$p" 2>/dev/null || true
  done < <(gpu_compute_pids)

  # Second pass if anything respawned / stuck.
  sleep 1
  kill_stale_bench_procs
  while read -r p; do
    [[ -n "$p" ]] || continue
    kill -9 "$p" 2>/dev/null || true
  done < <(gpu_compute_pids)

  wait_gpus_idle "$CLEANUP_WAIT_S" || true
  sleep "$SLEEP_AFTER_CLEANUP_S"
}

# Hard gate before every launch: GPUs + listen port must be free.
ensure_clean_before_launch() {
  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    return 0
  fi
  log "Pre-launch cleanup check (port=${PORT})"
  cleanup_servers
  local n
  n=$(gpu_compute_pids | wc -l | tr -d ' ')
  if [[ "${n:-0}" -ne 0 ]]; then
    log "ERROR: GPUs still busy after cleanup; refusing to launch"
    gpu_compute_pids | while read -r p; do
      ps -p "$p" -o pid,cmd --no-headers 2>/dev/null || echo "$p"
    done || true
    return 1
  fi
  if command -v lsof >/dev/null 2>&1; then
    if lsof -t -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      log "ERROR: port ${PORT} still in use after cleanup"
      return 1
    fi
  fi
  log "Pre-launch OK: GPUs idle, port ${PORT} free"
  return 0
}

wait_http_ready() {
  local url="$1"
  local timeout_s="$2"
  local i code
  for ((i = 0; i < timeout_s; i++)); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url" 2>/dev/null || true)
    if [[ "$code" == "200" ]]; then
      log "ready: $url"
      return 0
    fi
    # Fail fast if launcher died.
    if [[ -n "${LAUNCH_PID:-}" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      log "ERROR: launcher pid=$LAUNCH_PID exited before ready"
      return 1
    fi
    sleep 1
  done
  log "ERROR: not ready after ${timeout_s}s ($url last HTTP ${code:-000})"
  return 1
}

build_tag() {
  local case_base="$1" variant="$2" batched="$3" seqs="$4" nsys="$5"
  local tag="${TAG_PREFIX}_${case_base}_b${batched}_s${seqs}"
  if [[ "$variant" != "plain" ]]; then
    tag="${tag}_${variant}"
  fi
  if [[ "$nsys" -eq 1 ]]; then
    tag="${tag}_nsys"
  fi
  printf '%s' "$tag"
}

run_one() {
  local case_base="$1" variant="$2" devices="$3" draft_devices="$4"
  local seqs="$5" batched="$6" nsys="$7"

  local tag
  tag=$(build_tag "$case_base" "$variant" "$batched" "$seqs" "$nsys")

  if [[ -n "$ONLY_RE" ]] && ! [[ "$tag" =~ $ONLY_RE ]]; then
    return 0
  fi
  if [[ "$nsys" -eq 1 && "$SKIP_NSYS" -eq 1 ]]; then
    return 0
  fi
  if [[ "$nsys" -eq 0 && "$ONLY_NSYS" -eq 1 ]]; then
    return 0
  fi

  local case_arg="${case_base}_b${batched}"
  local -a launch_cmd=(
    "$LAUNCH" "$case_arg"
    --devices "$devices"
    --port "$PORT"
    --max-num-seqs "$seqs"
    --batched-tokens "$batched"
    # Real rejection sampling so gsm8k acceptance is meaningful.
    --no-synthetic
  )
  if [[ "$variant" == "disagg" ]]; then
    launch_cmd+=(--draft-devices "$draft_devices" --hs-nixl-sink --async-verify)
  fi
  if [[ "$nsys" -eq 1 ]]; then
    launch_cmd+=(--nsys)
  fi

  local tag_random="${tag}_random"
  local tag_gsm8k="${tag}_gsm8k"

  local -a random_cmd=(
    "$BENCH_RANDOM" --tag "$tag_random" --port "$PORT"
    --request-rates "$REQUEST_RATES"
    --duration-sec "$DURATION_SEC"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
  )
  local -a gsm8k_cmd=(
    "$BENCH_DATASET" --tag "$tag_gsm8k" --port "$PORT"
    --dataset gsm8k
    --request-rates "$REQUEST_RATES"
    --num-prompts "$GSM8K_NUM_PROMPTS"
    --temperature "$GSM8K_TEMPERATURE"
  )
  if [[ "$NO_WARMUP" -eq 1 ]]; then
    random_cmd+=(--no-warmup)
    gsm8k_cmd+=(--no-warmup)
  fi
  # nsys capture window is driven by random bench (/start_profile…/stop_profile).
  if [[ "$nsys" -eq 1 ]]; then
    random_cmd+=(--nsys)
  fi

  local cell_log="${SWEEP_LOG_DIR}/${tag}_$(date +%Y%m%d_%H%M%S).log"
  log "======== CELL tag=${tag} ========"
  log "launch: ${launch_cmd[*]}"
  if [[ "$SKIP_RANDOM" -eq 0 ]]; then
    log "random: ${random_cmd[*]}"
  fi
  if [[ "$SKIP_GSM8K" -eq 0 ]]; then
    log "gsm8k:  ${gsm8k_cmd[*]}"
  fi
  log "cell log -> ${cell_log}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  if ! ensure_clean_before_launch; then
    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
      return 0
    fi
    return 1
  fi

  # New session so we can kill the whole tree (sink+verify / nsys helper).
  if command -v setsid >/dev/null 2>&1; then
    setsid "${launch_cmd[@]}" >"$cell_log" 2>&1 &
  else
    "${launch_cmd[@]}" >"$cell_log" 2>&1 &
  fi
  LAUNCH_PID=$!
  # Session leader pgid == pid under setsid; fall back to process group.
  LAUNCH_PGID="$LAUNCH_PID"
  local _pg
  _pg=$(ps -o pgid= -p "$LAUNCH_PID" 2>/dev/null | tr -d ' ' || true)
  [[ -n "$_pg" ]] && LAUNCH_PGID="$_pg"
  log "launcher pid=${LAUNCH_PID} pgid=${LAUNCH_PGID:-?} log=${cell_log}"

  if ! wait_http_ready "http://127.0.0.1:${PORT}/v1/models" "$READY_TIMEOUT_S"; then
    log "ERROR: server failed to become ready for ${tag}"
    tail -n 80 "$cell_log" || true
    cleanup_servers
    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
      return 0
    fi
    return 1
  fi

  local bench_rc=0
  if [[ "$SKIP_RANDOM" -eq 0 ]]; then
    set +e
    "${random_cmd[@]}"
    bench_rc=$?
    set -e
    if [[ "$bench_rc" -ne 0 ]]; then
      log "ERROR: random bench failed rc=${bench_rc} tag=${tag_random}"
      cleanup_servers
      if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
        return 0
      fi
      return "$bench_rc"
    fi
    log "random OK -> ./bench_results/${tag_random}/"
  fi

  if [[ "$SKIP_GSM8K" -eq 0 ]]; then
    set +e
    "${gsm8k_cmd[@]}"
    bench_rc=$?
    set -e
    if [[ "$bench_rc" -ne 0 ]]; then
      log "ERROR: gsm8k bench failed rc=${bench_rc} tag=${tag_gsm8k}"
      cleanup_servers
      if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
        return 0
      fi
      return "$bench_rc"
    fi
    log "gsm8k OK -> ./bench_results/${tag_gsm8k}/"
  fi

  cleanup_servers
  return 0
}

trap 'log "trap: cleaning up"; cleanup_servers' INT TERM EXIT

log "Sweep start cwd=${SCRIPT_DIR}"
log "tag-prefix=${TAG_PREFIX} request-rates=${REQUEST_RATES} port=${PORT} dry_run=${DRY_RUN}"
log "random: duration=${DURATION_SEC}s input=${INPUT_LEN} output=${OUTPUT_LEN}"
log "gsm8k:  num_prompts=${GSM8K_NUM_PROMPTS} temperature=${GSM8K_TEMPERATURE}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  log "Initial pre-sweep GPU/port cleanup"
  ensure_clean_before_launch || {
    log "ERROR: cannot start sweep — clear GPUs/port ${PORT} manually"
    exit 1
  }
fi

failures=0
for case_row in "${CASES[@]}"; do
  IFS='|' read -r case_base variant devices draft_devices <<<"$case_row"
  for combo in "${COMBOS[@]}"; do
    read -r seqs batched <<<"$combo"
    for nsys in 0 1; do
      if ! run_one "$case_base" "$variant" "$devices" "$draft_devices" \
          "$seqs" "$batched" "$nsys"; then
        failures=$((failures + 1))
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
          log "Aborting after failure (use --continue-on-error to keep going)"
          exit 1
        fi
      fi
    done
  done
done

trap - INT TERM EXIT
cleanup_servers
log "Sweep done failures=${failures}"
exit "$failures"
