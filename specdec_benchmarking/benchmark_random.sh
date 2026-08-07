#!/usr/bin/env bash
#
# Sweeps vllm bench serve across a list of --request-rate values (random dataset).
# No --max-concurrency limit is applied (open-loop rate control + burstiness).
# Samples one or more vLLM /metrics endpoints (KV cache, queues, MFU counters)
# and GPU HW counters (DCGM) during each run.
#
# Usage:
#   ./benchmark_random.sh --tag <tag> [--model <model>] [--port <port>] \
#       [--request-rates 128,256,512] [--concurrencies 128,256,512] \
#       [--burstiness 1.0] \
#       [--metrics-url ROLE=URL ...] \
#       [--draft-metrics-url URL] \
#       [--prefill-metrics-url URL] [--decode-metrics-url URL]
#
# Examples:
#   # colocated / single serve (default: poll http://localhost:$PORT as "server")
#   ./benchmark_random.sh --tag PD1_b4096 --request-rates 128,256,512
#
#   # ~10s of arrivals at each rate (num_prompts = ceil(rate * duration))
#   ./benchmark_random.sh --tag PD1_b4096 --request-rates 128,256 --duration-sec 10
#
#   # P/D + specdec: scrape both engine servers (proxy still used for bench)
#   ./benchmark_random.sh --tag P1_D1S1_b8192 --port 8000 \
#       --prefill-metrics-url http://localhost:8100 \
#       --decode-metrics-url http://localhost:8200
#
#   # Nsight: pair with launch_bench_server.sh --nsys (cudaProfilerApi on serve)
#   ./benchmark_random.sh --tag PD1_b8192 --nsys --request-rates 4 \
#       --num-prompts 8 --no-warmup
#
# Results land in ./bench_results/<tag>/r<REQUEST_RATE>/
#   server_metrics.csv / draft_metrics.csv / prefill_metrics.csv / ...
set -euo pipefail

# ---- CLI --------------------------------------------------------------------
TAG=""
MODEL="openai/gpt-oss-20b"
INPUT_LEN=5000
OUTPUT_LEN=1000
PORT=8000
BURSTINESS="1"
DURATION_SEC=""   # if set: num_prompts = ceil(request_rate * duration_sec)
NUM_PROMPTS_OVERRIDE=""
DISABLE_WARMUP=0
# Pass --profile to vllm bench serve so a server started with
# launch_bench_server.sh --nsys (cudaProfilerApi) records the bench window.
NSYS=0
# Comma-separated request rates; default applied after parse if unset.
REQUEST_RATES_CSV=""
# ROLE=URL pairs; if none given after parsing, default to server=$BASE_URL
METRICS_URLS=()

usage() {
  echo "Usage: $0 --tag <tag> [--model <model>] [--port <port>]" >&2
  echo "         [--request-rates R1,R2,...] [--concurrencies R1,R2,...]" >&2
  echo "         [--burstiness FACTOR] [--duration-sec SEC] [--num-prompts N]" >&2
  echo "         [--no-warmup] [--nsys]" >&2
  echo "         [--metrics-url ROLE=URL]..." >&2
  echo "         [--draft-metrics-url URL]" >&2
  echo "         [--prefill-metrics-url URL] [--decode-metrics-url URL]" >&2
  echo "  --tag                  (required) run label, e.g. PD1_b4096" >&2
  echo "  --model                (default: openai/gpt-oss-20b)" >&2
  echo "  --port                 (default: 8000) client-facing vLLM / proxy port" >&2
  echo "  --request-rates        comma-separated rates to sweep (default: 128,256,512)" >&2
  echo "  --concurrencies        alias for --request-rates" >&2
  echo "  --burstiness           arrival burstiness (default: 1 = Poisson)" >&2
  echo "  --duration-sec SEC     approximate arrival window; sets" >&2
  echo "                         num_prompts = ceil(request_rate * SEC)" >&2
  echo "                         (run still waits for in-flight reqs to finish)" >&2
  echo "  --num-prompts N        fixed prompt count (overrides --duration-sec)" >&2
  echo "  --no-warmup            skip warmup requests (--num-warmups 0)" >&2
  echo "  --nsys                 pass --profile to vllm bench serve (triggers" >&2
  echo "                         cudaProfilerApi capture on servers launched" >&2
  echo "                         with launch_bench_server.sh --nsys)" >&2
  echo "  --metrics-url ROLE=URL extra /metrics scrape target (repeatable)" >&2
  echo "  --draft-metrics-url    shorthand for --metrics-url draft=URL" >&2
  echo "  --prefill-metrics-url  shorthand for --metrics-url prefill=URL" >&2
  echo "  --decode-metrics-url   shorthand for --metrics-url decode=URL" >&2
  echo "Note: --max-concurrency is NOT set (no concurrency cap)." >&2
  exit 1
}

add_metrics_url() {
  local role="$1" url="$2"
  if [ -z "$role" ] || [ -z "$url" ]; then
    echo "ERROR: metrics URL requires ROLE and URL" >&2
    usage
  fi
  if [[ ! "$role" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
    echo "ERROR: invalid metrics ROLE '$role' (use letters/digits/_)" >&2
    exit 1
  fi
  METRICS_URLS+=("${role}=${url}")
}

parse_rate_list() {
  local csv="$1"
  local -a out=()
  local tok
  IFS=',' read -ra _toks <<< "$csv"
  for tok in "${_toks[@]}"; do
    tok="${tok// /}"
    [[ -n "$tok" ]] || continue
    if ! awk -v r="$tok" 'BEGIN { exit !(r+0 == r && r > 0) }'; then
      echo "ERROR: invalid request rate '$tok' (need positive number)" >&2
      exit 1
    fi
    out+=("$tok")
  done
  if [ "${#out[@]}" -eq 0 ]; then
    echo "ERROR: empty request-rate list" >&2
    exit 1
  fi
  REQUEST_RATES=("${out[@]}")
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)    TAG="${2:?--tag requires a value}";    shift 2 ;;
    --model)  MODEL="${2:?--model requires a value}"; shift 2 ;;
    --port)   PORT="${2:?--port requires a value}";  shift 2 ;;
    --input-len)  INPUT_LEN="${2:?--input-len requires a value}"; shift 2 ;;
    --output-len) OUTPUT_LEN="${2:?--output-len requires a value}"; shift 2 ;;
    --burstiness) BURSTINESS="${2:?--burstiness requires a value}"; shift 2 ;;
    --duration-sec) DURATION_SEC="${2:?--duration-sec requires a value}"; shift 2 ;;
    --num-prompts) NUM_PROMPTS_OVERRIDE="${2:?--num-prompts requires a value}"; shift 2 ;;
    --no-warmup|--disable-warmup) DISABLE_WARMUP=1; shift ;;
    --nsys|--profile) NSYS=1; shift ;;
    --request-rates|--concurrencies)
      REQUEST_RATES_CSV="${2:?$1 requires a comma-separated list}"
      shift 2
      ;;
    --metrics-url)
      kv="${2:?--metrics-url requires ROLE=URL}"
      shift 2
      if [[ "$kv" != *=* ]]; then
        echo "ERROR: --metrics-url expects ROLE=URL, got: $kv" >&2
        usage
      fi
      add_metrics_url "${kv%%=*}" "${kv#*=}"
      ;;
    --draft-metrics-url)
      add_metrics_url "draft" "${2:?--draft-metrics-url requires a URL}"
      shift 2
      ;;
    --prefill-metrics-url)
      add_metrics_url "prefill" "${2:?--prefill-metrics-url requires a URL}"
      shift 2
      ;;
    --decode-metrics-url)
      add_metrics_url "decode" "${2:?--decode-metrics-url requires a URL}"
      shift 2
      ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "ERROR: --tag is required." >&2
  usage
fi

if ! awk -v b="$BURSTINESS" 'BEGIN { exit !(b+0 == b && b > 0) }'; then
  echo "ERROR: --burstiness must be a positive number (got: $BURSTINESS)" >&2
  exit 1
fi

if [ -n "$DURATION_SEC" ]; then
  if ! awk -v d="$DURATION_SEC" 'BEGIN { exit !(d+0 == d && d > 0) }'; then
    echo "ERROR: --duration-sec must be a positive number (got: $DURATION_SEC)" >&2
    exit 1
  fi
fi

if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
  if ! awk -v n="$NUM_PROMPTS_OVERRIDE" 'BEGIN { exit !(n+0 == n && n >= 1) }'; then
    echo "ERROR: --num-prompts must be >= 1 (got: $NUM_PROMPTS_OVERRIDE)" >&2
    exit 1
  fi
fi

if [ -n "$REQUEST_RATES_CSV" ]; then
  parse_rate_list "$REQUEST_RATES_CSV"
else
  REQUEST_RATES=(128 256 512)
fi

# ---- Configuration ---------------------------------------------------------
# Derive the HF cache snapshot path from the model name, e.g.
# openai/gpt-oss-20b -> models--openai--gpt-oss-20b
MODEL_CACHE_NAME="models--${MODEL//\//--}"
TOKENIZER=$(echo "${HF_HOME:-$HOME/.cache/huggingface}/hub/${MODEL_CACHE_NAME}/snapshots/"*)
if [ ! -d "$TOKENIZER" ]; then
  echo "WARNING: no local snapshot found for ${MODEL}; falling back to model name as tokenizer." >&2
  TOKENIZER="$MODEL"
fi
BASE_URL="http://localhost:${PORT}"
RESULT_ROOT="./bench_results/${TAG}"
SLEEP_BETWEEN_RUNS=2
METRICS_POLL_INTERVAL=5        # seconds between /metrics samples
# -----------------------------------------------------------------------------

# Always scrape the client-facing server unless the user already registered
# ROLE=server explicitly (or named it something else and only wants extras).
_has_server=0
for kv in "${METRICS_URLS[@]+"${METRICS_URLS[@]}"}"; do
  [ "${kv%%=*}" = "server" ] && _has_server=1
done
if [ "$_has_server" -eq 0 ]; then
  # Prepend so "server" is the primary scrape.
  METRICS_URLS=("server=${BASE_URL}" "${METRICS_URLS[@]+"${METRICS_URLS[@]}"}")
fi

echo "Model:        $MODEL"
echo "Tokenizer:    $TOKENIZER"
echo "Bench URL:    $BASE_URL"
echo "Request rates:${REQUEST_RATES[*]}"
echo "Burstiness:   $BURSTINESS"
if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
  echo "Num prompts:  $NUM_PROMPTS_OVERRIDE (fixed)"
elif [ -n "$DURATION_SEC" ]; then
  echo "Duration:     ${DURATION_SEC}s arrival window (num_prompts = ceil(rate × duration))"
else
  echo "Num prompts:  (heuristic by rate)"
fi
if [ "$DISABLE_WARMUP" -eq 1 ]; then
  echo "Warmups:      disabled (--num-warmups 0)"
fi
if [ "$NSYS" -eq 1 ]; then
  echo "Nsight:       ON (vllm bench serve --profile → server cudaProfilerApi)"
fi
echo "Max concur.:  (none — open-loop request-rate)"
echo "Metrics scrape targets:"
for kv in "${METRICS_URLS[@]}"; do
  echo "  ${kv%%=*} -> ${kv#*=}/metrics"
done

if [ -d "$RESULT_ROOT" ]; then
  echo "WARNING: $RESULT_ROOT already exists; files may be overwritten." >&2
fi
mkdir -p "$RESULT_ROOT"

# ---- Server-side metrics poller ---------------------------------------------
# Handles both metric name generations:
# vllm:kv_cache_usage_perc (V1 engine) and vllm:gpu_cache_usage_perc (older).
# KV GiB is derived as: blocks * vllm:kv_cache_block_size_bytes
#   total_blocks = num_gpu_blocks - 1  (null block reserved; from cache_config_info)
#   used_blocks  = usage_perc * total_blocks
# MFU counters require the server to be started with --enable-mfu-metrics.
scrape_metric() {
  # $1 = metrics dump file, $2 = regex of metric name(s)
  # Always exit 0: draft /metrics omits EngineCore queue/token counters, and
  # under `set -e` + `pipefail` a missing grep match would kill the poller
  # after the first successful curl (header-only CSV).
  grep -E "^vllm:(${2})(\{|[[:space:]])" "$1" 2>/dev/null \
    | awk '{s+=$NF} END {if (NR>0) printf "%.6g", s}' || true
}

# First matching sample (for per-engine gauges that must not be summed).
scrape_metric_first() {
  local file="$1" names="$2"
  grep -E "^vllm:(${names})(\{|[[:space:]])" "$file" 2>/dev/null \
    | awk 'NR==1 {printf "%.6g", $NF; exit}' || true
}

# Pull a string label off vllm:cache_config_info{...} (e.g. num_gpu_blocks).
scrape_cache_config_label() {
  local file="$1" label="$2"
  grep -E '^vllm:cache_config_info\{' "$file" 2>/dev/null | head -1 \
    | sed -n "s/.*${label}=\"\([^\"]*\)\".*/\1/p" || true
}

poll_metrics() {
  local metrics_base="$1"
  local out_csv="$2"
  local tmp
  tmp=$(mktemp)
  echo "unix_ts,kv_cache_usage_perc,kv_cache_block_size_bytes,kv_cache_num_blocks,kv_cache_usage_gib,kv_cache_total_gib,kv_cache_usage_bytes,kv_cache_total_bytes,num_requests_running,num_requests_waiting,preemptions_total,prompt_tokens_total,generation_tokens_total,prefix_cache_queries_total,prefix_cache_hits_total,estimated_flops_per_gpu_total,estimated_read_bytes_per_gpu_total,estimated_write_bytes_per_gpu_total" > "$out_csv"
  while true; do
    if curl -sf --max-time 2 "${metrics_base%/}/metrics" -o "$tmp"; then
      local ts kv bs nblocks_raw nblocks kv_gib kv_tot_gib kv_b kv_tot_b
      local n_run n_wait pre ptok gtok pcq pch flops rbytes wbytes
      ts=$(date +%s.%N)
      kv=$(scrape_metric  "$tmp" "kv_cache_usage_perc|gpu_cache_usage_perc")
      bs=$(scrape_metric_first "$tmp" "kv_cache_block_size_bytes")
      nblocks_raw=$(scrape_cache_config_label "$tmp" "num_gpu_blocks")
      # usable blocks exclude the reserved null block
      if [ -n "$nblocks_raw" ] && [ "$nblocks_raw" -gt 1 ] 2>/dev/null; then
        nblocks=$((nblocks_raw - 1))
      else
        nblocks=""
      fi
      # total = blocks * block_size; used = usage_perc * total
      # Skip until block_size is non-zero (gauge can be 0 before KV init).
      kv_tot_b=""; kv_b=""; kv_tot_gib=""; kv_gib=""
      if [ -n "$bs" ] && [ -n "$nblocks" ] \
          && awk -v b="$bs" -v n="$nblocks" 'BEGIN {exit !(b>0 && n>0)}'; then
        kv_tot_b=$(awk -v n="$nblocks" -v b="$bs" 'BEGIN {printf "%.0f", n*b}')
        kv_tot_gib=$(awk -v t="$kv_tot_b" 'BEGIN {printf "%.6g", t/(1024^3)}')
        if [ -n "$kv" ]; then
          kv_b=$(awk -v u="$kv" -v t="$kv_tot_b" 'BEGIN {printf "%.0f", u*t}')
          kv_gib=$(awk -v u="$kv" -v t="$kv_tot_gib" 'BEGIN {printf "%.6g", u*t}')
        fi
      fi
      n_run=$(scrape_metric "$tmp" "num_requests_running")
      n_wait=$(scrape_metric "$tmp" "num_requests_waiting")
      pre=$(scrape_metric "$tmp" "num_preemptions_total|num_preemptions")
      ptok=$(scrape_metric "$tmp" "prompt_tokens_total|prompt_tokens")
      gtok=$(scrape_metric "$tmp" "generation_tokens_total|generation_tokens")
      pcq=$(scrape_metric "$tmp" "gpu_prefix_cache_queries_total|prefix_cache_queries_total|prefix_cache_queries")
      pch=$(scrape_metric "$tmp" "gpu_prefix_cache_hits_total|prefix_cache_hits_total|prefix_cache_hits")
      flops=$(scrape_metric "$tmp" "estimated_flops_per_gpu_total")
      rbytes=$(scrape_metric "$tmp" "estimated_read_bytes_per_gpu_total")
      wbytes=$(scrape_metric "$tmp" "estimated_write_bytes_per_gpu_total")
      echo "$ts,$kv,$bs,$nblocks,$kv_gib,$kv_tot_gib,$kv_b,$kv_tot_b,$n_run,$n_wait,$pre,$ptok,$gtok,$pcq,$pch,$flops,$rbytes,$wbytes" >> "$out_csv"
    fi
    sleep "$METRICS_POLL_INTERVAL"
  done
}

# ---- GPU HW counter poller ----------------------------------------------------
# DCGM fields: 1004 = tensor-pipe active (MFU proxy),
#              1005 = DRAM active (memory-bandwidth proxy), 1002 = SM active.
start_gpu_poller() {
  local out_file="$1"
  if command -v dcgmi >/dev/null 2>&1; then
    dcgmi dmon -e 1002,1004,1005 -d $((METRICS_POLL_INTERVAL * 1000)) > "$out_file" 2>&1 &
  else
    nvidia-smi dmon -s pum -d "$METRICS_POLL_INTERVAL" -o T > "$out_file" 2>&1 &
  fi
  echo $!
}

for R in "${REQUEST_RATES[@]}"; do
  # Prompt count: explicit > duration-derived > rate heuristic.
  if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
    NUM_PROMPTS=$(awk -v n="$NUM_PROMPTS_OVERRIDE" 'BEGIN { printf "%d", n }')
  elif [ -n "$DURATION_SEC" ]; then
    NUM_PROMPTS=$(awk -v r="$R" -v d="$DURATION_SEC" \
      'BEGIN { n=int(r*d + 0.999999); if (n < 1) n=1; print n }')
  elif awk -v r="$R" 'BEGIN { exit !(r <= 8) }'; then
    NUM_PROMPTS=60
  elif awk -v r="$R" 'BEGIN { exit !(r <= 100) }'; then
    NUM_PROMPTS=500
  elif awk -v r="$R" 'BEGIN { exit !(r <= 255) }'; then
    NUM_PROMPTS=500
  else
    NUM_PROMPTS=1000
  fi

  # Safe directory token (128 or 128.5 -> 128 / 128p5)
  R_TAG="${R//./p}"
  RUN_DIR="$RESULT_ROOT/r${R_TAG}"
  mkdir -p "$RUN_DIR"

  OUT_FILE="result_${TAG}_r${R_TAG}.json"
  LOG_FILE="$RUN_DIR/log.txt"
  GPU_METRICS_LOG="$RUN_DIR/gpu_metrics.txt"

  if [ -n "$DURATION_SEC" ] && [ -z "$NUM_PROMPTS_OVERRIDE" ]; then
    echo "=== [${TAG}] model=${MODEL} request_rate=${R} burstiness=${BURSTINESS} duration_sec=${DURATION_SEC} num_prompts=${NUM_PROMPTS} ==="
  else
    echo "=== [${TAG}] model=${MODEL} request_rate=${R} burstiness=${BURSTINESS} num_prompts=${NUM_PROMPTS} ==="
  fi

  POLLER_PIDS=()
  for kv in "${METRICS_URLS[@]}"; do
    role="${kv%%=*}"
    url="${kv#*=}"
    csv="$RUN_DIR/${role}_metrics.csv"
    poll_metrics "$url" "$csv" &
    POLLER_PIDS+=($!)
  done
  GPU_POLLER_PID=$(start_gpu_poller "$GPU_METRICS_LOG")
  trap 'kill "${POLLER_PIDS[@]}" $GPU_POLLER_PID 2>/dev/null || true' EXIT

  # Warmups scale mildly with rate; --no-warmup forces 0.
  if [ "$DISABLE_WARMUP" -eq 1 ]; then
    NUM_WARMUPS=0
  else
    NUM_WARMUPS=$(awk -v r="$R" 'BEGIN { n=int(r+20); if (n<20) n=20; if (n>200) n=200; print n }')
  fi

  META_ARGS=("tag=${TAG}" "request_rate=${R}" "burstiness=${BURSTINESS}" "model=${MODEL}")
  if [ -n "$DURATION_SEC" ]; then
    META_ARGS+=("duration_sec=${DURATION_SEC}")
  fi

  # Run under a PTY when possible so the Rust indicatif progress bar still
  # renders (plain `... | tee` makes stdout a pipe → bar is hidden).
  BENCH_CMD=(
    vllm bench serve
    --backend openai
    --base-url "$BASE_URL"
    --model "$MODEL"
    --tokenizer "$TOKENIZER"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --seed 42
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$R"
    --burstiness "$BURSTINESS"
    --temperature 0
    --save-result
    --result-dir "$RUN_DIR"
    --result-filename "$OUT_FILE"
    --num-warmups "$NUM_WARMUPS"
    --metadata "${META_ARGS[@]}"
  )
  if [ "$NSYS" -eq 1 ]; then
    # Triggers /start_profile../stop_profile (cudaProfilerApi) on nsys-wrapped serve.
    BENCH_CMD+=(--profile)
  fi
  if command -v script >/dev/null 2>&1; then
    # -q quiet header, -e exit with child status, -f flush, -c run command
    script -qefc "$(printf '%q ' "${BENCH_CMD[@]}")" "$LOG_FILE"
  else
    echo "WARNING: 'script' not found; progress bar may be hidden under tee." >&2
    "${BENCH_CMD[@]}" 2>&1 | tee "$LOG_FILE"
  fi

  kill "${POLLER_PIDS[@]}" "$GPU_POLLER_PID" 2>/dev/null || true
  for pid in "${POLLER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  trap - EXIT

  echo "=== Done request_rate=${R} ==="
  sleep "$SLEEP_BETWEEN_RUNS"
done

echo
echo "All sweeps complete. Layout:"
echo "  $RESULT_ROOT/r<RATE>/result_${TAG}_r<RATE>.json"
for kv in "${METRICS_URLS[@]}"; do
  echo "  $RESULT_ROOT/r<RATE>/${kv%%=*}_metrics.csv"
done
echo "  $RESULT_ROOT/r<RATE>/gpu_metrics.txt"
echo "Next: python3 new_parse_sweep_results.py $RESULT_ROOT sweep_summary_${TAG}.csv --gpu <GPU>"
