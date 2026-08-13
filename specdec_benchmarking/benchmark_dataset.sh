#!/usr/bin/env bash
#
# Benchmark vllm bench serve on prepared custom datasets (GSM8K / MATH-500).
# Captures speculative acceptance stats in the result JSON (same fields as
# benchmark_random.sh). Use real rejection sampling on the server
# (launch_bench_server.sh --no-synthetic) when comparing acceptance rates.
#
# Usage:
#   ./prepare_math_datasets.py
#   ./benchmark_dataset.sh --tag PD1S1_gsm8k_remote --dataset gsm8k \
#       --request-rates 8 --num-prompts 100 --temperature 0
#   ./benchmark_dataset.sh --tag PD1S1_math500_colocated --dataset math500 \
#       --request-rates 8 --num-prompts 100 --temperature 0
#
#   # PD*S* + --hs-nixl-sink: also scrape sink draft KV
#   ./benchmark_dataset.sh --tag PD1S1_gsm8k_disagg --dataset gsm8k \
#       --request-rates 16 --num-prompts 100 --temperature 0 \
#       --draft-metrics-url http://127.0.0.1:9101
#
# Compare acceptance later:
#   python3 check_spec_quality.py compare \
#     --baseline 'bench_results/PD1S1_gsm8k_colocated/r8/*.json' \
#     --candidate 'bench_results/PD1S1_gsm8k_remote/r8/*.json'
#
set -euo pipefail

TAG=""
MODEL="openai/gpt-oss-20b"
DATASET_NAME="gsm8k"   # gsm8k | math500 | quality | path/to.jsonl
OUTPUT_LEN=1024
PORT=8000
BURSTINESS="1"
NUM_PROMPTS_OVERRIDE=""
TEMPERATURE="0"
DISABLE_WARMUP=0
REQUEST_RATES_CSV="8"
SEED=42
METRICS_URLS=()
METRICS_POLL_INTERVAL=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/bench_data"

usage() {
  echo "Usage: $0 --tag <tag> --dataset gsm8k|math500|quality|<jsonl>" >&2
  echo "         [--model MODEL] [--port PORT] [--request-rates R1,R2]" >&2
  echo "         [--num-prompts N] [--output-len N] [--temperature T]" >&2
  echo "         [--seed N] [--no-warmup]" >&2
  echo "         [--draft-metrics-url URL] [--metrics-url ROLE=URL]..." >&2
  echo "  --draft-metrics-url  scrape draft/sink /metrics → draft_metrics.csv" >&2
  echo "                       (HS NIXL sink default: http://127.0.0.1:9101)" >&2
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
      echo "ERROR: invalid request rate '$tok'" >&2
      exit 1
    fi
    out+=("$tok")
  done
  REQUEST_RATES=("${out[@]}")
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --dataset) DATASET_NAME="${2:?}"; shift 2 ;;
    --output-len) OUTPUT_LEN="${2:?}"; shift 2 ;;
    --num-prompts) NUM_PROMPTS_OVERRIDE="${2:?}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --burstiness) BURSTINESS="${2:?}"; shift 2 ;;
    --request-rates|--concurrencies) REQUEST_RATES_CSV="${2:?}"; shift 2 ;;
    --no-warmup|--disable-warmup) DISABLE_WARMUP=1; shift ;;
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
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$TAG" ] || usage
parse_rate_list "$REQUEST_RATES_CSV"

resolve_dataset() {
  case "$DATASET_NAME" in
    gsm8k) echo "${DATA_DIR}/gsm8k.jsonl" ;;
    math500|math-500) echo "${DATA_DIR}/math500.jsonl" ;;
    quality|quality_20) echo "${DATA_DIR}/quality_20.jsonl" ;;
    *) echo "$DATASET_NAME" ;;
  esac
}

DATASET_PATH="$(resolve_dataset)"
if [ ! -f "$DATASET_PATH" ]; then
  echo "Dataset not found: $DATASET_PATH" >&2
  echo "Run: python3 ${SCRIPT_DIR}/prepare_math_datasets.py" >&2
  # quality_20 can be created without HF download
  if [[ "$DATASET_NAME" == "quality" || "$DATASET_NAME" == "quality_20" ]]; then
    python3 "${SCRIPT_DIR}/prepare_math_datasets.py" --skip-download --out-dir "$DATA_DIR"
  else
    exit 1
  fi
fi

MODEL_CACHE_NAME="models--${MODEL//\//--}"
TOKENIZER=$(echo "${HF_HOME:-$HOME/.cache/huggingface}/hub/${MODEL_CACHE_NAME}/snapshots/"*)
if [ ! -d "$TOKENIZER" ]; then
  TOKENIZER="$MODEL"
fi
BASE_URL="http://localhost:${PORT}"
RESULT_ROOT="${SCRIPT_DIR}/bench_results/${TAG}"

# Always scrape the client-facing server unless already registered.
_has_server=0
for kv in "${METRICS_URLS[@]+"${METRICS_URLS[@]}"}"; do
  [ "${kv%%=*}" = "server" ] && _has_server=1
done
if [ "$_has_server" -eq 0 ]; then
  METRICS_URLS=("server=${BASE_URL}" "${METRICS_URLS[@]+"${METRICS_URLS[@]}"}")
fi

scrape_metric() {
  grep -E "^vllm:(${2})(\{|[[:space:]])" "$1" 2>/dev/null \
    | awk '{s+=$NF} END {if (NR>0) printf "%.6g", s}' || true
}
scrape_metric_first() {
  local file="$1" names="$2"
  grep -E "^vllm:(${names})(\{|[[:space:]])" "$file" 2>/dev/null \
    | awk 'NR==1 {printf "%.6g", $NF; exit}' || true
}
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
      if [ -n "$nblocks_raw" ] && [ "$nblocks_raw" -gt 1 ] 2>/dev/null; then
        nblocks=$((nblocks_raw - 1))
      else
        nblocks=""
      fi
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

DATASET_N=$(wc -l < "$DATASET_PATH" | tr -d ' ')
echo "Model:     $MODEL"
echo "Dataset:   $DATASET_PATH ($DATASET_N prompts)"
echo "Temp:      $TEMPERATURE"
echo "Out len:   $OUTPUT_LEN"
echo "Results:   $RESULT_ROOT"
echo "Metrics scrape targets:"
for kv in "${METRICS_URLS[@]}"; do
  echo "  ${kv%%=*} -> ${kv#*=}/metrics"
done
echo "NOTE: For acceptance-rate comparisons launch the server with --no-synthetic."

for R in "${REQUEST_RATES[@]}"; do
  if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
    NUM_PROMPTS="$NUM_PROMPTS_OVERRIDE"
  else
    NUM_PROMPTS="$DATASET_N"
  fi
  if [ "$NUM_PROMPTS" -gt "$DATASET_N" ]; then
    NUM_PROMPTS="$DATASET_N"
  fi

  R_TAG="${R//./p}"
  T_TAG="t${TEMPERATURE//./p}"
  RUN_DIR="$RESULT_ROOT/r${R_TAG}_${T_TAG}"
  mkdir -p "$RUN_DIR"
  OUT_FILE="result_${TAG}_r${R_TAG}_${T_TAG}.json"
  LOG_FILE="$RUN_DIR/log.txt"

  if [ "$DISABLE_WARMUP" -eq 1 ]; then
    NUM_WARMUPS=0
  else
    NUM_WARMUPS=10
  fi

  echo "=== [${TAG}] dataset=$(basename "$DATASET_PATH") rate=${R} temp=${TEMPERATURE} n=${NUM_PROMPTS} ==="

  POLLER_PIDS=()
  for kv in "${METRICS_URLS[@]}"; do
    role="${kv%%=*}"
    url="${kv#*=}"
    csv="$RUN_DIR/${role}_metrics.csv"
    poll_metrics "$url" "$csv" &
    POLLER_PIDS+=($!)
  done
  trap 'kill "${POLLER_PIDS[@]}" 2>/dev/null || true' EXIT

  BENCH_CMD=(
    vllm bench serve
    --backend openai
    --base-url "$BASE_URL"
    --model "$MODEL"
    --tokenizer "$TOKENIZER"
    --dataset-name custom
    --dataset-path "$DATASET_PATH"
    --custom-output-len "$OUTPUT_LEN"
    --seed "$SEED"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$R"
    --burstiness "$BURSTINESS"
    --temperature "$TEMPERATURE"
    --disable-shuffle
    --save-result
    --save-detailed
    --result-dir "$RUN_DIR"
    --result-filename "$OUT_FILE"
    --num-warmups "$NUM_WARMUPS"
    --metadata
      "tag=${TAG}"
      "dataset=$(basename "$DATASET_PATH")"
      "temperature=${TEMPERATURE}"
      "request_rate=${R}"
  )

  if command -v script >/dev/null 2>&1; then
    script -qefc "$(printf '%q ' "${BENCH_CMD[@]}")" "$LOG_FILE"
  else
    "${BENCH_CMD[@]}" 2>&1 | tee "$LOG_FILE"
  fi

  kill "${POLLER_PIDS[@]}" 2>/dev/null || true
  for pid in "${POLLER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  trap - EXIT

  # Attach a small acceptance summary sidecar for quick compare.
  RESULT_JSON="$RUN_DIR/$OUT_FILE"
  if [ -f "$RESULT_JSON" ]; then
    python3 - <<PY
import json
from pathlib import Path
p = Path("$RESULT_JSON")
d = json.loads(p.read_text())
summary = {
    "result": str(p),
    "tag": "$TAG",
    "dataset": "$(basename "$DATASET_PATH")",
    "temperature": float("$TEMPERATURE"),
    "request_rate": float("$R"),
    "num_prompts": d.get("num_prompts"),
    "output_throughput": d.get("output_throughput"),
    "spec_decode_acceptance_rate": d.get("spec_decode_acceptance_rate"),
    "spec_decode_acceptance_length": d.get("spec_decode_acceptance_length"),
    "spec_decode_per_position_acceptance_rates": d.get(
        "spec_decode_per_position_acceptance_rates"
    ),
}
out = p.with_name(p.stem + "_accept.json")
out.write_text(json.dumps(summary, indent=2))
print("Acceptance summary:", out)
print("  acc_rate:", summary["spec_decode_acceptance_rate"])
print("  acc_len:", summary["spec_decode_acceptance_length"])
print("  per_pos:", summary["spec_decode_per_position_acceptance_rates"])
PY
  fi
done

echo
echo "Done. Results under $RESULT_ROOT"
echo "Compare acceptance:"
echo "  python3 check_spec_quality.py compare \\"
echo "    --baseline 'bench_results/<colocated_tag>/r*_t0/result_*.json' \\"
echo "    --candidate 'bench_results/<remote_tag>/r*_t0/result_*.json'"
