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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/bench_data"

usage() {
  echo "Usage: $0 --tag <tag> --dataset gsm8k|math500|quality|<jsonl>" >&2
  echo "         [--model MODEL] [--port PORT] [--request-rates R1,R2]" >&2
  echo "         [--num-prompts N] [--output-len N] [--temperature T]" >&2
  echo "         [--seed N] [--no-warmup]" >&2
  exit 1
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

DATASET_N=$(wc -l < "$DATASET_PATH" | tr -d ' ')
echo "Model:     $MODEL"
echo "Dataset:   $DATASET_PATH ($DATASET_N prompts)"
echo "Temp:      $TEMPERATURE"
echo "Out len:   $OUTPUT_LEN"
echo "Results:   $RESULT_ROOT"
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
