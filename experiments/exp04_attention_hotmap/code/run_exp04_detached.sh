#!/usr/bin/env bash
# Exp04 attention heatmap experiment — detached (nohup-safe).
# Requires: GPU recommended; Qwen2.5-7B-Instruct at $MODEL (or pass MODEL=...).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL="${MODEL:-${ROOT}/models/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-256}"
DEVICE="${DEVICE:-cuda}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/exp04_attention_hotmap/results/run_${STAMP}}"
DATA_JSONL="${DATA_JSONL:-${ROOT}/experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl}"
LOG="${OUT_DIR}.log"
AB_RANDOM="${AB_RANDOM:-0}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT}/KVI/bin/python" ]]; then PY="${ROOT}/KVI/bin/python"
  else PY="python3"
  fi
fi

mkdir -p "$(dirname "$OUT_DIR")"
EXTRA=()
if [[ "${USE_HF_HOTPOT:-0}" == "1" ]]; then
  EXTRA+=(--use_hf_hotpot)
fi
if [[ "${AB_RANDOM}" == "1" ]]; then
  EXTRA+=(--ablation_random_kv)
fi

echo "[$(date -Iseconds)] Exp04 -> OUT_DIR=${OUT_DIR} LOG=${LOG}" | tee -a "${LOG}"
nohup env PYTHONUNBUFFERED=1 \
  "${PY}" -u "${ROOT}/experiments/exp04_attention_hotmap/code/run_exp04_attention.py" \
  --model "${MODEL}" \
  --out_dir "${OUT_DIR}" \
  --data_jsonl "${DATA_JSONL}" \
  --limit "${LIMIT}" \
  --device "${DEVICE}" \
  --dtype bfloat16 \
  "${EXTRA[@]}" \
  </dev/null >>"${LOG}" 2>&1 &

echo $! | tee -a "${LOG}"
echo "tail -f ${LOG}"
