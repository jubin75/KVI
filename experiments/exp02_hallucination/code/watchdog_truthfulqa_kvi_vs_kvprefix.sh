#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
EXP2="${ROOT}/experiments/exp02_hallucination"
OUT="${EXP2}/results/truthfulqa_kvi_vs_kvprefix_qwen25_7b"
RUN_LOG="${EXP2}/results/exp02_truthfulqa_kvi_vs_kvprefix_nohup.log"
WD_LOG="${EXP2}/results/watchdog_truthfulqa_kvi_vs_kvprefix.log"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
RESTART_COOLDOWN_SEC="${RESTART_COOLDOWN_SEC:-20}"
TARGET_N="${TARGET_N:-500}"

ts() { date -Iseconds; }

_count_lines() {
  if [ -f "${OUT}/predictions.jsonl" ]; then
    wc -l < "${OUT}/predictions.jsonl" || echo 0
  else
    echo 0
  fi
}

_find_pid() {
  pgrep -f "${ROOT}/KVI/bin/python3 -u ${ROOT}/experiments/exp01_main_qa/code/run_exp01.py --dataset ${EXP2}/data/truthfulqa_eval.jsonl --dataset_name TRUTHFULQA" | head -n 1 || true
}

_start_run() {
  nohup "${ROOT}/KVI/bin/python3" -u "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py" \
    --dataset "${EXP2}/data/truthfulqa_eval.jsonl" \
    --dataset_name TRUTHFULQA \
    --model "${ROOT}/models/Qwen2.5-7B-Instruct" \
    --graph_index "${EXP2}/artifacts/truthfulqa/graph_index.json" \
    --triple_kvbank_dir "${EXP2}/artifacts/truthfulqa/triple_kvbank" \
    --graph_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --ann_kv_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences" \
    --ann_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --ann_semantic_type_specs "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json" \
    --ann_pattern_index_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar" \
    --ann_sidecar_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar" \
    --methods kv_prefix,kvi \
    --out_dir "${OUT}" \
    --timeout_s 600 \
    --bootstrap_samples 1000 \
    --permutation_samples 2000 \
    --inference_service_url "http://127.0.0.1:18888" \
    --ann_inference_service_url "" \
    --ann_force_cpu \
    --truthfulqa_kvi_mc1_answer grounded \
    --truthfulqa_kvi_max_new_tokens 96 \
    >> "${RUN_LOG}" 2>&1 &
  echo "$!"
}

echo "[$(ts)] watchdog start interval=${CHECK_INTERVAL_SEC}s target_n=${TARGET_N}" >> "${WD_LOG}"

while true; do
  lines="$(_count_lines)"
  if [ "${lines}" -ge "${TARGET_N}" ]; then
    echo "[$(ts)] target reached ${lines}/${TARGET_N}; watchdog exit" >> "${WD_LOG}"
    exit 0
  fi

  pid="$(_find_pid)"
  if [ -n "${pid}" ] && [ -d "/proc/${pid}" ]; then
    echo "[$(ts)] alive pid=${pid} progress=${lines}/${TARGET_N}" >> "${WD_LOG}"
    sleep "${CHECK_INTERVAL_SEC}"
    continue
  fi

  echo "[$(ts)] run missing progress=${lines}/${TARGET_N}; restarting" >> "${WD_LOG}"
  new_pid="$(_start_run)"
  echo "[$(ts)] restarted pid=${new_pid}" >> "${WD_LOG}"
  sleep "${RESTART_COOLDOWN_SEC}"
done
