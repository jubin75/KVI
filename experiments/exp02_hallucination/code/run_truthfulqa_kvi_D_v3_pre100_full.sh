#!/usr/bin/env bash
# D v3: pre100 then full500, single serial job. Requires healthy graph resident.
#   RESIDENT_URL (default http://127.0.0.1:18888)
#   WAIT_RESIDENT (default 1): poll /health before run_exp01; abort if still down
#   RESIDENT_READY_GRACE_SEC (default 35): sleep after health OK before first infer
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
PY="${ROOT}/KVI/bin/python3"
EXP2="${ROOT}/experiments/exp02_hallucination"
RES="${EXP2}/results"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RESIDENT="${RESIDENT_URL-http://127.0.0.1:18888}"
WAIT_R="${WAIT_RESIDENT:-1}"
GRACE="${RESIDENT_READY_GRACE_SEC:-35}"

LOG="${RES}/exp02_truthfulqa_kvi_D_full_v3_${TS}.log"
OUT_PRE="${RES}/truthfulqa_kvi_optimize_D_v3_pre100_${TS}"
OUT_FULL="${RES}/truthfulqa_kvi_optimize_D_v3_full_${TS}"

mkdir -p "${RES}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

if [[ "${WAIT_R}" == "1" ]] && [[ -n "${RESIDENT}" ]]; then
  echo "[$(date -Iseconds)] waiting for resident ${RESIDENT}/health ..." | tee -a "${LOG}"
  ok=0
  for _i in $(seq 1 120); do
    if curl -sf --connect-timeout 2 "${RESIDENT}/health" >/dev/null 2>&1; then
      ok=1
      echo "[$(date -Iseconds)] resident healthy" | tee -a "${LOG}"
      break
    fi
    sleep 2
  done
  if [[ "${ok}" != "1" ]]; then
    echo "[$(date -Iseconds)] FATAL: resident not healthy after wait; abort (avoid partial D v3)" | tee -a "${LOG}"
    exit 1
  fi
  echo "[$(date -Iseconds)] grace sleep ${GRACE}s" | tee -a "${LOG}"
  sleep "${GRACE}"
fi

run_one () {
  local limit="$1"
  local outdir="$2"
  local tag="$3"
  echo "[$(date -Iseconds)] start ${tag} limit=${limit} out=${outdir}" | tee -a "${LOG}"
  ARGS=(
    -u experiments/exp01_main_qa/code/run_exp01.py
    --dataset experiments/exp02_hallucination/data/truthfulqa_eval.jsonl
    --dataset_name TRUTHFULQA
    --model models/Qwen2.5-7B-Instruct
    --graph_index experiments/exp02_hallucination/artifacts/truthfulqa/graph_index.json
    --triple_kvbank_dir experiments/exp02_hallucination/artifacts/truthfulqa/triple_kvbank
    --graph_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl
    --ann_kv_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences
    --ann_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl
    --ann_semantic_type_specs experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json
    --ann_pattern_index_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar
    --ann_sidecar_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar
    --methods graphrag,kv_prefix,kvi
    --out_dir "${outdir}"
    --timeout_s 1800
    --bootstrap_samples 500
    --permutation_samples 1000
    --ann_force_cpu
    --ann_inference_service_url ""
    --truthfulqa_kvi_mc1_answer grounded
    --truthfulqa_kvi_max_new_tokens 96
    --kvi_two_stage_kv_then_evidence
  )
  if [[ -n "${RESIDENT}" ]]; then
    ARGS+=(--inference_service_url "${RESIDENT}")
  fi
  if [[ "${limit}" != "0" ]]; then
    ARGS+=(--limit "${limit}")
  fi
  "${PY}" "${ARGS[@]}" 2>&1 | tee -a "${LOG}"
  echo "[$(date -Iseconds)] done limit=${limit} out=${outdir}" | tee -a "${LOG}"
}

echo "[$(date -Iseconds)] orchestrator TS=${TS}" | tee -a "${LOG}"
run_one 100 "${OUT_PRE}" "D v3 (balanced stage2)"
run_one 0 "${OUT_FULL}" "D v3 (balanced stage2)"
echo "[$(date -Iseconds)] all done pre100+full v3" | tee -a "${LOG}"
