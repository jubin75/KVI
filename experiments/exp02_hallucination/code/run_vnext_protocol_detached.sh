#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
PY="${ROOT}/KVI/bin/python3"
EXP1="${ROOT}/experiments/exp01_main_qa/code/run_exp01.py"
EXP2="${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py"
GRAPH_RUN="${ROOT}/scripts/run_graph_inference.py"
RES_DIR="${ROOT}/experiments/exp02_hallucination/results"
ART="${ROOT}/experiments/exp02_hallucination/artifacts"
DATA="${ROOT}/experiments/exp02_hallucination/data"
MODEL="${MODEL:-${ROOT}/models/Qwen2.5-7B-Instruct}"
RESIDENT_URL="${RESIDENT_URL:-http://127.0.0.1:18888}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RES_DIR}"
LOG="${RES_DIR}/exp02_vnext_protocol_${TS}.log"
PID_FILE="${RES_DIR}/exp02_vnext_protocol_${TS}.pid"

echo $$ > "${PID_FILE}"
echo "[$(date -Iseconds)] start vNext protocol pipeline ts=${TS}" | tee -a "${LOG}"

if ! curl -sf --connect-timeout 2 "${RESIDENT_URL}/health" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] resident not healthy, starting resident service..." | tee -a "${LOG}"
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
    bash "${ROOT}/experiments/exp01_main_qa/code/start_resident_service.sh" \
    >> "${RES_DIR}/resident_18888_gpu.log" 2>&1 &
  for _i in $(seq 1 90); do
    if curl -sf --connect-timeout 2 "${RESIDENT_URL}/health" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] resident healthy" | tee -a "${LOG}"
      break
    fi
    sleep 2
  done
fi

echo "[$(date -Iseconds)] step1 audit-only verifier smoke" | tee -a "${LOG}"
"${PY}" "${GRAPH_RUN}" \
  --model "${MODEL}" \
  --prompt "Who is faster in real life, a tortoise or a hare?" \
  --graph_index "${ART}/truthfulqa/graph_index.json" \
  --sentences_jsonl "${ART}/truthfulqa/sentences.tagged.jsonl" \
  --schema_blocks_jsonl "${ART}/truthfulqa/blocks.schema.jsonl" \
  --schema_kv_dir "${ART}/truthfulqa/kvbank_schema" \
  --triple_kvbank_dir "${ART}/truthfulqa/triple_kvbank" \
  --enable_kvi \
  --openqa_mode \
  --use_chat_template \
  --audit_only \
  --kvi_execution_mode schema_verify_then_evidence_write \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] step2 TruthfulQA smoke25" | tee -a "${LOG}"
"${PY}" "${EXP1}" \
  --dataset "${DATA}/truthfulqa_eval.jsonl" \
  --dataset_name TRUTHFULQA \
  --model "${MODEL}" \
  --graph_index "${ART}/truthfulqa/graph_index.json" \
  --triple_kvbank_dir "${ART}/truthfulqa/triple_kvbank" \
  --graph_sentences_jsonl "${ART}/truthfulqa/sentences.tagged.jsonl" \
  --graph_schema_blocks_jsonl "${ART}/truthfulqa/blocks.schema.jsonl" \
  --graph_schema_kv_dir "${ART}/truthfulqa/kvbank_schema" \
  --ann_kv_dir "${ART}/truthfulqa/kvbank_sentences" \
  --ann_sentences_jsonl "${ART}/truthfulqa/sentences.tagged.jsonl" \
  --ann_semantic_type_specs "${ART}/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json" \
  --ann_pattern_index_dir "${ART}/truthfulqa/kvbank_sentences/pattern_sidecar" \
  --ann_sidecar_dir "${ART}/truthfulqa/kvbank_sentences/pattern_sidecar" \
  --methods graphrag,kvi_triple_legacy,kvi_schema_verifier,kvi_noinject_planner \
  --limit 25 \
  --out_dir "${RES_DIR}/truthfulqa_vnext_smoke25_${TS}" \
  --timeout_s 1800 \
  --inference_service_url "${RESIDENT_URL}" \
  --ann_inference_service_url "" \
  --ann_force_cpu \
  --truthfulqa_kvi_mc1_answer grounded \
  --graph_audit_jsonl "${RES_DIR}/truthfulqa_vnext_smoke25_${TS}.audit.jsonl" \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] step3 FEVER smoke50" | tee -a "${LOG}"
"${PY}" "${EXP1}" \
  --dataset "${DATA}/fever_eval.jsonl" \
  --dataset_name FEVER \
  --model "${MODEL}" \
  --graph_index "${ART}/fever/graph_index.json" \
  --triple_kvbank_dir "${ART}/fever/triple_kvbank" \
  --graph_sentences_jsonl "${ART}/fever/sentences.tagged.jsonl" \
  --graph_schema_blocks_jsonl "${ART}/fever/blocks.schema.jsonl" \
  --graph_schema_kv_dir "${ART}/fever/kvbank_schema" \
  --ann_kv_dir "${ART}/fever/kvbank_sentences" \
  --ann_sentences_jsonl "${ART}/fever/sentences.tagged.jsonl" \
  --ann_semantic_type_specs "${ART}/fever/kvbank_sentences/pattern_sidecar/semantic_type_specs.json" \
  --ann_pattern_index_dir "${ART}/fever/kvbank_sentences/pattern_sidecar" \
  --ann_sidecar_dir "${ART}/fever/kvbank_sentences/pattern_sidecar" \
  --methods graphrag,kvi_triple_legacy,kvi_schema_verifier,kvi_noinject_planner \
  --limit 50 \
  --out_dir "${RES_DIR}/fever_vnext_smoke50_${TS}" \
  --timeout_s 1800 \
  --inference_service_url "${RESIDENT_URL}" \
  --ann_inference_service_url "" \
  --ann_force_cpu \
  --graph_audit_jsonl "${RES_DIR}/fever_vnext_smoke50_${TS}.audit.jsonl" \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] step4 Exp02 vnext smoke25" | tee -a "${LOG}"
"${PY}" "${EXP2}" \
  --root "${ROOT}" \
  --model "${MODEL}" \
  --resident_url "${RESIDENT_URL}" \
  --methods graphrag,kvi_triple_legacy,kvi_schema_verifier,kvi_noinject_planner \
  --result_tag "vnext_smoke25_${TS}" \
  --only_datasets truthfulqa,fever \
  --limit 25 \
  --skip_mirror_and_prepare \
  --reuse_artifacts \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] step5 Exp02 vnext pre100" | tee -a "${LOG}"
"${PY}" "${EXP2}" \
  --root "${ROOT}" \
  --model "${MODEL}" \
  --resident_url "${RESIDENT_URL}" \
  --methods graphrag,kvi_triple_legacy,kvi_schema_verifier,kvi_noinject_planner \
  --result_tag "vnext_pre100_${TS}" \
  --only_datasets truthfulqa,fever \
  --limit 100 \
  --skip_mirror_and_prepare \
  --reuse_artifacts \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] done vNext protocol pipeline ts=${TS}" | tee -a "${LOG}"
