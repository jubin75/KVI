#!/usr/bin/env bash
# TruthfulQA — graphrag + kv_prefix + KVI (Exp02 optimization verification), continues running after SSH disconnection.
# Prerequisite: experiments/exp02_hallucination/artifacts/truthfulqa/ and data/truthfulqa_eval.jsonl must exist
#
# Environment variables (optional):
#   TRUTHFULQA_LIMIT   default 25; 0 means no limit (uses full dataset from data)
#   TIMEOUT_S          default 1800
#   RESIDENT_URL       default http://127.0.0.1:18888; set to empty to skip resident mode (model loaded locally in subprocess, slower)
#   WAIT_RESIDENT      default 0; set to 1 to poll /health before starting and sleep RESIDENT_READY_GRACE_SEC
#   RESIDENT_READY_GRACE_SEC default 35
#   BUILD_ORACLE       default 1; set to 0 to skip weak oracle generation and omit --graph_audit_oracle_jsonl
#   RUN_AUDIT          default 0; set to 1 to write graph audit jsonl (used together with BUILD_ORACLE=1)
#   TRUTHFULQA_KVI_MC1_ANSWER default grounded; can be set to injected (only takes effect when --methods includes kv_prefix)
#   TRUTHFULQA_KVI_MAX_NEW_TOKENS default 96
#   KVI_MAX_KV_TRIPLES  not passed by default (uses run_exp01.py default; for ablation, set to 3/4/5 etc.)
#   KVI_DRM_THRESHOLD   not passed by default (uses run_exp01.py default; for ablation, set to 0.03/0.05 etc.)
#   KVI_TOP_K_RELATIONS not passed by default (uses run_exp01.py default; for ablation, set to 2/4 etc.)
#   KVI_MINIMAL_PROMPT  not passed by default; set to 1 to append --kvi_minimal_prompt (KVI KV-only / weak prompt control)
#
# Usage:
#   chmod +x experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh
#   nohup env TRUTHFULQA_LIMIT=25 RUN_AUDIT=1 WAIT_RESIDENT=1 bash experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh \
#     </dev/null >> experiments/exp02_hallucination/results/exp02_truthfulqa_kvi_opt_outer.log 2>&1 &
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
EXP2="${ROOT}/experiments/exp02_hallucination"
RES="${EXP2}/results"
PY="${ROOT}/KVI/bin/python3"
DATE="$(date +%Y%m%d_%H%M%S)"
LIMIT="${TRUTHFULQA_LIMIT:-25}"
TIMEOUT="${TIMEOUT_S:-1800}"
RESIDENT="${RESIDENT_URL:-http://127.0.0.1:18888}"
WAIT_R="${WAIT_RESIDENT:-0}"
GRACE="${RESIDENT_READY_GRACE_SEC:-35}"
BUILD_ORACLE="${BUILD_ORACLE:-1}"
RUN_AUDIT="${RUN_AUDIT:-0}"
KVI_MC1_ANSWER="${TRUTHFULQA_KVI_MC1_ANSWER:-grounded}"
KVI_MAX_NEW_TOKENS="${TRUTHFULQA_KVI_MAX_NEW_TOKENS:-96}"
KVI_MAX_KV_TRIPLES="${KVI_MAX_KV_TRIPLES:-}"
KVI_DRM_THRESHOLD="${KVI_DRM_THRESHOLD:-}"
KVI_TOP_K_RELATIONS="${KVI_TOP_K_RELATIONS:-}"
KVI_MINIMAL_PROMPT="${KVI_MINIMAL_PROMPT:-0}"

OUT="${RES}/truthfulqa_kvi_optimize_${DATE}"
LOG="${RES}/exp02_truthfulqa_kvi_optimize_${DATE}.log"
PID_FILE="${RES}/exp02_truthfulqa_kvi_optimize_${DATE}.pid"
ORACLE="${RES}/truthfulqa_weak_oracle_evidence_${DATE}.jsonl"
AUDIT_JSONL="${RES}/truthfulqa_graph_prompt_audit_kvi_opt_${DATE}.jsonl"

mkdir -p "${RES}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

ts() { date -Iseconds; }

{
  echo "[$(ts)] === run_truthfulqa_kvi_optimization_detached: start ==="
  echo "[$(ts)] OUT_DIR=${OUT} LOG=${LOG}"
} | tee -a "${LOG}"

if [[ "${WAIT_R}" == "1" ]] && [[ -n "${RESIDENT}" ]]; then
  echo "[$(ts)] waiting for resident ${RESIDENT}/health ..." | tee -a "${LOG}"
  ok=0
  for _i in $(seq 1 120); do
    if curl -sf --connect-timeout 2 "${RESIDENT}/health" >/dev/null 2>&1; then
      ok=1
      echo "[$(ts)] resident healthy" | tee -a "${LOG}"
      break
    fi
    sleep 2
  done
  if [[ "${ok}" != "1" ]]; then
    echo "[$(ts)] WARN: resident not healthy; proceeding anyway (graph may fail or local load)" | tee -a "${LOG}"
  else
    echo "[$(ts)] grace sleep ${GRACE}s" | tee -a "${LOG}"
    sleep "${GRACE}"
  fi
fi

if [[ "${BUILD_ORACLE}" == "1" ]]; then
  ORACLE_LIMIT="${LIMIT}"
  if [[ "${ORACLE_LIMIT}" == "0" ]]; then
    ORACLE_LIMIT="0"
  fi
  echo "[$(ts)] build_weak_oracle_evidence → ${ORACLE}" | tee -a "${LOG}"
  "${PY}" "${EXP2}/code/build_weak_oracle_evidence.py" \
    --dataset_jsonl "${EXP2}/data/truthfulqa_eval.jsonl" \
    --sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --out_jsonl "${ORACLE}" \
    --limit "${ORACLE_LIMIT}" \
    --top_k 3 >> "${LOG}" 2>&1
fi

ARGS=(
  -u "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py"
  --dataset "${EXP2}/data/truthfulqa_eval.jsonl"
  --dataset_name TRUTHFULQA
  --model "${ROOT}/models/Qwen2.5-7B-Instruct"
  --graph_index "${EXP2}/artifacts/truthfulqa/graph_index.json"
  --triple_kvbank_dir "${EXP2}/artifacts/truthfulqa/triple_kvbank"
  --graph_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl"
  --ann_kv_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences"
  --ann_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl"
  --ann_semantic_type_specs "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json"
  --ann_pattern_index_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar"
  --ann_sidecar_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar"
  --methods graphrag,kv_prefix,kvi
  --out_dir "${OUT}"
  --timeout_s "${TIMEOUT}"
  --bootstrap_samples 500
  --permutation_samples 1000
  --ann_force_cpu
  --truthfulqa_kvi_mc1_answer "${KVI_MC1_ANSWER}"
  --truthfulqa_kvi_max_new_tokens "${KVI_MAX_NEW_TOKENS}"
)

if [[ -n "${RESIDENT}" ]]; then
  ARGS+=(--inference_service_url "${RESIDENT}")
fi
ARGS+=(--ann_inference_service_url "")

if [[ -n "${KVI_MAX_KV_TRIPLES}" ]]; then
  ARGS+=(--kvi_max_kv_triples "${KVI_MAX_KV_TRIPLES}")
fi
if [[ -n "${KVI_DRM_THRESHOLD}" ]]; then
  ARGS+=(--kvi_drm_threshold "${KVI_DRM_THRESHOLD}")
fi
if [[ -n "${KVI_TOP_K_RELATIONS}" ]]; then
  ARGS+=(--kvi_top_k_relations "${KVI_TOP_K_RELATIONS}")
fi
if [[ "${KVI_MINIMAL_PROMPT}" == "1" ]]; then
  ARGS+=(--kvi_minimal_prompt)
fi

if [[ "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ "${RUN_AUDIT}" == "1" ]]; then
  ARGS+=(--graph_audit_jsonl "${AUDIT_JSONL}")
  if [[ "${BUILD_ORACLE}" == "1" ]]; then
    ARGS+=(--graph_audit_oracle_jsonl "${ORACLE}")
  fi
fi

echo "[$(ts)] launching run_exp01 (pid file ${PID_FILE})" | tee -a "${LOG}"
nohup env PYTHONUNBUFFERED=1 "${PY}" "${ARGS[@]}" >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[$(ts)] started pid=$(cat "${PID_FILE}")" | tee -a "${LOG}"
echo "log: ${LOG}"
echo "out: ${OUT}"
echo "pid: ${PID_FILE}"
