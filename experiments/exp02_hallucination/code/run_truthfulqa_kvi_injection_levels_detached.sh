#!/usr/bin/env bash
# TruthfulQA n=25 KVI injection-level ablation (resident/GPU-friendly, detached-safe).
# Compare kvi_max_kv_triples levels while keeping dataset/artifacts fixed.
#
# Env (optional):
#   TRUTHFULQA_LIMIT   default 25
#   TIMEOUT_S          default 1800
#   RESIDENT_URL       default http://127.0.0.1:18888
#   WAIT_RESIDENT      default 1
#   RESIDENT_READY_GRACE_SEC default 35
#   RUN_AUDIT          default 1
#   BUILD_ORACLE       default 1
#   KVI_LEVELS         default "0 2 4 6"  (space-separated)
#   KVI_DRM_THRESHOLD  default 0.05
#   KVI_TOP_RELATIONS  default 4
#
# Usage:
#   chmod +x experiments/exp02_hallucination/code/run_truthfulqa_kvi_injection_levels_detached.sh
#   nohup bash experiments/exp02_hallucination/code/run_truthfulqa_kvi_injection_levels_detached.sh \
#     </dev/null >> experiments/exp02_hallucination/results/exp02_truthfulqa_kvi_levels_outer.log 2>&1 &
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
EXP2="${ROOT}/experiments/exp02_hallucination"
RES="${EXP2}/results"
PY="${ROOT}/KVI/bin/python3"
DATE="$(date +%Y%m%d_%H%M%S)"
LIMIT="${TRUTHFULQA_LIMIT:-25}"
TIMEOUT="${TIMEOUT_S:-1800}"
RESIDENT="${RESIDENT_URL:-http://127.0.0.1:18888}"
WAIT_R="${WAIT_RESIDENT:-1}"
GRACE="${RESIDENT_READY_GRACE_SEC:-35}"
RUN_AUDIT="${RUN_AUDIT:-1}"
BUILD_ORACLE="${BUILD_ORACLE:-1}"
KVI_LEVELS="${KVI_LEVELS:-0 2 4 6}"
KVI_DRM="${KVI_DRM_THRESHOLD:-0.05}"
KVI_RELS="${KVI_TOP_RELATIONS:-4}"

RUN_ROOT="${RES}/truthfulqa_kvi_injection_levels_${DATE}"
LOG="${RES}/exp02_truthfulqa_kvi_injection_levels_${DATE}.log"
PID_FILE="${RES}/exp02_truthfulqa_kvi_injection_levels_${DATE}.pid"
ORACLE="${RUN_ROOT}/truthfulqa_weak_oracle_evidence_${DATE}.jsonl"
SUMMARY_MD="${RUN_ROOT}/injection_levels_summary.md"
SUMMARY_CSV="${RUN_ROOT}/injection_levels_summary.csv"

mkdir -p "${RES}" "${RUN_ROOT}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

ts() { date -Iseconds; }

echo "[$(ts)] === run_truthfulqa_kvi_injection_levels_detached: start ===" | tee -a "${LOG}"
echo "[$(ts)] RUN_ROOT=${RUN_ROOT}" | tee -a "${LOG}"
echo "[$(ts)] levels=${KVI_LEVELS} limit=${LIMIT} drm=${KVI_DRM} rels=${KVI_RELS}" | tee -a "${LOG}"

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
    echo "[$(ts)] WARN: resident not healthy; proceeding anyway" | tee -a "${LOG}"
  else
    echo "[$(ts)] grace sleep ${GRACE}s" | tee -a "${LOG}"
    sleep "${GRACE}"
  fi
fi

if [[ "${BUILD_ORACLE}" == "1" ]]; then
  echo "[$(ts)] build_weak_oracle_evidence -> ${ORACLE}" | tee -a "${LOG}"
  "${PY}" "${EXP2}/code/build_weak_oracle_evidence.py" \
    --dataset_jsonl "${EXP2}/data/truthfulqa_eval.jsonl" \
    --sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --out_jsonl "${ORACLE}" \
    --limit "${LIMIT}" \
    --top_k 3 >> "${LOG}" 2>&1
fi

echo "kvi_level,mc1_proxy,mc2_proxy,graphrag_mc2,kvprefix_mc2,delta_vs_graphrag,delta_vs_kvprefix,summary_json" > "${SUMMARY_CSV}"
{
  echo "# TruthfulQA KVI Injection Levels (n=${LIMIT})"
  echo
  echo "| KVI max_kv_triples | KVI MC1 | KVI MC2 | GraphRAG MC2 | KV Prefix MC2 | Δ(KVI-GR) | Δ(KVI-KVPrefix) |"
  echo "|---:|---:|---:|---:|---:|---:|---:|"
} > "${SUMMARY_MD}"

for LV in ${KVI_LEVELS}; do
  OUT="${RUN_ROOT}/level_${LV}"
  AUDIT="${RUN_ROOT}/audit_level_${LV}.jsonl"
  mkdir -p "${OUT}"
  echo "[$(ts)] >>> level=${LV} start" | tee -a "${LOG}"

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
    --truthfulqa_kvi_mc1_answer grounded
    --truthfulqa_kvi_max_new_tokens 96
    --kvi_max_kv_triples "${LV}"
    --kvi_drm_threshold "${KVI_DRM}"
    --kvi_top_k_relations "${KVI_RELS}"
    --no-truthfulqa_kvi_tuned_overrides
    --truthfulqa_kvi_minimal_prompt
    --truthfulqa_kvi_reconcile
    --limit "${LIMIT}"
    --ann_inference_service_url ""
  )
  if [[ -n "${RESIDENT}" ]]; then
    ARGS+=(--inference_service_url "${RESIDENT}")
  fi
  if [[ "${RUN_AUDIT}" == "1" ]]; then
    ARGS+=(--graph_audit_jsonl "${AUDIT}")
    if [[ "${BUILD_ORACLE}" == "1" ]]; then
      ARGS+=(--graph_audit_oracle_jsonl "${ORACLE}")
    fi
  fi

  "${PY}" "${ARGS[@]}" >> "${LOG}" 2>&1
  echo "[$(ts)] <<< level=${LV} done" | tee -a "${LOG}"

  "${PY}" - "${OUT}/summary.json" "${LV}" "${SUMMARY_CSV}" "${SUMMARY_MD}" >> "${LOG}" 2>&1 <<'PY'
import json,sys
from pathlib import Path
summary=Path(sys.argv[1]); lv=sys.argv[2]; csvp=Path(sys.argv[3]); mdp=Path(sys.argv[4])
j=json.loads(summary.read_text(encoding='utf-8'))
mm={r["method_key"]:r for r in j.get("methods",[])}
k=mm.get("kvi",{}); g=mm.get("graphrag",{}); p=mm.get("kv_prefix",{})
km1=float(k.get("truthfulqa_mc1_proxy",0)); km2=float(k.get("truthfulqa_mc2_proxy",0))
gm2=float(g.get("truthfulqa_mc2_proxy",0)); pm2=float(p.get("truthfulqa_mc2_proxy",0))
dkg=km2-gm2; dkp=km2-pm2
with csvp.open("a",encoding="utf-8") as f:
    f.write(f"{lv},{km1:.4f},{km2:.4f},{gm2:.4f},{pm2:.4f},{dkg:.4f},{dkp:.4f},{summary}\n")
with mdp.open("a",encoding="utf-8") as f:
    f.write(f"| {lv} | {km1:.4f} | {km2:.4f} | {gm2:.4f} | {pm2:.4f} | {dkg:+.4f} | {dkp:+.4f} |\n")
PY
done

echo "[$(ts)] all levels done" | tee -a "${LOG}"
echo "summary_md: ${SUMMARY_MD}" | tee -a "${LOG}"
echo "summary_csv: ${SUMMARY_CSV}" | tee -a "${LOG}"
echo $$ > "${PID_FILE}"
