#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
EXP="${ROOT}/experiments/exp02_hallucination"
OUT_ROOT="${EXP}/results/fever_kvi_safe_sweep"
LOG="${OUT_ROOT}/sweep.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "${OUT_ROOT}"

# Keep this sweep CPU-safe for local pipeline stability.
export CUDA_VISIBLE_DEVICES=""

DATASET="${EXP}/data/fever_eval.jsonl"
MODEL="${ROOT}/models/Qwen2.5-7B-Instruct"
GRAPH_INDEX="${EXP}/artifacts/fever/graph_index.json"
TRIPLE_KVBANK="${EXP}/artifacts/fever/triple_kvbank"
SENTS="${EXP}/artifacts/fever/sentences.tagged.jsonl"
ANN_KV="${EXP}/artifacts/fever/kvbank_sentences"
ANN_SPECS="${EXP}/artifacts/fever/kvbank_sentences/pattern_sidecar/semantic_type_specs.json"
ANN_SIDECAR="${EXP}/artifacts/fever/kvbank_sentences/pattern_sidecar"
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:18888}"

echo "[$(date -Iseconds)] FEVER KVI safe sweep start" | tee -a "${LOG}"
echo "dataset=${DATASET}" | tee -a "${LOG}"
echo "service=${SERVICE_URL}" | tee -a "${LOG}"

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: missing dataset ${DATASET}" | tee -a "${LOG}"
  exit 1
fi

printf "run_id,kvi_max_kv_triples,kvi_drm_threshold,kvi_top_k_relations,graphrag_em,kvi_em,delta_kvi_minus_graphrag\n" > "${SUMMARY_CSV}"

for T in 1 2; do
  for D in 0.10 0.15 0.20; do
    RUN_ID="t${T}_d${D}_r1"
    OUT_DIR="${OUT_ROOT}/${RUN_ID}"
    mkdir -p "${OUT_DIR}"
    echo "[$(date -Iseconds)] RUN ${RUN_ID} start" | tee -a "${LOG}"

    python "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py" \
      --dataset "${DATASET}" \
      --dataset_name FEVER \
      --model "${MODEL}" \
      --graph_index "${GRAPH_INDEX}" \
      --triple_kvbank_dir "${TRIPLE_KVBANK}" \
      --graph_sentences_jsonl "${SENTS}" \
      --ann_kv_dir "${ANN_KV}" \
      --ann_sentences_jsonl "${SENTS}" \
      --ann_semantic_type_specs "${ANN_SPECS}" \
      --ann_pattern_index_dir "${ANN_SIDECAR}" \
      --ann_sidecar_dir "${ANN_SIDECAR}" \
      --methods graphrag,kvi \
      --out_dir "${OUT_DIR}" \
      --timeout_s 600 \
      --bootstrap_samples 500 \
      --permutation_samples 1000 \
      --inference_service_url "${SERVICE_URL}" \
      --ann_inference_service_url "${SERVICE_URL}" \
      --kvi_max_kv_triples "${T}" \
      --kvi_drm_threshold "${D}" \
      --kvi_top_k_relations 1 \
      --kvi_reconcile_no_kv_decode \
      --no-kvi_minimal_prompt \
      >> "${LOG}" 2>&1

    python - <<'PY' "${OUT_DIR}/summary.json" "${SUMMARY_CSV}" "${RUN_ID}" "${T}" "${D}"
import json, sys
from pathlib import Path

summary = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
run_id = sys.argv[3]
t = sys.argv[4]
d = sys.argv[5]

obj = json.loads(summary.read_text(encoding="utf-8"))
rows = obj.get("methods", [])
emap = {str(r.get("method_key")): float(r.get("em") or 0.0) for r in rows if isinstance(r, dict)}
gr = emap.get("graphrag", 0.0)
kvi = emap.get("kvi", 0.0)
delta = kvi - gr
with csv_path.open("a", encoding="utf-8") as f:
    f.write(f"{run_id},{t},{d},1,{gr:.4f},{kvi:.4f},{delta:.4f}\n")
print(json.dumps({"run_id": run_id, "graphrag_em": gr, "kvi_em": kvi, "delta": delta}, ensure_ascii=False))
PY

    echo "[$(date -Iseconds)] RUN ${RUN_ID} done" | tee -a "${LOG}"
  done
done

echo "[$(date -Iseconds)] FEVER KVI safe sweep done. summary=${SUMMARY_CSV}" | tee -a "${LOG}"
