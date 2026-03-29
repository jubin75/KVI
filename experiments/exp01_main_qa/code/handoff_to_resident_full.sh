#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
VENV="${ROOT}/KVI/bin/activate"
RESULT_ROOT="${ROOT}/experiments/exp01_main_qa/results"
CODE_ROOT="${ROOT}/experiments/exp01_main_qa/code"
DATA_ROOT="${ROOT}/experiments/exp01_main_qa/data/benchmarks"
ART_ROOT="${ROOT}/experiments/exp01_main_qa/artifacts"
MODEL="${ROOT}/models/Qwen2.5-7B-Instruct"

SMOKE_PID="${1:-623345}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18888}"
SERVICE_URL="http://${HOST}:${PORT}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-2000}"

cd "${ROOT}"
source "${VENV}"
mkdir -p "${RESULT_ROOT}/official_full_resident"

echo "[handoff] wait smoke pid=${SMOKE_PID}"
while kill -0 "${SMOKE_PID}" 2>/dev/null; do
  sleep 20
done
echo "[handoff] smoke finished"

if ! python - <<'PY'
import json, urllib.request, sys
url = "http://127.0.0.1:18888/health"
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        obj = json.loads(r.read().decode("utf-8"))
    sys.exit(0 if isinstance(obj, dict) and obj.get("ok") else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "[handoff] starting resident service ${HOST}:${PORT}"
  nohup bash -lc "
    cd ${ROOT}
    source ${VENV}
    python ${CODE_ROOT}/exp01_resident_infer_service.py --host ${HOST} --port ${PORT}
  " > "${RESULT_ROOT}/official_full_resident/resident_service.log" 2>&1 &
  sleep 8
fi

echo "[handoff] run full hotpot"
python "${CODE_ROOT}/run_exp01.py" \
  --dataset "${DATA_ROOT}/hotpot_eval.jsonl" \
  --dataset_name HotpotQA \
  --model "${MODEL}" \
  --graph_index "${ART_ROOT}/hotpot/graph_index.json" \
  --triple_kvbank_dir "${ART_ROOT}/hotpot/triple_kvbank" \
  --graph_sentences_jsonl "${ART_ROOT}/hotpot/sentences.jsonl" \
  --ann_kv_dir "${ART_ROOT}/hotpot/kvbank_sentences" \
  --ann_sentences_jsonl "${ART_ROOT}/hotpot/sentences.tagged.jsonl" \
  --ann_semantic_type_specs "${ART_ROOT}/hotpot/kvbank_sentences/semantic_type_specs.json" \
  --ann_pattern_index_dir "${ART_ROOT}/hotpot/kvbank_sentences/pattern_sidecar" \
  --ann_sidecar_dir "${ART_ROOT}/hotpot/kvbank_sentences" \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_dir "${RESULT_ROOT}/official_full_resident/hotpot" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --permutation_samples "${PERMUTATION_SAMPLES}" \
  --inference_service_url "${SERVICE_URL}"

echo "[handoff] run full nq"
python "${CODE_ROOT}/run_exp01.py" \
  --dataset "${DATA_ROOT}/nq_eval.jsonl" \
  --dataset_name NQ \
  --model "${MODEL}" \
  --graph_index "${ART_ROOT}/nq/graph_index.json" \
  --triple_kvbank_dir "${ART_ROOT}/nq/triple_kvbank" \
  --graph_sentences_jsonl "${ART_ROOT}/nq/sentences.jsonl" \
  --ann_kv_dir "${ART_ROOT}/nq/kvbank_sentences" \
  --ann_sentences_jsonl "${ART_ROOT}/nq/sentences.tagged.jsonl" \
  --ann_semantic_type_specs "${ART_ROOT}/nq/kvbank_sentences/semantic_type_specs.json" \
  --ann_pattern_index_dir "${ART_ROOT}/nq/kvbank_sentences/pattern_sidecar" \
  --ann_sidecar_dir "${ART_ROOT}/nq/kvbank_sentences" \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_dir "${RESULT_ROOT}/official_full_resident/nq" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --permutation_samples "${PERMUTATION_SAMPLES}" \
  --inference_service_url "${SERVICE_URL}"

echo "[handoff] aggregate"
python "${CODE_ROOT}/aggregate_exp01.py" \
  --hotpot_summary "${RESULT_ROOT}/official_full_resident/hotpot/summary.json" \
  --nq_summary "${RESULT_ROOT}/official_full_resident/nq/summary.json" \
  --out_dir "${RESULT_ROOT}/official_full_resident/main_table"

echo "[handoff] done"
