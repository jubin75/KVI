#!/usr/bin/env bash
# FEVER minimal diagnostic: GraphRAG vs KVI, N=200, NO kvi_reconcile_no_kv_decode (avoids DrugBank-tie -> GraphRAG collapse).
set -euo pipefail

ROOT="/home/zd/dev/KVI"
EXP="${ROOT}/experiments/exp02_hallucination"
OUT="${EXP}/results/fever_kvi_minimal_noreconcile_limit200"
LOG="${OUT}/run.log"
mkdir -p "${OUT}"

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

{
  echo "[$(date -Iseconds)] start minimal FEVER run (limit=200, no reconcile)"
  echo "out_dir=${OUT}"
  echo "service=${SERVICE_URL}"
  "${ROOT}/KVI/bin/python" -u "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py" \
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
    --out_dir "${OUT}" \
    --limit 200 \
    --timeout_s 600 \
    --bootstrap_samples 500 \
    --permutation_samples 1000 \
    --inference_service_url "${SERVICE_URL}" \
    --ann_inference_service_url "${SERVICE_URL}" \
    --kvi_max_kv_triples 1 \
    --kvi_drm_threshold 0.10 \
    --kvi_top_k_relations 1 \
    --no-kvi_minimal_prompt
  echo "[$(date -Iseconds)] done"
} >> "${LOG}" 2>&1
