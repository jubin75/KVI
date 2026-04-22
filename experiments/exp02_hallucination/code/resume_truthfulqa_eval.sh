#!/usr/bin/env bash
# Restore a predictions.jsonl backup, strip cached MC proxy fields, resume run_exp01, then merge unified summaries + plots.
# Requires resident on RESIDENT_URL (default http://127.0.0.1:18888).
set -euo pipefail
ROOT="${ROOT:-/home/zd/dev/KVI}"
RESIDENT_URL="${RESIDENT_URL:-http://127.0.0.1:18888}"
EXP2="${ROOT}/experiments/exp02_hallucination"
OUT="${EXP2}/results/truthfulqa_fullmethods_qwen25_7b"
BAK="${1:-${OUT}/predictions.jsonl.bak_20260412_203109}"

if [[ ! -f "$BAK" ]]; then
  echo "Backup not found: $BAK" >&2
  exit 1
fi

cp -a "$BAK" "${OUT}/predictions.jsonl"
"${ROOT}/KVI/bin/python3" "${EXP2}/code/strip_truthfulqa_mc_proxy_fields.py" "${OUT}/predictions.jsonl"

export PYTHONUNBUFFERED=1
cd "${ROOT}"
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
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --out_dir "${OUT}" \
  --timeout_s 600 \
  --bootstrap_samples 1000 \
  --permutation_samples 2000 \
  --inference_service_url "${RESIDENT_URL}" \
  --ann_inference_service_url "" \
  --ann_force_cpu \
  --resume \
  >> "${EXP2}/results/exp02_truthfulqa_resume_eval.log" 2>&1 &

echo "Started resume PID=$! (log: ${EXP2}/results/exp02_truthfulqa_resume_eval.log)"
echo "When finished (wc -l predictions.jsonl == 500), run:"
echo "  ${ROOT}/KVI/bin/python3 ${EXP2}/code/merge_exp02_unified_summaries.py"
echo "  ${ROOT}/KVI/bin/python3 ${EXP2}/code/plot_unified_hallucination_bars.py"
echo "  ${ROOT}/KVI/bin/python3 ${EXP2}/code/plot_hallucination_proxy_bars.py --paper --three_panel_unified --fever_label_figure --truthfulqa_mc_figure"
