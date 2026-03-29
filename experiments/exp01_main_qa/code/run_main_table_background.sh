#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/exp01_main_qa/code/run_main_table_background.sh \
#     --model /path/to/base-llm \
#     --tag qwen25_7b \
#     --service_url http://127.0.0.1:18888
#
# Notes:
# - This script runs Exp01 for Hotpot/NQ/MedHop sequentially with --resume.
# - It then aggregates a main table under results/main_table_<tag>/.
# - Safe for unstable network when launched with nohup.

ROOT="/home/zd/dev/KVI"
VENV="$ROOT/KVI/bin/activate"

MODEL=""
TAG=""
SERVICE_URL=""
BOOTSTRAP_SAMPLES="800"
PERMUTATION_SAMPLES="1500"
TIMEOUT_S="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --service_url) SERVICE_URL="${2:-}"; shift 2 ;;
    --bootstrap_samples) BOOTSTRAP_SAMPLES="${2:-}"; shift 2 ;;
    --permutation_samples) PERMUTATION_SAMPLES="${2:-}"; shift 2 ;;
    --timeout_s) TIMEOUT_S="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TAG" ]]; then
  echo "Missing required args --model and --tag"
  exit 2
fi

cd "$ROOT"
source "$VENV"

HOTPOT_OUT="experiments/exp01_main_qa/results/multihop_hotpot_n120_fullmethods_${TAG}"
NQ_OUT="experiments/exp01_main_qa/results/nq_smoke100_fullmethods_${TAG}"
MEDHOP_OUT="experiments/exp01_main_qa/results/medhop_n40_fullmethods_${TAG}"
MAIN_OUT="experiments/exp01_main_qa/results/main_table_${TAG}"

mkdir -p "$HOTPOT_OUT" "$NQ_OUT" "$MEDHOP_OUT" "$MAIN_OUT"

SERVICE_ARGS=()
if [[ -n "${SERVICE_URL// }" ]]; then
  SERVICE_ARGS=(--inference_service_url "$SERVICE_URL")
fi

python -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp01_main_qa/data/multihop_hotpot/hotpot_eval_multihop.jsonl \
  --dataset_name HotpotQA_multihop \
  --model "$MODEL" \
  --graph_index experiments/exp01_main_qa/artifacts/hotpot_multihop/graph_index.json \
  --triple_kvbank_dir experiments/exp01_main_qa/artifacts/hotpot_multihop/triple_kvbank \
  --graph_sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot_multihop/sentences.jsonl \
  --ann_kv_dir experiments/exp01_main_qa/artifacts/hotpot_multihop/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot_multihop/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp01_main_qa/artifacts/hotpot_multihop/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp01_main_qa/artifacts/hotpot_multihop/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp01_main_qa/artifacts/hotpot_multihop/kvbank_sentences/pattern_sidecar \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --out_dir "$HOTPOT_OUT" \
  --resume \
  "${SERVICE_ARGS[@]}" \
  --timeout_s "$TIMEOUT_S" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --permutation_samples "$PERMUTATION_SAMPLES"

python -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp01_main_qa/data/benchmarks/nq_eval.jsonl \
  --dataset_name NQ \
  --model "$MODEL" \
  --graph_index experiments/exp01_main_qa/artifacts/nq/graph_index.json \
  --triple_kvbank_dir experiments/exp01_main_qa/artifacts/nq/triple_kvbank \
  --graph_sentences_jsonl experiments/exp01_main_qa/artifacts/nq/sentences.jsonl \
  --ann_kv_dir experiments/exp01_main_qa/artifacts/nq/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp01_main_qa/artifacts/nq/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp01_main_qa/artifacts/nq/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp01_main_qa/artifacts/nq/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp01_main_qa/artifacts/nq/kvbank_sentences/pattern_sidecar \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --limit 100 \
  --out_dir "$NQ_OUT" \
  --resume \
  "${SERVICE_ARGS[@]}" \
  --timeout_s "$TIMEOUT_S" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --permutation_samples "$PERMUTATION_SAMPLES"

python -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp01_main_qa/data/medhop_benchmarks_n40/medhop_eval.jsonl \
  --dataset_name MedHopQA_n40 \
  --model "$MODEL" \
  --graph_index experiments/exp01_main_qa/artifacts/medhop_n40/graph_index.json \
  --triple_kvbank_dir experiments/exp01_main_qa/artifacts/medhop_n40/triple_kvbank \
  --graph_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_kv_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --out_dir "$MEDHOP_OUT" \
  --resume \
  "${SERVICE_ARGS[@]}" \
  --timeout_s "$TIMEOUT_S" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --permutation_samples "$PERMUTATION_SAMPLES"

python experiments/exp01_main_qa/code/aggregate_exp01.py \
  --hotpot_summary "$HOTPOT_OUT/summary.json" \
  --medhop_summary "$MEDHOP_OUT/summary.json" \
  --nq_summary "$NQ_OUT/summary.json" \
  --out_dir "$MAIN_OUT"

echo "[done] main table generated at: $MAIN_OUT/main_table.md"
