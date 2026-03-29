#!/usr/bin/env bash
set -euo pipefail

# Official-style staged pipeline:
# 1) smoke (100 each)  2) full validation
# Designed for unstable remote sessions: run with nohup.

ROOT="/home/zd/dev/KVI"
VENV="${ROOT}/KVI/bin/activate"
MODEL="${ROOT}/models/Qwen2.5-7B-Instruct"
RESULT_ROOT="${ROOT}/experiments/exp01_main_qa/results"
DATA_ROOT="${ROOT}/experiments/exp01_main_qa/data/benchmarks"
ART_ROOT="${ROOT}/experiments/exp01_main_qa/artifacts"
CODE_ROOT="${ROOT}/experiments/exp01_main_qa/code"

SMOKE_LIMIT="${SMOKE_LIMIT:-100}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-2000}"
RUN_FULL_AFTER_SMOKE="${RUN_FULL_AFTER_SMOKE:-1}"

run_one() {
  local dataset_name="$1" dataset_jsonl="$2" out_dir="$3" art_dir="$4" extra_limit="$5"
  python "${CODE_ROOT}/run_exp01.py" \
    --dataset "${dataset_jsonl}" \
    --dataset_name "${dataset_name}" \
    --model "${MODEL}" \
    --graph_index "${art_dir}/graph_index.json" \
    --triple_kvbank_dir "${art_dir}/triple_kvbank" \
    --graph_sentences_jsonl "${art_dir}/sentences.jsonl" \
    --ann_kv_dir "${art_dir}/kvbank_sentences" \
    --ann_sentences_jsonl "${art_dir}/sentences.tagged.jsonl" \
    --ann_semantic_type_specs "${art_dir}/kvbank_sentences/semantic_type_specs.json" \
    --ann_pattern_index_dir "${art_dir}/kvbank_sentences/pattern_sidecar" \
    --ann_sidecar_dir "${art_dir}/kvbank_sentences" \
    --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
    --out_dir "${out_dir}" \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --permutation_samples "${PERMUTATION_SAMPLES}" \
    ${extra_limit}
}

main() {
  cd "${ROOT}"
  # shellcheck disable=SC1090
  source "${VENV}"

  mkdir -p "${RESULT_ROOT}/smoke100" "${RESULT_ROOT}/official_full"

  run_one "HotpotQA" "${DATA_ROOT}/hotpot_eval.jsonl" "${RESULT_ROOT}/smoke100/hotpot" "${ART_ROOT}/hotpot" "--limit ${SMOKE_LIMIT}"
  run_one "NQ" "${DATA_ROOT}/nq_eval.jsonl" "${RESULT_ROOT}/smoke100/nq" "${ART_ROOT}/nq" "--limit ${SMOKE_LIMIT}"
  python "${CODE_ROOT}/aggregate_exp01.py" \
    --hotpot_summary "${RESULT_ROOT}/smoke100/hotpot/summary.json" \
    --nq_summary "${RESULT_ROOT}/smoke100/nq/summary.json" \
    --out_dir "${RESULT_ROOT}/smoke100/main_table"

  if [[ "${RUN_FULL_AFTER_SMOKE}" == "1" ]]; then
    run_one "HotpotQA" "${DATA_ROOT}/hotpot_eval.jsonl" "${RESULT_ROOT}/official_full/hotpot" "${ART_ROOT}/hotpot" ""
    run_one "NQ" "${DATA_ROOT}/nq_eval.jsonl" "${RESULT_ROOT}/official_full/nq" "${ART_ROOT}/nq" ""
    python "${CODE_ROOT}/aggregate_exp01.py" \
      --hotpot_summary "${RESULT_ROOT}/official_full/hotpot/summary.json" \
      --nq_summary "${RESULT_ROOT}/official_full/nq/summary.json" \
      --out_dir "${RESULT_ROOT}/official_full/main_table"
  fi
}

main "$@"
