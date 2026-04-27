#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
PY="${ROOT}/KVI/bin/python3"
EXP2="${ROOT}/experiments/exp02_hallucination"
RES="${EXP2}/results"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RESIDENT="${RESIDENT_URL-}"
ANN_FORCE_CPU="${ANN_FORCE_CPU:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STRICT_REQUIRE_GPU="${STRICT_REQUIRE_GPU:-1}"
RUN_TAG="${RUN_TAG:-v36}"
LIMIT="${LIMIT:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ANN_VIA_RESIDENT="${ANN_VIA_RESIDENT:-1}"
AUTO_START_RESIDENT="${AUTO_START_RESIDENT:-1}"
RESIDENT_READY_GRACE_SEC="${RESIDENT_READY_GRACE_SEC:-5}"

LOG="${RES}/exp02_truthfulqa_kvi_D_${RUN_TAG}_pre100_${TS}.log"
OUT_PRE="${RES}/truthfulqa_kvi_optimize_D_${RUN_TAG}_pre100_${TS}"

mkdir -p "${RES}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

if [[ "${STRICT_REQUIRE_GPU}" == "1" && "${ANN_FORCE_CPU}" != "1" ]]; then
  "${PY}" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA is not available; refusing CPU fallback.", flush=True)
    sys.exit(2)
print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
PY
fi

if [[ -n "${RESIDENT}" ]]; then
  if ! curl -sf --connect-timeout 2 "${RESIDENT}/health" >/dev/null 2>&1; then
    if [[ "${AUTO_START_RESIDENT}" == "1" ]]; then
      nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 \
        bash "${ROOT}/experiments/exp01_main_qa/code/start_resident_service.sh" \
        >> "${RES}/resident_18888_gpu.log" 2>&1 &
      for _i in $(seq 1 60); do
        if curl -sf --connect-timeout 2 "${RESIDENT}/health" >/dev/null 2>&1; then
          sleep "${RESIDENT_READY_GRACE_SEC}"
          break
        fi
        sleep 2
      done
    fi
  fi
fi

echo "[$(date -Iseconds)] start D ${RUN_TAG} pre100 (closer-to-v3 stage2) out=${OUT_PRE}" | tee -a "${LOG}"

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
  --out_dir "${OUT_PRE}"
  --timeout_s 1800
  --bootstrap_samples 500
  --permutation_samples 1000
  --ann_inference_service_url ""
  --truthfulqa_kvi_mc1_answer grounded
  --truthfulqa_kvi_max_new_tokens 96
  --kvi_two_stage_kv_then_evidence
  --limit "${LIMIT}"
)

if [[ "${ANN_FORCE_CPU}" == "1" ]]; then
  ARGS+=(--ann_force_cpu)
fi
if [[ -n "${RESIDENT}" ]]; then
  ARGS+=(--inference_service_url "${RESIDENT}")
  if [[ "${ANN_VIA_RESIDENT}" == "1" ]]; then
    ARGS+=(--ann_inference_service_url "${RESIDENT}")
  fi
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARR[@]}")
fi

"${PY}" "${ARGS[@]}" 2>&1 | tee -a "${LOG}"
echo "[$(date -Iseconds)] done D ${RUN_TAG} pre100 out=${OUT_PRE}" | tee -a "${LOG}"
