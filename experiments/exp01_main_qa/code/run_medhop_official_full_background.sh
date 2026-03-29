#!/usr/bin/env bash
# Full MedHopQA official-NL Exp01 (342 ex × 5 methods) + artifact build.
# Intended for: nohup ... >> log 2>&1 &   (survives SSH drop)
#
# Env overrides:
#   MODEL                 Base LLM path (default: $ROOT/models/Qwen2.5-7B-Instruct)
#   SKIP_ARTIFACTS=1      Only run run_exp01 (expects artifacts/medhop_official ready)
#   START_RESIDENT=0      Do not start exp01_resident_infer_service; no --inference_service_url
#   RESIDENT_PORT=18888   HTTP port for resident graph cache
#   DOMAIN_ENCODER        sentence-transformers model for tagging + ANN (default all-MiniLM-L6-v2)
#   MEDHOP_OFFICIAL_OUT   run_exp01 --out_dir (default: .../medhop_official_fullmethods_qwen25_7b_kvituned)
#   KVI_DRM_THRESHOLD     default 0.12 (stricter triple gate vs global 0.05)
#   KVI_MAX_KV_TRIPLES    default 2
#   KVI_TOP_K_RELATIONS   default 1
#   KVI_MINIMAL_PROMPT    default 1 (set 0 to disable --kvi_minimal_prompt)

set -euo pipefail

ROOT="/home/zd/dev/KVI"
VENV="${ROOT}/KVI/bin/activate"
CODE="${ROOT}/experiments/exp01_main_qa/code"
DATA="${ROOT}/experiments/exp01_main_qa/data/medhop_official"
ART="${ROOT}/experiments/exp01_main_qa/artifacts/medhop_official"
RESULTS="${ROOT}/experiments/exp01_main_qa/results"
OUT="${MEDHOP_OFFICIAL_OUT:-${RESULTS}/medhop_official_fullmethods_qwen25_7b_kvituned}"
LOG="${RESULTS}/medhop_official_full_pipeline.log"
RESIDENT_LOG="${RESULTS}/medhop_official_resident.log"

MODEL="${MODEL:-${ROOT}/models/Qwen2.5-7B-Instruct}"
DOMAIN_ENCODER="${DOMAIN_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
RESIDENT_PORT="${RESIDENT_PORT:-18888}"
SPECS_FALLBACK="${ROOT}/experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar/semantic_type_specs.json"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

log() { echo "[$(date -Iseconds)] $*"; }

cd "${ROOT}"
# shellcheck disable=SC1090
source "${VENV}"

if [[ ! -f "${DATA}/medhop_eval.jsonl" ]]; then
  log "ERROR: missing ${DATA}/medhop_eval.jsonl — run prepare_medhop_official_from_raw.py first"
  exit 2
fi
if [[ ! -f "${MODEL}/config.json" ]] && [[ ! -f "${MODEL}" ]]; then
  log "WARN: MODEL path may be invalid: ${MODEL}"
fi

if [[ "${SKIP_ARTIFACTS:-0}" != "1" ]]; then
  mkdir -p "${ART}"

  if [[ ! -f "${SPECS_FALLBACK}" ]]; then
    log "ERROR: need semantic_type_specs at ${SPECS_FALLBACK}"
    exit 2
  fi
  cp -f "${SPECS_FALLBACK}" "${ART}/semantic_type_specs.json"

  # Refresh sentences/triples when data dir is newer
  if [[ ! -f "${ART}/sentences.jsonl" ]] || [[ "${DATA}/sentences_medhop.jsonl" -nt "${ART}/sentences.jsonl" ]]; then
    log "Copy sentences_medhop.jsonl -> artifacts/medhop_official/sentences.jsonl"
    cp -f "${DATA}/sentences_medhop.jsonl" "${ART}/sentences.jsonl"
  fi
  if [[ ! -f "${ART}/triples.jsonl" ]] || [[ "${DATA}/triples_medhop.jsonl" -nt "${ART}/triples.jsonl" ]]; then
    log "Copy triples_medhop.jsonl -> artifacts/medhop_official/triples.jsonl"
    cp -f "${DATA}/triples_medhop.jsonl" "${ART}/triples.jsonl"
  fi

  if [[ ! -f "${ART}/sentences.tagged.jsonl" ]] || [[ "${ART}/sentences.jsonl" -nt "${ART}/sentences.tagged.jsonl" ]]; then
    log "annotate_sentences_semantic_tags (CPU, large)"
    python "${ROOT}/scripts/annotate_sentences_semantic_tags.py" \
      --in_jsonl "${ART}/sentences.jsonl" \
      --out_jsonl "${ART}/sentences.tagged.jsonl" \
      --domain_encoder_model "${DOMAIN_ENCODER}" \
      --semantic_type_specs "${ART}/semantic_type_specs.json"
  else
    log "Skip annotate: sentences.tagged.jsonl up to date"
  fi

  if [[ ! -f "${ART}/kvbank_sentences/manifest.json" ]] || [[ "${ART}/sentences.tagged.jsonl" -nt "${ART}/kvbank_sentences/manifest.json" ]]; then
    log "build_kvbank_from_blocks_jsonl (GPU, long)"
    python "${ROOT}/scripts/build_kvbank_from_blocks_jsonl.py" \
      --blocks_jsonl "${ART}/sentences.tagged.jsonl" \
      --disable_enriched \
      --out_dir "${ART}/kvbank_sentences" \
      --base_llm "${MODEL}" \
      --domain_encoder_model "${DOMAIN_ENCODER}" \
      --layers 0,1,2,3 \
      --block_tokens 128 \
      --shard_size 1024 \
      --device cuda \
      --dtype bfloat16
  else
    log "Skip kvbank: manifest up to date"
  fi

  if [[ ! -f "${ART}/graph_index.json" ]] || [[ "${ART}/triples.jsonl" -nt "${ART}/graph_index.json" ]]; then
    log "build_knowledge_graph"
    python "${ROOT}/scripts/build_knowledge_graph.py" \
      --triples_jsonl "${ART}/triples.jsonl" \
      --out_graph "${ART}/graph_index.json"
  else
    log "Skip graph: graph_index.json up to date"
  fi

  if [[ ! -f "${ART}/triple_kvbank/manifest.json" ]] || [[ "${ART}/graph_index.json" -nt "${ART}/triple_kvbank/manifest.json" ]]; then
    log "triple_kv_compiler (GPU)"
    python "${ROOT}/src/graph/triple_kv_compiler.py" \
      --graph_index "${ART}/graph_index.json" \
      --model "${MODEL}" \
      --out_dir "${ART}/triple_kvbank" \
      --device cuda \
      --dtype bfloat16
  else
    log "Skip triple_kvbank: manifest up to date"
  fi

  log "Artifacts ready under ${ART}"
else
  log "SKIP_ARTIFACTS=1 — using existing ${ART}"
fi

SERVICE_ARGS=()
if [[ "${START_RESIDENT:-1}" == "1" ]]; then
  if curl -sf "http://127.0.0.1:${RESIDENT_PORT}/health" >/dev/null; then
    log "Resident already healthy on port ${RESIDENT_PORT}"
  else
    log "Starting exp01_resident_infer_service on ${RESIDENT_PORT} (see ${RESIDENT_LOG})"
    nohup bash -lc "cd '${ROOT}' && source '${VENV}' && exec python '${CODE}/exp01_resident_infer_service.py' --host 127.0.0.1 --port ${RESIDENT_PORT}" \
      >>"${RESIDENT_LOG}" 2>&1 &
    sleep 2
    if ! curl -sf "http://127.0.0.1:${RESIDENT_PORT}/health" >/dev/null; then
      log "ERROR: resident failed to expose /health on ${RESIDENT_PORT}"
      exit 2
    fi
  fi
  SERVICE_ARGS=(--inference_service_url "http://127.0.0.1:${RESIDENT_PORT}")
else
  log "START_RESIDENT=0 — each example will spawn subprocess graph (very slow)"
fi

mkdir -p "${OUT}"
KVI_DRM_THRESHOLD="${KVI_DRM_THRESHOLD:-0.12}"
KVI_MAX_KV_TRIPLES="${KVI_MAX_KV_TRIPLES:-2}"
KVI_TOP_K_RELATIONS="${KVI_TOP_K_RELATIONS:-1}"
KVI_MINIMAL_PROMPT="${KVI_MINIMAL_PROMPT:-1}"
KVI_EXTRA=()
if [[ "${KVI_MINIMAL_PROMPT}" == "1" ]]; then
  KVI_EXTRA+=(--kvi_minimal_prompt)
fi
log "run_exp01.py full MedHopQA_official (no limit, --resume) -> ${OUT}"
log "KVI tuning: drm_threshold=${KVI_DRM_THRESHOLD} max_kv_triples=${KVI_MAX_KV_TRIPLES} top_k_relations=${KVI_TOP_K_RELATIONS} kvi_minimal_prompt=${KVI_MINIMAL_PROMPT}"

python -u "${CODE}/run_exp01.py" \
  --dataset "${DATA}/medhop_eval.jsonl" \
  --dataset_name MedHopQA_official \
  --model "${MODEL}" \
  --graph_index "${ART}/graph_index.json" \
  --triple_kvbank_dir "${ART}/triple_kvbank" \
  --graph_sentences_jsonl "${ART}/sentences.tagged.jsonl" \
  --ann_kv_dir "${ART}/kvbank_sentences" \
  --ann_sentences_jsonl "${ART}/sentences.tagged.jsonl" \
  --ann_semantic_type_specs "${ART}/kvbank_sentences/pattern_sidecar/semantic_type_specs.json" \
  --ann_pattern_index_dir "${ART}/kvbank_sentences/pattern_sidecar" \
  --ann_sidecar_dir "${ART}/kvbank_sentences/pattern_sidecar" \
  --domain_encoder_model "${DOMAIN_ENCODER}" \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --out_dir "${OUT}" \
  --resume \
  --timeout_s 600 \
  --bootstrap_samples 1000 \
  --permutation_samples 2000 \
  --ann_force_cpu \
  --kvi_drm_threshold "${KVI_DRM_THRESHOLD}" \
  --kvi_max_kv_triples "${KVI_MAX_KV_TRIPLES}" \
  --kvi_top_k_relations "${KVI_TOP_K_RELATIONS}" \
  "${KVI_EXTRA[@]}" \
  "${SERVICE_ARGS[@]}"

log "Done. summary: ${OUT}/summary.json"
