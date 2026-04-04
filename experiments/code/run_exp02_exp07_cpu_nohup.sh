#!/usr/bin/env bash
# Sequential CPU pipeline: Exp02 smoke -> Exp02 full -> Exp07 smoke -> Exp07 full.
# Safe under SSH drop: run as  nohup bash ... &  and watch experiments/results/cpu_exp02_exp07_pipeline.log
set -euo pipefail

ROOT="/home/zd/dev/KVI"
export CUDA_VISIBLE_DEVICES=""

PY="${ROOT}/KVI/bin/python"
MAIN_LOG="${ROOT}/experiments/results/cpu_exp02_exp07_pipeline.log"
mkdir -p "${ROOT}/experiments/results"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${MAIN_LOG}"
}

run_py() {
  local label=$1
  shift
  log "${label}: ${PY} -u $*"
  "${PY}" -u "$@" >> "${MAIN_LOG}" 2>&1 || {
    log "FAILED: ${label} (see ${MAIN_LOG})"
    exit 1
  }
  log "OK: ${label}"
}

SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
SMOKE_TQA="${SMOKE_TQA:-40}"
SMOKE_FEVER="${SMOKE_FEVER:-40}"
SMOKE_CL_MAX="${SMOKE_CL_MAX:-40}"

RESIDENT="${RESIDENT_URL:-http://127.0.0.1:18888}"

log "======== start cpu pipeline (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-empty}) resident=${RESIDENT} ========"

run_py "SMOKE exp02" \
  "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cpu \
  --resident_url "${RESIDENT}" \
  --truthfulqa_max "${SMOKE_TQA}" \
  --fever_max "${SMOKE_FEVER}" \
  --limit "${SMOKE_LIMIT}"

run_py "FULL exp02" \
  "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cpu \
  --resident_url "${RESIDENT}"

run_py "SMOKE exp07" \
  "${ROOT}/experiments/exp07_clbench_longcontext/code/run_exp07_clbench_proxy.py" \
  --build_device cpu \
  --resident_url "${RESIDENT}" \
  --max_examples "${SMOKE_CL_MAX}" \
  --limit "${SMOKE_LIMIT}"

run_py "FULL exp07" \
  "${ROOT}/experiments/exp07_clbench_longcontext/code/run_exp07_clbench_proxy.py" \
  --build_device cpu \
  --resident_url "${RESIDENT}"

log "======== ALL PHASES COMPLETE ========"
