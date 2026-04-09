#!/usr/bin/env bash
# Exp02 FEVER-only on GPU0 + resident; safe under SSH disconnect (run via nohup … </dev/null &).
set -u
ROOT="/home/zd/dev/KVI"
LOG_DIR="${ROOT}/experiments/exp02_hallucination/results"
ORCH_LOG="${LOG_DIR}/exp02_fever_gpu_orchestrator.log"
RES_LOG="${ROOT}/experiments/results/resident_18888_gpu.log"
FEVER_LOG="${LOG_DIR}/exp02_fever_gpu.log"

mkdir -p "${LOG_DIR}" "${ROOT}/experiments/results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"

{
  echo "[$(date -Iseconds)] detached: cwd=${ROOT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[$(date -Iseconds)] detached: stopping old resident / exp02 (if any)"
} >> "${ORCH_LOG}"

pkill -f "/home/zd/dev/KVI/experiments/exp01_main_qa/code/exp01_resident_infer_service.py" 2>/dev/null || true
pkill -f "/home/zd/dev/KVI/experiments/exp02_hallucination/code/run_exp02_hallucination.py" 2>/dev/null || true
pkill -f "/home/zd/dev/KVI/experiments/exp01_main_qa/code/run_exp01.py" 2>/dev/null || true
sleep 3

{
  echo "[$(date -Iseconds)] detached: starting resident → ${RES_LOG}"
} >> "${ORCH_LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 \
  bash "${ROOT}/experiments/exp01_main_qa/code/start_resident_service.sh" \
  >> "${RES_LOG}" 2>&1 &

ok=0
for i in $(seq 1 120); do
  if curl -sf --connect-timeout 2 http://127.0.0.1:18888/health >/dev/null 2>&1; then
    ok=1
    echo "[$(date -Iseconds)] detached: resident healthy (try=${i})" >> "${ORCH_LOG}"
    break
  fi
  sleep 3
done

if [ "${ok}" != "1" ]; then
  echo "[$(date -Iseconds)] detached: FATAL resident did not become healthy; see ${RES_LOG}" >> "${ORCH_LOG}"
  exit 1
fi

# /health returns before the first /infer/graph loads weights; short wait avoids RemoteDisconnected / refused on first call.
GRACE_SEC="${RESIDENT_READY_GRACE_SEC:-45}"
echo "[$(date -Iseconds)] detached: resident up; grace sleep ${GRACE_SEC}s before FEVER eval" >> "${ORCH_LOG}"
sleep "${GRACE_SEC}"

{
  echo "[$(date -Iseconds)] detached: starting run_exp02 (fever-only, cuda, reuse, resume) → ${FEVER_LOG}"
} >> "${ORCH_LOG}"

# --resume_eval: append from existing predictions.jsonl; omits already-finished rows.
# For one uniform Graph/KVI prompt across all examples, delete predictions.jsonl first, then rerun without --resume_eval.
exec "${ROOT}/KVI/bin/python" -u "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cuda \
  --resident_url http://127.0.0.1:18888 \
  --skip_mirror_and_prepare \
  --reuse_artifacts \
  --resume_eval \
  --only_datasets fever >> "${FEVER_LOG}" 2>&1
