#!/usr/bin/env bash
# Full Exp02 (TruthfulQA + FEVER), SSH-safe: resident on GPU, eval uses resident + CPU ANN;
# run_exp02 clears CUDA for child processes so TruthfulQA MC likelihood loads once on CPU (avoids dual 7B on one GPU).
# Logs: results/exp02_full_orchestrator.log, results/exp02_full_pipeline.log, experiments/results/resident_18888_gpu.log
set -u
ROOT="/home/zd/dev/KVI"
LOG_DIR="${ROOT}/experiments/exp02_hallucination/results"
ORCH_LOG="${LOG_DIR}/exp02_full_orchestrator.log"
PIPE_LOG="${LOG_DIR}/exp02_full_pipeline.log"
RES_LOG="${ROOT}/experiments/results/resident_18888_gpu.log"
OUTER_LOG="${LOG_DIR}/exp02_full_nohup_outer.log"

mkdir -p "${LOG_DIR}" "${ROOT}/experiments/results"
export PYTHONUNBUFFERED=1
cd "${ROOT}"

ts() { date -Iseconds; }

{
  echo "[$(ts)] === exp02 full detached: start ==="
  echo "[$(ts)] stopping prior resident / exp02 / exp01 (best effort)"
} >> "${ORCH_LOG}"

pkill -f "${ROOT}/experiments/exp01_main_qa/code/exp01_resident_infer_service.py" 2>/dev/null || true
pkill -f "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" 2>/dev/null || true
pkill -f "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py" 2>/dev/null || true
sleep 4

{
  echo "[$(ts)] starting resident → ${RES_LOG}"
} >> "${ORCH_LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
  bash "${ROOT}/experiments/exp01_main_qa/code/start_resident_service.sh" \
  >> "${RES_LOG}" 2>&1 &

ok=0
for i in $(seq 1 120); do
  if curl -sf --connect-timeout 2 http://127.0.0.1:18888/health >/dev/null 2>&1; then
    ok=1
    echo "[$(ts)] resident healthy (try=${i})" >> "${ORCH_LOG}"
    break
  fi
  sleep 3
done

if [ "${ok}" != "1" ]; then
  echo "[$(ts)] FATAL: resident did not become healthy; see ${RES_LOG}" >> "${ORCH_LOG}"
  exit 1
fi

GRACE_SEC="${RESIDENT_READY_GRACE_SEC:-45}"
echo "[$(ts)] resident up; grace sleep ${GRACE_SEC}s before exp02" >> "${ORCH_LOG}"
sleep "${GRACE_SEC}"

{
  echo "[$(ts)] starting run_exp02_hallucination (both datasets, reuse artifacts, fresh eval — no --resume_eval)"
} >> "${ORCH_LOG}"

export CUDA_VISIBLE_DEVICES=""
set +e
"${ROOT}/KVI/bin/python" -u "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cpu \
  --resident_url http://127.0.0.1:18888 \
  --skip_mirror_and_prepare \
  --reuse_artifacts >> "${PIPE_LOG}" 2>&1
RC=$?
set -u
echo "[$(ts)] run_exp02_hallucination exit=${RC}" >> "${ORCH_LOG}"

if [ "${RC}" -eq 0 ]; then
  echo "[$(ts)] plotting figures" >> "${ORCH_LOG}"
  "${ROOT}/KVI/bin/python" "${ROOT}/experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py" \
    --paper --fever_label_figure --truthfulqa_mc_figure >> "${ORCH_LOG}" 2>&1
  "${ROOT}/KVI/bin/python" "${ROOT}/experiments/exp02_hallucination/code/plot_unified_hallucination_bars.py" >> "${ORCH_LOG}" 2>&1
  if "${ROOT}/KVI/bin/python" -c "import cairosvg" 2>/dev/null; then
    for svg in unified_hallucination_bars hallucination_proxy_bars_paper truthfulqa_mc_proxy_bars_paper fever_label_accuracy_bars_paper; do
      f="${LOG_DIR}/${svg}.svg"
      if [ -f "${f}" ]; then
        "${ROOT}/KVI/bin/python" -c "import cairosvg; cairosvg.svg2pdf(url='${f}', write_to='${LOG_DIR}/${svg}.pdf')" >> "${ORCH_LOG}" 2>&1 || true
      fi
    done
  else
    echo "[$(ts)] skip SVG→PDF: cairosvg not installed" >> "${ORCH_LOG}"
  fi
  echo "[$(ts)] === exp02 full detached: success ===" >> "${ORCH_LOG}"
else
  echo "[$(ts)] === exp02 full detached: FAILED (see ${PIPE_LOG}) ===" >> "${ORCH_LOG}"
fi

exit "${RC}"
