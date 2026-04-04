#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
EXP="${ROOT}/experiments/exp02_hallucination"
LOG="${EXP}/results/exp02_supervisor.log"
PIPE="${EXP}/results/exp02_pipeline_v2.log"
DONE="${EXP}/results/hallucination_proxy_summary.json"
mkdir -p "${EXP}/results"
export CUDA_VISIBLE_DEVICES=""

echo "[$(date -Iseconds)] Exp02 supervisor start" >> "${LOG}"
while true; do
  if [[ -f "${DONE}" ]]; then
    echo "[$(date -Iseconds)] Exp02 done ${DONE}" >> "${LOG}"
    exit 0
  fi
  echo "[$(date -Iseconds)] Exp02 launch" >> "${LOG}"
  if /home/zd/dev/KVI/KVI/bin/python -u "${EXP}/code/run_exp02_hallucination.py" --build_device cpu --resident_url http://127.0.0.1:18888 >> "${PIPE}" 2>&1; then
    echo "[$(date -Iseconds)] Exp02 run exit=0" >> "${LOG}"
  else
    rc=$?
    echo "[$(date -Iseconds)] Exp02 run failed rc=${rc}; retry in 120s" >> "${LOG}"
    sleep 120
  fi
done

