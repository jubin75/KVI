#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
EXP="${ROOT}/experiments/exp07_clbench_longcontext"
LOG="${EXP}/results/exp07_supervisor.log"
PIPE="${EXP}/results/exp07_pipeline_v2.log"
DONE="${EXP}/results/clbench_proxy_length_bucket_summary.json"
mkdir -p "${EXP}/results"
export CUDA_VISIBLE_DEVICES=""

echo "[$(date -Iseconds)] Exp07 supervisor start" >> "${LOG}"
while true; do
  if [[ -f "${DONE}" ]]; then
    echo "[$(date -Iseconds)] Exp07 done ${DONE}" >> "${LOG}"
    exit 0
  fi
  echo "[$(date -Iseconds)] Exp07 launch" >> "${LOG}"
  if /home/zd/dev/KVI/KVI/bin/python -u "${EXP}/code/run_exp07_clbench_proxy.py" --build_device cpu --resident_url http://127.0.0.1:18888 >> "${PIPE}" 2>&1; then
    echo "[$(date -Iseconds)] Exp07 run exit=0" >> "${LOG}"
  else
    rc=$?
    echo "[$(date -Iseconds)] Exp07 run failed rc=${rc}; retry in 180s" >> "${LOG}"
    sleep 180
  fi
done

