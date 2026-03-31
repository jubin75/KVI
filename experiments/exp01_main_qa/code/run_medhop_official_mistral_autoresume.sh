#!/usr/bin/env bash
# Keep running Mistral MedHop official eval in background until summary.json exists.
# Safe for unstable SSH: each loop uses --resume and continues from predictions.jsonl.

set -euo pipefail

ROOT="/home/zd/dev/KVI"
CODE="${ROOT}/experiments/exp01_main_qa/code"
RESULTS="${ROOT}/experiments/exp01_main_qa/results"
OUT="${RESULTS}/medhop_official_fullmethods_mistral7b_v0_3"
SUMMARY="${OUT}/summary.json"
PRED="${OUT}/predictions.jsonl"
PIPE_LOG="${RESULTS}/medhop_official_full_pipeline_mistral7b_v0_3.log"
SUP_LOG="${RESULTS}/medhop_official_supervisor_mistral7b_v0_3.log"
RUN_SCRIPT="${CODE}/run_medhop_official_full_background_mistral7b_v0_3.sh"

mkdir -p "${RESULTS}" "${OUT}"
touch "${PIPE_LOG}" "${SUP_LOG}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${SUP_LOG}" >/dev/null; }

line_count() {
  if [[ -f "${PRED}" ]]; then
    wc -l < "${PRED}"
  else
    echo 0
  fi
}

attempt=0
while true; do
  if [[ -f "${SUMMARY}" ]]; then
    log "summary exists -> done (${SUMMARY})"
    exit 0
  fi

  # Avoid duplicate runners.
  if pgrep -af "run_exp01.py.*medhop_official_fullmethods_mistral7b_v0_3" >/dev/null; then
    log "run_exp01 already running; sleep 120s (pred_lines=$(line_count))"
    sleep 120
    continue
  fi

  attempt=$((attempt + 1))
  log "attempt=${attempt} start resume run (pred_lines_before=$(line_count))"

  # Reuse built artifacts; only resume eval.
  if SKIP_ARTIFACTS=1 START_RESIDENT=1 MEDHOP_OFFICIAL_OUT="${OUT}" bash "${RUN_SCRIPT}" >> "${PIPE_LOG}" 2>&1; then
    log "attempt=${attempt} run exited 0 (pred_lines_after=$(line_count))"
  else
    rc=$?
    log "attempt=${attempt} run failed rc=${rc} (pred_lines_after=$(line_count)); sleep 60s then retry"
    sleep 60
  fi
done

