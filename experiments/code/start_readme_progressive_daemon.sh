#!/usr/bin/env bash
# Start README-aligned progressive search daemon under nohup (detach-safe).
# Prerequisites: resident inference service on RESIDENT_URL (default http://127.0.0.1:18888/health).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
MODEL="${MODEL:-${ROOT}/models/Qwen2.5-7B-Instruct}"
RESIDENT_URL="${RESIDENT_URL:-http://127.0.0.1:18888}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_LOG="${OUT_LOG:-${ROOT}/experiments/exp02_hallucination/results/readme_daemon_outer_${STAMP}.log}"
PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT}/KVI/bin/python" ]]; then PY="${ROOT}/KVI/bin/python"
  elif [[ -x "${ROOT}/KVI/bin/python3" ]]; then PY="${ROOT}/KVI/bin/python3"
  else PY="python3"
  fi
fi

mkdir -p "${ROOT}/experiments/exp02_hallucination/results"
echo "[${STAMP}] launching readme_progressive_search_daemon.py -> ${OUT_LOG}" | tee -a "${OUT_LOG}"

nohup env ROOT="${ROOT}" MODEL="${MODEL}" RESIDENT_URL="${RESIDENT_URL}" \
  "${PY}" -u "${ROOT}/experiments/code/readme_progressive_search_daemon.py" \
  --root "${ROOT}" \
  --model "${MODEL}" \
  --resident-url "${RESIDENT_URL}" \
  "$@" </dev/null >>"${OUT_LOG}" 2>&1 &

echo $! | tee -a "${OUT_LOG}"
echo "PID saved above; tail -f ${ROOT}/experiments/exp02_hallucination/results/README_DAEMON_JOURNAL.log"
