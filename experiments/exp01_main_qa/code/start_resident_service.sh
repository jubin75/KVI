#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18888}"

cd "${ROOT}"
source "${ROOT}/KVI/bin/activate"
python "${ROOT}/experiments/exp01_main_qa/code/exp01_resident_infer_service.py" --host "${HOST}" --port "${PORT}"
