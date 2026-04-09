#!/usr/bin/env bash
# Single-shot Exp02: skip dataset download/prepare if manifest matches; reuse compiled graph/KV artifacts.
# Requires resident inference on http://127.0.0.1:18888 (start_resident_service.sh).
set -euo pipefail
ROOT="/home/zd/dev/KVI"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
cd "${ROOT}"
exec "${ROOT}/KVI/bin/python" -u "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cpu \
  --resident_url "${RESIDENT_URL:-http://127.0.0.1:18888}" \
  --skip_mirror_and_prepare \
  --reuse_artifacts \
  "$@"
