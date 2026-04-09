#!/usr/bin/env bash
# FEVER Exp02 only: resident must already be healthy on 18888 (no orchestrator, no pkill).
# Uses CPU artifact paths + graph via resident (same as run_exp02_fast_once.sh).
set -euo pipefail
ROOT="/home/zd/dev/KVI"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
exec "${ROOT}/KVI/bin/python" -u "${ROOT}/experiments/exp02_hallucination/code/run_exp02_hallucination.py" \
  --build_device cpu \
  --resident_url "${RESIDENT_URL:-http://127.0.0.1:18888}" \
  --skip_mirror_and_prepare \
  --reuse_artifacts \
  --resume_eval \
  --only_datasets fever \
  "$@"
