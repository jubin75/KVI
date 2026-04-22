#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
OUT_DIR="$ROOT/experiments/exp02_hallucination/data"
LOG="$ROOT/experiments/exp02_hallucination/results/exp02_rebuild_data_offline.log"

mkdir -p "$ROOT/experiments/exp02_hallucination/results"

export HF_HOME="$ROOT/.hf_tmp"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$ROOT"
nohup python3 "$ROOT/experiments/exp02_hallucination/code/prepare_exp02_datasets.py" \
  --out_dir "$OUT_DIR" \
  --mirror_root "$ROOT/experiments/_mirror_data_resolved" \
  --mirror_data_root "$ROOT/experiments/_mirror_data" \
  --truthfulqa_max 500 \
  --fever_max 1000 \
  --offline_only \
  >> "$LOG" 2>&1 &

echo "started offline rebuild in background"
echo "log: $LOG"
