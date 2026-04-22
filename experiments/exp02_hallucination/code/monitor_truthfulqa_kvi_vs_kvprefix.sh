#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd/dev/KVI"
OUT_DIR="${ROOT}/experiments/exp02_hallucination/results/truthfulqa_kvi_vs_kvprefix_qwen25_7b"
PRED="${OUT_DIR}/predictions.jsonl"
LOG="${ROOT}/experiments/exp02_hallucination/results/monitor_truthfulqa_kvi_vs_kvprefix.log"
REPORT="${OUT_DIR}/monitor_report.md"
INTERVAL_SEC="${INTERVAL_SEC:-120}"
TARGET_N="${TARGET_N:-500}"

ts() { date -Iseconds; }

echo "[$(ts)] monitor start target_n=${TARGET_N} interval=${INTERVAL_SEC}s" >> "${LOG}"

while true; do
  lines=0
  if [ -f "${PRED}" ]; then
    lines=$(wc -l < "${PRED}" || echo 0)
  fi
  pct=$(python3 - <<PY
lines=${lines}
target=${TARGET_N}
print(f"{(100.0*lines/target if target>0 else 0.0):.1f}")
PY
)
  echo "[$(ts)] progress ${lines}/${TARGET_N} (${pct}%)" >> "${LOG}"

  if [ "${lines}" -gt 0 ]; then
    python3 - <<'PY' >> "${LOG}" 2>/dev/null
import json, re, pathlib
p = pathlib.Path("/home/zd/dev/KVI/experiments/exp02_hallucination/results/truthfulqa_kvi_vs_kvprefix_qwen25_7b/predictions.jsonl")
rows = []
with p.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
tail = rows[-5:] if len(rows) >= 5 else rows
noise_pat = re.compile(r"!\[[^\]]*\]\([^)]*\)|https?://|!{2,}")
k_noise = 0
k_short = 0
for r in rows:
    k = str((r.get("predictions") or {}).get("kvi") or "")
    if noise_pat.search(k):
        k_noise += 1
    if len(k.strip()) < 16:
        k_short += 1
print(f"[sample] total={len(rows)} kvi_noise={k_noise} kvi_short={k_short}")
for r in tail:
    q = str(r.get("question") or "")[:90]
    k = str((r.get("predictions") or {}).get("kvi") or "").replace("\n", " ")
    kvp = str((r.get("predictions") or {}).get("kv_prefix") or "").replace("\n", " ")
    print(f"  idx={r.get('idx')} q={q}")
    print(f"    kvi: {k[:140]}")
    print(f"    kv_prefix: {kvp[:140]}")
PY
  fi

  if [ "${lines}" -ge "${TARGET_N}" ]; then
    echo "[$(ts)] target reached; writing report and exit" >> "${LOG}"
    python3 - <<'PY' > "${REPORT}" 2>/dev/null
import json, re, pathlib
p = pathlib.Path("/home/zd/dev/KVI/experiments/exp02_hallucination/results/truthfulqa_kvi_vs_kvprefix_qwen25_7b/predictions.jsonl")
rows = []
with p.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
noise_pat = re.compile(r"!\[[^\]]*\]\([^)]*\)|https?://|!{2,}")
k_noise = sum(1 for r in rows if noise_pat.search(str((r.get("predictions") or {}).get("kvi") or "")))
k_short = sum(1 for r in rows if len(str((r.get("predictions") or {}).get("kvi") or "").strip()) < 16)
mc1_better = 0
mc2_better = 0
for r in rows:
    mc1 = r.get("truthfulqa_mc1_proxy") or {}
    mc2 = r.get("truthfulqa_mc2_proxy") or {}
    if float(mc1.get("kvi") or 0.0) > float(mc1.get("kv_prefix") or 0.0):
        mc1_better += 1
    if float(mc2.get("kvi") or 0.0) > float(mc2.get("kv_prefix") or 0.0):
        mc2_better += 1
print("# TruthfulQA KVI vs KV Prefix Monitor Report")
print("")
print(f"- total_rows: {len(rows)}")
print(f"- kvi_noise_rows: {k_noise}")
print(f"- kvi_short_rows: {k_short}")
print(f"- per_example_mc1_kvi_gt_kvprefix: {mc1_better}")
print(f"- per_example_mc2_kvi_gt_kvprefix: {mc2_better}")
print("")
print("## Last 10 samples")
for r in rows[-10:]:
    q = str(r.get("question") or "")[:120]
    k = str((r.get("predictions") or {}).get("kvi") or "").replace("\n", " ")
    kvp = str((r.get("predictions") or {}).get("kv_prefix") or "").replace("\n", " ")
    print(f"- idx={r.get('idx')} q={q}")
    print(f"  - kvi: {k[:200]}")
    print(f"  - kv_prefix: {kvp[:200]}")
PY
    break
  fi

  sleep "${INTERVAL_SEC}"
done

echo "[$(ts)] monitor exit" >> "${LOG}"
