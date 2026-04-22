#!/usr/bin/env bash
# Exp02 TruthfulQA — A→B→C sequential optimization (n=25, single resident, no concurrency).
# A: KVI ablation — assemble KV but disable past_key_values injection (prompt-only path).
# B: KV relevance — stricter DRM + tighter budget (no tuned overrides).
# C: Output shape — B + shorter decode + minimal prompt.
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
EXP2="${ROOT}/experiments/exp02_hallucination"
RES="${EXP2}/results"
PY="${ROOT}/KVI/bin/python3"
DATE="${DATE:-$(date +%Y%m%d_%H%M%S)}"

RESIDENT="${RESIDENT_URL:-http://127.0.0.1:18888}"
LIMIT="${TRUTHFULQA_LIMIT:-25}"
TIMEOUT="${TIMEOUT_S:-1800}"
GRACE="${RESIDENT_READY_GRACE_SEC:-35}"

mkdir -p "${RES}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

ts() { date -Iseconds; }

wait_resident() {
  if [[ -z "${RESIDENT}" ]]; then
    return 0
  fi
  echo "[$(ts)] waiting for resident ${RESIDENT}/health ..."
  for _i in $(seq 1 120); do
    if curl -sf --connect-timeout 2 "${RESIDENT}/health" >/dev/null 2>&1; then
      echo "[$(ts)] resident healthy; grace sleep ${GRACE}s"
      sleep "${GRACE}"
      return 0
    fi
    sleep 2
  done
  echo "[$(ts)] WARN: resident not healthy after wait; proceeding anyway"
}

run_one() {
  local label="$1"
  shift
  local out_dir="${RES}/truthfulqa_kvi_abc_${DATE}/${label}"
  local log="${RES}/exp02_truthfulqa_kvi_abc_${DATE}_${label}.log"
  mkdir -p "${out_dir}"
  echo "[$(ts)] === ABC ${label}: start ===" | tee -a "${log}"

  "${PY}" -u "${ROOT}/experiments/exp01_main_qa/code/run_exp01.py" \
    --dataset "${EXP2}/data/truthfulqa_eval.jsonl" \
    --dataset_name TRUTHFULQA \
    --model "${ROOT}/models/Qwen2.5-7B-Instruct" \
    --graph_index "${EXP2}/artifacts/truthfulqa/graph_index.json" \
    --triple_kvbank_dir "${EXP2}/artifacts/truthfulqa/triple_kvbank" \
    --graph_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --ann_kv_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences" \
    --ann_sentences_jsonl "${EXP2}/artifacts/truthfulqa/sentences.tagged.jsonl" \
    --ann_semantic_type_specs "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json" \
    --ann_pattern_index_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar" \
    --ann_sidecar_dir "${EXP2}/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar" \
    --methods graphrag,kv_prefix,kvi \
    --out_dir "${out_dir}" \
    --timeout_s "${TIMEOUT}" \
    --bootstrap_samples 500 \
    --permutation_samples 1000 \
    --ann_force_cpu \
    --truthfulqa_kvi_mc1_answer grounded \
    --truthfulqa_kvi_max_new_tokens 96 \
    --graph_audit_jsonl "${RES}/truthfulqa_graph_prompt_audit_kvi_abc_${DATE}_${label}.jsonl" \
    "$@" \
    ${RESIDENT:+--inference_service_url "${RESIDENT}"} \
    --ann_inference_service_url "" \
    ${LIMIT:+--limit "${LIMIT}"} \
    >> "${log}" 2>&1

  echo "[$(ts)] === ABC ${label}: done ===" | tee -a "${log}"
  echo "[$(ts)] out_dir=${out_dir}" | tee -a "${log}"
}

write_report() {
  local base="${RES}/truthfulqa_kvi_optimize_20260416_164904/summary.json"
  local root="${RES}/truthfulqa_kvi_abc_${DATE}"
  local out="${RES}/truthfulqa_kvi_abc_${DATE}_report.md"
  "${PY}" - <<PY
import json
from pathlib import Path

base = Path("${base}")
root = Path("${root}")
out = Path("${out}")

def load_summary(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = {m["method_key"]: m for m in d.get("methods", [])}
    st = d.get("statistics", {}) or {}
    def row(k: str):
        r = rows.get(k) or {}
        return {
            "mc1": float(r.get("truthfulqa_mc1_proxy") or 0.0),
            "mc2": float(r.get("truthfulqa_mc2_proxy") or 0.0),
            "em": float(r.get("em") or 0.0),
            "f1": float(r.get("f1_mean") or 0.0),
        }
    return {
        "path": str(p),
        "n": int(d.get("n") or 0),
        "kvi_cfg": {
            "kvi_max_kv_triples": st.get("kvi_max_kv_triples"),
            "kvi_drm_threshold": st.get("kvi_drm_threshold"),
            "kvi_top_k_relations": st.get("kvi_top_k_relations"),
            "truthfulqa_kvi_tuned_overrides": st.get("truthfulqa_kvi_tuned_overrides"),
            "truthfulqa_kvi_max_new_tokens": st.get("truthfulqa_kvi_max_new_tokens") or st.get("kvi_max_new_tokens_effective"),
            "truthfulqa_kvi_minimal_prompt": st.get("truthfulqa_kvi_minimal_prompt"),
            "truthfulqa_kvi_reconcile": st.get("truthfulqa_kvi_reconcile"),
            "kvi_disable_kv_injection": st.get("kvi_disable_kv_injection", None),
        },
        "graphrag": row("graphrag"),
        "kv_prefix": row("kv_prefix"),
        "kvi": row("kvi"),
    }

labels = ["A_no_kv_injection", "B_strict_kv_filter", "C_strict_filter_shorter"]
items = {"baseline_164904": load_summary(base)}
for lab in labels:
    p = root / lab / "summary.json"
    if not p.exists():
        raise SystemExit(f"missing summary: {p}")
    items[lab] = load_summary(p)

def fmt(x: float) -> str:
    return f"{x:.3f}"

base_kvi_mc2 = items["baseline_164904"]["kvi"]["mc2"]
md = []
md.append(f"## Exp02 TruthfulQA — ABC sequential report ({root.name})\\n")
md.append(f"- Baseline: `{base}`\\n")
md.append(f"- Runs root: `{root}`\\n\\n")
md.append("### Summary (MC2 primary)\\n\\n")
md.append("| Run | KVI cfg (key) | KVI MC1 | KVI MC2 | ΔMC2 vs baseline | GraphRAG MC2 | KV Prefix MC2 |\\n")
md.append("|---|---|---:|---:|---:|---:|---:|\\n")
for lab in labels:
    it = items[lab]
    cfg = it["kvi_cfg"]
    key = (
        f"disable_inj={cfg.get('kvi_disable_kv_injection')}, "
        f"triples={cfg.get('kvi_max_kv_triples')}, "
        f"drm={cfg.get('kvi_drm_threshold')}, "
        f"rels={cfg.get('kvi_top_k_relations')}, "
        f"tuned={cfg.get('truthfulqa_kvi_tuned_overrides')}, "
        f"max_new={cfg.get('truthfulqa_kvi_max_new_tokens')}, "
        f"min_prompt={cfg.get('truthfulqa_kvi_minimal_prompt')}, "
        f"reconcile={cfg.get('truthfulqa_kvi_reconcile')}"
    )
    dmc2 = it["kvi"]["mc2"] - base_kvi_mc2
    md.append(
        f"| {lab} | {key} | {fmt(it['kvi']['mc1'])} | {fmt(it['kvi']['mc2'])} | {fmt(dmc2)} | "
        f"{fmt(it['graphrag']['mc2'])} | {fmt(it['kv_prefix']['mc2'])} |\\n"
    )

md.append("\\n### Notes\\n\\n")
md.append("- A isolates the effect of KV injection by assembling KV but not injecting `past_key_values`.\\n")
md.append("- B disables tuned overrides and tightens KV selection (higher DRM, fewer relations/triples).\\n")
md.append("- C applies B plus shorter decode + minimal prompt to reduce long/off-topic tails.\\n")

out.write_text(''.join(md), encoding='utf-8')
print(json.dumps({\"ok\": True, \"report\": str(out)}, ensure_ascii=False))
PY
}

wait_resident

# A: isolate KV injection effect (disable past_key_values injection for KVI)
run_one "A_no_kv_injection" --kvi_disable_kv_injection

# B: improve KV relevance (stricter DRM + tighter budget) — disable tuned overrides for controlled ablation
run_one "B_strict_kv_filter" \
  --no-truthfulqa_kvi_tuned_overrides \
  --kvi_drm_threshold 0.08 \
  --kvi_top_k_relations 2 \
  --kvi_max_kv_triples 3

# C: B + shorter decode + minimal prompt (reduce long tails / hedging)
run_one "C_strict_filter_shorter" \
  --no-truthfulqa_kvi_tuned_overrides \
  --kvi_drm_threshold 0.08 \
  --kvi_top_k_relations 2 \
  --kvi_max_kv_triples 3 \
  --kvi_minimal_prompt \
  --truthfulqa_kvi_max_new_tokens 64

write_report
echo "[$(ts)] All ABC runs finished. Results under ${RES}/truthfulqa_kvi_abc_${DATE}/"
echo "[$(ts)] report: ${RES}/truthfulqa_kvi_abc_${DATE}_report.md"

