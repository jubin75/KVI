# Exp02 — Latest experiment requirements (rolling memo)

This document records the **current priority** experiment goals and reproducible conventions, to avoid relying solely on conversation memory. **On each program adjustment or run method change, append a section at the top** (date + short title), with old content retained as history.

---

## 2026-04-20 — D v3: Two-stage de-confliction (relaxed stage-2 constraints, retained correction bias)

**Motivation (from D v2 full-scale regression)**

- Stage 2 "single sentence + fixed refusal sentence + 48 token" constraint was too rigid, compressing fact-preserving expression, causing KVI to regress on full-scale MC1/MC2.
- Keep the two-stage framework unchanged, make only minimal changes, with the goal of restoring D v1's effective gains while continuing to suppress long-tail noise.

**v3 minimal changes (`scripts/run_graph_inference.py`)**

- Stage 2 instruction adjusted from "**exactly one sentence + exact refusal**" to "**1-2 short sentences**, state insufficiency in one sentence when evidence is insufficient."
- Stage 2 system instruction emphasizes "**evidence consistency > draft wording**", avoiding anchoring conflict between draft wording and final output.
- Stage 2 `max_new_tokens` relaxed from `min(48, ...)` to `min(72, ...)`.
- `_sanitize_two_stage_openqa_final` relaxed to: keep at most **2 sentences**, `max_chars` raised from 220 to 320, still retains `!` removal and basic denoising.

**Orchestration (`run_truthfulqa_kvi_D_v3_pre100_full.sh`)**

- Default **`WAIT_RESIDENT=1`**: before running, poll `RESIDENT_URL/health`; if not healthy, **exit directly** (avoids `Connection refused` leaving a half-finished `predictions.jsonl`); once healthy, wait `RESIDENT_READY_GRACE_SEC` seconds before launching `run_exp01.py`.

---

## 2026-04-19 — D two-stage: Tighten stage-2 final output (post full attribution)

**Attribution conclusion (TruthfulQA D full 500)**: The subset where KVI loses commonly shows **longer final output + exclamation noise**; stage 2 needs to enforce short sentences and remove `!`, and limit `max_new_tokens`.

**Code adjustments (`run_graph_inference.py`)**

- Stage 2 system/user instructions changed to: **single-sentence very short final output**, prohibit markdown/URL/`!`; output fixed refusal sentence when evidence is insufficient.
- Stage 2 `max_new_tokens` upper limit lowered to **`min(48, args.max_new_tokens)`**.
- New `_sanitize_two_stage_openqa_final`: removes `!`, truncates to first sentence, length cap; if empty output, fall back to cleaned stage-1 draft.

---

## 2026-04-17 — Round D approach: Two-stage (KV draft → evidence-corrected final)

**User-confirmed approach (for locating "how injected information is utilized by the generator")**

- **Stage 1 (KV draft)**: under `openqa + KVI`, give only `Question`, enable triple KV injection, no evidence text, generate 1-2 sentence draft first.
- **Stage 2 (evidence correction)**: feed "stage-1 draft + prompt evidence" to the model for correction and final output; stage 2 does not inject `past_key_values`.
- Goal: let KV spike signal anchor first, then correct bias via the evidence channel, reducing same-round dual-channel competition.

**Implementation switches**

- `run_graph_inference.py` added: `--kvi_two_stage_kv_then_evidence`
- `run_exp01.py` added same-named forwarding switch (only active on KVI path)

---

## 2026-04-16 — Execution constraints (anti-regression / anti-backtracking)

**New hard constraints (enforced by default from this section onward)**

- **GPU resident priority**: Before running Exp02, first ensure only one healthy resident (`127.0.0.1:18888`); if duplicate residents or other heavy GPU processes are found competing, clean up first before starting; do not blindly run under abnormal resource conditions.
- **Do not regress to low-value scale**: Once TruthfulQA has a `n=25` smoke baseline, subsequent debugging by default continues at `n=25` (may use `--resume`), unless explicitly reproducing a crash and temporarily reducing samples with the reason noted in the log.
- **Serial priority, avoid mutual interference**: Only run one `run_exp01.py` main task (or one optimization script main task) at a time, to avoid concurrent interference with resident/vram/IO and misdiagnosing issues as algorithmic degradation.
- **On failure, first examine the blocking point, not the scale**: Prioritize checking resident health, timeout locations, OOM, process exit codes; after confirming root cause, make minimal parameter adjustments (e.g. `timeout_s`, resume with `--resume`), and do not reduce to `n=10` and re-run first.

**Intent of this section**

- Formalize the process of "stabilize service and resources first, then run at the planned scale" as the default policy;
- Avoid fragmenting experiment conventions and conclusion comparability due to ad-hoc debugging.

---

## 2026-04-16 — TruthfulQA: A→B→C overnight serial optimization (for locating KVI vs GraphRAG gap)

**Goal**: Decompose the "KVI gap relative to GraphRAG" into three attributable experiment steps, avoiding changing many things based on intuition and making conclusions irreproducible.

- **A (ablation: disable KV injection)**: KVI **composes KV as usual**, but **does not inject `past_key_values`** during generation, to isolate the impact of "KV injection itself" on MC2. Parameter: `--kvi_disable_kv_injection` (forwarded to `run_graph_inference.py`).
- **B (improve relevance: stricter KV filtering)**: Turn off tuned overrides (`--no-truthfulqa_kvi_tuned_overrides`), explicitly raise `kvi_drm_threshold`, tighten `kvi_top_k_relations`, reduce `kvi_max_kv_triples`, with the goal of **better none than noisy**, reducing noisy triples.
- **C (output form: shorter and more convergent)**: On top of B, enable `--kvi_minimal_prompt` and lower `--truthfulqa_kvi_max_new_tokens`, reduce long-tail/off-topic/hedging, prioritizing MC2 proxy improvement.

**Run script**: `experiments/exp02_hallucination/code/run_truthfulqa_kvi_abc_sequential.sh` (serial, default n=25, depends on a single healthy `18888` resident).

---

## 2026-04-15 — Per-question error bucket + MC conditional text convergence (new comparison round started)

**Completed analysis**

- Added: `experiments/exp02_hallucination/code/analyze_truthfulqa_kvi_error_buckets.py`
- Based on `truthfulqa_kvi_injection_levels_20260415_163012/level_6/predictions.jsonl` output:
  - `results/truthfulqa_kvi_error_buckets_level6.md`
  - `results/truthfulqa_kvi_error_buckets_level6.json`
- Conclusion: Among samples where KVI trails GraphRAG, `has_many_exclaim` and hedging tone are common, supporting continued effort on conditional text convergence and output cleaning.

**Code convergence done (generic, not special-cased)**

- `run_exp01.py::_kvi_pred_for_truthfulqa_mc1_likelihood`
  - Added generic denoising (URL/rendering artifacts/generalized hedging prefixes).
  - Multi-sentence candidates changed to "fact density scoring" for sentence selection, rather than always picking the first sentence.

**New n=25 background comparison round started**

- Timestamp: `20260415_220828`
- Main log: `results/exp02_truthfulqa_kvi_optimize_20260415_220828.log`
- Output directory: `results/truthfulqa_kvi_optimize_20260415_220828/`

---

## 2026-04-15 — Injection level comparison (n=25) for verifying KVI noise sources

**Background judgment**: When retrieval/prompt funnels both hit, KVI may still experience generation drift due to triple graph injection level (`max_kv_triples`), dragging down `likelihood_proxy`.

**New experiment entry point**

- `experiments/exp02_hallucination/code/run_truthfulqa_kvi_injection_levels_detached.sh`
- Default comparison levels: `KVI_LEVELS="0 2 4 6"`, fixed `n=25`, fixed data artifacts, fixed `kvi_drm_threshold=0.05`, `kvi_top_k_relations=4`.
- Key: explicitly passes `--no-truthfulqa_kvi_tuned_overrides`, to avoid automatic tuned overrides overwriting the level ablation.

**Output**

- Root directory: `results/truthfulqa_kvi_injection_levels_<timestamp>/`
- One subdirectory per level: `level_<k>/summary.json`
- Summary: `injection_levels_summary.md` / `injection_levels_summary.csv`

---

## 2026-04-15 — Goal: KVI should outperform GraphRAG and KV Prefix on TruthfulQA proxy

**Goal**: On the same eval subset, KVI's `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy` (likelihood_proxy, **higher is better**) should be stably higher than GraphRAG and KV Prefix. To claim "significant", paired test or confidence intervals are needed; before expanding samples, do per-question audit and attribution first.

**Current round code**: TruthfulQA tightened KVI (DRM floor **0.046**, KV triples **4–5**), default **minimal prompt** and **reconcile** (open-QA uses evidence token overlap; fixed the issue where reconcile without DrugBank IDs always discards the KV decode); `run_graph_inference` for English openqa+KVI strengthens repetition penalty and light pre-decode cleaning; KVI answer cleaning weakened mixed Chinese-English.

---

## 2026-04-15 — Improve KVI performance on Exp02 (retrieval + template) and background run method

**Goal**

- Reduce the **relatively high hallucination proxy of KVI** on TruthfulQA (and subsequently FEVER), prioritizing improvements from both **evidence retrieval noise** and **English open-QA generation template** (consistent with audit funnel diagnosis).
- Without **changing data artifacts**, make graph-side default behavior more suitable for **TruthfulQA (`--openqa_mode`, not FEVER claim)**.

**Code change summary (2026-04-15)**

1. **`scripts/run_graph_inference.py`**
   - **Hybrid retrieval**: Under `--openqa_mode` and **non FEVER claim**, raise text retrieval `min_score` from `sentences.jsonl` to **at least 0.11** (still overridable via `--text_search_min_score`), reducing weak-relevance sentences flooding the prompt.
   - **Evidence count**: Under the same setting, before assembling the prompt, after DRM sorting, **default keep at most 8 evidence sentences** (`--max_openqa_evidence_sentences`, `0` means no truncation), alleviating evidence dilution and long generation.
   - **Template**: Under TruthfulQA branch, strengthen **use only evidence, prohibit fabrication / markdown / image links**, and require a brief explanation when evidence is insufficient in the user instruction.
2. **Background script**: `experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh`  
   - `nohup` runs `run_exp01.py` (`graphrag,kv_prefix,kvi`), logs and pid written to `results/`; optional **resident**, **weak oracle**, **audit jsonl** (see in-script comments and environment variables).

**Background run (survives SSH disconnect)**

```bash
cd /home/zd/dev/KVI
chmod +x experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh
nohup env TRUTHFULQA_LIMIT=25 WAIT_RESIDENT=1 RUN_AUDIT=1 \
  bash experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh \
  </dev/null >> experiments/exp02_hallucination/results/exp02_truthfulqa_kvi_opt_outer.log 2>&1 &
```

After completion, check: `results/exp02_truthfulqa_kvi_optimize_*.log`, `truthfulqa_kvi_optimize_*/summary.json`.

**One auto-started run (Agent, 2026-04-15)**

- Command: `run_truthfulqa_kvi_optimization_detached.sh`, `TRUTHFULQA_LIMIT=25`, `RUN_AUDIT=1`, `WAIT_RESIDENT=1`, resident `http://127.0.0.1:18888`.
- Main log: `results/exp02_truthfulqa_kvi_optimize_20260415_093135.log`; output directory: `results/truthfulqa_kvi_optimize_20260415_093135/`; child pid: `results/exp02_truthfulqa_kvi_optimize_20260415_093135.pid`; outer: `results/exp02_truthfulqa_kvi_opt_outer.log`.
- After completion, can run `analyze_graph_prompt_audit.py` on that run's `graph_audit_jsonl`.

**Verification suggestions**

- Compare KVI's `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy` (and GraphRAG as control) in `summary.json` against the same `limit` before changes.
- If `RUN_AUDIT=1` was on: run `analyze_graph_prompt_audit.py` on the output `graph_audit_jsonl` to check whether `MISS_RETRIEVAL` / `MISS_DRM` / `MISS_PROMPT` changed.

---

## 2026-04-15 — Audit smoke priority (dualchannelfix + post-code-adjustment verification)

**Goal**

- Focus on **TruthfulQA smoke with audit**, verifying the **graph-side / KVI path** still runs end-to-end after code changes and produces analyzable audit traces.
- Compare against "no-audit smoke50": the no-audit `truthfulqa_dualchannelfix_smoke50_20260414` has already succeeded; **audit version** previously had child process **600s timeout** (first question of GraphRAG or midway through KV Prefix), needs re-run or parameter adjustment under same settings.

**What "Audit smoke" means**

1. **Full pipeline (including generation)**: `experiments/exp01_main_qa/code/run_exp01.py`, TruthfulQA, `--methods graphrag,kv_prefix,kvi` (same as dualchannelfix subset), with:
   - `--graph_audit_jsonl` → appends retrieval / DRM / prompt audit rows (both GraphRAG and KVI child processes forward `run_graph_inference.py`'s audit parameters).
   - `--graph_audit_oracle_jsonl` (optional but recommended): weak oracle evidence, for R@retrieval / R@drm etc.; can be generated by `experiments/exp02_hallucination/code/build_weak_oracle_evidence.py` from `truthfulqa_eval.jsonl` (historically **NO_ORACLE** caused all-zero audit tables; confirm oracle path and `id` alignment).
2. **Audit only (no full generation)**: `experiments/exp02_hallucination/code/run_graph_prompt_audit_only.py` (internal `--audit_only`), for quickly checking whether retrieval/DRM/prompt entered the prompt, lower load than full pipeline.

**KVI / TruthfulQA eval side (consistent with recent runs)**

- `--truthfulqa_kvi_mc1_answer grounded`
- `--truthfulqa_kvi_max_new_tokens 96` (or consistent with `kvi_truthfulqa_runtime_overrides` in `summary.json`)
- Resident inference (if used): `--inference_service_url http://127.0.0.1:18888`; ANN side per local habit `--ann_force_cpu` or `--ann_via_resident`.

**Timeout and scale**

- Default `--timeout_s` in `run_exp01.py` is **300**, Exp02 main table commonly **600**; **if audit smoke still times out, first adjust to 1200–1800**, or **warm resident** first, or **`--limit` start from 3→10** and incrementally increase.
- Output directory suggested with date: `results/truthfulqa_dualchannelfix_smoke{N}_audit_YYYYMMDD/`, log `results/exp02_truthfulqa_dualchannelfix_smoke{N}_audit_YYYYMMDD.log`.

**Result analysis**

- Summary: `experiments/exp02_hallucination/code/analyze_graph_prompt_audit.py` (on audit JSONL / export md).
- Historically, `retrieval_drm_prompt_audit_smoke10_*_v2.md` showed GraphRAG had **MISS_DRM**, KVI had a different breakdown; interpret based on the **latest** audit output.

**Command skeleton (full pipeline audit smoke, adjust paths and limit to your machine)**

```bash
cd /home/zd/dev/KVI
# Optional: generate weak oracle first (id and dataset aligned, for R@retrieval / R@drm)
KVI/bin/python3 experiments/exp02_hallucination/code/build_weak_oracle_evidence.py \
  --dataset_jsonl experiments/exp02_hallucination/data/truthfulqa_eval.jsonl \
  --sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --out_jsonl experiments/exp02_hallucination/results/truthfulqa_weak_oracle_evidence_smoke10.jsonl \
  --limit 10 --top_k 3

KVI/bin/python3 -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp02_hallucination/data/truthfulqa_eval.jsonl \
  --dataset_name TRUTHFULQA \
  --model models/Qwen2.5-7B-Instruct \
  --graph_index experiments/exp02_hallucination/artifacts/truthfulqa/graph_index.json \
  --triple_kvbank_dir experiments/exp02_hallucination/artifacts/truthfulqa/triple_kvbank \
  --graph_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --ann_kv_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar \
  --methods graphrag,kv_prefix,kvi \
  --limit 10 \
  --out_dir experiments/exp02_hallucination/results/truthfulqa_dualchannelfix_smoke10_audit_$(date +%Y%m%d) \
  --timeout_s 1800 \
  --inference_service_url http://127.0.0.1:18888 \
  --ann_inference_service_url "" \
  --ann_force_cpu \
  --truthfulqa_kvi_mc1_answer grounded \
  --truthfulqa_kvi_max_new_tokens 96 \
  --graph_audit_jsonl experiments/exp02_hallucination/results/truthfulqa_graph_prompt_audit_smoke10_$(date +%Y%m%d).jsonl \
  --graph_audit_oracle_jsonl experiments/exp02_hallucination/results/truthfulqa_weak_oracle_evidence_smoke10.jsonl \
  2>&1 | tee experiments/exp02_hallucination/results/exp02_truthfulqa_dualchannelfix_smoke10_audit_$(date +%Y%m%d).log
```

(If `weak_oracle_evidence_smoke10.jsonl` does not exist yet, first generate it with `build_weak_oracle_evidence.py` for the same `--limit`, or temporarily remove `--graph_audit_oracle_jsonl` to see only non-oracle fields.)

---

## Maintenance notes

- **New conventions**: Copy the previous section template, change date, fill in "Goal / Parameters / Known issues".
- **Agent hint**: When handling Exp02 smoke or audit, read the latest section of this file first, then check `results/` and `run_exp01.py` current defaults.
