# Experiment 02 — Hallucination (proxy metrics)

**Current priority experiment conventions (rolling update by date, including audit smoke parameters and command skeletons)**: see [`LATEST_EXPERIMENT_REQUIREMENTS.md`](./LATEST_EXPERIMENT_REQUIREMENTS.md).

## Experiment approach and KVI analysis goals (quick reference)

When doing **KVI improvement** on Exp02 going forward, default to the following: **simultaneously compare against TruthfulQA and FEVER**, and derive directions for improving KVI (prompts, retrieval & KV, hyperparameters, structure, etc.) from both result sets, rather than optimizing only one set while degrading the other.

### Pipeline and scale

| Item | Convention |
|------|------|
| Main pipeline | `run_exp02_hallucination.py` → per-data `artifacts/<name>/` → `run_exp01.py` |
| Compared methods | `llm,rag,graphrag,kv_prefix,kvi` (five methods) |
| Default scale | TruthfulQA **500**, FEVER **1000**; actual **`n`** follows `results/*/summary.json` and `data/dataset_manifest.json` |
| Graph-side inference | Resident **`http://127.0.0.1:18888`** (`--resident_url`) |
| ANN (RAG / KV Prefix) | Default **local CPU**: `--ann_inference_service_url` empty and **`--ann_force_cpu`**; if explicit **`--ann_via_resident`**, shares the same resident as graph |
| Existing data and artifacts, no recompilation | **`--skip_mirror_and_prepare --reuse_artifacts`** (see `run_exp02_fast_once.sh`) |
| Eval checkpoint resume | Exp02 passes **`--resume_eval`** → forwarded to `run_exp01.py`'s **`--resume`** (validates `predictions.jsonl` prefix then appends) |
| FEVER-only under SSH + start resident | `code/run_fever_gpu_detached.sh` (includes **grace** and **`--resume_eval`**) |
| Resident already up locally, only resume FEVER | `code/run_fever_resume_eval.sh` |

### Metrics when reading results

- **TruthfulQA**: The table's **relaxed EM** is a **proxy**, not the official MC; for external comparison, prefer MC1/MC2 or the official script (see "Community conventions" below).  
- **FEVER**: Besides proxy EM, always check **`fever_label_accuracy`** (three-class label, closer to the veracity task).  
- **Graph/KVI prompts**: `scripts/run_graph_inference.py` uses a **single-line SUPPORTS / REFUTES / NOT ENOUGH INFO** tightened instruction for FEVER-style stems (those starting with `Claim:` and containing three-label instructions); TruthfulQA still uses open-ended English instructions. **If the same `predictions.jsonl` contains mixed rows from "before/after resume" or "delete file and re-run before/after", templates may be inconsistent** — for paper-grade unified convention, delete `predictions.jsonl` and re-run the full set (**not** `--resume_eval`).

### Analysis main line

1. Compare **KVI against other methods** (especially **GraphRAG**) on both datasets.  
2. Categorize errors: **format and label**, **retrieval/KV deviation**, **evidence-generation inconsistency**, **overly long or off-topic generation**, etc.  
3. When proposing changes, distinguish between **task-template-oriented** (may help FEVER but not TQA) and **mechanism-oriented** (e.g. KV selection and injection), and plan **cross-validation on both sets**.

---

## Full scale and background tasks

- **Default full scale** (when `run_exp02_hallucination.py` is called without `--limit`):  
  - **TruthfulQA**: `--truthfulqa_max` **500** (`generation` validation, see `prepare_exp02_datasets.py`)  
  - **FEVER**: `--fever_max` **1000** (local parquet from KILT/FEVER validation split, or HF fallback)  
- **Smoke test**: When `--limit N` is passed, `build_assets_from_dataset.py` only builds graph/KV for the first N samples; `run_exp01.py` also gets `--limit N`. Check **`n`** in `results/*/summary.json` to distinguish full scale from small sample.  
- **Auto-restart**: `code/run_exp02_autoresume.sh` loops the main script while **`results/hallucination_proxy_summary.json` does not exist**; once that file appears, it considers this round of Exp02 done and exits. To **force a full re-run**, first backup or delete the old `hallucination_proxy_summary.json` (and optionally clear result directories), then start the supervisor.  
- **Logs**: Main process stdout/stderr → `results/exp02_pipeline_v2.log`; supervisor line-by-line records → `results/exp02_supervisor.log` (`*.log` is `.gitignore`d, local only).

To check "whether background is still running": if local processes `run_exp02_autoresume.sh` / `run_exp02_hallucination.py` exist, and TruthfulQA or FEVER pipeline steps (e.g. `triple_kv_compiler.py`) are running, the full pipeline has not yet finished.

### Fast one-shot run (recommended)

If **`data/dataset_manifest.json` already has 500+1000** and **`artifacts/*` already have compiled graphs and triple KV**, do not use autoresume to re-compile from scratch repeatedly. First **stop** `run_exp02_autoresume.sh` and the old `run_exp02_hallucination.py`, start resident **18888**, then:

```bash
# Log: results/exp02_fast_run.log
nohup experiments/exp02_hallucination/code/run_exp02_fast_once.sh >> experiments/exp02_hallucination/results/exp02_fast_run.log 2>&1 &
```

Equivalent args: `--skip_mirror_and_prepare --reuse_artifacts` (see `run_exp02_hallucination.py`). Will still run **two passes** of `run_exp01` (500 + 1000 samples × five methods), with time dominated by inference, no repeated CPU KV compilation.

Optional: run only one dataset, e.g. `--only_datasets fever` (the other dataset will merge from existing `summary.json` into `hallucination_proxy_summary.json`).

### Survive SSH disconnect (recommended: FEVER fill-in + GPU0)

The resident **`/health` returns 200 before the model is loaded**, so starting eval immediately can easily cause **`RemoteDisconnected` / `Connection refused`**. Script **`code/run_fever_gpu_detached.sh`** starts resident first, polls health, then **`sleep 45`** (adjustable via env var **`RESIDENT_READY_GRACE_SEC`**), then **`exec`s FEVER-only Exp02**, the whole block on **nohup**, **detached from the SSH session**:

```bash
cd ~/dev/KVI
nohup bash experiments/exp02_hallucination/code/run_fever_gpu_detached.sh \
  </dev/null \
  >> experiments/exp02_hallucination/results/exp02_fever_gpu_orchestrator_outer.log 2>&1 &
```

- Orchestrator log: `results/exp02_fever_gpu_orchestrator.log`  
- Resident log: `experiments/results/resident_18888_gpu.log`  
- FEVER eval output: `results/exp02_fever_gpu.log`  

After disconnect, check with `pgrep -af 'run_fever_gpu_detached|run_exp02_hallucination|resident_infer'` and `wc -l results/fever_fullmethods_qwen25_7b/predictions.jsonl`.

---

## What is the "hallucination rate" in the current figure (and differences from community conventions)

In **`hallucination_proxy_summary.json`** written by `run_exp02_hallucination.py`: **TruthfulQA** is **`100 − relaxed EM`**, **FEVER** is **`100 − fever_label_accuracy`** (see that JSON's `note` and in-script comments).

- **relaxed EM** (`experiments/exp01_main_qa/code/metrics.py`): SQuAD-style normalization, checks whether **any gold appears as a substring in the model's full output** (with minor extensions for `yes`/`no`). Suitable for long-generation Hotpot/NQ-style QA, **not** the TruthfulQA or FEVER official main table convention.  
- **Community/official conventions to align with (recommended for paper or external comparison)**:  
  - **TruthfulQA**: Official/commonly used is **generation-based human or automated evaluation** (e.g. MC1/MC2, or official scripts), not "whether the reference sentence appears in the long generation."  
  - **FEVER**: The shared task commonly uses **three-class label accuracy** (SUPPORTS / REFUTES / NOT ENOUGH INFO); a full pipeline can also plug into **fever-scorer** (requires predicted Wikidata evidence etc.), which differs from this repo's "label-only" setting.

### Recommended implementation order (settled)

1. **FEVER (priority, landed)**  
   - `run_exp01.py` additionally computes **label accuracy** when `--dataset_name FEVER`: finds the **first occurrence** of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` in the model's full response (`metrics.parse_fever_label`), compares against gold written by `prepare_exp02_datasets.py`.  
   - See each method's **`fever_label_accuracy`**, **`fever_label_ci95_*`** in `results/fever_fullmethods_qwen25_7b/summary.json`; per-question see **`fever_label_em`** in `predictions.jsonl`.  
   - Closer than **relaxed EM** to the shared task's **veracity label** convention; **still not** the official scorer with evidence submission.  
2. **TruthfulQA (second step, integrated)**  
   - `prepare_exp02_datasets.py` now supports merging `multiple_choice` `mc1_targets` / `mc2_targets` (if locally or online readable) into `truthfulqa_eval.jsonl`.  
   - If `multiple_choice` is unavailable, it self-builds MC targets from the generation split's `correct_answers/incorrect_answers`, guaranteeing coverage.  
   - `run_exp01.py` now outputs `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`, default `--truthfulqa_mc_mode likelihood_proxy` (log-likelihood scoring on candidate options).  
   - Note: this implementation is closer to the MC convention than pure string matching, but is still noted as **proxy** — not fully equivalent item-by-item to the official TruthfulQA release pipeline.

Exp02's **`hallucination_proxy_summary.json`** (written by `run_exp02_hallucination.py`) convention is: **TruthfulQA = `100 − relaxed EM`**, **FEVER = `100 − fever_label_accuracy`** (consistent with the JSON's `note`). Therefore **the "hallucination rates" for the two tasks in the same file are not horizontally comparable**: TruthfulQA's column is often very high (unlikely to substring-match reference answers in long generation), which does not imply the model is more "honest" on FEVER. If the paper's main figure needs to juxtapose FEVER side by side and TruthfulQA hopes to approach community MC semantics, use the **`results/summary.json` + three-panel figure** below (MC1 / MC2 / FEVER label).

---

## Actual parameters: `run_exp02_hallucination.py` → `run_exp01.py` (line-by-line mapping)

The following are items the main script **explicitly passes** or matches **defaults** when calling `run_exp01.py` for **each dataset** after building `artifacts/<name>/` (excerpted from `run_exp02_hallucination.py`).

| Parameter | Value / Note |
|------|-----------|
| `--dataset` | `data/truthfulqa_eval.jsonl` or `data/fever_eval.jsonl` |
| `--dataset_name` | `TRUTHFULQA` or `FEVER` |
| `--model` | Default `.../models/Qwen2.5-7B-Instruct` |
| `--graph_index` | `artifacts/<name>/graph_index.json` |
| `--triple_kvbank_dir` | `artifacts/<name>/triple_kvbank` |
| `--graph_sentences_jsonl` | `artifacts/<name>/sentences.tagged.jsonl` |
| `--ann_kv_dir` and other ANN paths | Corresponding dataset `kvbank_sentences` and pattern sidecar |
| `--methods` | `llm,rag,graphrag,kv_prefix,kvi` |
| `--out_dir` | `results/<name>_fullmethods_qwen25_7b` |
| `--timeout_s` | `600` |
| `--bootstrap_samples` / `--permutation_samples` | `1000` / `2000` |
| `--inference_service_url` / `--ann_inference_service_url` | If command line provided `--resident_url` (autoresume uses `http://127.0.0.1:18888`), both are passed |
| `--limit` | **Only** when `run_exp02_hallucination.py --limit K` is given, restricts eval sample count |
| `--em_mode` | **Not passed**, uses `run_exp01.py` default **`relaxed`** |
| `--openqa_mode` | **Not passed**, uses `run_exp01.py` default **`True`** (`BooleanOptionalAction`, meaning Graph/KVI use English open-domain prompts; see `run_exp01.py` help text) |

Unlisted `run_exp01` parameters use their in-file defaults (e.g. KVI's `kvi_max_kv_triples=3`, `kvi_reconcile_no_kv_decode=False`, etc.).

---

## FEVER: Where does gold come from? Why does it coexist with `openqa_mode`?

### How gold (`answer` / `answers`) is constructed

Logic is in **`prepare_exp02_datasets.py`** (same args as the `run_exp02_hallucination.py` invocation: `--fever_max`, `--mirror_root`, `--mirror_data_root`, `--streaming`).

1. Load rows from local parquet (e.g. `kilt_fever_validation.parquet`) or fallback HF config.  
2. **Label mapping** (when integer-encoded):  

   `label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}`  

   If column is string, `strip().upper()`; if KILT-style `output` list contains an `answer` field, take its uppercase string.  
3. **Fields written to JSONL**:  
   - `question`: Manually assembled English instructions + claim, requiring the model to **output only one of the three labels** (see in-script `q = f'Claim: "{claim}"\nBased on evidence, ...'`).  
   - `answer` / `answers`: **Single gold string**, i.e. the above **`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`** (only this one string placed in the `answers` list).

Therefore, the **supervision signal remains FEVER's three-class labels**; at eval time `run_exp01` uses relaxed EM to check whether the normalized gold substring (e.g. `supports`) appears in the model's **long output**, coexisting with "open-domain long answers."

### Why `openqa_mode=True` by default (unrelated to Chinese medical prompts)

- `openqa_mode` only affects the **Graph/KVI path** prompt template in `run_graph_inference`: default **English open-domain**, avoiding the **Chinese medical system prompt** (prepared for MedHop in Exp01) from polluting TruthfulQA/FEVER.  
- FEVER's **claim+instructions** are already injected via the dataset's **`question` field**; gold is still **three-class labels**, not open-domain free-text facts.

To **strictly reproduce MedHop-style Chinese closed-domain graph reasoning**, you would need to **explicitly pass** `run_exp01.py`'s `--no-openqa_mode` for Exp02 (currently `run_exp02_hallucination.py` does **not** pass this, so all use default open-domain English graph-side prompts).

---

## Results and figures (paper figures — look here)

**Absolute path (this machine)**: `/home/zd/dev/KVI/experiments/exp02_hallucination/results/`

### Currently available results (TruthfulQA + FEVER)

> Convention note: The EM in the table below is `relaxed EM`; Exp02 hallucination rate is `100 - relaxed EM` (proxy).

| Dataset | Method | EM (%) | 95% CI | F1 Mean | Proxy Hallucination (%) |
|---|---|---|---:|---:|---:|---:|
| TruthfulQA | LLM | 5.0 | [3.2, 7.0] | 0.176 | 95.0 |
| TruthfulQA | RAG | 7.4 | [5.0, 9.6] | 0.135 | 92.6 |
| TruthfulQA | GraphRAG | 17.8 | [14.6, 21.2] | 0.195 | 82.2 |
| TruthfulQA | KV Prefix | 3.8 | [2.2, 5.6] | 0.016 | 96.2 |
| TruthfulQA | KVI | 11.8 | [9.0, 14.6] | 0.115 | 88.2 |
| FEVER | LLM | 38.3 | [35.2, 41.2] | 0.173 | 61.7 |
| FEVER | RAG | 92.6 | [91.0, 94.1] | 0.288 | 7.4 |
| FEVER | GraphRAG | 73.0 | [70.4, 76.0] | 0.576 | 27.0 |
| FEVER | KV Prefix | 74.3 | [71.7, 76.8] | 0.312 | 25.7 |
| FEVER | KVI | 89.3 | [87.3, 91.2] | 0.893 | 10.7 |

FEVER additional label metrics (closer to veracity task):

| FEVER Method | FEVER Label Accuracy (%) | 95% CI |
|---|---|---:|
| LLM | 30.9 | [28.0, 33.7] |
| RAG | 92.5 | [90.9, 94.1] |
| GraphRAG | 68.8 | [66.1, 72.0] |
| KV Prefix | 72.3 | [69.6, 75.1] |
| KVI | 89.3 | [87.3, 91.2] |

Corresponding files:

- `results/truthfulqa_fullmethods_qwen25_7b/summary.json`
- `results/fever_fullmethods_qwen25_7b/summary.json`
- `results/hallucination_proxy_summary.json`

### Tables and raw metrics

| Content | Path |
|------|------|
| Cross-dataset summary (proxy metrics) | `results/hallucination_proxy_summary.json`, `hallucination_proxy_summary.md` (generated after `run_exp02_hallucination.py` completes; **if missing**, use the two `summary.json` files below to manually construct figures) |
| TruthfulQA per-method EM / F1 / CI (and `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`) | `results/truthfulqa_fullmethods_qwen25_7b/summary.json`, `results.md`, `results.csv` |
| FEVER same as above + **label accuracy** `fever_label_accuracy` | `results/fever_fullmethods_qwen25_7b/summary.json`, `results.md`, `results.csv` |
| Per-question predictions | `predictions.jsonl` under each dataset directory |

### Vector figures for paper publication (SVG, pluggable into LaTeX / Word)

**Two-panel vs three-panel (do not mix files)**

| Figure | Panels | TruthfulQA convention | FEVER convention |
|----|------|-----------------|------------|
| `hallucination_proxy_bars_paper.svg` | **2** | `100 − relaxed EM` (substring proxy, bars often **80–96%**) | `100 − fever_label_accuracy` (consistent with `hallucination_proxy_summary.json`) |
| `hallucination_proxy_three_panel_paper.svg` or `unified_hallucination_bars.svg` | **3** | Left two panels: `100 − MC1 / MC2 likelihood proxy` | Right panel: `100 − fever_label_accuracy` |

Three-panel data comes from **`results/summary.json`** generated after running Exp02 (different from `hallucination_proxy_summary.json`: the latter still uses relaxed EM for TruthfulQA).

Script: `experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py`

```bash
# Recommended: generate paper two-panel + three-panel + optional split figures in one pass (requires existing results/summary.json)
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --paper \
  --three_panel_unified \
  --fever_label_figure \
  --truthfulqa_mc_figure

# Or generate from two per-dataset summary.json only (when hallucination_proxy_summary.json is missing)
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --truthfulqa_summary experiments/exp02_hallucination/results/truthfulqa_fullmethods_qwen25_7b/summary.json \
  --fever_summary experiments/exp02_hallucination/results/fever_fullmethods_qwen25_7b/summary.json \
  --paper \
  --three_panel_unified \
  --fever_label_figure \
  --truthfulqa_mc_figure
```

(Three-panel can also be generated standalone via `code/plot_unified_hallucination_bars.py` outputting `unified_hallucination_bars.svg`, with slightly simpler styling; uses the same `results/summary.json` as `--three_panel_unified`.)

| Output file | Use |
|----------|------|
| **`hallucination_proxy_three_panel_paper.svg`** | **Recommended paper side-by-side main figure**: three panels with unified convention (TQA MC1 + TQA MC2 + FEVER label → hallucination rate) |
| `hallucination_proxy_three_panel.svg` | Same as above, non-`paper` style |
| `unified_hallucination_bars.svg` | Same-class three-panel (`plot_unified_hallucination_bars.py`) |
| **`hallucination_proxy_bars_paper.svg`** | **Two-panel only**: left TQA **relaxed EM** (prone to "inflated" values), right FEVER label; figure caption has noted the difference |
| `hallucination_proxy_bars.svg` | Screen preview / non-print |
| **`fever_label_accuracy_bars_paper.svg`** | FEVER **three-class label accuracy** (requires `summary.json` with `fever_label_accuracy`; old results need re-run of `run_exp01.py`) |
| `fever_label_accuracy_bars.svg` | Same as above, non-`paper` style |
| **`truthfulqa_mc_proxy_bars_paper.svg`** | TruthfulQA **MC1/MC2 proxy** (from `truthfulqa_mc*_proxy`, not official MC script scores) |
| `truthfulqa_mc_proxy_bars.svg` | Same as above, non-`paper` style |

**PDF**: Editorial offices often require PDF/EPS. Use Inkscape (`inkscape file.svg --export-filename=file.pdf`) or `rsvg-convert -f pdf -o file.pdf file.svg` to convert from **SVG**, preserving vector quality.

**Wording to make clear in figures**: In two-panel figures, **TruthfulQA = relaxed EM proxy**, which is not the same number as **MC1/MC2** or human evaluation; in three-panel figures, TruthfulQA is **likelihood MC proxy**. On the FEVER side, use **label accuracy** as primary; still not the official fever-scorer with evidence.

Also: `*.html` is for browser preview only; submissions generally use **SVG/PDF**.

---

## Code entry points

| Step | Script |
|------|------|
| Mirror download | `experiments/code/download_mirror_datasets.py` |
| Data JSONL | `code/prepare_exp02_datasets.py` |
| Main orchestrator | `code/run_exp02_hallucination.py` |
| Evaluation | `experiments/exp01_main_qa/code/run_exp01.py` |
| EM definition | `experiments/exp01_main_qa/code/metrics.py` |
