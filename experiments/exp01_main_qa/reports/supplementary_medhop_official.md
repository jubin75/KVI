# Supplement: MedHopQA official-style evaluation set (built from medhop_raw)

> **Repo path**: `experiments/exp01_main_qa/reports/supplementary_medhop_official.md`. `data/medhop_*`, `artifacts/` etc. are local build/data disk artifacts and are **not checked into Git** by default; the paths in the tables below represent the pipeline convention locations.

This supplementary table documents the **natural language query + short / long answer + supporting_facts** evaluation format built from `medhop_raw`, and its relationship to the main table **MedHop-ID** variant. This split can directly interface with the Exp01 pipeline (`medhop_eval.jsonl` + `sentences_medhop.jsonl` + `triples_medhop.jsonl`).

## Data locations

| Artifact | Path |
|------|------|
| Source data (this repo) | `experiments/exp01_main_qa/data/medhop_raw/medhop_source_validation.parquet.jsonl` |
| Supplementary-table-level schema (paper/appendix) | `experiments/exp01_main_qa/data/medhop_official/medhop_official_eval.jsonl` |
| Exp01 runnable `dataset` | `experiments/exp01_main_qa/data/medhop_official/medhop_eval.jsonl` |
| Sentences, triples | `experiments/exp01_main_qa/data/medhop_official/sentences_medhop.jsonl`, `triples_medhop.jsonl` |
| Build manifest | `experiments/exp01_main_qa/data/medhop_official/manifest_medhop_official.json` |

## Table S — `medhop_official_eval.jsonl` fields

| Field | Description |
|------|------|
| `id` | Consistent with raw, e.g. `MH_dev_0` |
| `question` | Natural language template: `Which DrugBank-listed drug interacts with DBxxxx? …`; by default also includes a line constraining output to **only the partner DrugBank ID** (consistent with the main table MedHop-ID EM convention) |
| `raw_query` | Original `interacts_with DBxxxx?` |
| `answer` / `answers` | Gold answer (partner DrugBank ID) |
| `short_answer` | Same as gold ID (this release does **not** map to generic drug names; if natural language gold like `Ritonavir` etc. is needed, an external ID→name table is required) |
| `long_answer` | First **2** `supports` passages truncated and concatenated (≤2500 characters), for appendix or manual verification |
| `type` | Fixed tag `drug_drug_interaction_completion` |
| `supporting_facts` | `{title, sentence}` pairs split from `supports`: if the line starts with `DBxxxxx :`, `title` is that ID; otherwise `title` is `passage_k` |
| `candidates` | Multi-choice ID list preserved from raw |
| `dataset` | `MedHopQA_official_nl` |
| `gold_note` | Brief note that gold standard remains IDs and what EM means |

`medhop_eval.jsonl` is the Exp01 entry point: each line contains at minimum `id`, `question`, `answer`/`answers`; `question` is identical to that in `medhop_official_eval.jsonl`.

## Scale (current build)

Based on `manifest_medhop_official.json`, a typical build yields:

- Samples: **342**
- Sentence blocks: **105098** (split from `supports` strings by sentence)
- Triples: **488** (1–2 per sample; when some evidence sentences do not simultaneously hit query/answer IDs, only 1 triple is kept)

## Rebuild command

```bash
cd experiments/exp01_main_qa/code
python3 prepare_medhop_official_from_raw.py \
  --medhop_raw ../data/medhop_raw/medhop_source_validation.parquet.jsonl \
  --out_dir ../data/medhop_official
```

Optional: `--no_append_id_only_hint` does not append the "output only DB ID" hint to the model (more open-ended queries; EM convention may not be fully comparable with the main table strict-ID convention).

## Differences from main table MedHop-ID

| Item | MedHop-ID (main table) | Official NL (this supplementary table) |
|------|-------------------|------------------------|
| Query form | `interacts_with DBxxxx?` + ID output instruction | Natural language "Which DrugBank-listed drug…" + same-style ID output instruction by default |
| Gold | DrugBank partner ID | Same |
| EM | Substring / existing pipeline EM | Same when using the same `medhop_eval.jsonl` |

The main table still retains the smaller **MedHopQA_n40** subset results; the full **342** samples can be used for extended experiments or appendix reporting.

## Exp01 run tips

Point `--dataset` to `data/medhop_official/medhop_eval.jsonl`, point graph and KVI related paths to the newly generated `sentences_medhop.jsonl` / `triples_medhop.jsonl` in the **same directory**, and build `graph_index`, tagged sentences, kvbank, etc. following the existing MedHop workflow (same usage as `prepare_medhopqa_assets.py` outputs). The full 342 samples and ~105k sentences have significantly higher resident/embedding cost than n40; set `limit` or batch as needed.

## Exp01 smoke test (`--limit 2`, runnable)

Execute from repo root (the example reuses existing **MedHopQA_n40** artifacts: `MH_dev_0` / `MH_dev_1` have the same IDs as the first two official samples, and graph/KV are aligned with these two; for the **formal full 342-sample run**, build dedicated `medhop_official` artifacts):

```bash
source KVI/bin/activate   # if using project venv
python -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp01_main_qa/data/medhop_official/medhop_eval.jsonl \
  --dataset_name MedHopQA_official_smoke \
  --model /path/to/Qwen2.5-7B-Instruct \
  --graph_index experiments/exp01_main_qa/artifacts/medhop_n40/graph_index.json \
  --triple_kvbank_dir experiments/exp01_main_qa/artifacts/medhop_n40/triple_kvbank \
  --graph_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_kv_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --limit 2 \
  --out_dir experiments/exp01_main_qa/results/medhop_official_smoke_limit2 \
  --ann_force_cpu
```

**One successful smoke test output on this machine**: `experiments/exp01_main_qa/results/medhop_official_smoke_limit2/` (contains `predictions.jsonl`, `summary.json`, `results.md`). Base model is local `Qwen2.5-7B-Instruct`; under `relaxed` EM, GraphRAG/KVI achieves 100% on the 2 samples (substring match on gold ID), other methods yield 0 on this small sample — only for pipeline connectivity verification.

### Full formal experiment (background, survives SSH disconnect)

- **Script**: `experiments/exp01_main_qa/code/run_medhop_official_full_background.sh`  
  - Under `artifacts/medhop_official/`, sequentially: semantic tagging → `kvbank_sentences` → `graph_index.json` → `triple_kvbank` (same structure as n40).  
  - If `127.0.0.1:18888` has no `/health`, it will **nohup launch** `exp01_resident_infer_service.py` (graph-side singleton loads 7B, logs below).  
  - Then **`run_exp01.py` full 342 samples, `--resume`**; default output directory **`results/medhop_official_fullmethods_qwen25_7b_kvituned/`**, and default **KVI tuning**: `kvi_drm_threshold=0.12`, `max_kv_triples=2`, `top_k_relations=1`, `--kvi_minimal_prompt`. **Pre-tuning baseline** (main table Panel A row) at `results/medhop_official_fullmethods_qwen25_7b/summary.json`.
- **Recommended startup method** (choose either this or the script's internal `nohup`; below wraps with an extra tee):

```bash
cd /home/zd/dev/KVI
nohup bash experiments/exp01_main_qa/code/run_medhop_official_full_background.sh \
  >> experiments/exp01_main_qa/results/medhop_official_full_pipeline.log 2>&1 &
```

- **Main log**: `experiments/exp01_main_qa/results/medhop_official_full_pipeline.log`  
- **Resident service log** (if launched by script): `experiments/exp01_main_qa/results/medhop_official_resident.log`  
- **Environment variables**: `MODEL`, `SKIP_ARTIFACTS=1` (eval only), `START_RESIDENT=0` (don't start resident, very slow), `RESIDENT_PORT`, `MEDHOP_OFFICIAL_OUT`, `KVI_DRM_THRESHOLD`, `KVI_MAX_KV_TRIPLES`, `KVI_TOP_K_RELATIONS`, `KVI_MINIMAL_PROMPT` (0/1).
