## Experiment 1
### Main QA Performance

**Purpose**: Verify whether KVI improves question-answering accuracy (Exact Match, EM).  
**Datasets**: `HotpotQA`, `MedHopQA` and `NQ (Natural Questions)`.  
**Presentation format**: Unified results table (comparing EM and CI95 across `HotpotQA / MedHopQA / NQ` by method).

**Git check-in notes**: Scoring and build artifacts under `results/`, `artifacts/`, `data/` are **not committed** by default (see repo root `.gitignore`). Paper-facing **Markdown main tables and supplementary tables** are placed in **`reports/`** (see `reports/README.md`); if `results` is locally a symlink to a data disk, use `cp -aL` to sync into `reports/` before committing.

---

### Evaluation Table (target format)

| Method | Retrieval | Injection | HotpotQA EM | HotpotQA CI95 | MedHopQA EM | MedHopQA CI95 | NQ EM | NQ CI95 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| LLM | none | none | 32.4 | [30.1, 34.8] | 28.1 | [26.0, 30.0] |
| RAG | ANN | prompt | 55.2 | [52.4, 58.1] | 47.8 | [45.3, 50.1] |
| GraphRAG | graph | prompt | 58.7 | [56.0, 61.3] | 50.3 | [48.0, 52.9] |
| KV Prefix | ANN | KV | 57.9 | [55.4, 60.6] | 49.1 | [46.9, 51.6] |
| KVI | graph | KV + prompt | 66.4 | [63.7, 69.0] | 56.2 | [53.8, 58.5] |

> Note: The values above are example formats from the paper/documentation. When reproducing experiments in this repo, replace them with actual EM values from your runs.

---

### Method Definitions (for Exp01)

- **LLM**: no retrieval, no injection
- **RAG**: ANN retrieval + prompt evidence
- **GraphRAG**: graph retrieval + prompt evidence
- **KV Prefix**: ANN retrieval + KV-only injection
- **KVI**: graph retrieval + KV injection + prompt evidence (dual-channel)

### KVI vs RAG / analysis (important)

- **Root cause analysis**: see `KVI_FAILURE_ANALYSIS.md` (domain mismatch, synthetic graph, retrieval channel inconsistency, etc.).
- **Default** `run_exp01.py` enables **`--openqa_mode`** for Graph/KVI (English open-domain prompt, avoiding leftover Chinese medical system prompt).
- **Ablation**: **`--kvi_minimal_prompt`** — when injecting KV, removes the long evidence list from the prompt, to test whether "KV + question only" reduces attention deflection.
- **Exp3 / Exp6**: Independent directories `experiments/exp03_retrieval_quality/`, `experiments/exp06_ablation/`; summary table at repo root `experiments/RESULTS_COMBINED.md` (`python experiments/combine_experiment_results.py`).

---

### Inputs Required

- HotpotQA eval split (JSONL / converted format)
- MedHopQA eval split (JSONL / converted format)
- NQ eval split (JSONL / converted format)
- Base LLM (local path recommended)
- Retrieval artifacts (ANN / graph index)
- KV artifacts (for KV Prefix / KVI), e.g. `triple_kvbank/`

---

### Dataset Preparation (actual, not toy)

Use mirror-accessible dataset names:

- HotpotQA: `hotpot_qa` (`distractor` / `validation`)
- MedHopQA: local `.json` / `.jsonl` (schema-adapted by script below)
- NQ: `natural_questions` (`validation`)

Prepare command:

```bash
python experiments/exp01_main_qa/code/prepare_hotpot_nq.py \
  --out_dir experiments/exp01_main_qa/data/benchmarks \
  --hotpot_config distractor \
  --hotpot_split validation \
  --nq_split validation \
  --hotpot_max 0 \
  --nq_max 0 \
  --streaming
```

> `--hotpot_max 0 --nq_max 0` means pull the full official validation set (or the complete subset the mirror can provide).  
> For a smoke test, temporarily change to `--hotpot_max 100 --nq_max 100`.

Generated files:

- `experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl`
- `experiments/exp01_main_qa/data/benchmarks/nq_eval.jsonl`
- `experiments/exp01_main_qa/data/benchmarks/dataset_manifest.json`

### MedHopQA build (Hotpot-like construction)

Convert MedHopQA to Exp01 format and build sentence/triple assets in one step:

```bash
python experiments/exp01_main_qa/code/prepare_medhopqa_assets.py \
  --medhop_input /path/to/medhopqa.jsonl \
  --out_dir experiments/exp01_main_qa/data/medhop_benchmarks \
  --max_examples 0
```

Outputs:

- `experiments/exp01_main_qa/data/medhop_benchmarks/medhop_eval.jsonl`
- `experiments/exp01_main_qa/data/medhop_benchmarks/sentences_medhop.jsonl`
- `experiments/exp01_main_qa/data/medhop_benchmarks/triples_medhop.jsonl`
- `experiments/exp01_main_qa/data/medhop_benchmarks/manifest_medhop.json`

Then build artifacts under `experiments/exp01_main_qa/artifacts/medhop/` with the same graph/ANN/KV commands used for Hotpot/NQ (replace paths accordingly).

Unified JSONL schema:

```json
{"id":"...","question":"...","answer":"...","answers":["...","..."],"dataset":"HotpotQA|NQ"}
```

### Hotpot multi-hop asset build (for KV value verification)

To avoid QA-template synthetic artifacts and preserve Hotpot's original long context,
build a multi-hop-oriented asset pack from raw `context` + `supporting_facts`:

```bash
python experiments/exp01_main_qa/code/prepare_hotpot_multihop_assets.py \
  --out_dir experiments/exp01_main_qa/data/multihop_hotpot \
  --hotpot_config distractor \
  --hotpot_split validation \
  --hotpot_max 120 \
  --streaming
```

Outputs:

- `experiments/exp01_main_qa/data/multihop_hotpot/hotpot_eval_multihop.jsonl`
- `experiments/exp01_main_qa/data/multihop_hotpot/sentences_multihop.jsonl`
- `experiments/exp01_main_qa/data/multihop_hotpot/triples_multihop.jsonl`
- `experiments/exp01_main_qa/data/multihop_hotpot/manifest_multihop.json`

This dataset is intended for validating whether KVI benefits from multi-hop, long-text evidence.

---

### Run Strategy (single `--dataset`)

As requested, the experiment script uses **single-dataset single-evaluation**:

- `run_exp01.py --dataset <one_dataset>`: evaluates all 5 methods (LLM / RAG / GraphRAG / KV Prefix / KVI) on that dataset in one pass.
- Run once each for HotpotQA, MedHopQA, NQ.
- Then use `aggregate_exp01.py` to auto-merge into the main table (HotpotQA + MedHopQA + NQ).
- Each method automatically outputs `EM`, `95% bootstrap CI`, and `summary.json` includes paired permutation test `p-value` of KVI vs other methods.

---

### Official-style staged execution (recommended)

For unstable remote environments, a two-phase approach is recommended:

1. **Phase A (verification)**: Hotpot/NQ each with `--limit 100`, confirm pipeline and statistical output are correct.  
2. **Phase B (formal)**: Remove `--limit`, run full validation in background (survives SSH disconnect).

---

### Commands

#### 1) Run on HotpotQA

```bash
python experiments/exp01_main_qa/code/run_exp01.py \
  --dataset /path/to/hotpot_eval.jsonl \
  --dataset_name HotpotQA \
  --model /home/zd/dev/KVI/models/Qwen2.5-7B-Instruct \
  --graph_index /path/to/graph_index.json \
  --triple_kvbank_dir /path/to/triple_kvbank \
  --graph_sentences_jsonl /path/to/sentences.jsonl \
  --ann_kv_dir /path/to/kvbank_sentences \
  --ann_sentences_jsonl /path/to/sentences.tagged.jsonl \
  --ann_semantic_type_specs /path/to/semantic_type_specs.json \
  --ann_pattern_index_dir /path/to/pattern_sidecar \
  --ann_sidecar_dir /path/to/work_dir \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_dir experiments/exp01_main_qa/results/hotpot \
  --bootstrap_samples 1000 \
  --permutation_samples 2000
```

#### 2) Run on NQ

```bash
python experiments/exp01_main_qa/code/run_exp01.py \
  --dataset /path/to/nq_eval.jsonl \
  --dataset_name NQ \
  --model /home/zd/dev/KVI/models/Qwen2.5-7B-Instruct \
  --graph_index /path/to/graph_index.json \
  --triple_kvbank_dir /path/to/triple_kvbank \
  --graph_sentences_jsonl /path/to/sentences.jsonl \
  --ann_kv_dir /path/to/kvbank_sentences \
  --ann_sentences_jsonl /path/to/sentences.tagged.jsonl \
  --ann_semantic_type_specs /path/to/semantic_type_specs.json \
  --ann_pattern_index_dir /path/to/pattern_sidecar \
  --ann_sidecar_dir /path/to/work_dir \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_dir experiments/exp01_main_qa/results/nq \
  --bootstrap_samples 1000 \
  --permutation_samples 2000
```

#### 3) Aggregate into final main table

```bash
python experiments/exp01_main_qa/code/aggregate_exp01.py \
  --hotpot_summary experiments/exp01_main_qa/results/hotpot/summary.json \
  --medhop_summary experiments/exp01_main_qa/results/medhop/summary.json \
  --nq_summary experiments/exp01_main_qa/results/nq/summary.json \
  --out_dir experiments/exp01_main_qa/results/main_table
```

Outputs:
- `main_table.md`
- `main_table.csv`
- `main_table_summary.json`

---

### Auto-build artifacts (graph + ANN + triple KV)

For each dataset JSONL, build required artifacts under `artifacts/<dataset>/`:

```bash
# Hotpot (high-quality mode: avoid synthetic triples, use LLM extraction)
python experiments/exp01_main_qa/code/build_assets_from_dataset.py \
  --dataset_jsonl experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl \
  --out_dir experiments/exp01_main_qa/artifacts/hotpot \
  --max_examples 200

python scripts/extract_triples.py \
  --sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.jsonl \
  --out_triples experiments/exp01_main_qa/artifacts/hotpot/triples.jsonl \
  --model /home/zd/dev/KVI/models/Qwen2.5-7B-Instruct \
  --batch_size 3 \
  --device cuda

python scripts/annotate_sentences_semantic_tags.py \
  --in_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.jsonl \
  --out_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.tagged.jsonl \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --semantic_type_specs experiments/exp01_main_qa/artifacts/hotpot/semantic_type_specs.json

python scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.tagged.jsonl \
  --disable_enriched \
  --out_dir experiments/exp01_main_qa/artifacts/hotpot/kvbank_sentences \
  --base_llm /home/zd/dev/KVI/models/Qwen2.5-7B-Instruct \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --layers 0,1,2,3 \
  --block_tokens 128 \
  --shard_size 1024 \
  --device cuda \
  --dtype bfloat16

python scripts/build_knowledge_graph.py \
  --triples_jsonl experiments/exp01_main_qa/artifacts/hotpot/triples.jsonl \
  --out_graph experiments/exp01_main_qa/artifacts/hotpot/graph_index.json

python src/graph/triple_kv_compiler.py \
  --graph_index experiments/exp01_main_qa/artifacts/hotpot/graph_index.json \
  --model /home/zd/dev/KVI/models/Qwen2.5-7B-Instruct \
  --out_dir experiments/exp01_main_qa/artifacts/hotpot/triple_kvbank \
  --device cuda \
  --dtype bfloat16
```

Repeat the same for NQ by replacing `hotpot` paths with `nq`.

> Note: `extract_triples.py` (LLM extraction) is preferred over simple template-based triple synthesis and can significantly improve upstream graph quality for GraphRAG/KVI.  
> On the ANN side, keeping the semantic tags in `sentences.tagged.jsonl` and higher-coverage corpora (rather than minimal samples) is recommended for better recall.

---

### Plan A: Resident inference service (reducing runtime)

`run_exp01.py` defaults to calling inference scripts via subprocess, which is stable but slower. A Plan A switch is provided:

- Start a resident service process (`exp01_resident_infer_service.py`).
- The batch eval script switches to HTTP calls to the service (`--inference_service_url`).
- The service process caches `transformers.from_pretrained`, avoiding reload per query.
- After remote disconnect, the service and batch tasks continue running (`nohup` + logs).

Start service (example):

```bash
nohup bash -lc '
cd /home/zd/dev/KVI
source /home/zd/dev/KVI/KVI/bin/activate
python experiments/exp01_main_qa/code/exp01_resident_infer_service.py --host 127.0.0.1 --port 18888
' > experiments/exp01_main_qa/results/resident_service.log 2>&1 &
```

Eval script switching to resident service (example):

```bash
python experiments/exp01_main_qa/code/run_exp01.py \
  ... \
  --inference_service_url http://127.0.0.1:18888
```

Background execution template (100 samples first):

```bash
nohup bash -lc '
cd /home/zd/dev/KVI
source /home/zd/dev/KVI/KVI/bin/activate
python experiments/exp01_main_qa/code/run_exp01.py ... --limit 100 \
  --inference_service_url http://127.0.0.1:18888
' > experiments/exp01_main_qa/results/hotpot_100.log 2>&1 &
```

---

### Current reproducible table artifact (pilot run)

A fully automated end-to-end pilot (5 methods, Hotpot+NQ, each `--limit 3`) is available at:

- `experiments/exp01_main_qa/results/hotpot/summary.json`
- `experiments/exp01_main_qa/results/nq/summary.json`
- `experiments/exp01_main_qa/results/main_table/main_table.md`

> Note: this pilot is for pipeline verification and table generation.  
> For paper numbers, increase `--limit` and/or use full prepared splits.

---

### Outputs Required

- `results.md` / `results.csv`: main table (5 methods × 3 datasets)
- `summary.json`: aggregated EM metrics
- `predictions_*.jsonl`: per-example outputs for audit

### Case study extraction (KVI wins)

After running Exp01 with all methods, extract cases where KVI succeeds while GraphRAG (and optionally RAG) fails:

```bash
python experiments/exp01_main_qa/code/collect_kvi_win_cases.py \
  --predictions_jsonl experiments/exp01_main_qa/results/smoke100/hotpot/predictions.jsonl \
  --dataset_jsonl experiments/exp01_main_qa/data/multihop_hotpot/hotpot_eval_multihop.jsonl \
  --out_md experiments/exp06_ablation/results/case_studies/kvi_win_over_graphrag_rag.md \
  --out_json experiments/exp06_ablation/results/case_studies/kvi_win_over_graphrag_rag.json \
  --max_cases 30 \
  --require_rag_fail
```

The output markdown is paper-ready for qualitative case analysis.

---

### Notes

- Exp01 is **not** toy-only; toy runs are only for smoke tests.
- Final reported Exp01 numbers should include **HotpotQA + MedHopQA + NQ**.
- Paper main tables should use large subsets or the full official validation set, reporting CI + significance; 100 samples are only for pipeline and parameter verification.
