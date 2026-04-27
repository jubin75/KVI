# Experiment 3 — Retrieval Quality (HotpotQA)

**Purpose**: Compare **ANN** vs **Graph+hybrid** on official supporting sentences coverage (Recall@k / MRR).

## Directory

| Path | Description |
|------|------|
| `data/benchmarks/` | Hotpot JSONL (must contain `gold_supporting_sentences`) |
| `code/run_exp03_retrieval.py` | Main script |
| `results/` | `metrics.json`, `metrics.md` |

## Prepare Data (shared with Exp01 via `prepare_hotpot_nq.py`)

```bash
python experiments/exp01_main_qa/code/prepare_hotpot_nq.py \
  --out_dir experiments/exp03_retrieval_quality/data/benchmarks \
  --hotpot_config distractor --hotpot_split validation \
  --hotpot_max 500 --streaming \
  --include_hotpot_supporting_sentences \
  --hotpot_only
```

## Run (must use the same Hotpot artifact paths as Exp01)

```bash
python experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py \
  --dataset_jsonl experiments/exp03_retrieval_quality/data/benchmarks/hotpot_eval.jsonl \
  --graph_index experiments/exp01_main_qa/artifacts/hotpot/graph_index.json \
  --sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.jsonl \
  --out_dir experiments/exp03_retrieval_quality/results \
  --limit 0
```

`--limit N`: Only evaluate the first N entries (for debugging).
