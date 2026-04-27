# Experiment 3 & 6 (migrated to independent directories)

- **Exp3**: `experiments/exp03_retrieval_quality/README.md`
- **Exp6**: `experiments/exp06_ablation/README.md`
- **Merge**: `python experiments/combine_experiment_results.py` → `experiments/RESULTS_COMBINED.md`

The content below is kept for quick reference.

## Exp3 — Retrieval Quality (HotpotQA)

```bash
python experiments/exp01_main_qa/code/prepare_hotpot_nq.py \
  --out_dir experiments/exp03_retrieval_quality/data/benchmarks \
  --hotpot_config distractor --hotpot_split validation \
  --hotpot_max 500 --streaming \
  --include_hotpot_supporting_sentences \
  --hotpot_only
```

```bash
python experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py \
  --dataset_jsonl experiments/exp03_retrieval_quality/data/benchmarks/hotpot_eval.jsonl \
  --graph_index experiments/exp01_main_qa/artifacts/hotpot/graph_index.json \
  --sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.jsonl \
  --out_dir experiments/exp03_retrieval_quality/results
```

## Exp6 — Ablation

```bash
python experiments/exp06_ablation/code/run_exp06_ablation.py \
  --exp01_summary experiments/exp01_main_qa/results/smoke100/hotpot/summary.json \
  --out_dir experiments/exp06_ablation/results
```
