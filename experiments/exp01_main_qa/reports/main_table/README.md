# Exp01 aggregate main table

This directory holds **`main_table.md`** / **`main_table.csv`**: HotpotQA EM + NQ EM (5 methods), merged from two `run_exp01.py` runs + `aggregate_exp01.py`.

## Current source (smoke100, N=100 per dataset)

- Hotpot: `experiments/exp01_main_qa/results/smoke100/hotpot/summary.json`
- NQ: `experiments/exp01_main_qa/results/smoke100/nq/summary.json`

Regenerate:

```bash
python experiments/exp01_main_qa/code/aggregate_exp01.py \
  --hotpot_summary experiments/exp01_main_qa/results/smoke100/hotpot/summary.json \
  --nq_summary experiments/exp01_main_qa/results/smoke100/nq/summary.json \
  --out_dir experiments/exp01_main_qa/results/main_table
```

For full validation (no `--limit`), use `run_exp01_official_pipeline.sh` or run Hotpot/NQ each without `--limit`, then point aggregate at those `summary.json` files.

## Combined report

`python experiments/combine_experiment_results.py` → `experiments/RESULTS_COMBINED.md` (includes this main table + optional latest verify + Exp3 + Exp6).
