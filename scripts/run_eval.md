# Script Description: Run Evaluation (AB/Regression)

Purpose: Execute evaluation protocol (see `docs/40_evaluation_protocol.md`), output quality/latency/stability/interpretability reports.

## Input
- `--eval_set`: evaluation set path
- `--config`: config file (e.g. `configs/example.yaml`)

## Key Parameters (Suggested)
- `--ab_groups`: baseline (inject.off) vs injection (inject.on)
- `--k_values`: 8,16,32
- `--export_debug`: whether to output citation and γ statistics

## Output
- Report: Markdown/JSON
- Sample analysis: hit chunk and citation correctness examples


