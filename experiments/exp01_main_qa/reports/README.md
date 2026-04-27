# Exp01 — Repo-check-in-able reports (Markdown)

`experiments/exp01_main_qa/results/` is often a **symlink** pointing to a data disk on this machine, so Git cannot track files inside it. Paper/reviewer-facing **Markdown tables and notes** are placed in this directory, separate from scoring artifacts.

- `main_table/main_table.md` — Dual-panel main table (Qwen + Mistral summary tables under `main_table_mistral7b_v0_3/`)
- `supplementary_medhop_official.md` — MedHop official eval set documentation

Update workflow: After running experiments locally, copy the corresponding `.md` files from `results/` to this directory before committing (use `cp -aL results/... reports/...`).
