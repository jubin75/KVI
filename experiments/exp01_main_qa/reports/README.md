# Exp01 — 可入库的报告（Markdown）

`experiments/exp01_main_qa/results/` 在本机常设为指向数据盘的**符号链接**，Git 无法跟踪其内部路径。论文/审稿用的 **Markdown 表与说明** 放在本目录，与跑分产物分离。

- `main_table/main_table.md` — 双 Panel 主表（Qwen + Mistral 摘要表见 `main_table_mistral7b_v0_3/`）
- `supplementary_medhop_official.md` — MedHop official 评测集说明

更新流程：在本地跑完实验后，将 `results/` 下对应 `.md` 复制到本目录再提交（可用 `cp -aL results/... reports/...`）。
