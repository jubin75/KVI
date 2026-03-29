## Experiments

This directory contains **reproducible experiment code**, **small test datasets**, and **saved results** for the KVI paper.

### Remote env (Linux) — network & model cache

- **HuggingFace 连通性**：直连 `huggingface.co` 不可达（超时/Network unreachable）。**改用镜像可行**：`export HF_ENDPOINT=https://hf-mirror.com` 后，`curl https://hf-mirror.com` 返回 200（约 0.5s），且 `huggingface_hub` 会使用该镜像生成下载 URL（如 `https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/...`）。建议在跑实验的终端或 `~/.bashrc` 中设置并 `source ~/.bashrc`，以便从镜像拉取模型。
- **Encoder 本地缓存**：`sentence-transformers/all-MiniLM-L6-v2` 已存在于 **`/data/huggingface-cache/user/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`**，用户 `zd` 可读。使用方式：`export HF_HOME=/data/huggingface-cache/user/huggingface` 后，transformers 会从该目录加载 encoder，**无需外网**。
- **BASE_LLM（Qwen2.5-7B-Instruct）**：`config/topics/SFTSV/config.json` 中 `build.base_llm` 已设为 **`/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct`**（本地目录）。首次使用请先下载：`export HF_ENDPOINT=https://hf-mirror.com` 后运行 `python scripts/download_qwen25_7b_local.py`（日志：`experiments/logs/download_qwen25_7b.log`）。下载完成后该路径可直接作为 `from_pretrained` 的本地路径使用。

### Exp01 dataset construction (HotpotQA + NQ)

- **Source repos used in mirror**:
  - HotpotQA: `hotpot_qa` (config=`distractor`, split=`validation`)
  - NQ: `natural_questions` (split=`validation`)
- **Why these names**: in current mirror, `hotpotqa/hotpotqa` and `google/natural-questions` are unavailable; above names are reachable and verified.
- **Conversion script**: `experiments/exp01_main_qa/code/prepare_hotpot_nq.py`
- **Command used**:
  - `python experiments/exp01_main_qa/code/prepare_hotpot_nq.py --out_dir experiments/exp01_main_qa/data/benchmarks --hotpot_config distractor --hotpot_split validation --nq_split validation --hotpot_max 500 --nq_max 500 --streaming`
- **Output schema (paper-facing unified format)**:
  - each line: `{"id","question","answer","answers","dataset"}`
  - `answer`: first normalized answer used for compatibility
  - `answers`: all available gold aliases (used by EM via best-match)
- **Current local dataset size**:
  - `experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl`: 500
  - `experiments/exp01_main_qa/data/benchmarks/nq_eval.jsonl`: 500
  - manifest: `experiments/exp01_main_qa/data/benchmarks/dataset_manifest.json`

### Structure

- `exp01_main_qa/`: Experiment 1 — main QA performance (Exact Match).
  - `data/`: datasets (toy + pointers to full datasets)
  - `code/`: runnable scripts
  - `results/`: generated outputs (JSONL/CSV/Markdown tables)
- `exp03_retrieval_quality/`: Experiment 3 — ANN vs Graph retrieval (Recall@k, MRR on Hotpot supporting sentences).
  - `data/benchmarks/`: Hotpot JSONL with `gold_supporting_sentences` (via `prepare_hotpot_nq.py --hotpot_only --include_hotpot_supporting_sentences`)
  - `code/run_exp03_retrieval.py`
  - `results/metrics.json`, `metrics.md`
- `exp06_ablation/`: Experiment 6 — ablation table (template + optional fill from Exp01).
  - `code/run_exp06_ablation.py`
  - `results/ablation_table.md`, `ablation_table.json`
- `RESULTS_COMBINED.md`: merged Exp1 + Exp3 + Exp6 (run `python experiments/combine_experiment_results.py`).

