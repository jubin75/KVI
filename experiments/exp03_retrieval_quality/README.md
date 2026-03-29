# Experiment 3 — Retrieval Quality (HotpotQA)

**目的**：对比 **ANN** 与 **Graph+hybrid** 对官方 supporting sentences 的覆盖（Recall@k / MRR）。

## 目录

| Path | 说明 |
|------|------|
| `data/benchmarks/` | Hotpot JSONL（需含 `gold_supporting_sentences`） |
| `code/run_exp03_retrieval.py` | 主脚本 |
| `results/` | `metrics.json`、`metrics.md` |

## 准备数据（与 Exp01 共用 `prepare_hotpot_nq.py`）

```bash
python experiments/exp01_main_qa/code/prepare_hotpot_nq.py \
  --out_dir experiments/exp03_retrieval_quality/data/benchmarks \
  --hotpot_config distractor --hotpot_split validation \
  --hotpot_max 500 --streaming \
  --include_hotpot_supporting_sentences \
  --hotpot_only
```

## 运行（需与 Exp01 相同 Hotpot 工件路径）

```bash
python experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py \
  --dataset_jsonl experiments/exp03_retrieval_quality/data/benchmarks/hotpot_eval.jsonl \
  --graph_index experiments/exp01_main_qa/artifacts/hotpot/graph_index.json \
  --sentences_jsonl experiments/exp01_main_qa/artifacts/hotpot/sentences.jsonl \
  --out_dir experiments/exp03_retrieval_quality/results \
  --limit 0
```

`--limit N`：只评测前 N 条（调试用）。
