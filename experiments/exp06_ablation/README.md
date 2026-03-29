# Experiment 6 — Ablation Study

**目的**：验证各模块贡献（Full KVI vs −graph / −DRM / −KV / −dual-channel）。

## 目录

| Path | 说明 |
|------|------|
| `code/run_exp06_ablation.py` | 生成消融表模板 + 可选从 Exp01 `summary.json` 填入一行 |
| `results/` | `ablation_table.md`、`ablation_table.json` |

## 与 `run_graph_inference.py` 的对应关系

| Variant | CLI 思路 |
|--------|----------|
| Full KVI | `--enable_kvi --triple_kvbank_dir ... --openqa_mode` |
| − KV injection | `--max_kv_triples 0` 或 不传 `--enable_kvi`（即 GraphRAG） |
| − DRM（放宽） | 调低 `--drm_threshold`（如 `0.0`）作代理 |
| − dual channel | `--enable_kvi --kvi_minimal_prompt --openqa_mode` |
| − graph retrieval | 使用 `run_kvi2_runtime_test.py --pipeline modeA_rag`（仅 ANN） |

**Accuracy / Hallucination** 需从各自运行的 `predictions.jsonl` + 你的幻觉判定规则填写；本脚本先产出表头与占位符。

## 运行

```bash
python experiments/exp06_ablation/code/run_exp06_ablation.py \
  --exp01_summary experiments/exp01_main_qa/results/smoke100/hotpot/summary.json \
  --out_dir experiments/exp06_ablation/results
```

`--exp01_summary` 可选；若提供且含 `kvi` 方法，会将 **Full KVI** 一行 EM 填入（relaxed EM 与 Exp01 一致）。
