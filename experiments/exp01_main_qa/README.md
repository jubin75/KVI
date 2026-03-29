## Experiment 1
### Main QA Performance

**目的**：验证 KVI 是否提升问答准确率（Exact Match, EM）。  
**数据集**：`HotpotQA`、`MedHopQA` 与 `NQ (Natural Questions)`。  
**展示形式**：统一结果表（按方法对比 `HotpotQA / MedHopQA / NQ` 的 EM 与 CI95）。

**Git 入库说明**：`results/`、`artifacts/`、`data/` 下的跑分与构建产物默认 **不提交**（见仓库根目录 `.gitignore`）。论文用 **Markdown 主表与附表** 放在 **`reports/`**（见 `reports/README.md`）；本地若将 `results` 指到数据盘符号链接，更新表后请 `cp -aL` 同步到 `reports/` 再提交。

---

### Evaluation Table (target format)

| Method | Retrieval | Injection | HotpotQA EM | HotpotQA CI95 | MedHopQA EM | MedHopQA CI95 | NQ EM | NQ CI95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LLM | none | none | 32.4 | [30.1, 34.8] | 28.1 | [26.0, 30.0] |
| RAG | ANN | prompt | 55.2 | [52.4, 58.1] | 47.8 | [45.3, 50.1] |
| GraphRAG | graph | prompt | 58.7 | [56.0, 61.3] | 50.3 | [48.0, 52.9] |
| KV Prefix | ANN | KV | 57.9 | [55.4, 60.6] | 49.1 | [46.9, 51.6] |
| KVI | graph | KV + prompt | 66.4 | [63.7, 69.0] | 56.2 | [53.8, 58.5] |

> 说明：上表数值是论文/文档中的示例格式。你在本仓库复现实验时，需替换为实际跑出来的 EM。

---

### Method Definitions (for Exp01)

- **LLM**: no retrieval, no injection
- **RAG**: ANN retrieval + prompt evidence
- **GraphRAG**: graph retrieval + prompt evidence
- **KV Prefix**: ANN retrieval + KV-only injection
- **KVI**: graph retrieval + KV injection + prompt evidence (dual-channel)

### KVI vs RAG / analysis (重要)

- **原因说明**：见 `KVI_FAILURE_ANALYSIS.md`（域错配、synthetic 图谱、检索通道不一致等）。
- **默认** `run_exp01.py` 对 Graph/KVI 开启 **`--openqa_mode`**（英文开放域提示，避免沿用医学中文 system prompt）。
- **消融**：**`--kvi_minimal_prompt`** —— KVI 注入 KV 时从 prompt 中去掉长证据列表，用于检验「仅 KV + 问题」是否减轻注意力偏转。
- **Exp3 / Exp6**：独立目录 `experiments/exp03_retrieval_quality/`、`experiments/exp06_ablation/`；汇总表见仓库根目录 `experiments/RESULTS_COMBINED.md`（`python experiments/combine_experiment_results.py`）。

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

> `--hotpot_max 0 --nq_max 0` 表示尽量拉取官方 validation 全量（或镜像可提供的完整子集）；  
> 若先做 smoke test，可临时改成 `--hotpot_max 100 --nq_max 100`。

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

按你的要求，实验脚本采用 **单数据集单次评估**：

- `run_exp01.py --dataset <one_dataset>`：在该数据集上一次性评估 5 个方法（LLM / RAG / GraphRAG / KV Prefix / KVI）
- 对 HotpotQA、MedHopQA、NQ 各跑一次
- 再用 `aggregate_exp01.py` 自动合并成主表（HotpotQA + MedHopQA + NQ）
- 每个方法自动输出 `EM`、`95% bootstrap CI`，并在 `summary.json` 中输出 `KVI` 相对其他方法的配对置换检验 `p-value`

---

### Official-style staged execution (recommended)

在远程不稳定场景下，建议两阶段：

1. **阶段A（校验）**：Hotpot/NQ 各 `--limit 100`，确认链路与统计输出正常  
2. **阶段B（正式）**：去掉 `--limit`，后台跑完整 validation（断线不影响）

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

输出：
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

> 说明：`extract_triples.py`（LLM抽取）优先于简单模板合成三元组，可显著提高 GraphRAG/KVI 上游图谱质量。  
> ANN 侧建议保留 `sentences.tagged.jsonl` 的语义标签与更高覆盖语料（而非极小样本）来提升召回。

---

###方案A：常驻推理服务（降低耗时）

`run_exp01.py` 默认通过子进程调用推理脚本，稳定但较慢。已提供方案A开关：

- 启一个常驻服务进程（`exp01_resident_infer_service.py`）
- 批量评测脚本改为 HTTP 调用服务（`--inference_service_url`）
- 服务进程内对 `transformers.from_pretrained` 做缓存，避免每条 query 重载
- 远程断开后，服务与批任务仍继续运行（`nohup` + 日志）

启动服务（示例）：

```bash
nohup bash -lc '
cd /home/zd/dev/KVI
source /home/zd/dev/KVI/KVI/bin/activate
python experiments/exp01_main_qa/code/exp01_resident_infer_service.py --host 127.0.0.1 --port 18888
' > experiments/exp01_main_qa/results/resident_service.log 2>&1 &
```

评测脚本切到常驻服务（示例）：

```bash
python experiments/exp01_main_qa/code/run_exp01.py \
  ... \
  --inference_service_url http://127.0.0.1:18888
```

后台执行模板（先100条）：

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
- 论文主表建议使用官方 validation 大子集/全量，并报告 CI + 显著性；100 条仅用于链路与参数确认。

