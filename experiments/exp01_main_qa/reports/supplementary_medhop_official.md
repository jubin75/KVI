# Supplement: MedHopQA 官方风格评测集（自 medhop_raw）

> **仓库路径**：`experiments/exp01_main_qa/reports/supplementary_medhop_official.md`。`data/medhop_*`、`artifacts/` 等为本地构建/数据盘产物，默认 **不入 Git**；下表路径表示流水线约定位置。

本附表说明从 `medhop_raw` 构建的 **自然语言问句 + short / long answer + supporting_facts** 评测格式，以及与主表 **MedHop-ID** 变体的关系。该 split 可与 Exp01 流水线直接对接（`medhop_eval.jsonl` + `sentences_medhop.jsonl` + `triples_medhop.jsonl`）。

## 数据位置

| 产物 | 路径 |
|------|------|
| 源数据（本仓库） | `experiments/exp01_main_qa/data/medhop_raw/medhop_source_validation.parquet.jsonl` |
| 附表级 schema（论文/附录） | `experiments/exp01_main_qa/data/medhop_official/medhop_official_eval.jsonl` |
| Exp01 可跑 `dataset` | `experiments/exp01_main_qa/data/medhop_official/medhop_eval.jsonl` |
| 句子、三元组 | `experiments/exp01_main_qa/data/medhop_official/sentences_medhop.jsonl`，`triples_medhop.jsonl` |
| 构建清单 | `experiments/exp01_main_qa/data/medhop_official/manifest_medhop_official.json` |

## 表 S — `medhop_official_eval.jsonl` 字段

| 字段 | 说明 |
|------|------|
| `id` | 与 raw 一致，如 `MH_dev_0` |
| `question` | 自然语言模板：`Which DrugBank-listed drug interacts with DBxxxx? …`；默认另附一行 **仅输出伙伴 DrugBank ID** 的约束（与主表 MedHop-ID 的 EM 口径一致） |
| `raw_query` | 原始 `interacts_with DBxxxx?` |
| `answer` / `answers` | 标准答案（伙伴 DrugBank ID） |
| `short_answer` | 与 gold ID 相同（本发行版 **不**映射通用药品名；若需 `Ritonavir` 类等自然语言金标，需外部 ID→名称表） |
| `long_answer` | 使用前 **2** 条 `supports` 全文截断拼接（≤2500 字符），供附录或人工核对 |
| `type` | 固定标记 `drug_drug_interaction_completion` |
| `supporting_facts` | `supports` 拆成的 `{title, sentence}`：若以 `DBxxxxx :` 起头则 `title` 为该 ID，否则 `title` 为 `passage_k` |
| `candidates` | 保留 raw 中的多选 ID 列表 |
| `dataset` | `MedHopQA_official_nl` |
| `gold_note` | 简短说明金标仍为 ID 及 EM 含义 |

`medhop_eval.jsonl` 为 Exp01 入口：每行至少含 `id`、`question`、`answer`/`answers`；`question` 与 `medhop_official_eval.jsonl` 中一致。

## 规模（当前构建）

以 `manifest_medhop_official.json` 为准，典型一次构建为：

- 例数：**342**
- 句子块：**105098**（由 `supports` 字符串按句切分）
- 三元组：**488**（每例 1–2 条；部分证据句未同时命中 query/answer ID 时仅保留 1 条）

## 重建命令

```bash
cd experiments/exp01_main_qa/code
python3 prepare_medhop_official_from_raw.py \
  --medhop_raw ../data/medhop_raw/medhop_source_validation.parquet.jsonl \
  --out_dir ../data/medhop_official
```

可选：`--no_append_id_only_hint` 不对模型追加「仅输出 DB ID」提示（问句更开放，EM 口径可能与主表 strict-ID 不完全可比）。

## 与主表 MedHop-ID 的差异

| 项目 | MedHop-ID（主表） | Official NL（本附表） |
|------|-------------------|------------------------|
| 问句形式 | `interacts_with DBxxxx?` + ID 输出说明 | 自然语言 “Which DrugBank-listed drug…” + 默认同款 ID 输出说明 |
| 金标 | DrugBank 伙伴 ID | 相同 |
| EM | 子串 / 管道既有 EM | 使用同一 `medhop_eval.jsonl` 时与 MedHop-ID **一致** |

主表仍保留较小子集 **MedHopQA_n40** 的已跑结果；全量 **342** 条可用于扩展实验或附录报告。

## Exp01 运行提示

将 `--dataset` 指向 `data/medhop_official/medhop_eval.jsonl`，图与 KVI 相关路径指向 **同目录下** 新生成的 `sentences_medhop.jsonl` / `triples_medhop.jsonl`，并按既有 MedHop 流程构建 `graph_index`、tagged sentences、kvbank 等（与 `prepare_medhopqa_assets.py` 产物用法相同）。全量 342 条与 ~105k 句对 resident/嵌入耗时显著高于 n40，建议按需设 `limit` 或分批。

## Exp01 冒烟（`--limit 2`，已跑通）

在仓库根目录执行（示例复用既有 **MedHopQA_n40** 产物：`MH_dev_0` / `MH_dev_1` 与全量 official 前两条 id 一致，图与 KV 与这两条对齐；**正式全量 342 条**应对 `medhop_official` 单独建 artifacts）：

```bash
source KVI/bin/activate   # 若使用项目 venv
python -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp01_main_qa/data/medhop_official/medhop_eval.jsonl \
  --dataset_name MedHopQA_official_smoke \
  --model /path/to/Qwen2.5-7B-Instruct \
  --graph_index experiments/exp01_main_qa/artifacts/medhop_n40/graph_index.json \
  --triple_kvbank_dir experiments/exp01_main_qa/artifacts/medhop_n40/triple_kvbank \
  --graph_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_kv_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp01_main_qa/artifacts/medhop_n40/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar \
  --methods llm,rag,graphrag,kv_prefix,kvi \
  --limit 2 \
  --out_dir experiments/exp01_main_qa/results/medhop_official_smoke_limit2 \
  --ann_force_cpu
```

**本机一次成功冒烟产物**：`experiments/exp01_main_qa/results/medhop_official_smoke_limit2/`（含 `predictions.jsonl`、`summary.json`、`results.md`）。基座模型为本地 `Qwen2.5-7B-Instruct`；`relaxed` EM 下 GraphRAG/KVI 在 2 例上为 100%（子串命中金标 ID），其余方法在该小样本上为 0，仅作管道连通性验证。

### 全量正式实验（后台，断 SSH 可继续）

- **脚本**：`experiments/exp01_main_qa/code/run_medhop_official_full_background.sh`  
  - 在 `artifacts/medhop_official/` 下依次：语义标注 → `kvbank_sentences` → `graph_index.json` → `triple_kvbank`（与 n40 同构）。  
  - 若本机 `127.0.0.1:18888` 无 `/health`，会 **nohup 拉起** `exp01_resident_infer_service.py`（图侧单例加载 7B，日志见下）。  
  - 然后 **`run_exp01.py` 全 342 条、`--resume`**；默认输出目录 **`results/medhop_official_fullmethods_qwen25_7b_kvituned/`**，且默认 **KVI 调参**：`kvi_drm_threshold=0.12`、`max_kv_triples=2`、`top_k_relations=1`、`--kvi_minimal_prompt`。**调参前基线**（主表 Panel A 行）见 `results/medhop_official_fullmethods_qwen25_7b/summary.json`。
- **推荐启动方式**（与脚本内 `nohup` 二选一即可；下面为再包一层 tee）：

```bash
cd /home/zd/dev/KVI
nohup bash experiments/exp01_main_qa/code/run_medhop_official_full_background.sh \
  >> experiments/exp01_main_qa/results/medhop_official_full_pipeline.log 2>&1 &
```

- **主日志**：`experiments/exp01_main_qa/results/medhop_official_full_pipeline.log`  
- **常驻服务日志**（若由脚本启动）：`experiments/exp01_main_qa/results/medhop_official_resident.log`  
- **环境变量**：`MODEL`、`SKIP_ARTIFACTS=1`（仅跑评测）、`START_RESIDENT=0`（不启 resident、极慢）、`RESIDENT_PORT`、`MEDHOP_OFFICIAL_OUT`、`KVI_DRM_THRESHOLD`、`KVI_MAX_KV_TRIPLES`、`KVI_TOP_K_RELATIONS`、`KVI_MINIMAL_PROMPT`（0/1）。
