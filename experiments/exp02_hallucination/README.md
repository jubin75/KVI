# Experiment 02 — Hallucination (proxy metrics)

## 全量规模与后台任务

- **默认全量**（`run_exp02_hallucination.py` 未传 `--limit` 时）  
  - **TruthfulQA**：`--truthfulqa_max` **500**（`generation` validation，见 `prepare_exp02_datasets.py`）  
  - **FEVER**：`--fever_max` **1000**（KILT/FEVER 校验 split 的本地 parquet 或 HF 备选）  
- **烟雾测试**：传 `--limit N` 时，`build_assets_from_dataset.py` 只对前 N 条建图/KV；`run_exp01.py` 也会 `--limit N`。看 `results/*/summary.json` 里的 **`n`** 即可区分全量还是小样本。  
- **自动重启**：`code/run_exp02_autoresume.sh` 在 **`results/hallucination_proxy_summary.json` 不存在** 时循环执行主脚本；一旦该文件生成即认为本轮 Exp02 结束并退出。若要**强制重跑全量**，需先备份或删除旧的 `hallucination_proxy_summary.json`（以及按需清结果目录），再启动 supervisor。  
- **日志**：主进程 stdout/stderr → `results/exp02_pipeline_v2.log`；督导一行行记录 → `results/exp02_supervisor.log`（`*.log` 已 `.gitignore`，仅存本地）。

判断「后台是否还在跑」：本机存在 `run_exp02_autoresume.sh` / `run_exp02_hallucination.py` 进程，且 TruthfulQA 或 FEVER 管线（如 `triple_kv_compiler.py`）在跑，即全量流水线尚未收尾。

---

## 当前图中的「幻觉率」是什么（与社区口径的差异）

流水线末尾写的是 **Hallucination Rate (proxy) = 100 − relaxed EM**（见 `run_exp02_hallucination.py` 写 `hallucination_proxy_summary.json` 的注释）。

- **relaxed EM**（`experiments/exp01_main_qa/code/metrics.py`）：SQuAD 风格归一化后，**任一条 gold 是否作为子串出现在模型整段输出中**（并对 `yes`/`no` 有少量扩展）。适合长段生成的 Hotpot/NQ 风格 QA，**不是** TruthfulQA 或 FEVER 官方主表口径。  
- **应对齐的社区/官方口径（建议写论文或对外对比时使用）**  
  - **TruthfulQA**：官方/常用为 **generation 上的人类或自动化评测**（如 MC1/MC2、或官方脚本），而不是「参考句是否出现在长生成里」。  
  - **FEVER**：共享任务常用 **三分类标签准确率**（SUPPORTS / REFUTES / NOT ENOUGH INFO）；完整流水线还可接 **fever-scorer**（需预测 Wikidata 证据等），与本仓库的「仅标签」设定不同。

### 推荐实现顺序（已定）

1. **FEVER（优先，已落地）**  
   - `run_exp01.py` 在 `--dataset_name FEVER` 时额外计算 **标签准确率**：在模型全文里找 **首次出现** 的 `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`（`metrics.parse_fever_label`），与 `prepare_exp02_datasets.py` 写入的 gold 比较。  
   - 见 `results/fever_fullmethods_qwen25_7b/summary.json` 中每方法的 **`fever_label_accuracy`**、**`fever_label_ci95_*`**；逐题见 `predictions.jsonl` 的 **`fever_label_em`**。  
   - 比 **relaxed EM** 更接近共享任务的 **veracity 标签**口径；**仍不是**带证据提交的官方 scorer。  
2. **TruthfulQA（第二步，待做）**  
   - 接入 **MC 题目格式**（MC1/MC2）或 TruthfulQA **官方评测脚本**；依赖与数据格式改动更多，排在 FEVER 之后。

Exp02 的 **`hallucination_proxy_summary.json` / 柱状图** 仍为 **`100 − relaxed EM`**；读 FEVER 的「社区友好」指标时请直接看 **`fever_label_accuracy`**（或可后续加第二张图）。`plot_hallucination_proxy_bars.py` 对 TruthfulQA 一栏也仍应标注 **proxy**。

---

## `run_exp02_hallucination.py` → `run_exp01.py` 实际参数（逐条对照）

以下为主脚本对 **每个数据集** 构建完 `artifacts/<name>/` 后，调用 `run_exp01.py` 时**明确传入**或与**默认值**一致的项（摘自 `run_exp02_hallucination.py`）。

| 参数 | 值 / 说明 |
|------|-----------|
| `--dataset` | `data/truthfulqa_eval.jsonl` 或 `data/fever_eval.jsonl` |
| `--dataset_name` | `TRUTHFULQA` 或 `FEVER` |
| `--model` | 默认 `.../models/Qwen2.5-7B-Instruct` |
| `--graph_index` | `artifacts/<name>/graph_index.json` |
| `--triple_kvbank_dir` | `artifacts/<name>/triple_kvbank` |
| `--graph_sentences_jsonl` | `artifacts/<name>/sentences.tagged.jsonl` |
| `--ann_kv_dir` 等 ANN 路径 | 对应该数据集的 `kvbank_sentences` 与 pattern sidecar |
| `--methods` | `llm,rag,graphrag,kv_prefix,kvi` |
| `--out_dir` | `results/<name>_fullmethods_qwen25_7b` |
| `--timeout_s` | `600` |
| `--bootstrap_samples` / `--permutation_samples` | `1000` / `2000` |
| `--inference_service_url` / `--ann_inference_service_url` | 若命令行传了 `--resident_url`（autoresume 为 `http://127.0.0.1:18888`）则一并传入 |
| `--limit` | **仅当** `run_exp02_hallucination.py --limit K` 时追加，限制评测条数 |
| `--em_mode` | **未传**，沿用 `run_exp01.py` 默认 **`relaxed`** |
| `--openqa_mode` | **未传**，沿用 `run_exp01.py` 默认 **`True`**（`BooleanOptionalAction`，即 Graph/KVI 走英文开放域提示，见 `run_exp01.py` 帮助文案） |

未列出的 `run_exp01` 参数均取其文件内默认值（如 KVI 的 `kvi_max_kv_triples=3`、`kvi_reconcile_no_kv_decode=False` 等）。

---

## FEVER：gold 从哪来？为什么和 `openqa_mode` 同时出现？

### Gold（`answer` / `answers`）如何构造

逻辑在 **`prepare_exp02_datasets.py`**（与 `run_exp02_hallucination.py` 调用参数一致：`--fever_max`、`--mirror_root`、`--mirror_data_root`、`--streaming`）。

1. 从本地 parquet（如 `kilt_fever_validation.parquet`）或备选 HF 配置加载行。  
2. **标签映射**（整型标号时）：  

   `label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}`  

   若列为字符串则 `strip().upper()`；若 KILT 风格 `output` 列表里带 `answer` 字段则取其大写字符串。  
3. **写入 JSONL 的字段**：  
   - `question`：人为拼装的英文说明 + claim，要求模型 **只输出三标签之一**（见脚本内 `q = f'Claim: "{claim}"\nBased on evidence, ...'`）。  
   - `answer` / `answers`：**单一标准串**，即上述 **`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`**（仅此一字串放入 `answers` 列表）。

因此，**监督信号仍是 FEVER 的三类标签**；评测时 `run_exp01` 用 relaxed EM 检查模型**长输出**里是否出现归一化后的 gold 子串（例如 `supports`），与「开放域长答案」并存。

### 为何默认 `openqa_mode=True`（和医学中文无关）

- `openqa_mode` 只影响 **Graph/KVI 路径**里 `run_graph_inference` 的提示模板：默认 **英文开放域**，避免 Exp01 为 MedHop 准备的 **中文医学 system prompt** 污染 TruthfulQA/FEVER。  
- FEVER 的 **claim+指令** 已通过数据集里的 **`question` 字段** 注入；gold 仍是 **三分类标签**，不是开放域自由文本事实。

如需**严格复现 MedHop 式中文闭域图推理**，需对 Exp02 **显式传入** `run_exp01.py` 的 `--no-openqa_mode`（当前 `run_exp02_hallucination.py` **没有**传这一项，故全为默认开放域英文图侧提示）。

---

## 结果与图（论文用图看这里）

**绝对路径（本机）**：`/home/zd/dev/KVI/experiments/exp02_hallucination/results/`

### 表格与原始指标

| 内容 | 路径 |
|------|------|
| 跨数据集汇总（代理指标） | `results/hallucination_proxy_summary.json`、`hallucination_proxy_summary.md`（跑完 `run_exp02_hallucination.py` 后生成；**无则**可用下面两个 `summary.json` 手搓图） |
| TruthfulQA 逐方法 EM / F1 / CI | `results/truthfulqa_fullmethods_qwen25_7b/summary.json`、`results.md`、`results.csv` |
| FEVER 同上 + **标签准确率** `fever_label_accuracy` | `results/fever_fullmethods_qwen25_7b/summary.json`、`results.md`、`results.csv` |
| 逐题预测 | 各数据集目录下 `predictions.jsonl` |

### 发表论文用的矢量图（SVG，可插 LaTeX / Word）

脚本：`experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py`

```bash
# 推荐：全量跑完后有 hallucination_proxy_summary.json 时
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --paper \
  --fever_label_figure

# 或仅从两个 summary.json 生成（无 proxy 汇总文件时）
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --truthfulqa_summary experiments/exp02_hallucination/results/truthfulqa_fullmethods_qwen25_7b/summary.json \
  --fever_summary experiments/exp02_hallucination/results/fever_fullmethods_qwen25_7b/summary.json \
  --paper \
  --fever_label_figure
```

| 输出文件 | 用途 |
|----------|------|
| **`hallucination_proxy_bars_paper.svg`** | **论文优先**：白底、Helvetica、图注说明「100 − relaxed EM（proxy）」 |
| `hallucination_proxy_bars.svg` | 屏幕预览 / 非印刷 |
| **`fever_label_accuracy_bars_paper.svg`** | FEVER **三分类标签准确率**（需 `summary.json` 含 `fever_label_accuracy`；老结果需重跑 `run_exp01.py`） |
| `fever_label_accuracy_bars.svg` | 同左，非 `paper` 样式 |

**PDF**：编辑部常要 PDF/EPS。可用 Inkscape（`inkscape file.svg --export-filename=file.pdf`）或 `rsvg-convert -f pdf -o file.pdf file.svg` 从 **SVG** 转换，保持矢量。

**图中要写清的表述**：TruthfulQA 子图对应的是 **relaxed EM 的代理**，不是官方 MC；FEVER 子图若用 proxy 同上；若另附 **`fever_label_accuracy`** 图，更接近 **共享任务 veracity 标签**口径（仍非带证据的官方 scorer）。

另：`*.html` 仅为浏览器预览，投稿一般用 **SVG/PDF**。

---

## 代码入口

| 步骤 | 脚本 |
|------|------|
| 镜像下载 | `experiments/code/download_mirror_datasets.py` |
| 数据 JSONL | `code/prepare_exp02_datasets.py` |
| 主 orchestrator | `code/run_exp02_hallucination.py` |
| 评测 | `experiments/exp01_main_qa/code/run_exp01.py` |
| EM 定义 | `experiments/exp01_main_qa/code/metrics.py` |
