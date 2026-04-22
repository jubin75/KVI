# Experiment 02 — Hallucination (proxy metrics)

**当前优先的实验约定（按日期滚动更新，含 audit 冒烟参数与命令骨架）**：见 [`LATEST_EXPERIMENT_REQUIREMENTS.md`](./LATEST_EXPERIMENT_REQUIREMENTS.md)。

## 实验方式与 KVI 分析目标（速查）

后续若在 Exp02 上做 **KVI 提升**，默认以此为准：**同时对照 TruthfulQA 与 FEVER**，从两套结果里归纳能改进 KVI 的方向（提示、检索与 KV、超参、结构等），而不是只优化其中一集并把另一集带崩。

### 管线与规模

| 项目 | 约定 |
|------|------|
| 主流程 | `run_exp02_hallucination.py` → 各数据 `artifacts/<name>/` → `run_exp01.py` |
| 对比方法 | `llm,rag,graphrag,kv_prefix,kvi`（五方法） |
| 默认规模 | TruthfulQA **500**、FEVER **1000**；实际 **`n`** 以 `results/*/summary.json` 与 `data/dataset_manifest.json` 为准 |
| Graph 侧推理 | 常驻 **`http://127.0.0.1:18888`**（`--resident_url`） |
| ANN（RAG / KV Prefix） | 默认 **本机 CPU**：`--ann_inference_service_url` 为空且 **`--ann_force_cpu`**；若显式 **`--ann_via_resident`** 则与 graph 共用同一 resident |
| 已有数据与工件、不重复编译 | **`--skip_mirror_and_prepare --reuse_artifacts`**（见 `run_exp02_fast_once.sh`） |
| 评测断点续跑 | Exp02 传 **`--resume_eval`** → 透传 `run_exp01.py` 的 **`--resume`**（校验 `predictions.jsonl` 前缀后追加） |
| SSH 下 FEVER-only + 起 resident | `code/run_fever_gpu_detached.sh`（已含 **grace** 与 **`--resume_eval`**） |
| 本机 resident 已就绪、仅续跑 FEVER | `code/run_fever_resume_eval.sh` |

### 读结果时的指标

- **TruthfulQA**：表中 **relaxed EM** 是 **proxy**，不是官方 MC；对外对比宜另接 MC1/MC2 或官方脚本（见下文「社区口径」）。  
- **FEVER**：除 proxy EM 外，务必看 **`fever_label_accuracy`**（三分类标签，更接近 veracity 任务）。  
- **Graph/KVI 提示**：`scripts/run_graph_inference.py` 对 FEVER 式题干（如以 `Claim:` 起头且含三标签说明）会用 **单行 SUPPORTS / REFUTES / NOT ENOUGH INFO** 的收紧说明；TruthfulQA 仍为开放式英文说明。**同一 `predictions.jsonl` 内若混有「续跑前后」或「删文件重跑前后」的行，模板可能不一致**——论文级统一口径可删 `predictions.jsonl` 后全量重跑（**不要** `--resume_eval`）。

### 分析主线

1. 在两套数据集上对比 **KVI 与其余方法**（尤其 **GraphRAG**）。  
2. 归类错误：**格式与标签**、**检索/KV 是否偏离**、**证据与生成不一致**、**过长或跑题生成** 等。  
3. 提出改动时区分：更偏 **任务模板**（对 FEVER 友好可能对 TQA 无用）与更偏 **机制**（如 KV 选择与注入），并计划在 **两集上交叉验证**。

---

## 全量规模与后台任务

- **默认全量**（`run_exp02_hallucination.py` 未传 `--limit` 时）  
  - **TruthfulQA**：`--truthfulqa_max` **500**（`generation` validation，见 `prepare_exp02_datasets.py`）  
  - **FEVER**：`--fever_max` **1000**（KILT/FEVER 校验 split 的本地 parquet 或 HF 备选）  
- **烟雾测试**：传 `--limit N` 时，`build_assets_from_dataset.py` 只对前 N 条建图/KV；`run_exp01.py` 也会 `--limit N`。看 `results/*/summary.json` 里的 **`n`** 即可区分全量还是小样本。  
- **自动重启**：`code/run_exp02_autoresume.sh` 在 **`results/hallucination_proxy_summary.json` 不存在** 时循环执行主脚本；一旦该文件生成即认为本轮 Exp02 结束并退出。若要**强制重跑全量**，需先备份或删除旧的 `hallucination_proxy_summary.json`（以及按需清结果目录），再启动 supervisor。  
- **日志**：主进程 stdout/stderr → `results/exp02_pipeline_v2.log`；督导一行行记录 → `results/exp02_supervisor.log`（`*.log` 已 `.gitignore`，仅存本地）。

判断「后台是否还在跑」：本机存在 `run_exp02_autoresume.sh` / `run_exp02_hallucination.py` 进程，且 TruthfulQA 或 FEVER 管线（如 `triple_kv_compiler.py`）在跑，即全量流水线尚未收尾。

### 快速一次性跑完（推荐）

若 **`data/dataset_manifest.json` 已是 500+1000** 且 **`artifacts/*` 已编译过图与 triple KV**，不要用 autoresume 从头反复编译。先**停掉** `run_exp02_autoresume.sh` 与旧的 `run_exp02_hallucination.py`，再起常驻 **18888**，然后：

```bash
# 日志：results/exp02_fast_run.log
nohup experiments/exp02_hallucination/code/run_exp02_fast_once.sh >> experiments/exp02_hallucination/results/exp02_fast_run.log 2>&1 &
```

等价参数：`--skip_mirror_and_prepare --reuse_artifacts`（见 `run_exp02_hallucination.py`）。仍会跑 **两遍** `run_exp01`（500 + 1000 条 × 五方法），耗主要取决于推理，不再重复 CPU 编译 KV。

可选：只跑某一数据集，例如 `--only_datasets fever`（另一数据集会从已有 `summary.json` 并入 `hallucination_proxy_summary.json`）。

### SSH 断线仍跑（推荐：FEVER 补跑 + GPU0）

常驻的 **`/health` 在模型未载入前就会返回 200**，若立刻开跑容易 **`RemoteDisconnected` / `Connection refused`**。脚本 **`code/run_fever_gpu_detached.sh`** 会先起 resident、轮询 health，再 **`sleep 45`**（可用环境变量 **`RESIDENT_READY_GRACE_SEC`** 改），最后 **`exec` 跑 FEVER-only Exp02**，整段挂在 **nohup** 上，**与 SSH 会话脱钩**：

```bash
cd ~/dev/KVI
nohup bash experiments/exp02_hallucination/code/run_fever_gpu_detached.sh \
  </dev/null \
  >> experiments/exp02_hallucination/results/exp02_fever_gpu_orchestrator_outer.log 2>&1 &
```

- 编排日志：`results/exp02_fever_gpu_orchestrator.log`  
- 常驻日志：`experiments/results/resident_18888_gpu.log`  
- FEVER 评测输出：`results/exp02_fever_gpu.log`  

断线后用 `pgrep -af 'run_fever_gpu_detached|run_exp02_hallucination|resident_infer'` 与 `wc -l results/fever_fullmethods_qwen25_7b/predictions.jsonl` 自查。

---

## 当前图中的「幻觉率」是什么（与社区口径的差异）

`run_exp02_hallucination.py` 写出的 **`hallucination_proxy_summary.json`** 中：**TruthfulQA** 为 **`100 − relaxed EM`**，**FEVER** 为 **`100 − fever_label_accuracy`**（见该 JSON 的 `note` 与脚本内注释）。

- **relaxed EM**（`experiments/exp01_main_qa/code/metrics.py`）：SQuAD 风格归一化后，**任一条 gold 是否作为子串出现在模型整段输出中**（并对 `yes`/`no` 有少量扩展）。适合长段生成的 Hotpot/NQ 风格 QA，**不是** TruthfulQA 或 FEVER 官方主表口径。  
- **应对齐的社区/官方口径（建议写论文或对外对比时使用）**  
  - **TruthfulQA**：官方/常用为 **generation 上的人类或自动化评测**（如 MC1/MC2、或官方脚本），而不是「参考句是否出现在长生成里」。  
  - **FEVER**：共享任务常用 **三分类标签准确率**（SUPPORTS / REFUTES / NOT ENOUGH INFO）；完整流水线还可接 **fever-scorer**（需预测 Wikidata 证据等），与本仓库的「仅标签」设定不同。

### 推荐实现顺序（已定）

1. **FEVER（优先，已落地）**  
   - `run_exp01.py` 在 `--dataset_name FEVER` 时额外计算 **标签准确率**：在模型全文里找 **首次出现** 的 `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`（`metrics.parse_fever_label`），与 `prepare_exp02_datasets.py` 写入的 gold 比较。  
   - 见 `results/fever_fullmethods_qwen25_7b/summary.json` 中每方法的 **`fever_label_accuracy`**、**`fever_label_ci95_*`**；逐题见 `predictions.jsonl` 的 **`fever_label_em`**。  
   - 比 **relaxed EM** 更接近共享任务的 **veracity 标签**口径；**仍不是**带证据提交的官方 scorer。  
2. **TruthfulQA（第二步，已接入）**  
   - `prepare_exp02_datasets.py` 现已支持把 `multiple_choice` 的 `mc1_targets` / `mc2_targets`（若本地或在线可读）并入 `truthfulqa_eval.jsonl`。  
   - 若 `multiple_choice` 不可用，会从 generation split 的 `correct_answers/incorrect_answers` 自建 MC targets，保证覆盖率。  
   - `run_exp01.py` 现已输出 `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`，默认 `--truthfulqa_mc_mode likelihood_proxy`（对候选选项做对数似然打分）。  
   - 注意：该实现比纯字符串匹配更接近 MC 口径，但仍记为 **proxy**，与官方 TruthfulQA 发布链路并非逐项完全等价。

Exp02 的 **`hallucination_proxy_summary.json`**（由 `run_exp02_hallucination.py` 写出）口径是：**TruthfulQA = `100 − relaxed EM`**，**FEVER = `100 − fever_label_accuracy`**（与 JSON 内 `note` 一致）。因此 **同一文件里两套任务的「幻觉率」不可横向类比**：TruthfulQA 这一列往往极高（长生成里很难子串命中参考句），并不代表 FEVER 上模型更「诚实」。论文主图若要与 FEVER 并列、且 TruthfulQA 希望接近社区 MC 语义，请用下面的 **`results/summary.json` + 三栏图**（MC1 / MC2 / FEVER label）。

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

### 当前已跑结果（TruthfulQA + FEVER）

> 口径说明：下表 EM 为 `relaxed EM`；Exp02 的 hallucination rate 为 `100 - relaxed EM`（proxy）。

| Dataset | Method | EM (%) | 95% CI | F1 Mean | Proxy Hallucination (%) |
|---|---|---:|---:|---:|---:|
| TruthfulQA | LLM | 5.0 | [3.2, 7.0] | 0.176 | 95.0 |
| TruthfulQA | RAG | 7.4 | [5.0, 9.6] | 0.135 | 92.6 |
| TruthfulQA | GraphRAG | 17.8 | [14.6, 21.2] | 0.195 | 82.2 |
| TruthfulQA | KV Prefix | 3.8 | [2.2, 5.6] | 0.016 | 96.2 |
| TruthfulQA | KVI | 11.8 | [9.0, 14.6] | 0.115 | 88.2 |
| FEVER | LLM | 38.3 | [35.2, 41.2] | 0.173 | 61.7 |
| FEVER | RAG | 92.6 | [91.0, 94.1] | 0.288 | 7.4 |
| FEVER | GraphRAG | 73.0 | [70.4, 76.0] | 0.576 | 27.0 |
| FEVER | KV Prefix | 74.3 | [71.7, 76.8] | 0.312 | 25.7 |
| FEVER | KVI | 89.3 | [87.3, 91.2] | 0.893 | 10.7 |

FEVER 额外标签指标（更接近 veracity 任务）：

| FEVER Method | FEVER Label Accuracy (%) | 95% CI |
|---|---:|---:|
| LLM | 30.9 | [28.0, 33.7] |
| RAG | 92.5 | [90.9, 94.1] |
| GraphRAG | 68.8 | [66.1, 72.0] |
| KV Prefix | 72.3 | [69.6, 75.1] |
| KVI | 89.3 | [87.3, 91.2] |

对应文件：

- `results/truthfulqa_fullmethods_qwen25_7b/summary.json`
- `results/fever_fullmethods_qwen25_7b/summary.json`
- `results/hallucination_proxy_summary.json`

### 表格与原始指标

| 内容 | 路径 |
|------|------|
| 跨数据集汇总（代理指标） | `results/hallucination_proxy_summary.json`、`hallucination_proxy_summary.md`（跑完 `run_exp02_hallucination.py` 后生成；**无则**可用下面两个 `summary.json` 手搓图） |
| TruthfulQA 逐方法 EM / F1 / CI（以及 `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`） | `results/truthfulqa_fullmethods_qwen25_7b/summary.json`、`results.md`、`results.csv` |
| FEVER 同上 + **标签准确率** `fever_label_accuracy` | `results/fever_fullmethods_qwen25_7b/summary.json`、`results.md`、`results.csv` |
| 逐题预测 | 各数据集目录下 `predictions.jsonl` |

### 发表论文用的矢量图（SVG，可插 LaTeX / Word）

**两栏 vs 三栏（不要混用文件）**

| 图 | 栏数 | TruthfulQA 口径 | FEVER 口径 |
|----|------|-----------------|------------|
| `hallucination_proxy_bars_paper.svg` | **2** | `100 − relaxed EM`（子串 proxy，柱子常 **80–96%**） | `100 − fever_label_accuracy`（与 `hallucination_proxy_summary.json` 一致） |
| `hallucination_proxy_three_panel_paper.svg` 或 `unified_hallucination_bars.svg` | **3** | 左两栏：`100 − MC1 / MC2 likelihood proxy` | 右栏：`100 − fever_label_accuracy` |

三栏数据来自跑完 Exp02 后生成的 **`results/summary.json`**（与 `hallucination_proxy_summary.json` 不同：后者 TruthfulQA 仍是 relaxed EM）。

脚本：`experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py`

```bash
# 推荐：一次生成 paper 两栏 + 三栏 + 可选拆分图（需已有 results/summary.json）
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --paper \
  --three_panel_unified \
  --fever_label_figure \
  --truthfulqa_mc_figure

# 或仅从两个 per-dataset summary.json 生成（无 hallucination_proxy_summary.json 时）
python3 experiments/exp02_hallucination/code/plot_hallucination_proxy_bars.py \
  --truthfulqa_summary experiments/exp02_hallucination/results/truthfulqa_fullmethods_qwen25_7b/summary.json \
  --fever_summary experiments/exp02_hallucination/results/fever_fullmethods_qwen25_7b/summary.json \
  --paper \
  --three_panel_unified \
  --fever_label_figure \
  --truthfulqa_mc_figure
```

（三栏也可单独用 `code/plot_unified_hallucination_bars.py` 写出 `unified_hallucination_bars.svg`，样式略简；与 `--three_panel_unified` 使用同一 `results/summary.json`。）

| 输出文件 | 用途 |
|----------|------|
| **`hallucination_proxy_three_panel_paper.svg`** | **论文并列主图推荐**：三栏统一口径（TQA MC1 + TQA MC2 + FEVER label → 幻觉率） |
| `hallucination_proxy_three_panel.svg` | 同左，非 `paper` 样式 |
| `unified_hallucination_bars.svg` | 与上同类三栏（`plot_unified_hallucination_bars.py`） |
| **`hallucination_proxy_bars_paper.svg`** | **仅两栏**：左 TQA **relaxed EM**（易显「虚高」）、右 FEVER label；图注已写明差异 |
| `hallucination_proxy_bars.svg` | 屏幕预览 / 非印刷 |
| **`fever_label_accuracy_bars_paper.svg`** | FEVER **三分类标签准确率**（需 `summary.json` 含 `fever_label_accuracy`；老结果需重跑 `run_exp01.py`） |
| `fever_label_accuracy_bars.svg` | 同左，非 `paper` 样式 |
| **`truthfulqa_mc_proxy_bars_paper.svg`** | TruthfulQA **MC1/MC2 proxy**（来自 `truthfulqa_mc*_proxy`，非官方 MC 脚本分） |
| `truthfulqa_mc_proxy_bars.svg` | 同左，非 `paper` 样式 |

**PDF**：编辑部常要 PDF/EPS。可用 Inkscape（`inkscape file.svg --export-filename=file.pdf`）或 `rsvg-convert -f pdf -o file.pdf file.svg` 从 **SVG** 转换，保持矢量。

**图中要写清的表述**：两栏图中 **TruthfulQA = relaxed EM 代理**，与 **MC1/MC2** 或人类评测不是同一数字；三栏图中 TruthfulQA 为 **likelihood MC proxy**。FEVER 侧以 **标签准确率** 为主；仍非带证据的官方 fever-scorer。

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
