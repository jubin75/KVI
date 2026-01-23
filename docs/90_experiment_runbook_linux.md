# Linux(/home/jb/KVI) 实验 Runbook：数据集→训练→测试（可复制执行）

> 约定：所有命令在 `/home/jb/KVI` 执行。  
> 目标：你可以按本文从零跑通：PDF→raw context→blocks→KVBank→（可选）projector/gate 训练→单步/多步注入测试。  
> **架构规范**：slot-aware schema 注入遵循 `docs/slot_enum.md`；schema code review 遵循 `docs/73_schema_code_review.md`。

## 0) 安装依赖（Python + 系统包）

### 0.1 Python 环境

```bash
cd /home/jb/KVI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

如果你用的是旧版 `transformers`，加载 Qwen2/DeepSeek 可能报：
`Tokenizer class Qwen2Tokenizer does not exist ...`。建议升级：

```bash
pip install -U "transformers>=4.41" accelerate safetensors tokenizers sentencepiece
```

### 0.2 系统依赖（OCR）

如果你要处理扫描 PDF（OCR），需要安装 `tesseract`：

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## 1) 数据构建（生产推荐）：PDF → raw_chunks(4096) → blocks(256) → KVBank

### 1.1 设置实验参数

```bash
# 建议使用绝对路径；如果你已经 cd /home/jb/KVI，也可以用相对路径 ./pdfs
export PDF_DIR="/home/jb/KVI/pdfs"
export WORK_DIR="/home/jb/KVI/_exp_prod"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# DeepSeek（知识含量过滤）
export DEEPSEEK_API_KEY="sk-bc1bf3f7edd344c69ca74b2279340434"
```

### 1.1.1（强烈推荐）专题库模式：以后统一“分库做数据集”

> 结论：External KV injection 是高增益、低容错；**语料不相关/命中乱码块**会被注入放大。  
> 所以我们推荐把“医学大库”拆成多个“专题库”（SFTSV、SARS‑CoV‑2 等），并且每次只加载/构建一个专题库。

这里有两条等价链路：

- **推荐主线（更省心）**：`rebuild_topic_kvbank_from_config.py`  
  一条命令完成：doc-level DeepSeek（摘要）筛选 → raw_chunks → blocks → KVBank（可选 split_tables）
- **调试主线（可拆解）**：`build_topic_pdf_subset_deepseek.py` → `build_raw_context_from_pdfs.py` → `build_kvbank_from_pdf_dir_multistep.py`  
  适合你逐段确认“筛选/抽取/切块/建库”哪一步出了问题

两条链路的关系：
- `rebuild_topic_kvbank_from_config.py` 本质上是把“调试主线”的步骤**串起来**（内部会调用 doc-filter，并直接调用 pipeline 函数生成 raw_chunks/blocks/KVBank）。
- 所以：**结果等价**，只是 `rebuild` 更一键；三段式更容易定位问题。

### 1.1.2（你只需要改 config.json）专题库的配置文件位置与目录约定

> 约定：本 runbook 针对你的远程目录（flat layout）`/home/jb/KVI`。  
> 如果你在另一个环境是 monorepo（repo root 下还有 `external_kv_injection/` 子目录），把下面的相对路径整体加上前缀即可。

模板位置（远程 `/home/jb/KVI`）：
- `config/topics/SFTSV/config.json`
- `config/topics/SARS2/config.json`

推荐目录结构（你已经在采用）：
- **专题源 PDF**（你人工整理/或放软链）：  
  - SFTSV：`/home/jb/KVI/pdfs/sftsvpdf`（或 `/home/jb/KVI/pdfs/SFTSV`）  
  - SARS2：`/home/jb/KVI/pdfs/sarspdf`（或 `/home/jb/KVI/pdfs/SARS2`）
- **专题产物目录**：  
  - SFTSV：`/home/jb/KVI/topics/SFTSV/`  
  - SARS2：`/home/jb/KVI/topics/SARS2/`

重要提示（重复 PDF / 软链接 / results 写入方式）：
- doc-level 筛选阶段默认 `mode=symlink`：`out_pdf_dir` 下会看到 **KEEP 的 PDF 软链接**，指向 `source_pdf_dir`（省磁盘、速度快）。
- 如果目录里有同名 PDF：`dedupe_by_basename=true` 会跳过重复项，并在 results.jsonl 标记 `DUPLICATE/SKIP`。
- **results_jsonl 默认是追加写**：方便保留历史记录。若你想每次重跑都得到“干净的一份结果”，在 config 的 `doc_filter` 里加：
  - `"overwrite_results": true`

### 1.2（推荐先做）快速验证：只跑 PDF → raw_chunks，确保抽取/解析正常

如果这一步失败，说明是 PDF 抽取/OCR/依赖问题（不是 block 切分问题）。

```bash
python scripts/build_raw_context_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_DIR/raw_chunks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

### 1.2 一键构建 raw context + KVBank（表格优先 + DeepSeek 过滤）

注意：如果开启 `--knowledge_filter`，会对**每个段落**调用一次 DeepSeek 接口（串行），因此在网络/限流情况下可能很慢、CPU/GPU 负载也会很低，这是正常的。
建议首次验证链路时先不加 `--knowledge_filter`，确认 PDF 抽取与分块没问题后再开启过滤。

```bash
python scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR" \
  --work_dir "$WORK_DIR" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --block_tokens 256 \
  --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

### 1.2.1（专题库）一键构建：SFTSV / SARS‑CoV‑2 两个专题 KVBank

> 约定目录结构（推荐）：
> - SFTSV：`$WORK_DIR/topics/sftsv/...`
> - SARS‑CoV‑2：`$WORK_DIR/topics/sarscov2/...`

```bash
export WORK_DIR_TOPIC="$WORK_DIR/topics"

python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR_SFTSV" \
  --work_dir "$WORK_DIR_TOPIC/sftsv" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024

python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR_SARSCOV2" \
  --work_dir "$WORK_DIR_TOPIC/sarscov2" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024
```

### 1.2.2（推荐主线）一条命令重建专题库：doc-level DS(abstract) → pipeline → KVBank

> 你只需要改 config.json 里的：
> - `goal`：建立专题库的目标（用户输入）
> - `extract_tables`：是否抽取表格
> - 路径：`source_pdf_dir / out_pdf_dir / build.work_dir`

```bash
# SFTSV 专题库（输出目录：/home/jb/KVI/topics/SFTSV）
python -u scripts/rebuild_topic_kvbank_from_config.py \
  --config config/topics/SFTSV/config.json

# SARS2 专题库（输出目录：/home/jb/KVI/topics/SARS2）
python -u scripts/rebuild_topic_kvbank_from_config.py \
  --config config/topics/SARS2/config.json
```

可选：从“完全干净”开始（仅删目标产物）

如果你想确保没有旧文件干扰（非必须），可以先删 evidence 相关两项再跑重建：

```bash
rm -rf /home/jb/KVI/topics/SFTSV/work/kvbank_evidence
rm -f /home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl
```

完成后，如何判断成功（从头构建 + 双库，一共 6 个硬产物）：
- `topics/<TOPIC>/doc_filter_results.jsonl`：doc-level 筛选记录（KEEP/DROP/UNCERTAIN）
- `topics/<TOPIC>/pdfs/`：KEEP 的 PDF（默认软链）
- `topics/<TOPIC>/work/raw_chunks.jsonl`、`topics/<TOPIC>/work/blocks.jsonl`
- `topics/<TOPIC>/work/kvbank_blocks/manifest.json`（以及 `kvbank_tables/manifest.json` 若 `split_tables=true`）
- `topics/<TOPIC>/work/blocks.evidence.jsonl`（DeepSeek 抽取式证据句，推荐从 raw_chunks 段落抽取）
- `topics/<TOPIC>/work/kvbank_evidence/manifest.json`（evidence KVBank）

#### 1.2.2.1（新增，推荐）Evidence 版本：blocks.evidence + kvbank_evidence

为了解决“raw block 噪声太大、知识碎片化导致注入退化”的问题，我们引入 **evidence-first 双库策略**：

- **evidence 库**：DeepSeek **抽取式**证据句（extractive-only）→ `blocks.evidence.jsonl` → `kvbank_evidence/`
- **raw 库**：保留 `blocks.jsonl` + `kvbank_blocks/` 用于回溯与补上下文

现在 `rebuild_topic_kvbank_from_config.py` 会在 **PDF→raw_chunks→blocks** 之后，默认继续生成 evidence（从头构建）：

- `topics/<TOPIC>/work/blocks.evidence.jsonl`
- `topics/<TOPIC>/work/kvbank_evidence/manifest.json`

如果你只想在已有产物基础上**单独补建 evidence**（不重跑 PDF→raw_chunks→blocks），可以用（但不推荐作为“最干净”的主线）：

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

# 1) raw_chunks -> evidence blocks（DeepSeek 抽取式证据句；更干净，推荐）
python -u scripts/build_evidence_blocks_from_raw_chunks_jsonl_deepseek.py \
  --raw_chunks_jsonl "$WORK_DIR/raw_chunks.jsonl" \
  --out_jsonl "$WORK_DIR/blocks.evidence.jsonl" \
  --topic_goal "$(jq -r .goal config/topics/SFTSV/config.json)" \
  --max_sentences_per_paragraph 3

# 提示：如果你发现 evidence 覆盖不足（抽取偏保守），可以把上面的 max_sentences_per_paragraph 提到 4 或 5，
# 会增加证据句密度（也可能带来少量噪声，需要用 1.3 的关键词抽样再确认）。

# 2) evidence blocks -> kvbank_evidence
python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$WORK_DIR/blocks.evidence.jsonl" \
  --out_dir "$WORK_DIR/kvbank_evidence" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024
```

#### 1.2.2.1.a（调试必备）mini evidence.txt 一键建库：evidence.txt → blocks → pattern sidecar → pattern_contract → kvbank_blocks

> 用途：你只有几行 `evidence.txt`，想先跑通“检索→Evidence Units→注入→回答”的最小闭环（用于架构调试/补漏洞），不依赖 PDF。
>
> 约定目录（可替换成任意 topic）：  
> - `TOPIC_DIR=/home/jb/KVI/config/topics/SFTSV`  
> - 输入：`$TOPIC_DIR/evidence.txt`  
> - 产物：`$TOPIC_DIR/blocks.jsonl`、`$TOPIC_DIR/blocks.enriched.jsonl`、`$TOPIC_DIR/pattern_contract.json`、`$TOPIC_DIR/kvbank_blocks/`

```bash
cd /home/jb/KVI

export TOPIC_DIR="/home/jb/KVI/config/topics/SFTSV"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# 1) evidence.txt -> blocks.jsonl
python scripts/build_blocks_from_raw_text.py \
  --raw_text "$TOPIC_DIR/evidence.txt" \
  --out "$TOPIC_DIR/blocks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 \
  --keep_last_incomplete_block

# 2) blocks.jsonl -> blocks.enriched.jsonl + pattern sidecar（pattern_out_dir 放在 topic_dir 里即可）
python scripts/build_pattern_index_from_blocks_v2.py \
  --blocks_jsonl_in "$TOPIC_DIR/blocks.jsonl" \
  --blocks_jsonl_out "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_out_dir "$TOPIC_DIR"

# 3) blocks.enriched.jsonl -> pattern_contract.json（供 PatternContractLoader + matcher/scoring 使用）
python scripts/pattern_contract_autogen.py \
  --blocks_jsonl_in "$TOPIC_DIR/blocks.enriched.jsonl" \
  --out "$TOPIC_DIR/pattern_contract.json" \
  --topic SFTSV \
  --min_abbr_count 1 \
  --min_slot_count 1 \
  --max_abbr 50 \
  --max_slots 50

# 4) blocks.enriched.jsonl -> kvbank_blocks（mini KVBank）
python scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --out_dir "$TOPIC_DIR/kvbank_blocks" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 \
  --device cuda \
  --dtype bfloat16
```

##### 1.2.2.1.a.1 验证（simple pipeline）：routing + Evidence Units + injected output

> 目标：验证“检索命中 + Evidence Units 数量 + 注入输出”都正常（输出 JSON 里会包含 step_debug）。
> 注意：simple pipeline 是架构调试模式，核心链路是：prompt → similarity retrieval（kvbank_blocks）→ Evidence Units（sentence-level）→ 多步注入 → 文本回答。

```bash
cd /home/jb/KVI

export TOPIC_DIR="/home/jb/KVI/config/topics/SFTSV"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# A) 症状（应出现 selected_unit_counts>0 且 evidence_units_shown 有症状枚举句）
python scripts/run_kvi2_runtime_test.py \
  --pipeline simple \
  --model "$BASE_LLM" \
  --prompt "SFTSV的主要临床症状有哪些？" \
  --kv_dir "$TOPIC_DIR/kvbank_blocks" \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_index_dir "$TOPIC_DIR" \
  --sidecar_dir "$TOPIC_DIR" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --use_chat_template \
  --top_k 8 \
  --simple_use_evidence_units \
  --simple_require_units \
  --simple_max_steps 1 \
  --simple_max_blocks_per_step 4 \
  --simple_max_unit_sentences 4 \
  --show_baseline

# B) 地区 + 症状（multi-intent，evidence_units_shown 应同时包含地区与症状句）
python scripts/run_kvi2_runtime_test.py \
  --pipeline simple \
  --model "$BASE_LLM" \
  --prompt "2009-2014年SFTSV在我国的主要发病地区有哪些？主要临床症状有哪些？" \
  --kv_dir "$TOPIC_DIR/kvbank_blocks" \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_index_dir "$TOPIC_DIR" \
  --sidecar_dir "$TOPIC_DIR" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --use_chat_template \
  --top_k 8 \
  --simple_use_evidence_units \
  --simple_require_units \
  --simple_max_steps 1 \
  --simple_max_blocks_per_step 4 \
  --simple_max_unit_sentences 6 \
  --show_baseline
```

#### 1.2.2.2（新增，推荐）Schema 版本：blocks.schema + kvbank_schema（slot-aware 注入）

为了解决"重复注入同一语义维度导致答案坍缩"的问题，我们在 evidence 之上引入 **schema-first + slot-aware 策略**：

- **schema 库**：从 evidence 聚合的高信息密度槽位约束 → `blocks.schema.jsonl` → `kvbank_schema/`
- **evidence 库**：保留用于 grounding/citation（retrieval-only，**永不注入**）
- **raw 库**：保留用于回溯/fallback 上下文（retrieval-only，**永不注入**）

**核心约束（见 `docs/slot_enum.md`）**：
- Schema KV 是**唯一允许注入**的 cache
- Evidence/raw **只能 append prompt**，不可注入 KV
- 每步最多注入 1 个 schema；slot 覆盖用完即停止

##### 1) 从 evidence blocks 生成 schema blocks

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/build_schema_blocks_from_evidence_jsonl.py \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --out_jsonl "$WORK_DIR/blocks.schema.jsonl"
```

说明：
- 该脚本会从 evidence 文本中**启发式推断** `answerable_slots`（用于 slot-aware 选择），包括但不限于：
  - `transmission` / `pathogenesis` / `diagnosis` / `treatment`
  - `disease_full_name`（全称/缩写展开，taxonomy.definition）
  - `geographic_distribution`（地区分布，epidemiology.geography）
- schema text 中已移除 `vector` 字段（避免物种中文俗名误译/越权）。

⚠️ 重要：如果你升级了 slot 或 schema 编译逻辑，必须 **重建**：
- `blocks.schema.jsonl`
- `kvbank_schema`

##### 2) 构建 schema KVBank

```bash
python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$WORK_DIR/blocks.schema.jsonl" \
  --out_dir "$WORK_DIR/kvbank_schema" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024
```

##### 3) 检查 schema blocks 质量

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$WORK_DIR/blocks.schema.jsonl" \
  --sample 10
```

完成后，如何判断成功（schema-first 增加的 2 个硬产物）：
- `$WORK_DIR/blocks.schema.jsonl`（schema blocks，带 `slots` 字段）
- `$WORK_DIR/kvbank_schema/manifest.json`（schema KVBank）

### 1.2.3（调试主线）把链路拆成三段，逐段排错

当你需要定位“到底是筛选不准、抽取不行、还是建库出错”时，用这三段式：

1) **doc-level 筛选（摘要）**：源 PDF → `topics/<TOPIC>/pdfs/` + `doc_filter_results.jsonl`

```bash
python -u scripts/build_topic_pdf_subset_deepseek.py \
  --config config/topics/SFTSV/config.json \
  --max_pdfs 200
```

2) **抽取 raw_chunks（不建库）**：KEEP PDFs → raw_chunks

```bash
python -u scripts/build_raw_context_from_pdfs.py \
  --pdf_dir "/home/jb/KVI/topics/SFTSV/pdfs" \
  --out "/home/jb/KVI/topics/SFTSV/work/raw_chunks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

3) **一键 blocks+KVBank**：KEEP PDFs → work_dir（raw_chunks/blocks/kvbank）

```bash
python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "/home/jb/KVI/topics/SFTSV/pdfs" \
  --work_dir "/home/jb/KVI/topics/SFTSV/work" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024
```

为什么 runbook 里同时有 `rebuild_topic_kvbank_from_config.py` 和 `build_kvbank_from_pdf_dir_multistep.py`？

- **`rebuild_topic_kvbank_from_config.py`（推荐主线）**：专题库“一条命令”流水线。它会按 `config/topics/<TOPIC>/config.json` 先做 doc-level DeepSeek（摘要）筛选，然后跑 PDF→raw_chunks→blocks→KVBank，并默认继续生成 `blocks.evidence.jsonl` + `kvbank_evidence/`（双库策略）。
- **`build_kvbank_from_pdf_dir_multistep.py`（调试/拆解入口）**：不依赖 topic config 的通用脚本，用来在你**已经有一批 PDF** 时直接做 PDF→work_dir→KVBank，适合定位问题（比如抽取/OCR、切块、建库哪一步异常），也适合你临时做一个非专题的实验目录。

结论：**正常情况下只跑 `rebuild_topic_kvbank_from_config.py` 就够了**；只有当你要“逐段排错/做非专题 quick experiment”时才跑 `build_kvbank_from_pdf_dir_multistep.py`。

### 1.3 质量检查（evidence-first）：如何确认 blocks 文本“抽取质量好”

> 结论：在“双库策略”下，你**优先验证 evidence**（决定检索相关性与注入噪声），raw 只在需要表格/补上下文时再看。  
> 简单原则：**你最终推理/注入优先用哪个库，就先检查哪个库对应的 blocks 文件。**

先设定当前专题的 work 目录（下面以 SFTSV 为例）：

```bash
export TOPIC_WORK_DIR="/home/jb/KVI/topics/SFTSV/work"
```

#### 1.3.1（优先）检查 evidence blocks：`blocks.evidence.jsonl`

1) **整体统计 + 抽样**（空块率、重复率、疑似乱码比例、抽样原文）

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.evidence.jsonl" \
  --sample 10
```

> 说明：evidence blocks 目标是“更短、更单意图、更可直接回答”，通常不强调 `--tables_only`。

2) **关键词抽样验证：库里是否包含某类证据句（例如“发病机制/免疫机制”）**

> 用途：快速回答“evidence 库里有没有机制/pathogenesis/immune 相关证据句”，避免只凭感觉猜。  
> 脚本会流式扫描 `blocks.evidence.jsonl`，统计每个关键词命中次数，并随机抽样打印命中的 block（含 `doc_id/source_uri/line_no` + snippet）。

```bash
python -u scripts/sample_blocks_by_keywords.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.evidence.jsonl" \
  --keywords "pathogenesis,mechanism,immune,cytokine,MODS,multi-organ,致病,发病机制,免疫,细胞因子,器官功能衰竭" \
  --sample 20 \
  --seed 0 \
  --max_chars 600
```

#### 1.3.2（可选）回看 raw blocks：`blocks.jsonl`（表格/上下文/定位碎片化问题）

1) **整体统计 + 抽样**

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.jsonl" \
  --sample 10
```

2) **只抽样表格相关 blocks**（仅当你开启 `extract_tables/split_tables`，并且确实希望表格进入 raw 库/表格路由时才需要）

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.jsonl" \
  --tables_only \
  --sample 10
```

> 如果你的仓库是“monorepo 布局”（即 repo root 下还有 `external_kv_injection/` 子目录），则把上述命令里的脚本路径改为：
> `python -u external_kv_injection/scripts/inspect_blocks_quality.py ...`

产物（专题 work_dir 下）：
- `$TOPIC_WORK_DIR/raw_chunks.jsonl`（PDF 抽取后的 raw context，不进 attention）
- `$TOPIC_WORK_DIR/blocks.evidence.jsonl`（DeepSeek 抽取式证据句，推荐优先检索/注入）
- `$TOPIC_WORK_DIR/kvbank_evidence/`（evidence KVBank）
- （可选回溯）`$TOPIC_WORK_DIR/blocks.jsonl` + `$TOPIC_WORK_DIR/kvbank_blocks/`（raw blocks/KVBank，用于补上下文/表格）

### 1.4 后台构建 KVBank（evidence-first）：blocks → kvbank，nohup + 实时看日志

> `blocks_to_kvbank` 阶段计算量大且**非常吃内存**。建议启用**方案A：分片 KVBank**（`--shard_size`），让它边处理边落盘，避免一次性 `np.stack` 把内存打爆。

同样先设定当前专题 work 目录（下面以 SFTSV 为例）：

```bash
export TOPIC_WORK_DIR="/home/jb/KVI/topics/SFTSV/work"
```

#### 1.4.1（优先）构建 evidence KVBank：`blocks.evidence.jsonl` → `kvbank_evidence/`

1) 启动后台任务（日志同时写文件，方便随时 `tail -f`）

```bash
mkdir -p "$TOPIC_WORK_DIR/logs"

nohup bash -lc "python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl '$TOPIC_WORK_DIR/blocks.evidence.jsonl' \
  --out_dir '$TOPIC_WORK_DIR/kvbank_evidence' \
  --base_llm '$BASE_LLM' \
  --domain_encoder_model '$DOMAIN_ENCODER' \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 2>&1 | tee -a '$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log'" \
  >/dev/null 2>&1 &
echo "started, log=$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log"
```

2) 在当前终端“实时看输出”（不影响后台运行）

```bash
tail -f "$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log"
```

3) 检查是否落盘成功（分片模式下，会出现 `kvbank_evidence/manifest.json` + `kvbank_evidence/shards/00000/...`）

```bash
ls -alh "$TOPIC_WORK_DIR/kvbank_evidence"
ls -alh "$TOPIC_WORK_DIR/kvbank_evidence/shards" | head
```

#### 1.4.2（可选）构建 raw KVBank：`blocks.jsonl` → `kvbank_blocks/`（表格/补上下文）

> 只有当你需要 raw fallback（或表格路由）时才跑这一段；否则可跳过。

```bash
mkdir -p "$TOPIC_WORK_DIR/logs"

nohup bash -lc "python -u scripts/build_kvbank_from_blocks.py \
  --blocks '$TOPIC_WORK_DIR/blocks.jsonl' \
  --out_dir '$TOPIC_WORK_DIR/kvbank_blocks' \
  --base_llm '$BASE_LLM' \
  --retrieval_encoder_model '$DOMAIN_ENCODER' \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 2>&1 | tee -a '$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log'" \
  >/dev/null 2>&1 &
echo "started, log=$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log"
```

```bash
tail -f "$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log"
```

```bash
ls -alh "$TOPIC_WORK_DIR/kvbank_blocks"
ls -alh "$TOPIC_WORK_DIR/kvbank_blocks/shards" | head
```

## 2) 测试：多步注入

### 2.0（推荐，新增）Schema-first 注入（slot-aware，最严格）

> 核心规则（见 `docs/slot_enum.md`）：
> - **只有 schema KV 可注入**（schema text forward → cache）
> - evidence/raw **只能 append prompt**（grounding），不可注入 KV
> - 每步最多注入 1 个 schema；slot 覆盖用完即停止

必须同时提供三库：`kvbank_schema`（注入）+ `kvbank_evidence`（grounding）+ `kvbank_blocks`（fallback）

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_blocks" \
  --kv_dir_evidence "$WORK_DIR/kvbank_evidence" \
  --kv_dir_schema "$WORK_DIR/kvbank_schema" \
  --blocks_jsonl_schema "$WORK_DIR/blocks.schema.jsonl" \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "SFTSV 的主要传播途径是什么？请用中文回答，并逐字引用1句证据原文（英文也可以）。" \
  --schema_required_slots "transmission,pathogenesis" \
  --use_struct_slots \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --enable_layer2 \
  --layer1_max_new_tokens 256 \
  --layer2_max_new_tokens 192 \
  --max_new_tokens 256 \
  --debug_print_candidates 10
```

验证要点：
- 日志应出现 `[retriever] routing=schema->evidence->raw ...`
- StepDebug.note 应包含 `schema_selector=...`（slot 覆盖信息）
- 若选中 schema 不引入新 slot → 停止（`stop_reason=no_new_slots`）
- 答案末尾只会 append `【证据句】/【回退上下文（raw）】`，**schema text 不会出现在 prompt**

三层知识输出（强制格式）：
- 注入后的回答会严格分成三段（永不跳层/合并）：
  - `### L0｜证据支持的结论`（Evidence-Bound：仅基于 evidence/文档，不允许扩写；证据不足必须写“暂无证据支持”）
  - `### L1｜领域共识（LLM 内部知识）`（Domain Prior：教科书级共识解释；不得与 L0 冲突；禁止“最新研究/假说”）
  - `### L2｜推测性或解释性补充`（Speculative：默认关闭；开启后必须显式“推测/尚未完全证实”，不得覆盖 L0/L1）

开启 L2（可选）：
- 默认 L2 不生成，只保留占位；如需开启推测层，给 demo 加：
  - `--enable_layer2`
  - 可调 token 预算：`--layer2_max_new_tokens 192`
  - L1 token 预算：`--layer1_max_new_tokens 256`

### 2.1（legacy）Evidence-first 注入

> 结论：默认用 **evidence KVBank** 做检索与注入（噪声更低、相关性更强），必要时再回退 raw 库补上下文。

下面以 SFTSV 为例（flat layout：`/home/jb/KVI`）：

```bash
python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --topic sftsv --topic_work_dir "/home/jb/KVI/topics" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "SFTSV 的主要传播途径是什么？请用中文回答，并逐字引用1句证据原文（英文也可以）。" \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --allowed_langs "zh,en" \
  --layers 0,1,2,3 \
  --max_steps 1 \
  --max_blocks_per_step 1 \
  --top_k_blocks 16 \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --max_new_tokens 256
```

说明：
- `--topic ... --topic_work_dir ...`：脚本会优先探测 `kvbank_evidence`（并在需要时回退 raw 的 `kvbank_blocks`）。
- `--blocks_jsonl(_evidence) + --allowed_langs`：强烈建议开启，避免混语料导致命中非目标语言 block 后注入退化。
- `--max_steps=1 + --max_blocks_per_step=1`：先用“最小注入”验证相关性与稳定性；稳定后再把 `--max_steps` 提到 2/4。

### 2.1.1（等价写法）显式指定 evidence + raw（不使用 topic mode）

如果你的专题库 work_dir 下存在：

- `kvbank_evidence/manifest.json`
- `blocks.evidence.jsonl`

如果你不想用 `--topic` 自动探测，也可以显式指定两个库的路径（效果等价）：

```bash
python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "/home/jb/KVI/topics/SFTSV/work/kvbank_blocks" \
  --kv_dir_evidence "/home/jb/KVI/topics/SFTSV/work/kvbank_evidence" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "SFTSV 的主要传播途径是什么？请用中文回答，并逐字引用1句证据原文（英文也可以）。" \
  --max_steps 1 \
  --max_blocks_per_step 1 \
  --top_k_blocks 16 \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --allowed_langs "zh,en" \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --max_new_tokens 256
```

验证要点：
- `=== Step Debug ===` 的 `selected_block_ids` 更倾向 evidence block（更短、更单意图）。
- 答案应更少出现“unknown routes / mink bite”等 raw 噪声带来的退化。

### 2.2（新增）评测集 A/B：强制 JSON 协议输出 + faithfulness/overclaim 指标

准备评测集 `prompts.jsonl`（每行至少包含 `prompt`）：

```json
{"id":"sftsv_tx_001","prompt":"SFTSV 的主要传播途径是什么？"}
{"id":"sftsv_tx_002","prompt":"SFTSV 是否存在人传人？如果有，主要通过什么接触？"}
```

运行 A/B（baseline vs injection），并自动计算覆盖率/overclaim：

```bash
python -u scripts/run_ab_eval_protocol.py \
  --model "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --kv_dir "/home/jb/KVI/topics/SFTSV/work/kvbank_blocks" \
  --kv_dir_evidence "/home/jb/KVI/topics/SFTSV/work/kvbank_evidence" \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --prompts_jsonl "/home/jb/KVI/topics/SFTSV/eval/prompts.jsonl" \
  --out_jsonl "/home/jb/KVI/topics/SFTSV/eval/ab_results.jsonl" \
  --max_examples 0
```

结果与验证：
- `ab_results.jsonl` 每条包含 baseline/injected 的原始输出、解析后的 JSON、以及 `covered/overclaim` 指标。
- 终端末尾会打印 summary：`baseline_valid/inj_valid`、`baseline_covered/inj_covered`、`baseline_over/inj_over`。

### 2.3（新增）单元测试：Evidence Recall（检索命中率冒烟测试）

目的：
- 验证 `blocks.evidence.jsonl` 产物非空、结构正常
- 验证 `kvbank_evidence` 对一组常见中文问法具有基本“命中能力”（不要求逐条证据匹配，只做召回 sanity check）
- 及时发现“检索向量退化/总是命中同一批证据”等问题

运行（以 SFTSV 为例）：

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/test_evidence_recall.py \
  --kv_dir_evidence "$WORK_DIR/kvbank_evidence" \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --top_k 16
```

说明：
- 默认会对“传播途径/病原体/发病机制/流行病学/诊断防控”五类 query 做检索，并断言总体 hit_rate 不低于阈值
- 若你发现 evidence 基本都是英文，可保留中文 query：该测试使用英文 anchor 做命中判断（检索是否拉回相关英文证据句）

## 3) （可选）训练 Projector（对齐到 past_key_values 空间，max_kv_tokens=256）

> 这条训练链路使用 `ChunkStore`（见 `external_kv_injection/scripts/build_chunkstore_from_pdfs.py`）。  
> 如果你希望训练数据也采用“DeepSeek 过滤后的 raw context”，建议先用 1) 的 raw_chunks 产物做一次转换生成 chunkstore（后续可加脚本）。

```bash
export WORK_TRAIN="external_kv_injection/_exp_train"
mkdir -p "$WORK_TRAIN"
```

### 3.1 PDF → chunkstore.jsonl（快速训练集）

```bash
python external_kv_injection/scripts/build_chunkstore_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_TRAIN/chunkstore.jsonl" \
  --dataset_version v0
```

### 3.2 生成 teacher KV dataset

```bash
python external_kv_injection/scripts/build_teacher_kv_dataset.py \
  --chunkstore "$WORK_TRAIN/chunkstore.jsonl" \
  --out "$WORK_TRAIN/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --layers 0,1,2,3 \
  --max_kv_tokens 256 \
  --max_samples 200
```

### 3.3 训练 projector

```bash
python external_kv_injection/scripts/train_projector_kv.py \
  --dataset "$WORK_TRAIN/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --out_dir "$WORK_TRAIN/projector_ckpt" \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 1
```

### 3.4 用 projector 构建 KVBank（检索 key 使用 DomainEncoder）

```bash
python external_kv_injection/scripts/build_kvbank_with_projector.py \
  --chunkstore "$WORK_TRAIN/chunkstore.jsonl" \
  --out_dir "$WORK_TRAIN/kvbank_projector" \
  --base_model "$BASE_LLM" \
  --projector_ckpt "$WORK_TRAIN/projector_ckpt/projector_kv.pt" \
  --max_kv_tokens 256 \
  --max_chunks 200 \
  --retrieval_encoder_model "$DOMAIN_ENCODER"
```

## 4) （可选）训练/使用 Gate（DomainEncoder(query) embedding）

```bash
python external_kv_injection/scripts/train_gate_query.py \
  --kv_dir "$WORK_TRAIN/kvbank_projector" \
  --out "$WORK_TRAIN/gate_query.pt"
```

推理验证（单步注入 demo）：

```bash
python external_kv_injection/scripts/run_qwen_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_TRAIN/kvbank_projector" \
  --prompt "请根据知识库内容回答：SFTSV 的主要传播途径是什么？并给出依据。" \
  --layers 0,1,2,3 \
  --top_k 4 \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --gate_ckpt "$WORK_TRAIN/gate_query.pt" \
  --gate_mode scale_v \
  --max_new_tokens 128
```

