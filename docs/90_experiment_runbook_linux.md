# Linux(/home/jb/KVI) 实验 Runbook：数据集→训练→测试（可复制执行）

> 约定：所有命令在 `/home/jb/KVI` 执行。  
> 目标：你可以按本文从零跑通：PDF→raw context→blocks→KVBank→（可选）projector/gate 训练→单步/多步注入测试。

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

完成后，如何判断成功（四个硬产物）：
- `topics/<TOPIC>/doc_filter_results.jsonl`：doc-level 筛选记录（KEEP/DROP/UNCERTAIN）
- `topics/<TOPIC>/pdfs/`：KEEP 的 PDF（默认软链）
- `topics/<TOPIC>/work/raw_chunks.jsonl`、`topics/<TOPIC>/work/blocks.jsonl`
- `topics/<TOPIC>/work/kvbank_blocks/manifest.json`（以及 `kvbank_tables/manifest.json` 若 `split_tables=true`）

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

### 1.3 质量检查：如何确认 blocks 文本“抽取质量好”

在跑 `blocks.jsonl` 生成后，建议做两步：

1) **整体统计**（空块率、token 分布、重复率、疑似乱码比例、表格覆盖率）

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$WORK_DIR/blocks.jsonl" \
  --sample 10
```

2) **只抽样表格相关 blocks**（医学场景优先确认表格是否保留下来）

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$WORK_DIR/blocks.jsonl" \
  --tables_only \
  --sample 10
```

> 如果你的仓库是“monorepo 布局”（即 repo root 下还有 `external_kv_injection/` 子目录），则把上述命令里的脚本路径改为：
> `python -u external_kv_injection/scripts/inspect_blocks_quality.py ...`

产物：
- `$WORK_DIR/raw_chunks.jsonl`（raw context 存储层，不进 attention）
- `$WORK_DIR/blocks.jsonl`（256-token memory blocks）
- `$WORK_DIR/kvbank_blocks/`（FAISS KVBank：embedding + K/V + metadata）

### 1.4 后台构建 KVBank（blocks → kvbank，nohup + 实时看日志）

> `blocks_to_kvbank` 阶段计算量大且**非常吃内存**。建议启用**方案A：分片 KVBank**（`--shard_size`），让它边处理边落盘，避免一次性 `np.stack` 把内存打爆。

1) 启动后台任务（日志同时写文件，方便随时 `tail -f`）

```bash
mkdir -p "$WORK_DIR/logs"

nohup bash -lc "python -u scripts/build_kvbank_from_blocks.py \
  --blocks '$WORK_DIR/blocks.jsonl' \
  --out_dir '$WORK_DIR/kvbank_blocks' \
  --base_llm '$BASE_LLM' \
  --retrieval_encoder_model '$DOMAIN_ENCODER' \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 2>&1 | tee -a '$WORK_DIR/logs/blocks_to_kvbank.log'" \
  >/dev/null 2>&1 &
echo "started, log=$WORK_DIR/logs/blocks_to_kvbank.log"
```

2) 在当前终端“实时看输出”（不影响后台运行）

```bash
tail -f "$WORK_DIR/logs/blocks_to_kvbank.log"
```

3) 检查是否落盘成功（分片模式下，会出现 `kvbank_blocks/manifest.json` + `kvbank_blocks/shards/00000/...`）

```bash
ls -alh "$WORK_DIR/kvbank_blocks"
ls -alh "$WORK_DIR/kvbank_blocks/shards" | head
```

## 2) 测试：多步注入（Multi-step Injection，2×V100 友好）

```bash
python external_kv_injection/scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_blocks" \
  --kv_dir_tables "$WORK_DIR/kvbank_tables" \
  --enable_table_routing \
  --table_top_k 4 \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "请结合知识库逐步推理回答：SFTSV 的主要传播途径是什么？并给出依据。" \
  --blocks_jsonl "$WORK_DIR/blocks.jsonl" \
  --allowed_langs "zh,en" \
  --layers 0,1,2,3 \
  --max_steps 8 \
  --max_step_tokens 1024 \
  --max_total_tokens 2048 \
  --top_k_blocks 8 \
  --max_blocks_per_step 1 \
  --use_attention_entropy \
  --entropy_threshold 0.35 \
  --max_new_tokens 128
```

说明：
- `--blocks_jsonl + --allowed_langs`：在混语料（尤其包含日文）PDF 上强烈建议开启，避免检索命中非目标语言 block 后“注入导致语义退化/乱码/重复输出 prompt”。
- `--max_blocks_per_step 1`：对 RoPE 模型（如 Qwen2）建议先从 1 开始，稳定后再逐步增大到 2/4。

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

如果决定新开 chat，开头请贴这 6 行（足够cursor无缝上）
### 你的远程工作目录：/home/jb/KVI
### 你的 WORK_DIR：例如 /home/jb/KVI/_exp_prod
### 你最终的 blocks 文件：$WORK_DIR/blocks.v2.jsonl
### 你最终的 KVBank：$WORK_DIR/kvbank_blocks_v2，以及（如有）$WORK_DIR/kvbank_tables_v2
### 你用的模型：BASE_LLM=...，DOMAIN_ENCODER=...
### 你要做的下一步（比如“跑 multistep demo + 看 step debug”）


