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

产物：
- `$WORK_DIR/raw_chunks.jsonl`（raw context 存储层，不进 attention）
- `$WORK_DIR/blocks.jsonl`（256-token memory blocks）
- `$WORK_DIR/kvbank_blocks/`（FAISS KVBank：embedding + K/V + metadata）

## 2) 测试：多步注入（Multi-step Injection，2×V100 友好）

```bash
python external_kv_injection/scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_blocks" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "请结合知识库逐步推理回答：SFTSV 的主要传播途径是什么？并给出依据。" \
  --layers 0,1,2,3 \
  --max_steps 8 \
  --max_step_tokens 1024 \
  --max_total_tokens 2048 \
  --top_k_blocks 8 \
  --use_attention_entropy \
  --entropy_threshold 0.35 \
  --max_new_tokens 128
```

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


