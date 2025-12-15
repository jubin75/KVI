# Raw Context 构建流程（PDF→raw_chunks→blocks→KVBank）

本流程对齐 `PRD/raw context构建流程.md`，明确 **raw context 与 KV Bank 的职责边界**：

- **Raw context（存储层）**：离线产物是 `raw_chunks.jsonl`（4096-token chunks，可 overlap），用于后续构建 KV Bank；**不直接参与 attention**。
- **KV Bank（长期记忆层）**：存储单位是 `blocks.jsonl` 对应的 **256-token memory blocks** 的：
  - `block embedding`（ANN 检索用）
  - `K_ext/V_ext`（注入用，默认来自 base LLM 的 past_key_values，layers 0..3）
  - `metadata`（doc_id/段落类型/疾病/日期等）

## 脚本

- PDF → raw_chunks：`external_kv_injection/scripts/build_raw_context_from_pdfs.py`
- raw_chunks → blocks：`external_kv_injection/scripts/build_blocks_from_raw_chunks.py`
- blocks → KVBank：`external_kv_injection/scripts/build_kvbank_from_blocks.py`
- 一键：`external_kv_injection/scripts/build_kvbank_from_pdf_dir_multistep.py`

## 一键命令（推荐）

```bash
cd /home/jb/KVI

export PDF_DIR="/path/to/pdfs"
export WORK_DIR="external_kv_injection/_work_raw"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

python external_kv_injection/scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR" \
  --work_dir "$WORK_DIR" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --block_tokens 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

## OCR（扫描 PDF）
本工程已实现扫描 PDF 的 OCR（`pytesseract` + `tesseract` 二进制）。

- Python 依赖：`pytesseract`, `pillow`（已加入 `requirements.txt`）
- 系统依赖：需要安装 `tesseract` 命令

在 Linux 上常见安装方式（示例）：

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

启用 OCR（PDF→raw_chunks 阶段）：
- 使用 `external_kv_injection/scripts/build_raw_context_from_pdfs.py --ocr auto|on`

## 表格优先（医疗场景推荐）
本工程已实现 **表格抽取优先**（基于 `pdfplumber`）：
- 会把每页抽取到的表格转换成 markdown，并追加到该页文本后，进入 `raw_chunks.jsonl`。
- 同时清洗逻辑会弱化/移除图例（Figure）、引用标注、公式噪声，避免稀释表格信息密度。

依赖：
- `pdfplumber`（已加入 `requirements.txt`）

关闭表格抽取（不推荐）：
- `external_kv_injection/scripts/build_raw_context_from_pdfs.py --no_tables`

## DeepSeek 知识含量过滤（推荐用于生产 raw context）
如果你希望把“前言泛背景、病例患者叙事、未来展望、方法学局限/问题”等低知识密度内容剔除，只保留摘要/结果/结论/指南与表格信息，
可以启用 DeepSeek 过滤器（段落级 KEEP/DROP）。

### 依赖与环境变量
- Python 依赖：`requests`（已加入 `requirements.txt`）
- 需要设置 API Key（默认读取环境变量 `DEEPSEEK_API_KEY`）：

```bash
export DEEPSEEK_API_KEY="YOUR_KEY"
```

### 启用方式（PDF→raw_chunks 阶段）

```bash
python external_kv_injection/scripts/build_raw_context_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_DIR/raw_chunks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

更激进的过滤（UNCERTAIN 也丢弃）：
- 加 `--strict_drop_uncertain`

### 隐私提示
开启该过滤会把段落文本发送到 DeepSeek API；若有合规要求，请先在本地做脱敏/或改成私有化部署的 OpenAI 兼容 endpoint。

产物：
- `$WORK_DIR/raw_chunks.jsonl`（raw context）
- `$WORK_DIR/blocks.jsonl`（memory blocks）
- `$WORK_DIR/kvbank_blocks/`（FAISS KVBank，存 blocks 的 embedding+K/V+metadata）


