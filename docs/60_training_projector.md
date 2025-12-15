# 训练：DomainEncoder + Projector + Gate（当前实现：Projector 对齐到 past_key_values）

本仓库当前提供一条“符合你约束”的可跑通训练路径：
- **不训练 LLM layers**：冻结 base model
- 训练 **KVProjector**：从 base model 的 last_hidden 预测每个注入层的 `past_key_values` 空间 K/V
- 注入层固定为 `L={0,1,2,3}`（可改）
- `max_kv_tokens=256`

> Gate 与 DomainEncoder 的训练可后续接入；Projector 是“对齐到 cache 空间”的关键模块。

## Quickstart：可复制粘贴的命令串（推荐）

> 约定：以下命令都在仓库根目录 `/home/jb/KVI` 执行。

### 0) 安装依赖（建议在 Linux/L40 服务器上执行）

```bash
cd /home/jb/KVI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 1) 设置路径与模型（按需修改）

```bash
export PDF_DIR="/path/to/pdfs"
export WORK_DIR="external_kv_injection/_work_train"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
# 推荐：一个通用 DomainEncoder（用于检索 + gate 输入，需与建库一致）
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"
```

### 2) PDF → ChunkStore

```bash
python external_kv_injection/scripts/build_chunkstore_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_DIR/chunkstore.jsonl" \
  --dataset_version v0
```

### 3) 构建 teacher KV dataset（监督信号：past_key_values，layers=0..3，max_kv_tokens=256）

```bash
python external_kv_injection/scripts/build_teacher_kv_dataset.py \
  --chunkstore "$WORK_DIR/chunkstore.jsonl" \
  --out "$WORK_DIR/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --layers 0,1,2,3 \
  --max_kv_tokens 256 \
  --max_samples 200
```

### 4) 训练 KVProjector（不训练 LLM layers）

```bash
python external_kv_injection/scripts/train_projector_kv.py \
  --dataset "$WORK_DIR/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --out_dir "$WORK_DIR/projector_ckpt" \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 1
```

### 5) 用 Projector 构建 KVBank（检索 key 使用 DomainEncoder，推荐）

```bash
python external_kv_injection/scripts/build_kvbank_with_projector.py \
  --chunkstore "$WORK_DIR/chunkstore.jsonl" \
  --out_dir "$WORK_DIR/kvbank_projector" \
  --base_model "$BASE_LLM" \
  --projector_ckpt "$WORK_DIR/projector_ckpt/projector_kv.pt" \
  --max_kv_tokens 256 \
  --max_chunks 200 \
  --retrieval_encoder_model "$DOMAIN_ENCODER"
```

### 6) （可选）训练 Gate（输入为 DomainEncoder(query) embedding，demo 伪标签）

```bash
python external_kv_injection/scripts/train_gate_query.py \
  --kv_dir "$WORK_DIR/kvbank_projector" \
  --out "$WORK_DIR/gate_query.pt"
```

### 7) 注入推理验证（DomainEncoder 用于检索+gate 输入）

```bash
python external_kv_injection/scripts/run_qwen_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_projector" \
  --prompt "请根据知识库内容回答：SFTSV 的主要传播途径是什么？并给出依据。" \
  --layers 0,1,2,3 \
  --top_k 4 \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --gate_ckpt "$WORK_DIR/gate_query.pt" \
  --gate_mode scale_v \
  --max_new_tokens 128
```

## Step 0：准备 ChunkStore
从 PDF 构建 chunkstore：
- `python external_kv_injection/scripts/build_chunkstore_from_pdfs.py --pdf_dir /path/to/pdfs --out external_kv_injection/_work_train/chunkstore.jsonl --dataset_version v0`

## Step 1：构建 teacher KV dataset（监督信号）
用冻结 base model 对每条 chunk forward，抽取 layers=[0..3] 的 `past_key_values` 作为 teacher：

- `python external_kv_injection/scripts/build_teacher_kv_dataset.py \`
  `--chunkstore external_kv_injection/_work_train/chunkstore.jsonl \`
  `--out external_kv_injection/_work_train/teacher_kv_dataset.pt \`
  `--model Qwen/Qwen2.5-7B-Instruct \`
  `--layers 0,1,2,3 \`
  `--max_kv_tokens 256 \`
  `--max_samples 200`

产物：
- `teacher_kv_dataset.pt`（list of samples，包含 teacher_k/teacher_v）

## Step 2：训练 KVProjector（不训练 LLM）

- `python external_kv_injection/scripts/train_projector_kv.py \`
  `--dataset external_kv_injection/_work_train/teacher_kv_dataset.pt \`
  `--model Qwen/Qwen2.5-7B-Instruct \`
  `--out_dir external_kv_injection/_work_train/projector_ckpt \`
  `--batch_size 1 --lr 1e-4 --epochs 1`

产物：
- `external_kv_injection/_work_train/projector_ckpt/projector_kv.pt`

## Step 3：用 Projector 构建 KVBank（不再跑 teacher cache）

- `python external_kv_injection/scripts/build_kvbank_with_projector.py \`
  `--chunkstore external_kv_injection/_work_train/chunkstore.jsonl \`
  `--out_dir external_kv_injection/_work_train/kvbank_projector \`
  `--base_model Qwen/Qwen2.5-7B-Instruct \`
  `--projector_ckpt external_kv_injection/_work_train/projector_ckpt/projector_kv.pt \`
  `--max_kv_tokens 256 \`
  `--max_chunks 200`

## Step 4：推理回归测试（必须做）

### A) 注入关闭回归
- `inject.enabled=false` 或不传 `past_key_values`，输出应与原模型一致（固定 seed/禁采样）。

### B) γ=0 回归（gate 路线）
- 当引入 gate 后：γ=0 必须等价于不注入。

### C) 延迟与显存
- 统计：检索耗时、注入开销、tokens/s；top_k 与 max_kv_tokens 是主要控制旋钮。

## Gate（γ）：用 query embedding 驱动（当前实现：demo 级）

### 推理期输入
- 推荐：`q = DomainEncoder(query)` embedding（与检索空间一致）。
- 兼容：`q = pooled last_hidden`（仅当 KVBank 的 retrieval_keys 也是用同语义构建时才建议使用）。

### 输出
- `gamma ∈ [0, clamp_max]`（默认 clamp_max=0.1）

### demo 级注入控制方式（不改写 attention 的工程近似）
- `scale_v`：只缩放外部 `V_ext *= gamma`（近似控制外部贡献强度）
- `onoff`：当 gamma 很小直接不注入（快速 AB）

### 训练（可选）
- 当前提供无标注的伪标签训练脚本：`external_kv_injection/scripts/train_gate_query.py`
- 后续可升级为带标注监督：哪些 query 需要外部知识（或用 QA 质量/引用正确率作间接信号）

## DomainEncoder 一致性（非常重要）
如果你选择用 `DomainEncoder(query)` 作为 gate 输入，建议同时让 **KVBank 的 retrieval_keys** 也来自同一个 DomainEncoder，
否则检索与 gate 所依据的“语义空间”不同，会导致：
- 检索命中质量下降（Recall@k 变差）
- gamma 学到的规律与实际检索不一致

已支持的建库参数：
- `build_kvbank_from_chunkstore.py --retrieval_encoder_model <HF_ENCODER>`
- `build_kvbank_with_projector.py --retrieval_encoder_model <HF_ENCODER>`


