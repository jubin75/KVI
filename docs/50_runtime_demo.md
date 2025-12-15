# 运行 Demo：PDF → KVBank（真实 past_key_values）→ Qwen 注入生成

本 Demo 的目标：在“完整可跑通”的工程实现上，验证以下链路：

- PDF → chunk（ChunkStore）
- chunk →（用目标基模 forward）→ **真实 past_key_values(K/V cache)** 作为外部 KV
- 建立 FAISS 索引（retrieval_key）→ 在线检索 top-k
- 将 top-k 的 K/V 作为 **past_key_values 前缀**喂给 HF 模型，完成注入与生成

## 依赖
- Python 3.10+
- `torch`
- `transformers`
- `faiss-cpu` 或 `faiss-gpu`
- `numpy`
- `pymupdf`（PDF 抽取）

## 1) 端到端构建：PDF → ChunkStore → KVBank

运行（在 Linux `/home/jb/KVI`）：

- `external_kv_injection/scripts/build_kvbank_from_pdfs.py`

参数示例：
- `--pdf_dir /path/to/pdfs`
- `--work_dir external_kv_injection/_work_demo`
- `--model Qwen/Qwen2.5-7B-Instruct`
- `--layers 0,1,2,3`
- `--max_kv_tokens 128`
- `--max_chunks 200`

输出：
- `work_dir/chunkstore.jsonl`
- `work_dir/kvbank/`（包含 `index.faiss`、`k_ext.npy`、`v_ext.npy`、`metas.jsonl`、`manifest.json`）

## 2) 在线注入推理：加载 KVBank → 检索 top-k → 注入生成

运行（在 Linux `/home/jb/KVI`）：

- `external_kv_injection/scripts/run_qwen_inject_demo.py`

关键参数：
- `--model`：与建库时相同（确保 cache 空间一致）
- `--kv_dir`：上一步输出的 `work_dir/kvbank`
- `--prompt`：用户输入
- `--layers`：注入层集合（必须与建库 layer_ids 兼容）
- `--top_k`：检索条数

## 重要说明（为什么这个实现“更完整”）
- 本方案把外部 KV 定义为**目标基模真实产生的 past_key_values**：
  - 解决了 RoPE/kv_heads/head_dim 对齐问题（无需在 demo 阶段自己实现 rotary 对齐）
  - 让注入路径在 HF 上可直接跑通（past_key_values 前缀）
- 生产级演进：
  - 用独立 DomainEncoder 做检索 embedding（而不是用基模 pooled hidden）
  - 用 projector/gate 做可控注入（目前 demo 注入等价于 concat 前缀 KV）
  - 用 IVF/HNSW/PQ 与分片/压缩支持更大规模


