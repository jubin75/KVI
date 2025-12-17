# 多步注入（Multi-step Injection）工程实现说明（2×V100 友好）

本实现对应 `PRD/多步注入的工程实现.md` 的关键约束：

- **三层结构**：Raw Context（仅建库）→ External KV Bank（长期记忆）→ Attention Injection（工作记忆）
- **Raw context 不进 attention**：只用于构建 KVBank
- **切分规则**：4096-token chunks（overlap=256）→ 256-token memory blocks（KVBank 存 blocks）
- **注入上限**：单步 ≤1024 tokens（推荐），绝对 ≤2048；注入层默认 `0..3`
- **检索是推理的一部分**：每步基于当前状态重新检索
- **Stopping policy**：必须至少 2 个信号 + 安全上限（本实现：边际收益 + 冗余 + 上限）

## 代码入口

- Raw→blocks：`external_kv_injection/src/pipelines/raw_context_to_blocks.py`
- blocks→KVBank：`external_kv_injection/src/pipelines/blocks_to_kvbank.py`
- Multi-step runtime：`external_kv_injection/src/runtime/multistep_injector.py`
- 一键建库（raw text）：`external_kv_injection/scripts/build_kvbank_from_raw_text.py`
- 运行多步注入：`external_kv_injection/scripts/run_multistep_inject_demo.py`

## 每步注入 token 数（明确标注）
- 记忆块固定为 **256-token**（tokenizer 级别截断）
- 每步注入 `≤ max_step_tokens`（默认 1024）：
  - 约等于每步最多注入 4 个 blocks（若每块 kv_len≈256）

## stopping policy（判定点）
在 `MultiStepInjector.run()` 每一步 forward 后计算：
- **边际收益**：`logit_delta` 与 `hidden_delta`（新注入前后变化的平均绝对差）
- **注意力收敛（可选硬指标）**：external KV attention entropy（归一化到 0~1），连续下降且低于阈值
- **检索冗余**：新 step 的 query embedding 与历史 query embedding 的 cosine 相似度超过阈值（demo 近似；生产级可改为 block embedding 近似）
- **安全上限**：max_steps、max_total_tokens

当满足 **至少两个信号** 时停止继续检索注入，并进入最终回答生成阶段。

## Demo 命令（raw text → KVBank → multi-step 推理）

```bash
cd /home/jb/KVI

export RAW_TEXT="/path/to/raw.txt"
export WORK_DIR="external_kv_injection/_work_multistep"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

python external_kv_injection/scripts/build_kvbank_from_raw_text.py \
  --raw_text "$RAW_TEXT" \
  --work_dir "$WORK_DIR" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --block_tokens 256

python external_kv_injection/scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_blocks" \
  --kv_dir_tables "$WORK_DIR/kvbank_tables" \
  --enable_table_routing \
  --table_top_k 4 \
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

## 检索侧路由（Keep+Split Tables）
当你在建库阶段启用了 `--split_tables` 产生 `kvbank_tables/`，运行多步注入时可以开启检索侧路由：

- **默认只查主库**（`kvbank_blocks`），用于一般叙述/机制/指南类问题
- 当 prompt 中出现 **数值/统计/阈值/对照** 等提示（例如 AUC/OR/HR/95%CI/p-value/阈值/敏感度/特异度/表1），会 **额外查询表格库**（`kvbank_tables`）并合并结果
- 用 `--table_top_k` 控制表格候选上限（建议小一些，避免噪声淹没正文）

## 禁止项对照（已规避）
- 不把 raw context 直接注入 attention（raw 仅建库）
- 不一次性注入整库（每步≤1024，总≤2048）
- 不做 RAG prompt 拼接（注入的是 K/V cache 前缀）


