# 概览：External KV Injection（Dual-channel：Memory-level 注入 + 图检索与 prompt 证据）

本文档给出**产品级**端到端数据流与模块边界，用于指导实现与验收（不包含具体代码细节）。

## 与论文主旨对齐的设计立场

- **主旨不是**「用注入 KV **替代**长 evidence 文本中的**全部**信息」。外部 KV 承载的是**编译后的结构化信号**（如 schema / 关系型 capsule 对应的注意力兼容表示），用于**引导推理偏置与关系一致性**，而不是把整篇语料压缩进 cache 后不再依赖显式文本。
- **系统强依赖双通道**：（1）**图检索 / GraphRAG 等**在语料上取回**原文或近原文片段**，经 **prompt 侧 grounding** 提供可核对、可引用的证据；（2）在上述文本通道之外，**Memory-level** 注入将 capsule 编译得到的 **K_ext/V_ext** 写入 Transformer 注意力路径，使结构化知识与检索到的文本**共同**参与生成。
- **「Schema-only 注入」**描述的是**在线注意力侧的安全与工程约束**（哪些块的 K/V 允许进 attention），**不是**论文的唯一卖点句。可选地称 schema/capsule 侧为相对紧凑的**结构化记忆注入**；**实验表（如 Exp01/Exp02）衡量的是端到端方法对比**，不必单独等同于「记忆压缩」命题的证明。

## 端到端数据流（离线 → 在线）

### 离线构建（3 库分层）

1. **Raw blocks（回溯/扩展用）**
   - PDF/文档 → 抽取/OCR → raw_chunks(长段) → blocks(256 token) → `blocks.jsonl`
   - `blocks.jsonl` → `kvbank_blocks/`（ANN 检索向量 + K/V 存储）

2. **Evidence blocks（grounding/citation 用；在线默认不进 attention 注入）**
   - 从 raw_chunks 或 blocks 进行抽取式证据句生成 → `blocks.evidence.jsonl`
   - `blocks.evidence.jsonl` → `kvbank_evidence/`（ANN 检索向量 + K/V 存储）
   - 约束：Evidence 主要用于**检索与拼接到 prompt**（grounding / citation）。**禁止将 evidence 块的 K/V 作为外部前缀注入到 attention**（与 schema 注入路径区分，避免「用一段 KV 冒充全文证据」的歧义）。

3. **Schema blocks（结构化约束用；在线允许注入 attention 的主路径）**
   - 从 evidence 聚合/编译得到 schema / 关系型 capsule → `blocks.schema.jsonl`
   - `blocks.schema.jsonl` → `kvbank_schema/`
   - 约束：在线推理中，**允许注入 attention 的外部前缀 KV** 默认仅来自 **`kvbank_schema`（及与其同构的编译 triple / capsule 库）** 的预计算 `K_ext/V_ext`。

### 在线推理（QueryPlan → 图检索与证据融合 → prompt grounding + schema KV 注入 → 输出）

1. **问题解析（QueryPlan）**
   - 输入：用户问题 `q`
   - 输出：
     - `required_slots`：从问题推断需要裁决的 slot（多意图一次性抽出）
     - `fact_types`：用于策略判断（例如哪些类型允许领域共识、哪些必须证据）
     - `retrieval_queries`：一个全局 query + 若干 per-slot query（通常会偏英文以匹配论文 evidence）

2. **检索路由（图 / schema → evidence → raw）**
   - **图索引与 GraphRAG 式检索**：在实体与关系上做多跳扩展，筛出与 query 对齐的句子或块，**提高进入 prompt 的文本证据命中率**（与「仅扁平 ANN」相对）。
   - schema 检索：用于选择可注入的 schema / capsule KV（`kvbank_schema`）
   - evidence 检索：用于 grounding 文本候选（`kvbank_evidence`）
   - raw 检索：用于回溯/扩展（`kvbank_blocks`，可选）

3. **证据融合（文本通道）**
   - 对 `retrieval_queries` 做多路检索（global + per-slot），对 evidence hits 做 union + 去重，并与图检索结果对齐。
   - 产出：
     - `global_evidence`：**拼接到 prompt** 的 grounding 片段（主证据面）
     - `slot_evidence_map`：slot → evidence 子集（用于「该 slot 是否有证据」「如何裁决」）

4. **Schema / capsule KV 注入（Memory-level 通道）**
   - 将选中 capsule 对应的 `K_ext/V_ext` 作为静态前缀参与注意力计算，**约束关系与推理走向**；与步骤 3 的**显式文本证据并行**，二者互补而非互斥。

5. **三层输出（docs/74_three_knowledge_layers.md）**
   - **L0｜证据支持的结论**：按 slot 做裁决（证据不足则明确说明）
   - **L1｜领域共识（LLM 内部知识）**：教科书级共识，禁止补「需要证据的数据事实」（如具体地理范围）
   - **L2｜推测性或解释性补充**：可选，必须带不确定性措辞，且受安全规则限制

## 关键约束（必须满足）

- **不训练主模型参数**（可训练外部模块：encoder/projector/gate）
- **维度不变**：`K_ext/V_ext` 的 `head_dim` 必须与目标模型 attention head_dim 一致
- **KV cache 兼容**：prefill/decode 阶段注入行为一致；外部 KV 视为「静态前缀 KV」（不写入可学习 cache）
- **Schema-only attention injection**：在线向 attention 注入的外部前缀 KV **仅**来自 schema/capsule 编译路径；evidence/raw 的 K/V **不作为**该类注入源
- **Evidence 的文本角色**：evidence（及 raw）须通过 **检索 + prompt grounding** 进入模型；这是 **事实对齐与可解释引用** 的主载体之一

## 补充关键

Schema slot / capsule 不是知识的全集，而是「可裁决事实的最小结构化集合」；**全文与细节**仍由 **图检索与 prompt 证据** 承担。
