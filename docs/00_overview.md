# 概览：External KV Injection（Schema-first / Evidence-grounded）

本文档给出**产品级**端到端数据流与模块边界，用于指导实现与验收（不包含具体代码细节）。

## 端到端数据流（离线 → 在线）

### 离线构建（3 库分层）

1. **Raw blocks（回溯/扩展用）**
   - PDF/文档 → 抽取/OCR → raw_chunks(长段) → blocks(256 token) → `blocks.jsonl`
   - `blocks.jsonl` → `kvbank_blocks/`（ANN 检索向量 + K/V 存储）

2. **Evidence blocks（grounding/citation 用，永不注入）**
   - 从 raw_chunks 或 blocks 进行抽取式证据句生成 → `blocks.evidence.jsonl`
   - `blocks.evidence.jsonl` → `kvbank_evidence/`（ANN 检索向量 + K/V 存储）
   - 约束：Evidence 只允许用于**检索/拼接到 prompt**做 grounding/citation，**禁止注入到 attention KV**。

3. **Schema blocks（约束用，唯一允许注入）**
   - 从 evidence 聚合/编译得到 schema → `blocks.schema.jsonl`
   - `blocks.schema.jsonl` → `kvbank_schema/`
   - 约束：在线推理中**只允许注入 schema KV**（来自 `kvbank_schema` 的预计算 K/V）。

### 在线推理（QueryPlan → fusion retrieval → schema KV injection → 三层输出）

1. **问题解析（QueryPlan）**
   - 输入：用户问题 `q`
   - 输出：
     - `required_slots`：从问题推断需要裁决的 slot（多意图一次性抽出）
     - `fact_types`：用于策略判断（例如哪些类型允许领域共识、哪些必须证据）
     - `retrieval_queries`：一个全局 query + 若干 per-slot query（通常会偏英文以匹配论文 evidence）

2. **检索路由（schema → evidence → raw）**
   - schema 检索：用于选择可注入的 schema KV（`kvbank_schema`）
   - evidence 检索：用于 grounding（`kvbank_evidence`）
   - raw 检索：用于回溯/扩展（`kvbank_blocks`，可选；不进入注入）

3. **Evidence fusion（一次性到位）**
   - 对 `retrieval_queries` 做多路检索（global + per-slot），对 evidence hits 做 union + 去重
   - 产出：
     - `global_evidence`：用于（可选）拼接到 prompt 的 grounding 片段
     - `slot_evidence_map`：slot → evidence 子集（用于“该 slot 是否有证据”“L0 如何裁决”）

4. **Schema KV 注入（约束信号）**
   - 只允许注入 `kvbank_schema` 返回的 `K_ext/V_ext`
   - evidence/raw 的 K/V **永不注入**

5. **三层输出（docs/74_three_knowledge_layers.md）**
   - **L0｜证据支持的结论**：按 slot 做裁决（证据不足则明确说明）
   - **L1｜领域共识（LLM 内部知识）**：教科书级共识，禁止补“需要证据的数据事实”（如具体地理范围）
   - **L2｜推测性或解释性补充**：可选，必须带不确定性措辞，且受安全规则限制

## 关键约束（必须满足）

- **不训练主模型参数**（可训练外部模块：encoder/projector/gate）
- **维度不变**：`K_ext/V_ext` 的 `head_dim` 必须与目标模型 attention head_dim 一致
- **KV cache 兼容**：prefill/decode 阶段注入行为一致；外部 KV 视为“静态前缀 KV”（不写入 cache）
- **Schema-only injection**：在线推理中，**schema KV 是唯一允许注入的 cache**
- **Evidence is retrieval-only**：evidence/raw 只允许用于检索与 grounding/citation，禁止作为 KV 注入

## 补充关键

Schema slot 不是知识的全集，而是“可裁决事实的最小集合”

