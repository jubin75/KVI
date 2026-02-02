# KVI vs 传统 RAG

## KVI 的主要优点（相对 RAG）
1. 更强的“局部对齐/约束”能力：把证据写进 KV cache（attention 侧），模型在生成时更“被迫看到”这些内容；RAG 只是把证据拼到 prompt 末尾，容易被忽略或被覆盖。
多步注入可做“逐轮修正”：KVI 的 multi-step injection 能根据上一轮生成动态改写检索/query，适合复杂问答；RAG 多数是一次检索+一次生成。
2. 可做更严格的“槽位/语义维度控制”：你们的 schema-first / slot-aware、语义类型路由、list_feature 排序等属于“可工程化的可控对齐”；传统 RAG 通常靠 prompt 约束，粒度更粗。

### “意图 tag + 可控检索”的具体设计（用于解释上面的“槽位/语义维度控制”）
在 KVI（KV 注入）里，**一旦检索命中噪声并被注入**，模型会被强引导到错误方向；因此需要把“检索侧的意图/维度控制”做成工程化能力，而不是依赖 base LLM 在回答阶段自我纠错。

这里给出一个和本工程实现一致的、可落地的设计：

#### 2.1 建库阶段：对每条 sentence 做语义意图识别（离线）
- **输入**：`sentences.jsonl`（每条 sentence 一行，`text` 为知识句）
- **配置**：`semantic_type_specs.json`（可编辑的意图 taxonomy）
  - 每个 type 至少包含：
    - `description`：该维度的自然语言定义
    - `threshold`：该维度的判定阈值（embedding 相似度）
- **做法（不使用 base LLM）**：
  - 用检索 encoder（如 `sentence-transformers/all-MiniLM-L6-v2`）对每个 type 的 anchor（`description`）编码
  - 对每条 sentence 编码，得到对每个 type 的相似度 `semantic_scores[type]`
  - 生成 `semantic_tags`（>=threshold 的 type；若全不满足，保留 top1 作为兜底）
- **落盘**：把 tag/score 写进 sentence 的 metadata（跟随 KVBank metas 一起存）
  - `metadata.semantic_scores: {type: score}`
  - `metadata.semantic_tags: [type...]`
  - `metadata.semantic_primary: type`

这样做的意义是：**意图体系从“代码硬编码”迁移到“配置 + 数据”**，新增/调整维度不需要改 runtime 代码，只需要更新 `semantic_type_specs.json` 并重新编译 sentence-kvbank。

#### 2.2 在线检索阶段：用 intent tag 做 rerank / filter（不依赖 base LLM）
- **Query→Intent**：同样用 encoder + `semantic_type_specs.json` 做“意图推断”（选 top1 type + debug 相似度）
- **候选池**：ANN 先取较大的候选（例如 top_k*3 或 top_k*10）
- **重排策略**：
  - 优先使用“建库时写入”的 `semantic_scores[target_type]` 作为 rerank 分数（稳定、可解释、低时延）
  - 若缺失 tag（兼容老库），再回退到在线计算“query anchor vs sentence text”相似度
  - 最终排序：`(intent_match_score, ann_score)` 组合排序

#### 2.3 “检测违规→再注入/再生成一轮”：第二轮只加强检索约束，不让模型补知识
当检测到回答存在明显违规（例如缩写/别名括号扩写不在证据中、维度漂移等）时，第二轮的改进点应放在检索侧：
- 使用更严格的 intent filter（例如要求 `semantic_scores[target_type] >= threshold`）
- 扩大候选池后再 rerank（提高找到正确 sentence 的概率）
- 使用违规类型作为负信号（例如“机制问题出现症状漂移”则对 symptom 维度强降权）

关键原则：**第二轮不依赖 base LLM 生成新的“改写 query / 补全症状”，而是通过配置化、可解释的检索约束把正确的 sentence 注入进去。**
更容易做“补漏洞”形态的产品策略：当识别到高风险/高价值点时注入；否则走 base LLM（你们已在 simple pipeline 里验证这个方向的可行性）。
3. 更容易把“证据-结论”绑定到可审计协议：Evidence Units + 引用后处理能把“哪些句子由哪些证据支持”做得更硬，方便专家审阅与合规评估。

## KVI 的主要缺点/成本（相对 RAG）
1. 工程复杂度与调参面更大：KV 注入、投影、gate、pattern/contract、list_feature、cleaner、multi-step 交互，任何一环错都会放大退化；RAG 的链路更短更稳。
2. 失败模式更“隐蔽且破坏性强”：一旦检索命中噪声并注入，模型会被强引导到错误方向；RAG 至少还能“看起来像在引用”，但 KVI 更可能把噪声当真并扩写。
3. 可移植性/可解释性挑战：KV 注入对不同模型架构、cache实现、tokenization敏感；RAG 基本模型无关。
4.算力/时延与系统边界更难控：多步注入 + 可能的多轮检索/重排会增加时延；RAG 常见形态更容易做低时延优化。
5. 评测更难：KVI 既要评测检索，又要评测注入有效性与“注入副作用”（overclaim/语义污染）；RAG 的评测维度更标准化。

## 什么时候 KVI 更合适？什么时候 RAG 更合适？
1. KVI 更合适：高风险领域（医疗/法律）、强约束输出、需要 slot/意图路由、需要“证据绑定 + 限制幻觉”、以及你们这种“默认 base，必要时补漏洞注入”的产品策略。
2. RAG 更合适：快速落地、知识库规模大且异构、对模型无侵入、追求稳定性与可维护性、或证据只需“参考”而非强约束生成。

