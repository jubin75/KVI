## 背景与目标

我们要解决的是：**在外部 KV 注入 / RAG 场景下，模型输出不稳定、容易被噪声块扰动、难评测难 debug**。

本设计文档定义两个“产品化”的稳定接口：

- **统一输出协议（Output Protocol）**：对用户问题千变万化，系统仍能返回可消费、可评测、可追溯的结构化输出。
- **DeepSeek 抽取式证据（Extractive Evidence）约束**：把原始文献块（raw blocks）转换为“高纯度、可核验”的证据句块（evidence blocks），降低注入噪声与语义碎片化风险。

本设计刻意把“呈现模板”和“协议”拆开：协议稳定、模板可变且由前端/后处理决定。

---

## 1）统一输出协议（Output Protocol）

### 1.1 设计原则

- **协议稳定**：不随问题类型变化（避免“每个问题一套 prompt 模版”）。
- **证据可核验**：任何关键结论都要能指向可引用的 evidence quote。
- **允许不确定**：证据不足时必须明确声明，不得编造。
- **便于 A/B 与自动评测**：结构字段固定，方便比较 baseline vs injection。
- **呈现解耦**：最终 UI 展示（段落/要点/表格）由渲染层完成，模型只输出协议结构。

### 1.2 协议结构（JSON）

下述 JSON 是**唯一**推荐的 machine-readable 输出。文本展示可以由渲染层把 JSON 渲染成自然语言。

```json
{
  "final_answer": "string (面向用户的最终答案，简洁明确)",
  "answer_type": "one_of: fact|procedure|comparison|definition|triage|other",
  "confidence": "one_of: high|medium|low",
  "key_points": [
    "string (3-7条要点，和 final_answer 一致，不新增事实)"
  ],
  "evidence_quotes": [
    {
      "quote": "string (必须逐字来自原文；可以是英文/中文)",
      "claim_supported": "string (这条quote支持的具体主张；必须可对齐到 key_points/ final_answer 的一部分)",
      "source": {
        "topic": "string (例如 sftsv)",
        "doc_id": "string",
        "block_id": "string",
        "source_uri": "string|null",
        "locator": {
          "page": "int|null",
          "char_start": "int|null",
          "char_end": "int|null"
        }
      }
    }
  ],
  "limitations": [
    "string (证据缺口/不确定性；必须具体，不要空话)"
  ],
  "safety_notes": [
    "string (医疗相关：提醒就医/不替代医生；只在需要时出现)"
  ],
  "debug": {
    "retrieval_query": "string|null",
    "selected_block_ids": ["string"],
    "injected": "boolean",
    "notes": ["string"]
  }
}
```

### 1.3 协议硬约束（必须满足）

- **JSON-only 输出**：输出必须是合法 JSON；不得混入额外自然语言/代码块/多版本尝试。
- **无编造引用**：`evidence_quotes[].quote` 必须逐字出自原文证据（raw/evidence block），不得改写、不得“翻译后当引用”。
- **主张可对齐**：每条 `claim_supported` 必须能指向 `final_answer` 或某条 `key_points`，否则算无效证据。
- **证据不足要承认**：如果 evidence 无法支持关键结论，必须把结论降级为不确定，并写入 `limitations`。
- **语言策略**：`final_answer/key_points/limitations` 用用户语言输出；`quote` 保持原文语言（可中可英）。

### 1.4 渲染层（给产品/前端的建议）

渲染层可以把 JSON 稳定渲染为：

- **简洁版**：`final_answer` + 1-2 条 `evidence_quotes`（可折叠）+ `limitations`（可折叠）
- **审计版**：全部 `key_points` + 全部 `evidence_quotes` + `debug` 信息

这样用户不需要写“输出约束 prompt”，产品也不需要频繁改模版。

### 1.5 A/B 评测建议（baseline vs injection）

对同一问题，强制两次输出都遵循协议：

- **baseline**：`debug.injected=false`
- **injection**：`debug.injected=true`，并记录 `selected_block_ids`

自动指标建议：

- **Faithfulness**：`final_answer/key_points` 中每个原子主张是否被某条 `evidence_quotes` 支持
- **Overclaim rate**：出现无法被 quote 支持的新主张比例
- **Citation quality**：quote 是否包含关键谓词/对象（例如“tick bites are the main route…”）

---

## 2）DeepSeek 抽取式证据（Extractive Evidence）约束

目标：把 raw blocks 变成可注入、低噪声、单意图的 evidence blocks，并且**最大化可核验性**。

### 2.1 核心原则：只抽取，不总结

- **必须 extractive**：输出的证据句必须逐字来自输入文本（允许最小清洗：合并断行、修正多余空格）。
- **禁止 abstractive**：不得把多个句子“总结成一句”，不得意译，不得添加原文不存在的事实。
- **单块单意图**：一个 evidence block 只回答一个原子事实/主张，避免语义混杂。

### 2.2 输入/输出契约

**输入**（给 DeepSeek）：

- `user_query`：用户问题（原样）
- `topic_goal`：专题目标（例如 SFTSV）
- `raw_block_text`：来自论文/文档的原始段落（可能有噪声、断行）
- `source_meta`：doc_id / block_id / source_uri / page 等（如有）

**输出**（DeepSeek 必须返回 JSON-only）：

```json
{
  "keep": true,
  "evidence_sentences": [
    {
      "quote": "string (逐字证据句，来自 raw_block_text)",
      "relevance": "one_of: direct|supporting|background",
      "claim": "string (这句支持的原子主张，必须可直接回答 user_query 的某个部分)",
      "span": { "char_start": 0, "char_end": 0 }
    }
  ],
  "reject_reason": "string|null"
}
```

### 2.3 DeepSeek 约束清单（必须写进 system/prompt）

- **必须 JSON-only**：不得输出解释文字、不得输出 Markdown、不得输出多版本。
- **quote 必须逐字**：`quote` 必须是 `raw_block_text` 的子串（允许空格/换行正规化后匹配）。
- **必须给 span**：`char_start/char_end` 指向原文位置，便于程序校验与追溯。
- **最多 3 句**：每个 raw block 最多抽取 1-3 句，且优先 `direct`。
- **拒绝机制**：如果找不到可直接支持 user_query 的句子，`keep=false` 并填 `reject_reason`（例如“only discusses genetics, not transmission route”）。
- **医学安全**：不得生成诊疗建议；只做“证据抽取/是否相关”。

### 2.4 推荐的抽取 prompt（模板示例）

下面是“抽取式证据”的**通用**指令模板（供实现时使用；不是让用户手写）：

```text
System:
You are an information extraction engine. You must output JSON only.
You MUST NOT paraphrase or summarize. Only copy exact substrings from the provided text.

User:
user_query: {USER_QUERY}
topic_goal: {TOPIC_GOAL}
raw_block_text:
{RAW_BLOCK_TEXT}

Task:
Select up to 3 exact evidence sentences from raw_block_text that directly answer user_query (or directly support the answer).
Return JSON with spans (char_start, char_end) pointing to the exact substring in raw_block_text.
If none, return keep=false with a short reject_reason.
```

### 2.5 自动校验（强烈建议）

在落盘为 evidence blocks 前做强校验：

- **substring 校验**：正规化空白后，`quote` 必须能在 `raw_block_text` 中找到
- **长度阈值**：过滤过短 quote（例如 < 40 chars）与“只有数字/表格” quote
- **语言过滤（可选）**：按 topic/用户语言策略选择 `zh/en`，避免日文等

### 2.6 evidence blocks 的落盘格式（用于重建 KVBank）

建议生成新的 `blocks.evidence.jsonl`，每行一个 evidence block：

```json
{
  "block_id": "doc_id::raw_block_id::ev0",
  "doc_id": "10.1186_s12985-024-02387-x",
  "source_uri": null,
  "text": "Ticks have been confirmed to be the main vector of transmission for SFTSV.",
  "meta": {
    "from_raw_block_id": "10.1186_s12985-024-02387-x_chunk1_t3840-5121_t768-1024",
    "span": { "char_start": 0, "char_end": 0 },
    "relevance": "direct",
    "claim": "SFTSV main transmission vector is ticks"
  }
}
```

注：这样一个 evidence block 语义更纯，注入噪声更小；raw blocks 仍保留用于追溯与 debug。

---

## 3）为什么这两件事能解决你当前痛点

- **“统一输出协议”**解决：用户问题千变万化时，输出仍可消费/可评测/可追溯，不依赖用户写 prompt 模版。
- **“抽取式证据块”**解决：KV 注入被噪声放大、语义碎片化干扰 attention 的问题，让注入更像“短证据前缀”而不是“随机背景记忆”。

下一阶段实现建议（不在本文展开代码细节）：

- 构建 evidence blocks → 重建 topic KVBank（evidence 版本）
- injection 默认优先 evidence KVBank；必要时回退 raw KVBank
- 评测集对同一 query 输出协议 JSON，做 faithfulness/overclaim 指标


