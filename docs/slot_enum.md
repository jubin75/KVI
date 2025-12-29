# Slot Enum 定义（全局语义槽位规范）

> 本文档是 schema-first KV-injection 系统的**权威语义契约**。  
> 所有 slot-aware 选择、stop-rule、输出结构化都必须遵循本规范。

---

## 1. Slot Enum（全局固定枚举）

```python
SCHEMA_SLOT_ENUM = (
    "transmission",       # 传播途径
    "pathogenesis",       # 发病机制
    "clinical_features",  # 临床表现
    "diagnosis",          # 诊断
    "treatment",          # 治疗
)
```

### 规则

- Slot 是**语义维度**，不是文本片段
- Slot enum 全局固定，**只能追加，不可修改/删除**
- Topic 只能**启用** enum 子集，不可自定义新 slot

---

## 2. Schema Block JSON 规范

每条 schema block 必须包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `block_id` | string | 唯一标识（建议 `<doc_id>::schema`） |
| `text` | string | schema 文本（用于 forward → KV cache） |
| `slots` | list[string] | 覆盖的 slot 标识符（来自全局 enum） |

### 示例

```json
{
  "block_id": "10.1186_s12985-024-02387-x::schema",
  "doc_id": "10.1186_s12985-024-02387-x",
  "text": "Confirmed evidence: primary transmission = tick bite; vector = Haemaphysalis longicornis; human-to-human via body fluid contact documented.",
  "slots": ["transmission"],
  "token_count": 42
}
```

---

## 3. Slot-Aware 选择规则

### 3.1 选择条件（严格）

一个 schema 可被选择 **当且仅当**：

```
slots(schema) ∩ (required_slots − answered_slots) ≠ ∅
```

- `required_slots`：上游给定（或从 query 推断）
- `answered_slots`：已注入 schema 累积覆盖的 slots

### 3.2 次级排序

- ANN 相似度/overlap **仅用于召回池内排序**
- **不可覆盖** slot-based gating

### 3.3 无候选时

如果严格 slot gating 下无 schema 满足覆盖 → **返回空 list，停止注入**

---

## 4. Stop-Rule（多步注入终止条件）

满足以下**任一**条件时必须停止：

1. **no selectable schema**：无 schema 可覆盖剩余 required_slots
2. **no new slots**：选中的 schema 不引入新 slot
3. **redundancy_hits > 0**：检测到冗余
4. **logit_delta_vs_zero_prefix < ε**（默认 0.05）：注入无语义增益

停止原因**必须**写入 `StepDebug.note`。

---

## 5. 禁止事项（硬约束）

| 禁止行为 | 说明 |
|----------|------|
| ❌ 把 schema text 拼到 prompt | schema 只能 forward → cache 注入 |
| ❌ 注入 evidence/raw KV | evidence/raw 只能 append prompt grounding |
| ❌ 对未覆盖 slot 生成答案 | 必须写"证据未提及" |
| ❌ 把 slot 当作 topic label | slot 是语义维度，topic 是库分区 |
| ❌ 修改/删除已有 slot | enum 只能追加 |

---

## 6. Generation 规则（Slot → Answer 映射）

- 最终答案**必须**按 slot 结构化
- 只允许回答**被注入 schema 覆盖的 slot**
- 未覆盖 slot 必须明确写"证据未提及"
- evidence/raw 可 append prompt 做 grounding，但**不可注入 KV**

---

## 7. 架构原则（不可协商）

> **Slots 定义"什么可以被回答"。**  
> **Schemas 定义"什么是已知的"。**  
> **Evidence 定义"为什么可信"。**

