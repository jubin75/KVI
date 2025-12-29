# CODE_REVIEW.md

# ✅ Schema-First KV Injection

## 快速 Code Review Checklist（10 项）

---

## A. KV 注入边界（最重要）

### ⬜ 1. 只有 schema 进入 `inject_kv(...)`？

* 搜索关键字：`inject_kv(`
* 合格标准：

  * 只接受 schema forward 得到的 cache
  * ❌ evidence / raw / prompt 文本不得出现

### ⬜ 2. Schema 是否真的走了 forward？

* 是否存在：

```python
model.forward(schema_text, use_cache=True)
```

* 是否使用返回的 `past_key_values` 注入

---

## B. Schema 选择逻辑（避免隐性退化）

### ⬜ 3. Schema selection 是否不依赖 ANN 排序分数？

* `choose_answerable_schema()`：

  * 不读取 `score / distance`
  * 只判断 answerability

### ⬜ 4. 每 step 是否最多注入 1 个 schema？

* 检查：

```python
schema_max_selected_per_step == 1
```

* 是否标记 `used_schema_ids`

---

## C. Evidence / Raw 的职责隔离

### ⬜ 5. Evidence / raw 是否永不进入 KV 注入路径？

* 搜索：evidence / raw
* 确认：只用于 retrieval / prompt append

### ⬜ 6. grounding_retriever 是否强制执行？

* 若 evidence retrieval 为空：

  * 是否报错或明确 fallback
* 不允许静默跳过

### ⬜ 7. Evidence 与 raw 的 prompt append 是否语义分区？

* 是否区分：

  * 【证据句】
  * 【回退上下文（raw）}

---

## D. Multistep 行为可解释性

### ⬜ 8. StepDebug 是否记录 schema 选择原因？

* `StepDebug.note` 是否包含：

  * answerability 命中原因
  * schema block_id

### ⬜ 9. 是否防止 schema KV 重复注入？

* 是否存在：

```python
used_schema_ids.add(schema_id)
```

* step > 1 时是否跳过已注入 schema

---

## E. 架构防回退（未来安全）

### ⬜ 10. 是否存在明确的禁止性注释 / assert？

至少满足一个：

* 注释：

  > “Schema text must NOT be concatenated into prompt.”
* 或 assert：

```python
assert schema_text not in prompt
```

---

## 使用说明

* **PR 评审**：任一 ❌ 红旗 → 直接要求修改
* **日常自检**：10 项 ≥ 9 项通过 → 架构安全
* **新人接手代码**：这份 checklist 就是隐性架构文档

---

## 架构级确认

> 如果这 10 条长期成立，你的系统不会再退化为 prompt-RAG，也不会被 evidence 绑死生成能力。
