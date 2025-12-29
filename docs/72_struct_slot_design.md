# 结构化槽位（Evidence → Schema → Generation）设计说明

> 本文档用于指导 **KV-injection / evidence-first LLM 系统** 的代码改造，目标是在保持证据严格性的前提下，恢复回答的层次性、完整性与表达能力。

---

## 1. 问题背景

当前系统采用 **evidence-first KV 注入策略**：

```
PDF → raw_chunks → blocks
             ↘ evidence blocks → KVBank_evidence
```

在推理阶段直接将 **extractive evidence 原句** 注入 base LLM attention 层，虽然：

* 保证了事实对齐
* 降低了噪声

但同时引发了以下问题：

1. 回答中出现**重复句式 / 语义回环**
2. 回答内容**明显比 baseline LLM 简单**
3. 发病机制、解释性内容难以展开

根因在于：

> **Evidence 被直接当作“答案文本”使用，而不是“约束条件”使用。**

---

## 2. 总体改造思路

将当前逻辑：

```
Evidence sentence → KV injection → generation
```

升级为：

```
Evidence sentence
   ↓
Evidence → Schema（结构化槽位）
   ↓
Schema-based KV injection（抽象语义）
   ↓
Schema-guided generation（允许扩展）
```

**核心原则：**

* Evidence 只负责“填槽”
* Schema 才是注入与生成的中介表示
* 自然语言生成不再直接复述 evidence 原句

---

## 3. 结构化槽位（Evidence Schema）设计

### 3.1 最小可行 Schema

以传染病领域为例：

```python
class EvidenceSchema:
    transmission_primary: List[str]      # 已确认的主要传播途径
    transmission_secondary: List[str]    # 次要/有限证据传播途径
    vector: List[str]                    # 已确认媒介
    pathogenesis_notes: List[str]         # 与发病机制相关的证据线索
```

### 3.2 Schema 设计原则

1. **中间表示，不是答案**
2. 字段语义稳定，可扩展
3. 支持多 evidence 累积、去重
4. 不要求完整覆盖所有问题维度

---

## 4. Evidence → Schema 填槽逻辑

### 4.1 函数定义

```python
def build_schema_from_evidence(selected_evidence_blocks) -> EvidenceSchema:
    ...
```

### 4.2 行为约束

* 只做：

  * 抽取（extract）
  * 归一化（normalize）
  * 去重（deduplicate）
* 不生成完整自然语言句子
* 不引入 evidence 中不存在的新事实

### 4.3 示例逻辑

```python
if "tick" in evidence.text.lower():
    schema.transmission_primary.append("tick bite")

if "Haemaphysalis" in evidence.text:
    schema.vector.append("Haemaphysalis longicornis")
```

---

## 5. 基于 Schema 的 KV 注入

### 5.1 禁止的做法（现有问题源头）

```python
inject_kv(text=evidence_sentence)  # ❌ 不允许
```

### 5.2 推荐做法

```python
inject_kv(
    text="Confirmed evidence: primary transmission = tick bite; vector = Haemaphysalis longicornis"
)
```

### 5.3 注入文本设计原则

* 摘要化
* 抽象化
* 去表面重复
* 只表达“已确认事实约束”

> 注：**当前阶段实现**是将“被选中的 schema block 原文”直接送入 base LLM forward，获取对应的 attention K/V cache 后再注入指定层；将 evidence/规则填槽后再“编译”为更短更稳的 schema KV（slot-compiled schema KV）属于后续可选的性能/稳定性优化。

---

## 6. Layer-wise 注入衰减（Injection Decay）

### 6.1 设计目的

* 低层：强制事实对齐
* 高层：释放叙事与组织能力

### 6.2 推荐配置

```python
injection_decay = {
    0: 1.0,
    1: 0.8,
    2: 0.5,
    3: 0.3
}
```

---

## 7. 语义级反重复机制

### 7.1 问题说明

`no_repeat_ngram_size` 只能约束 token 表面重复，
无法避免语义级复述。

### 7.2 解决方案

* 对连续生成句子计算 embedding cosine similarity
* 若相似度 > 0.9：

  * 对相关 token logits 施加 penalty
  * 而不是 hard stop

---

## 8. Schema-Guided Generation 约束规则

生成阶段应隐式遵守以下规则（system / hidden prompt）：

```
You must respect confirmed evidence slots.
You may elaborate mechanisms and explanations using general biomedical knowledge.
Do NOT contradict confirmed evidence.
Absence of evidence does not imply negation.
```

其效果是：

* 允许模型补充发病机制、免疫反应等解释性内容
* 同时不编造传播途径等关键事实

---

## 9. 设计收益总结

该结构化槽位方案能够：

1. 从根本上减少重复生成
2. 恢复 baseline LLM 的层次化表达能力
3. 保留 evidence-first 的学术严谨性
4. 为后续领域扩展（肿瘤、药物、合成路径）提供统一中间层

---

## 10. 架构定位

> **Evidence 是“约束”，不是“答案”；**
> **Schema 是 LLM 与证据之间的真正接口。**

该文档作为后续代码重构与 prompt 设计的规范依据。
