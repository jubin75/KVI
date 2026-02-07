# Knowledge Signature & Complementary Injection Architecture (v2)

> v2 重写说明：v1 的 Signature Schema 提出了 primary_axis / role_axis 的结构分离，
> 但在实际运行中暴露了三个根本缺陷：
> 1. Tag 分类体系依赖人工预设，无法跨 topic 复用
> 2. Flat sentence 丢失了 evidence 之间的关系结构
> 3. KV 注入与 RAG 注入相同内容导致 attention 竞争
>
> v2 保留 v1 的 Signature 不变式，但在此基础上引入 **互补注入（Complementary Injection）**
> 和 **Graph-based Knowledge Index** 两层升级。

---

## Part I — Signature Schema（保留自 v1）

### 1. 设计不变式

- Signature 只描述"语义维度 + 使用权限"，不描述问题类型、推理策略、具体实体
- Signature 在编译期 / Authoring Layer 确定，运行时禁止修改
- `role_axis` 决定 Evidence 的"可用行为"——不是"内容是什么"，是"系统可以用它来做什么"

### 2. Signature JSON 结构

```json
{
  "signature_id": "string",
  "primary_axis": "string",
  "role_axis": "string",
  "quality": {
    "granularity": "atomic | sentence | paragraph",
    "assertiveness": "fact | consensus | hypothesis",
    "scope": "local | general"
  },
  "constraints": {
    "allow_generation": false,
    "allow_reasoning": false,
    "allow_projection": true
  }
}
```

### 3. primary_axis / role_axis / constraints 映射

> 与 v1 一致，此处不再重复。核心规则：
> - 每条 Evidence 只能有一个 primary_axis（≤ 10 种）
> - role_axis → constraints 映射不可覆盖
> - Mode A 只允许 `reasoning_input` / `free_generation`
> - Mode B 只允许 `grounding_only` / `reference_only`

---

## Part II — KVI 尖峰信号的本质与误用分析

### 4. KV 注入的机制

KV 注入在 Transformer 的 attention 计算中插入外部 KV 向量，充当"虚拟 token"。
模型的 attention head 对这些位置产生高权重（**尖峰信号**），
在 token 预测的概率分布上产生定向拉偏——让模型更倾向于生成与注入内容语义相关的 token。

### 5. 尖峰信号的有效场景与无效场景

| 场景 | 有效性 | 原因 |
|------|--------|------|
| 注入具体事实（"法维拉韦"、"血小板减少"） | **强** | 尖峰直接影响 token 选择概率 |
| 注入实体定义（"SFTSV = 发热伴血小板减少综合征病毒"） | **强** | 锚定命名，消除幻觉 |
| 注入跨证据关系摘要 | **中** | 提供全局视角，辅助组织回答 |
| 注入抽象约束（"不得推测"） | **弱** | 约束是行为指令，attention 拉偏无法影响行为 |
| 注入与 RAG 相同的 evidence 文本 | **负面** | attention 竞争，导致 token 扭曲和语言混乱 |

### 6. 双通道干扰问题（v1 教训）

v1 实验中，Mode A 使用 RAG + KV 注入 **同一份 evidence**，产生了：

- **Token 扭曲**："单核" → "单胞"，"免疫抑制" → "免疫耐受"
- **语言混乱**："流感样症状" → "influenza-like symptoms"
- **Meta-reasoning**：模型生成"需要结合 E2"而非直接回答
- **Grounding 下降**：Mode A 的 grounding score 反而低于纯 RAG

**根因**：Prompt 中的 evidence 文本通过正常 self-attention 影响生成；
KV prefix 从浅层注入相同语义的 attention 信号。两组信号指向同一内容但向量位置不同，
导致 attention head 分裂，token 选择不稳定。

---

## Part III — 互补注入架构（Complementary Injection）

### 7. 核心原则

> **KV 注入的内容必须与 RAG prompt 中的文本互补，不得重复。**

| 通道 | 注入内容 | 利用的能力 |
|------|---------|-----------|
| RAG（prompt 文本） | 具体 evidence 句子 | 模型的文本理解和指令遵循 |
| KV 注入 | RAG 文本中 **没有** 的补充信息 | 尖峰信号的事实拉偏 |

### 8. KV 注入的三类补充信息

#### 8.1 Entity Priming（实体锚定）

编译一组"实体定义"KV cache：

```
"SFTSV 的全称是发热伴血小板减少综合征病毒，由新型布尼亚病毒引起。"
"法维拉韦（Favipiravir）是一种广谱抗 RNA 病毒药物。"
```

**作用**：消除命名幻觉。Prompt 里不需要写定义句，但 attention 空间中
模型会"看到"正确名称的尖峰，避免用参数记忆中的错误名称（如"鼠疫"、"塞卡夫病毒"）。

**编译方式**：每个 topic 维护一个 `entity_priming.jsonl`，
存放实体全称、缩写映射、关键术语解释。编译时与 evidence 分开生成独立的 KV bank。

#### 8.2 Cross-Evidence Synthesis（跨证据关系摘要）

编译一组"关系摘要"KV cache：

```
"SFTSV 感染导致发热、血小板减少，临床治疗以对症支持为主，法维拉韦为重要抗病毒选择。"
```

**作用**：提供 evidence 之间的"大图"。RAG 的各条独立 sentence 提供细节，
KV 尖峰让模型在组织回答时有全局视角，减少遗漏和重复。

**编译方式**：可由 LLM 从 evidence 集合自动生成 1-3 条摘要，人工审核后入库。

#### 8.3 Domain Terminology（领域术语锚定）

编译关键术语的 KV cache：

```
"流感样症状指发热、乏力、食欲不振等类似流感的临床表现。"
```

**作用**：防止模型在生成时替换领域术语（如"流感样症状" → "influenza-like symptoms"）。

### 9. 运行时流程

```
Query → Evidence Routing → top_k evidence sentences
                              ↓                        ↓
                         RAG prompt              KV injection
                    (evidence 原文锚定)     (entity priming + synthesis)
                              ↓                        ↓
                         ┌─────────────────────────────┐
                         │   LLM Generation (合并)      │
                         │   prompt: evidence text      │
                         │   past_key_values: priming   │
                         └──────────┬──────────────────┘
                                    ↓
                          Post-generation grounding filter
                                    ↓
                              Final output
```

### 10. 与 v1 的差异

| 维度 | v1 | v2（互补注入） |
|------|----|----|
| KV 注入内容 | evidence 原文 | entity priming + synthesis |
| RAG 内容 | evidence 原文 | evidence 原文（不变） |
| 是否重复 | 完全重复 → 干扰 | 互补 → 协同 |
| 命名幻觉 | 无法解决 | entity priming 直接消除 |
| 跨 evidence 组织 | 依赖 LLM 自由发挥 | synthesis KV 提供全局引导 |

---

## Part IV — Graph-based Knowledge Index（方案 C · 远景）

### 11. 动机

互补注入（Part III）解决了 KV/RAG 协同问题，但仍依赖人工 tag 分类。
方案 C 通过 Knowledge Graph 彻底消除 tag 依赖。

### 12. 编译时

1. **Entity Extraction**：用 LLM 从每条 evidence 抽取三元组 `(subject, predicate, object)`
   - "法维拉韦用于治疗 SFTSV" → `(法维拉韦, treats, SFTSV)`
   - "SFTSV 导致血小板减少" → `(SFTSV, causes, 血小板减少)`

2. **Graph 构建**：
   - Nodes = entities（SFTSV, 法维拉韦, 血小板减少, 发热 ...）
   - Edges = relations (causes, treats, manifests, prevents ...)
   - 每条 edge 关联原始 evidence sentence 作为 provenance
   - 每个 node 附带 entity priming text

3. **KV 编译**：
   - Evidence sentence → KV cache（保留，用于 RAG 锚定）
   - Entity priming text → KV cache（用于 KV 注入）
   - Graph 关系路径 → synthesis KV cache（自动生成）

### 13. 检索时

```
Query: "SFTSV 的临床症状有哪些？"
  ↓
Entity recognition: SFTSV
Relation type: causes / manifests
  ↓
Graph walk: SFTSV --causes--> 发热
            SFTSV --causes--> 血小板减少
            SFTSV --causes--> 恶心呕吐腹泻
            SFTSV --causes--> 血管通透性增加
  ↓
Collect provenance sentences → RAG prompt
Collect entity priming → KV injection
```

### 14. 对当前系统的替换关系

```
当前:   Query → intent tag → ANN + soft_bonus → sentences → KV inject (重复)
方案C:  Query → entity + relation → graph walk → sentences → KV inject (互补)
```

替换的组件：
- `semantic_type_specs.json` + `soft_filter` → **graph traversal**
- `annotate_sentences_semantic_tags.py` → **entity/relation extraction**
- 保留：KV injection 管线、grounding filter、Mode A/B 行为约束

### 15. 渐进迁移路径

```
Stage 0 (当前): tag + soft_bonus routing + 同内容 KV/RAG
Stage 1 (路线B): tag routing 不变 + 互补 KV (entity priming)  ← 下一步实现
Stage 2:        自动 entity extraction + 简单 graph index
Stage 3 (方案C): 完整 graph retrieval + 互补 KV + 无 tag
```

---

## Part V — 校验规则与禁止设计（继承自 v1）

### 16. 编译期强校验

- `primary_axis ∉ allowed set` → 拒绝
- `role_axis ∉ allowed set` → 拒绝
- `role_axis` 与 `constraints` 不匹配 → 拒绝
- 同一 Evidence 同时标记 `reasoning_input` 与 `grounding_only` → 拒绝

### 17. 明确禁止的设计

- ❌ 在 signature 中引入 intent / task / question
- ❌ role_axis 与 primary_axis 语义重叠
- ❌ 运行时基于 evidence 内容推断 role
- ❌ KV 注入与 RAG prompt 注入相同内容（v2 新增）
- ❌ 用 KV 注入抽象行为约束（如"不得推测"）

### 18. 一句话总结

> Signature 声明"语义位置 + 行为许可"；
> RAG 传递 evidence 事实；KV 注入传递互补信号（实体锚定 + 跨证据关系）；
> 三者分工明确，不重叠，不干扰。
