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

### 14. Context-Unaware Injection 问题（已确认 · 待方案 C 解决）

#### 14.1 问题定义

当用户查询的目标实体 **不属于** 当前加载的 topic 时，系统仍会盲目注入该 topic 的
entity priming KV 和 evidence，产生错误输出。

**实测案例**：

| 查询 | 当前 topic | 注入的 entity priming | 结果 |
|------|-----------|---------------------|------|
| "SARS2 的临床症状有哪些？" | SFTSV | ep_sftsv_name, ep_favipiravir | LLM 把 SFTSV 的症状嫁接到 SARS2 上，完全错误 |

**根因**：当前架构中 entity priming 和 evidence retrieval 都在 **单 topic 上下文** 中执行，
没有 **query-topic 相关性门控**。系统不会检查"用户问的是不是这个 topic 的内容"。

#### 14.2 为什么 Stage 0 / Stage 1 无法解决

| 架构 | 问题 |
|------|------|
| Stage 0（tag routing） | Tag + ANN 在当前 topic 的 sentence pool 里搜索，无法判断 query 实体是否匹配 topic |
| Stage 1（互补注入） | Entity priming 无条件注入当前 topic 的术语锚点，不关心 query 实体 |

即使加一个"topic 名称匹配"的硬规则，也无法处理实体别名、近义词、跨 topic 关联等复杂场景。

#### 14.3 方案 C (GraphRAG) 的自然解决

在 Graph-based Knowledge Index 中，检索的第一步就是 **Entity Recognition**：

```
Query: "SARS2 的临床症状有哪些？"
  ↓
Entity recognition: SARS2
  ↓
Graph lookup: SARS2 ∉ graph nodes → 没有任何匹配的 evidence
  ↓
Result: "当前知识库不包含 SARS2 的相关信息。"
```

- **Entity 锚定检索**：只有 query 中的实体命中 graph 中的 node 时，才会触发 graph walk 和 evidence 收集
- **不命中 = 不注入**：如果实体不在 graph 中，则不会检索任何 evidence，也不会注入任何 KV
- **跨 topic 扩展**：多个 topic 的 knowledge graph 可以合并为一个统一 graph，query 中的实体自动路由到正确的 topic 子图

这是 GraphRAG 相对于 flat tag routing 的结构性优势：**实体是检索的入口，不是 topic 是检索的入口**。

### 15. 三元 KVI 理论与实验修正

GPT 专家提出的"三元 KVI"在 attention / KV Bank 层面有三个核心主张，
经我们的实验验证后做出以下采纳/修正决策：

| 主张 | 核心思想 | 决策 | 理由 |
|------|---------|------|------|
| **Subject Anchoring (Key 侧)** | 将 subject 映射为 Key 偏置向量，提升 subject token 显著性 | **修正** → prompt 级实体上下文 | 多轮实验证明：任何 KV prefix 注入在 RAG 证据充足时均造成 token 腐蚀（"血小板"→"血板"）。改用 prompt 级 entity context 替代 |
| **Relation as Attention Routing** | Relation embedding 调制 cross-token attention，引导信息在 subject-object 之间流动 | **部分采纳** → 关系引导检索，非引导 attention | 修改 attention forward 违反铁律。改为：relation type 决定 graph walk 方向（检索路由），而非 attention head 激活 |
| **Triple-structured knowledge** | (s, r, o) 三元组结构化知识表达 | **完全采纳** | 这是 Scheme C 的基础——替代 flat sentence + tag 的平面结构 |

> **关键实验结论**：Relation 的价值在于 **检索路由**（决定哪些 evidence 进入 prompt），
> 而非 **attention 路由**（调制哪些 head 被激活）。
> 前者在 RAG 架构中可工程化实现；后者需要修改模型权重或 attention forward。

### 16. 对当前系统的替换关系

```
当前:   Query → intent tag → ANN + soft_bonus → sentences → KV inject (重复/干扰)
方案C:  Query → entity recognition → graph walk (relation-typed) → sentences → RAG prompt
                                                                              ↑
                                                            prompt-level entity context
                                                            (替代 KV 注入的 subject anchoring)
```

替换的组件：
- `semantic_type_specs.json` + `soft_filter` → **graph traversal (relation-typed)**
- `annotate_sentences_semantic_tags.py` → **triple extraction (`extract_triples.py`)**
- Entity priming KV injection → **prompt-level entity context（`graph_retriever._build_entity_context`）**
- **Context-Unaware Injection 问题** → **entity-anchored retrieval（实体不命中 = 不检索 = 不注入）**
- 保留：grounding filter、Mode A/B 行为约束、base LLM 管线

### 17. Scheme C 模块结构

```
src/graph/
  schema.py             # Triple, Entity, GraphNode, KnowledgeGraphIndex
  triple_extractor.py   # LLM-based (s,r,o) 抽取
  knowledge_graph.py    # Graph 构建 + entity index
  graph_retriever.py    # 运行时 entity recognition + graph walk + evidence 收集

scripts/
  extract_triples.py        # CLI: sentences.jsonl → triples.jsonl
  build_knowledge_graph.py  # CLI: triples.jsonl → graph_index.json
```

### 18. 编译时流程

```
sentences.jsonl
      ↓
  extract_triples.py (base LLM structured extraction)
      ↓
  triples.jsonl  (每条: subject, predicate, object, provenance)
      ↓
  build_knowledge_graph.py
      ↓
  graph_index.json  (nodes + edges + entity_index)
      ↓
  [可选] aliases.jsonl → 补充实体别名映射
```

### 19. 运行时流程

```
Query: "SFTSV的临床症状有哪些？"
  ↓
Entity Recognition: SFTSV → node_0001
Intent: symptom → relations = [causes, manifests_as]
  ↓
Graph Walk: node_0001 --causes--> 发热, 血小板减少, ...
            node_0001 --manifests_as--> 乏力, 食欲不振, ...
  ↓
Collect provenance sentences → RAG prompt evidence
Build entity context → "SFTSV（发热伴血小板减少综合征病毒）：属于新型布尼亚病毒属。"
  ↓
Prompt = entity_context + evidence_block + 要求
  ↓
LLM Generation (无 KV 注入) → Grounding Filter → Output
```

### 20. 渐进迁移路径

```
Stage 0 (当前): tag + soft_bonus routing + 同内容 KV/RAG
                ⚠ Context-Unaware: 不检查 query 实体是否属于当前 topic
                ⚠ KV 注入在 RAG 充足时造成 token 腐蚀
Stage 1 (已验证): tag routing + 互补 KV (entity priming) + RAG sufficiency gate
                ⚠ Context-Unaware 仍然存在
                ✓ RAG sufficiency gate 避免了非 definition 查询的 KV 干扰
Stage 2 (进行中): triple extraction + graph index + entity-anchored retrieval
                → 解决 Context-Unaware（entity 不命中 = 不检索）
                → prompt-level entity context 替代 KV 注入
Stage 3 (方案C): 完整 graph retrieval + 多 topic 统一 graph + 无 tag
                → 彻底解决 Context-Unaware + 跨 topic 实体路由
```

---

## Part V — 校验规则与禁止设计（继承自 v1）

### 17. 编译期强校验

- `primary_axis ∉ allowed set` → 拒绝
- `role_axis ∉ allowed set` → 拒绝
- `role_axis` 与 `constraints` 不匹配 → 拒绝
- 同一 Evidence 同时标记 `reasoning_input` 与 `grounding_only` → 拒绝

### 18. 明确禁止的设计

- ❌ 在 signature 中引入 intent / task / question
- ❌ role_axis 与 primary_axis 语义重叠
- ❌ 运行时基于 evidence 内容推断 role
- ❌ KV 注入与 RAG prompt 注入相同内容（v2 新增）
- ❌ 用 KV 注入抽象行为约束（如"不得推测"）
- ❌ 不检查 query 实体与 topic 的匹配即注入 KV / evidence（v2 新增 · Context-Unaware 禁令）

### 19. 一句话总结

> Signature 声明"语义位置 + 行为许可"；
> RAG 传递 evidence 事实；KV 注入传递互补信号（实体锚定 + 跨证据关系）；
> 检索以实体为锚点，不以 topic 为锚点；
> 三者分工明确，不重叠，不干扰，不盲注。
