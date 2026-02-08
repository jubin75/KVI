# Triple KVI — 三元知识值注入架构 (v3)

> v3 重写说明：在 v2 的互补注入 + GraphRAG 基础上，回归 KVI 的核心价值主张。
>
> **铁律**：KV 注入是本框架的核心能力，不可废弃。
>
> v2 实验证明：KV 注入失败的原因是 **实现方式**（长文本 blob、上下文无关注入、
> 双通道重复），而非 KV 注入本身。v3 将 GPT 专家的"三元 KVI"理论与 Scheme C
> 图谱架构融合，实现 **图谱引导的结构化 KV 注入**。

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

### 15. 三元 KVI 理论与实验修正（v3 更新）

GPT 专家提出的"三元 KVI"在 attention / KV Bank 层面有三个核心主张。
v2 基于 token 腐蚀实验一度放弃 KV 注入，v3 经重新分析后回归：

**v2 token 腐蚀的真正根因**（重新归因）：
- ❌ KV 注入本身 → 不是根因
- ✅ 长文本 blob（70+ tokens，中英混合）→ 是根因
- ✅ 上下文无关注入（全局注入，不管 query 是否匹配）→ 是根因
- ✅ 双通道重复（KV 和 prompt 注入相同内容）→ 是根因

| 主张 | 核心思想 | v3 决策 | 理由 |
|------|---------|--------|------|
| **Subject Anchoring (Key 侧)** | 将 subject 映射为 Key 偏置向量，提升 subject token 显著性 | **重新采纳** → 图谱引导的短三元组 KV 注入 | v2 失败的是"长 blob 全局注入"，图谱引导的极短（~15 token）纯中文 KV 可避免腐蚀 |
| **Relation as Layer Routing** | Relation 类型调制注入层位置 | **新增采纳** → RELATION_LAYER_MAP | 不修改 attention forward，而是按 relation 类型选择 KV 注入的 Transformer 层段 |
| **Relation as Retrieval Routing** | Relation 类型决定 graph walk 方向 | **完全采纳**（v2 已实现） | relation type 决定检索哪些 evidence 进入 prompt |
| **Triple-structured knowledge** | (s, r, o) 三元组结构化知识表达 | **完全采纳**（v2 已实现） | Scheme C 的基础 |

> **v3 关键原则**：KV 承载 **注意力结构约束**（短、聚焦、不重复），Prompt 承载 **详细证据内容**。
> 两个通道功能互补，绝不重叠。

### 16. 对当前系统的替换关系（v3 更新）

```
旧 Mode A: Query → intent tag → ANN → sentences → KV inject (长 blob, 重复)
新 Scheme C: Query → entity recognition → graph walk → sentences + triples
                                                          ↓              ↓
                                                     RAG prompt     三元组 KV 注入
                                                    (证据原文)   (subject anchoring + relation-layer)
                                                          ↓              ↓
                                                     ┌───────────────────────┐
                                                     │ LLM Generation (合并)  │
                                                     │ prompt: evidence text  │
                                                     │ past_kv: triple KV    │
                                                     └───────────────────────┘
```

替换的组件：
- `semantic_type_specs.json` + `soft_filter` → **graph traversal (relation-typed)**
- `annotate_sentences_semantic_tags.py` → **triple extraction (`extract_triples.py`)**
- 旧 entity priming KV（长 blob 全局注入）→ **三元组 KV 注入（短、图谱选择性、按层注入）**
- **Context-Unaware Injection 问题** → **entity-anchored retrieval（实体不命中 = 不检索 = 不注入）**
- 保留：grounding filter、base LLM 管线、KV bank 编译基础设施

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

### 19. 运行时流程（v3：含三元组 KV 注入）

```
Query: "SFTSV的临床症状有哪些？"
  ↓
Entity Recognition: SFTSV → node_0001
Intent: symptom → relations = [causes, manifests_as]
  ↓
Graph Walk: node_0001 --causes--> 发热, 血小板减少, ...
            node_0001 --manifests_as--> 乏力, 食欲不振, ...
  ↓
┌──────────────────┬────────────────────────────┐
│ KV 注入通道       │ Prompt 通道                 │
│                  │                            │
│ Subject Anchor:  │ Entity Context:            │
│ "SFTSV（发热伴血  │ "SFTSV：发热伴血小板减少..."  │
│  小板减少综合征    │                            │
│  病毒）"         │ Evidence Block:             │
│  → layers 0-3   │ 1. 患者通常以急性发热...     │
│                  │ 2. 核心的病理特征是...       │
│ Triple KV:       │                            │
│ "SFTSV导致发热"  │ 问题 + 要求                  │
│  → layers 4-7   │                            │
│ (causes关系层段)  │                            │
└──────────────────┴────────────────────────────┘
                    ↓
        LLM Generation (KV prefix + prompt)
                    ↓
              Grounding Filter
                    ↓
                 Output
```

### 20. 渐进迁移路径

```
Stage 0: tag + soft_bonus routing + 同内容 KV/RAG
         ⚠ Context-Unaware + KV 双通道干扰

Stage 1: tag routing + 互补 KV (entity priming) + RAG sufficiency gate
         ⚠ Context-Unaware 仍然存在
         ✓ RAG sufficiency gate 避免了非 definition 查询的 KV 干扰

Stage 2: triple extraction + graph index + entity-anchored retrieval
         ✓ 解决 Context-Unaware（entity 不命中 = 不检索）
         ⚠ 移除了 KV 注入（仅 prompt 级 entity context）

Stage 3 (v3 · 当前目标): 完整三元 KVI
         ✓ 图谱引导的三元组 KV 注入（Subject Anchoring + Relation Layer Routing）
         ✓ KV/Prompt 互补（KV 传注意力结构，Prompt 传证据内容）
         ✓ entity-anchored retrieval 保持 Context-Aware
         ✓ 多 topic 统一 graph 扩展就绪
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

---

## Part VI — 三元 KVI 在 Scheme C 中的集成方案（v3 核心）

### 21. 设计哲学

三元 KVI 的核心思想：**三元组 ≠ 文本，而是注意力结构约束**。

| 传统 RAG | 三元 KVI |
|----------|---------|
| 往 prompt 里塞信息，希望模型自己看懂 | 提前告诉模型：哪些概念该互相关注 |
| 所有信息都走同一通道 | KV 走结构约束通道，Prompt 走内容通道 |

### 22. 三元组 → KV 注入的拆解

对知识图谱中每条三元组 `t = (s, r, o)` 及其 subject entity：

#### 22.1 Subject Anchoring（Key 侧）

**做法**：将 subject 的定义/别名映射为一组短 KV 向量，注入到 Transformer **浅层**（layers 0-3）。

**编译源**：
```
entity.description or entity.aliases → "SFTSV（发热伴血小板减少综合征病毒）"
```

**约束**：
- 纯中文，不超过 20 tokens
- 每个 entity 最多一条 subject anchor
- 注入层 = 浅层（负责 token embedding alignment）

**效果**：模型在生成过程中更容易"反复回到 subject"，减少主题漂移。
对应 GPT 专家所说的"信号尖峰（high saliency）"。

#### 22.2 Relation as Layer Routing（Relation 侧）

**做法**：Relation 类型不作为 token 注入，而是决定三元组 KV 注入到哪些 Transformer 层段。

**RELATION_LAYER_MAP**（32 层 Transformer 为例）：

| Relation 类型 | 注入层段 | 语义理由 |
|--------------|---------|---------|
| `is_a`, `has_subtype` | layers 0-7 | 定义/分类 → 浅层 token 对齐 |
| `causes`, `manifests_as`, `leads_to` | layers 8-15 | 因果/机制 → 中层语义推理 |
| `treats`, `prevents`, `diagnosed_by` | layers 12-19 | 治疗/诊断 → 中高层行为推理 |
| `located_in`, `part_of`, `transmission_route` | layers 4-11 | 结构/位置 → 早中层空间推理 |
| 其他 / fallback | layers 0-7 | 默认注入浅层 |

> **关键**：不修改 attention forward，仅选择性地将 KV 前缀注入到对应层段，
> 其他层段对应位置的 KV 值为零或不注入。这在标准 `past_key_values` 接口中
> 通过 **per-layer masking** 实现。

#### 22.3 Triple KV（Object 侧 / 三元组整句）

**做法**：将三元组凝练为极短的中文陈述句，编译为 KV cache，按 relation type 注入对应层段。

**编译源**：
```
(SFTSV, causes, 发热) → "SFTSV导致发热"
(SFTSV, is_a, 布尼亚病毒) → "SFTSV属于布尼亚病毒"
(法维拉韦, treats, SFTSV) → "法维拉韦治疗SFTSV"
```

**约束**：
- 每条三元组 KV ≤ 15 tokens
- 纯中文，无英文术语混入
- 仅注入与 query 匹配的 entity 相关的三元组（图谱选择性）

### 23. KV 与 Prompt 的互补分工

| 通道 | 内容 | Token 量 | 作用 |
|------|------|---------|------|
| **KV prefix** | Subject anchor + Triple KV | ~50-100 tokens（总计） | 注意力结构约束：锚定主题、建立概念连接 |
| **Prompt** | Entity context + Evidence 原文 + Query | ~200-500 tokens | 详细事实内容：证据细节、生成指令 |

**互补规则**：
- KV 中的三元组句（"SFTSV导致发热"）与 Prompt 中的 evidence 原文（"患者通常以急性发热、乏力、食欲不振等流感样症状起病"）**语义相关但措辞不同**
- KV 提供方向性（"导致发热"），Prompt 提供细节（"急性发热、乏力、食欲不振等流感样症状"）
- 绝不将 evidence 原文编译为 KV（v1/v2 的教训）

### 24. 编译时流程

```
graph_index.json + aliases.jsonl
      ↓
  triple_kv_compiler.py
      ↓
  ┌─────────────────────────────────┐
  │ 对每个 entity:                    │
  │   1. 生成 subject_anchor text    │
  │      (description or aliases,   │
  │       ≤20 tokens, 纯中文)       │
  │   2. 收集关联三元组               │
  │      → triple_text (≤15 tokens) │
  │      → relation_type            │
  │   3. Tokenize → forward model   │
  │      → 提取 KV cache            │
  │   4. 标记 layer_range           │
  │      (来自 RELATION_LAYER_MAP)   │
  └─────────────────────────────────┘
      ↓
  triple_kvbank/
    manifest.json           # {entity_name → [kv_item_id, ...]}
    anchor_{entity}.safetensors   # subject anchor KV (layers 0-3)
    triple_{entity}_{rel}.safetensors  # triple KV (relation-dependent layers)
    meta.jsonl              # text, entity, relation, layer_range
```

### 25. 运行时流程

```
Query
  ↓
Entity Recognition → matched_entities
Intent Classification → target_relations
  ↓
Graph Walk → evidence_sentences, related_triples
  ↓
┌─ KV Assembly ─────────────────────────────┐
│ for entity in matched_entities:            │
│   load anchor_{entity}.safetensors        │
│   for triple in related_triples[entity]:  │
│     load triple_{entity}_{rel}.safetensors│
│   merge by layer (per-layer concat)       │
│ → past_key_values (per-layer KV tensors)  │
└───────────────────────────────────────────┘
  ↓
Prompt = entity_context + evidence_block + query + 要求
  ↓
LLM.generate(prompt, past_key_values=assembled_kv)
  ↓
Grounding Filter → Output
```

### 26. RELATION_LAYER_MAP 实现细节

```python
# 32-layer model (e.g. Qwen2.5-7B)
RELATION_LAYER_MAP = {
    # 定义/分类 → 浅层
    "is_a":        (0, 7),
    "has_subtype":  (0, 7),
    "also_known_as": (0, 7),

    # 因果/机制 → 中层
    "causes":       (8, 15),
    "manifests_as": (8, 15),
    "leads_to":     (8, 15),
    "associated_with": (8, 15),

    # 治疗/诊断 → 中高层
    "treats":       (12, 19),
    "prevents":     (12, 19),
    "diagnosed_by": (12, 19),

    # 结构/位置 → 早中层
    "located_in":   (4, 11),
    "part_of":      (4, 11),
    "transmission_route": (4, 11),
    "distributed_in": (4, 11),
}

SUBJECT_ANCHOR_LAYERS = (0, 3)  # subject anchoring 固定浅层
DEFAULT_LAYERS = (0, 7)          # 未知 relation 的 fallback
```

### 27. Per-Layer KV Masking 实现

标准 HuggingFace `past_key_values` 是 `tuple[tuple[Tensor, Tensor]]`，
每层一个 `(key, value)` pair。

**策略**：对于不在该三元组的 `layer_range` 内的层，将对应的 KV 设为零向量。

```python
def assemble_kv(kv_items, num_layers):
    """
    kv_items: list of (kv_cache, layer_start, layer_end)
    每个 kv_cache: tuple of (key[L,S,H,D], value[L,S,H,D])
    返回: past_key_values per-layer assembled
    """
    merged = [[] for _ in range(num_layers)]
    for kv_cache, l_start, l_end in kv_items:
        for layer_idx in range(num_layers):
            if l_start <= layer_idx <= l_end:
                merged[layer_idx].append(
                    (kv_cache[0][layer_idx], kv_cache[1][layer_idx])
                )
    # concat per-layer along sequence dim
    result = []
    for layer_idx in range(num_layers):
        if merged[layer_idx]:
            keys = torch.cat([k for k, v in merged[layer_idx]], dim=1)
            vals = torch.cat([v for k, v in merged[layer_idx]], dim=1)
            result.append((keys, vals))
        else:
            result.append(None)  # no KV for this layer
    return tuple(result)
```

### 28. 防止 Token 腐蚀的设计约束

| 约束 | 规则 | 理由 |
|------|------|------|
| Token 长度限制 | Subject anchor ≤ 20 tokens, Triple KV ≤ 15 tokens | v2 的 70+ token blob 是腐蚀主因 |
| 纯中文 | KV 编译文本不含英文/数字 | 中英混合在 Qwen tokenizer 中产生碎片 token |
| 图谱选择性 | 仅 query 匹配的 entity 的 KV 被加载 | Context-Aware，不盲注 |
| 不重复 Prompt | KV 中的三元组句 ≠ Prompt 中的 evidence 原文 | 避免双通道干扰 |
| 层段隔离 | 按 relation type 分层注入 | 不同语义类型的信号不在同一层竞争 |

### 29. 代码模块结构（新增）

```
src/graph/
  triple_kv_compiler.py    # 新增: 三元组 → KV bank 编译
                            #   compile_triple_kvbank(graph_index, model, tokenizer)
                            #   RELATION_LAYER_MAP
                            #   SUBJECT_ANCHOR_LAYERS

scripts/
  run_graph_inference.py   # 修改: 集成 KV 注入
                            #   load_triple_kvbank()
                            #   assemble_kv()
                            #   model.generate(past_key_values=...)
```

### 30. 与 v2 的关键差异

| 维度 | v2 (无 KV) | v3 (三元 KVI) |
|------|-----------|--------------|
| KV 注入 | 完全移除 | 回归：短三元组 KV，图谱引导 |
| Subject Anchoring | prompt 级 entity context | **双通道**：KV (浅层 anchor) + prompt (entity context) |
| Relation 作用 | 仅检索路由 | **双重作用**：检索路由 + 层段选择 |
| Token 腐蚀风险 | 无（不注入） | 极低（≤15 token/条，纯中文，选择性注入） |
| Grounding | 仅 prompt 证据 | prompt 证据 + entity context + KV 三元组句 |
