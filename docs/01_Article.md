# External Knowledge Value Injection (KVI): Architecture & System Design

> 本文为论文写作用的系统架构总结。基于 13_Triple_KVI.md / 14_Scheme_C.md / 15_架构调整.md 的演化过程，
> 提炼出**真正起作用的模块架构**。

---

## 1. Motivation

Large language models (LLMs) in knowledge-intensive domains face three structural limitations:
bounded context windows, implicit and uncontrollable knowledge storage, and weak dependence on
precise external evidence during generation. Existing approaches — prompt engineering and
retrieval-augmented generation (RAG) — treat external knowledge as textual hints appended to
the input, rather than as integral components of the model's internal computation.

We propose **Knowledge Value Injection (KVI)**, a framework that introduces external knowledge
directly into the attention mechanism of frozen LLMs at the key-value (KV) level, enabling
reliable and controllable knowledge utilization **without modifying model parameters**.

### 1.1 Key Insight

Transformer attention computes weighted aggregations over key-value pairs. Long-term domain
memory can therefore be modeled as an **extension of the KV space** — injected knowledge
becomes a first-class participant in attention computation, not merely visible text in the
prompt window.

### 1.2 What Failed and Why

| Approach | Failure Mode | Root Cause |
|----------|-------------|------------|
| v1: RAG + KV injection of same evidence text | Token corruption ("血小板" → "血板"), language mixing | Dual-channel interference — same content in both prompt and KV prefix causes attention head splitting |
| v2: Abandon KV, prompt-only entity context | Loses attention-level knowledge guidance | KV injection itself was not the problem; long blobs + context-unaware injection were |
| Flat tag routing (intent → ANN → sentences) | Context-unaware injection; cross-topic hallucination | No entity-level gate; topic-level routing cannot handle entity aliases or multi-topic queries |

### 1.3 Design Principles (Lessons Learned)

1. **KV carries attention structure constraints; Prompt carries detailed evidence content** — the two channels are complementary, never duplicated
2. **Short, focused KV** (≤15 tokens per triple) — avoids the token corruption caused by long-blob injection
3. **Entity-anchored retrieval** — entity not matched in graph → no retrieval → no injection (solves context-unaware problem)
4. **DRM filtering before KV injection** — irrelevant triples must be pruned before entering attention
5. **Graph structure + sentence index** — triples provide structural semantics; sentence index provides traceable evidence

### 1.4 Open Direction — KV as Reasoning-Chain / Memory Trace (Not Only Attention Bias)

**Empirical context (MedHopQA-ID *n=40* vs. *official* NL split, Exp01):** GraphRAG largely remains **evidence-in-prompt**: retrieved sentences (plus graph anchors) are exposed as natural language in the prompt. KVI **adds** a second channel: **triple → compiled KV prefix** on top of the same visible evidence. Under **NL queries + a large graph** (see gloss below), triple selection (DRM, relation gating, KV budget) is more brittle than text retrieval alignment: a **wrong triple** becomes a **high-salience wrong prior** in attention, and can **conflict** with correct sentences already in the prompt — yielding **smaller EM drop for GraphRAG than for KVI** on that split.

**Design hypothesis for future implementation (no code commitment here):** For settings with **long NL context** and **multi-turn or long-horizon** use, KV should not be conceptualised solely as **static prefix that reshapes attention**. It may need to behave as **explicit intermediate states** on a **reasoning chain** or **memory trace**:

- **NL-supplemented nodes**: each injected unit is not only tensor KV but carries a **short natural-language gloss** (what was committed, what remains open, provenance, confidence), readable and auditable.
- **Chain semantics**: nodes are **ordered or linked** (e.g. anchor entity → candidate relations → disambiguation after evidence) rather than an unordered bag of triple-KV blocks.
- **Lifecycle**: **session-local** traces can live in **extended KV cache** (prefix state); **cross-session** traces can be **serialised to disk** and **re-loaded** as conditioning for the next run (external memory), analogous to scratchpad + long-term store.

This direction targets the failure mode **“wrong triple dominates attention”** by making memory **revocable, inspectable, and structurally part of inference**, rather than only an attention bias injected once at the front.

**Gloss — “NL + large graph” (NL + 大图):** *Large graph* means a **large knowledge-graph instance for the benchmark build**: many **entities** and **relation edges (triples)** *and* a **large sentence index** (e.g. on the order of **10⁵** evidence sentences for the full MedHop official split), as opposed to a **small subgraph** built for a tiny subset (e.g. *n=40*). It is **not** primarily about the PNG figure size; it is about **graph + text index scale and density**, which increase **candidate triples**, **neighbourhood noise**, and **selection error rate** for the KV path.

---

## 2. System Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  Compile-Time Pipeline                   │
                    │                                                         │
  PDF Corpus ──→ raw_chunks ──→ DeepSeek Evidence Extraction ──→ blocks.evidence.jsonl
                    │                 (section-aware + noise filter)           │
                    │                         ↓                               │
                    │              Triple Extraction (LLM)                    │
                    │                         ↓                               │
                    │              Knowledge Graph Build                      │
                    │           (nodes + edges + sentence_index)              │
                    │                         ↓                               │
                    │              Triple KV Bank Compile                     │
                    │    (subject anchor KV + relation-layer triple KV)       │
                    └─────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────┐
                    │                   Query-Time Pipeline                    │
                    │                                                         │
  User Query ──→ Entity Recognition ──→ Intent Classification                │
                    │         ↓                    ↓                          │
                    │    Graph Walk          Relation Mapping                 │
                    │    (multi-hop)         (intent → relations)             │
                    │         ↓                                               │
                    │    DRM Scoring → Relation Gating → KV Budget           │
                    │         ↓                    ↓                          │
                    │   ┌─────────────┐   ┌──────────────────┐               │
                    │   │ KV Channel  │   │ Prompt Channel   │               │
                    │   │             │   │                  │               │
                    │   │ Subject     │   │ Entity Context   │               │
                    │   │ Anchor KV   │   │ (description)    │               │
                    │   │ (layers 0-3)│   │                  │               │
                    │   │             │   │ Evidence Block    │               │
                    │   │ Triple KV   │   │ (ranked by DRM)  │               │
                    │   │ (relation-  │   │                  │               │
                    │   │  dependent  │   │ Query + 要求      │               │
                    │   │  layers)    │   │                  │               │
                    │   └──────┬──────┘   └────────┬─────────┘               │
                    │          └──────────┬─────────┘                         │
                    │                     ↓                                   │
                    │          LLM.generate(prompt, past_key_values)          │
                    │                     ↓                                   │
                    │             Grounding Filter                            │
                    │                     ↓                                   │
                    │              Final Output                               │
                    └─────────────────────────────────────────────────────────┘
```

---

## 3. Compile-Time Pipeline

### 3.1 Evidence Extraction (PDF → Evidence Sentences)

```
PDF Corpus → pdf_to_raw_context_chunks (4096-token chunks)
                        ↓
              DeepSeek Extractive Evidence
              (topic_goal guided, section-aware)
                        ↓
              blocks.evidence.jsonl + docs.meta.jsonl
```

**Quality control layers:**

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| Section filter | `allowed_paragraph_types = abstract, results, conclusion` | Exclude references, methods, acknowledgements |
| Paragraph noise gate | Regex: DOI, citation patterns, journal metadata, year clusters | Reject bibliography-like paragraphs before API call |
| Sentence noise gate | Post-extraction filter on DeepSeek output | Reject citation-heavy, DOI-containing, fragment quotes |
| Topic-goal guidance | `--topic_goal` text embedded in DeepSeek prompt | Focus extraction on clinically relevant dimensions |

### 3.2 Triple Extraction (Evidence → Knowledge Triples)

```
blocks.evidence.jsonl → sentences.jsonl (dedup + compile)
                              ↓
                   Noise gate (reject citation/DOI sentences)
                              ↓
                   LLM Triple Extraction (DeepSeek or local)
                   (structured JSON: subject, predicate, object,
                    confidence, sentence_index)
                              ↓
                        triples.jsonl
```

Each triple carries **provenance**: `sentence_id`, `sentence_text`, `source_block_id`, `source_doc_id`.

### 3.3 Knowledge Graph Build

```
triples.jsonl + aliases.jsonl
              ↓
    KnowledgeGraphBuilder
              ↓
    graph_index.json
      ├── nodes (entities with type, aliases, description)
      ├── triples (edges with relation type + provenance)
      ├── entity_index (normalised name/alias → node_id)
      ├── sentence_index (sentence_id → text + doc_id + triple_ids)
      └── triple_sentence_index (triple_id → [sentence_id])
```

**Key design: `(S, R, O) + I` — Quadruple structure.**

Triples provide structural semantics for graph traversal; the sentence index (`I`) provides
traceable evidence links back to original text. This avoids the problem of triples being
too sparse to cover sentence-level semantics while maintaining structured retrieval capability.

**Relation type vocabulary** (medical domain):

| Category | Relations | Layer Range (28-layer model) |
|----------|-----------|------------------------------|
| Definition/taxonomy | `is_a`, `has_subtype` | layers 0-7 |
| Causation/mechanism | `causes`, `manifests_as`, `associated_with` | layers 8-15 |
| Treatment/diagnosis | `treats`, `prevents`, `diagnosed_by` | layers 12-19 |
| Structure/location | `located_in`, `part_of`, `transmits_via` | layers 4-11 |

### 3.4 Triple KV Bank Compilation

```
graph_index.json + base LLM
              ↓
    triple_kv_compiler.py
              ↓
    triple_kvbank/
      ├── manifest.json (entity → [kv_item_id, ...])
      └── *.pt (per-item KV cache tensors)
```

For each entity in the graph:

1. **Subject Anchor KV**: Entity description/aliases → short Chinese text (≤20 tokens) →
   forward through base LLM → extract KV cache → assign to **layers 0-3** (shallow, token alignment)

2. **Triple KV**: Each triple `(S, R, O)` → condensed Chinese sentence (≤15 tokens,
   e.g., "SFTSV导致血小板减少") → forward through base LLM → extract KV cache →
   assign to **relation-dependent layer range** (via RELATION_LAYER_MAP)

**Anti-corruption constraints:**

| Constraint | Rule | Rationale |
|-----------|------|-----------|
| Token length | Subject anchor ≤ 20 tokens, Triple KV ≤ 15 tokens | Long blobs (70+ tokens) were the primary cause of token corruption in v1 |
| Language | Pure Chinese, no English/number mixing | Mixed tokens cause tokenizer fragmentation |
| Selectivity | Only matched-entity KV loaded | Context-aware; no blind injection |
| Non-duplication | KV triple text ≠ prompt evidence text | Avoid dual-channel interference |
| Layer isolation | Different relation types → different layer ranges | Semantic signal types don't compete in the same layers |

---

## 4. Query-Time Pipeline

### 4.1 Entity Recognition + Intent Classification

```
Query: "SFTSV的临床症状有哪些？"
  ↓
Entity Recognition: longest-match against entity_index → SFTSV (node_0001)
Intent Classification: keyword-based → "symptom"
  ↓
Relation Mapping: symptom → [causes, manifests_as, manifestation_of]
```

**Entity-anchored gate**: If no entity matches the graph, retrieval stops immediately
(returns "knowledge base does not contain relevant information"). This structurally
prevents context-unaware injection — the failure mode of flat tag routing.

### 4.2 Graph Walk (Multi-hop Retrieval)

```
Graph Walk from matched entities:
  1. Outgoing edges along target relations (max_hops configurable)
  2. Incoming edges if outgoing yields nothing
  3. Broad 1-hop fallback if intent-specific walk is empty
  4. Topic-scoped relation scan: find triples by relation type
     across ALL graph triples (catches proxy-subject triples)
```

### 4.3 DRM Scoring → Relation Gating → KV Budget

This is the critical **pre-injection filter** that prevents irrelevant KV from entering attention.

```
Graph Walk triples
      ↓
DRM Scoring: bigram overlap(query, provenance_sentence) per triple
      ↓
Filter: drop triples below drm_threshold (default 0.05)
      ↓
Relation Gating: group by relation type, rank by aggregate DRM score,
                 keep top-k relation groups (default 2)
      ↓
KV Budget: from selected groups, take top-N triples by DRM score
           (default 3) for KV injection
      ↓
KV Assembly: load subject anchor + selected triple KV tensors,
             merge per-layer via concatenation
```

### 4.4 双通道生成（Dual-Channel Generation）

| 通道 | 内容 | Token 预算 | 作用 |
|------|------|-----------|------|
| **KV prefix** (`past_key_values`) | Subject anchor + Triple KV | ~50-100 tokens | 注意力结构约束：锚定主题、建立概念连接 |
| **Prompt** (文本输入) | Entity context + Evidence sentences (DRM排序) + Query + 生成指令 | ~200-500 tokens | 详细事实内容：证据细节、生成要求 |

**互补规则**：KV 提供方向性引导（"SFTSV导致发热"），Prompt 提供细节素材
（"患者通常以急性发热、乏力、食欲不振等流感样症状起病"）。
两者语义相关但措辞不同——绝不重复。

#### 4.4.1 KV 注入在 Attention 计算中的作用机制

Transformer 每一层的注意力计算为：

```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

其中 Q 来自当前正在生成的 token，K 和 V 来自所有"可见"的历史位置。

**传统 RAG 的做法**：将 evidence 文本拼入 prompt → tokenize 后产生一组 K, V 向量 →
模型通过 self-attention "看到"这些 token。知识以**文本 token** 的身份进入 attention，
和 query、指令、标点等所有 token 一起竞争 attention 权重。模型需要"理解"文本才能利用知识。

**KVI 的做法**：在编译期，将三元组短句（如"SFTSV导致血小板减少"，仅 8 个 token）通过
base LLM 做一次前向传播，提取每层的 (K, V) 张量并保存到磁盘：

```
text → tokenize → model.forward(use_cache=True) → past_key_values
                                                    ↓
                                          每层一组 (Key, Value) 张量
                                          shape: [batch, heads, seq_len, head_dim]
                                                    ↓
                                              保存为 .pt 文件
```

推理时，将预编译的 KV 张量通过 `past_key_values` 参数注入 `model.generate()`。
此时 attention 的 K, V 空间被扩展：

```
K = [K_external | K_prompt],  V = [V_external | V_prompt]
```

模型在生成每个 token 时，Q 会**同时 attend 到**：
- **外部注入的 KV**（来自预编译三元组）—— 这是预计算的注意力向量，不经过 token 理解
- **Prompt 文本的 KV**（来自 evidence 原文）—— 这是模型对文本的正常理解

外部 KV 不是文本，而是**已经编码为模型内部 attention 信号格式的向量**，
直接参与 softmax 权重分配，影响每个生成 token 的概率分布。

**这就是 "first-class citizen"（一等公民）的含义**：
知识不是被当作"文本提示"让模型自己去理解和可能忽略，
而是以模型内部计算格式（KV 向量）直接参与 attention 权重计算——
模型**无法忽略**这些信号，它们和模型自身产生的 KV 在数学上完全等价。

#### 4.4.2 维度对齐：为什么外部 KV 能与 Prompt KV 无缝拼接

一个自然的疑问：K_external 和 K_prompt 来源不同，它们的张量维度能对齐吗？

**能。因为它们都由同一个模型的同一层 attention 产生，架构维度完全相同。**

每一层 KV cache 的张量形状是 `[batch, num_heads, seq_len, head_dim]`。
其中 `batch`、`num_heads`、`head_dim` 是**模型架构常量**——无论输入什么文本，
这三个维度始终不变。唯一不同的是 `seq_len`（序列长度，取决于输入了多少 token）。

```
K_external: [1, 32,   8, 128]   ← 编译期：三元组短句（8 个 token）前向传播的 KV
K_prompt:   [1, 32, 200, 128]   ← 推理时：prompt 文本（200 个 token）前向传播的 KV
                  ↓ concat(dim=2)  沿序列长度维度拼接
K_full:     [1, 32, 208, 128]   ← 合并后的完整 KV
```

`num_heads=32` 和 `head_dim=128` 完全一致，拼接只在 `seq_len` 维度上发生。
这与 HuggingFace 自身的 KV cache 机制完全相同——模型在自回归生成时，
每一步也是把新 token 的 KV 沿 seq_len 维度 concat 到已有 cache 上。
KVI 只是在**生成开始前**预先放入了一段"虚拟历史 token"的 KV。

#### 4.4.3 Softmax 竞争机制：KV 注入是"精准吸引"而非"强制拉偏"

另一个核心问题：既然每个生成 token 的 Q 都会 attend 到 K_full 的所有位置，
是否意味着 KV 注入**强迫**每个 token 都被外部知识语义拉偏？

**不是强制拉偏，而是有选择权的吸引力。** 关键在于 softmax 是**竞争性**的——
所有位置瓜分 100% 的 attention 权重：

```
对于生成位置 t 的每个 token：
  Q_t:     [1, 32,   1, 128]   ← 当前 token 的 query 向量
  K_full:  [1, 32, 208, 128]   ← 外部 KV（8 位置）+ Prompt KV（200 位置）

  attention_weights = softmax(Q_t · K_full^T / √128)
                    → [1, 32, 1, 208]
                       ↑
                  208 个位置竞争分配 attention 权重

  output_t = attention_weights · V_full   ← 按权重加权求和
```

三种典型场景：

**场景 A：Q_t 与 K_external 语义高度对齐（尖峰信号）**

模型正在决定"SFTSV导致____"的下一个 token 时，Q_t 的向量方向恰好与
K_external 中"导致血小板减少"的某个位置高度相似：

```
Q_t · K_external[pos=3]^T = 高分 → softmax 后该位置权重大
→ V_external[pos=3] 强烈影响输出 → token 概率向"血小板减少"偏移
```

这就是 13_Triple_KVI.md 中描述的**"尖峰信号"**——
短 KV 与当前生成语义对齐时，在 attention 权重分布中产生一个高峰，
定向拉偏 token 选择概率。

**场景 B：Q_t 与 K_external 语义无关（信号透明）**

模型正在生成"请根据上述证据____"中的功能词"上述"时，
Q_t 和三元组知识毫无语义关联：

```
Q_t · K_external[all]^T = 低分 → softmax 后外部 KV 位置权重趋近 0
→ V_external 几乎不影响输出 → attention 权重落在 K_prompt 的相关位置上
```

此时 KV prefix 近乎透明，不干扰正常生成。

**场景 C：零向量层（Relation Layer Routing 的非活跃层）**

对于不在该三元组 `[layer_start, layer_end]` 范围内的层，
KV 被填充为零向量：

```
Q_t · 0^T = 0 → softmax 后权重 ≈ 1/N（均匀分散，淹没在其他位置中）
→ 该层对这些知识完全"无感"
```

**这解释了为什么短 KV 是核心设计约束：**

| KV 长度 | Softmax 行为 | 结果 |
|---------|-------------|------|
| **8 tokens**（三元组短句） | 少数位置可能产生尖峰，多数 token 不受影响 | **精准的方向性吸引**：语义对齐时生效，不对齐时透明 |
| **70+ tokens**（v1 长 blob） | 大量位置争夺 attention → 权重碎片化 → 多个矛盾信号 | **注意力碎片化**：token 选择不稳定 → "血小板"→"血板" |

短 KV = 少数高聚焦的 attention 尖峰 = 语义对齐时精准引导，不对齐时自动退让。
这是"每个 token 都看得到 KV"这个特性从**负面**（v1 的全局干扰）
变为**正面**（当前设计的选择性引导）的关键约束。

#### 4.4.4 关系类型层段路由（Relation Layer Routing）

KV 注入并非对所有层一视同仁。基于 Transformer 不同层段的语义分工假设：
- **浅层**（layers 0-7）：负责 token 级特征——"是什么"（定义、分类）
- **中层**（layers 8-15）：负责语义关系——"为什么"（因果、机制）
- **中高层**（layers 12-19）：负责行为推理——"怎么办"（治疗、诊断）

系统通过 `RELATION_LAYER_MAP` 将不同关系类型的三元组 KV 路由到对应层段：

```
三元组 "SFTSV属于布尼亚病毒" (is_a)     → 注入 layers 0-7  → 影响实体识别和定义对齐
三元组 "SFTSV导致血小板减少" (causes)    → 注入 layers 8-15 → 影响因果推理链构建
三元组 "法维拉韦治疗SFTSV" (treats)      → 注入 layers 12-19 → 影响治疗行为推理
```

实现方式：对不在该三元组 `[layer_start, layer_end]` 范围内的层，
对应位置的 KV 填充为零向量（不产生 attention 信号）。
HuggingFace 的 `past_key_values` 接口要求所有层具有相同的序列长度，
因此非活跃层用零张量填充到统一长度——零向量不产生有效 attention 权重，
相当于该层"看不到"这些知识。

**效果**：不同语义类型的知识信号被隔离到各自的"认知层段"，
避免在同一层产生 attention 竞争。

#### 4.4.5 互补双通道设计如何消除干扰

**v1 的失败根因**：同一条 evidence（如"患者通常以急性发热、乏力、食欲不振等流感样症状起病"）
同时出现在两个通道：
- Prompt 文本 → 产生 K_prompt, V_prompt
- KV prefix → 产生 K_external, V_external

两组向量指向**相同语义但占据不同位置**。模型的 attention head 在做 softmax 时，
对同一个概念出现了两个"竞争源"，导致权重分裂 → token 选择不稳定 →
"血小板" 变成 "血板"，"免疫抑制" 变成 "免疫耐受"。

**当前设计的互补分工**：

| 通道 | 内容示例 | 功能比喻 |
|------|---------|---------|
| **KV prefix** | "SFTSV导致发热"（8 tokens） | **潜意识指南针**：在模型的注意力空间中植入方向性信号——"SFTSV 和发热之间存在因果关系" |
| **Prompt text** | "患者通常以急性发热、乏力、食欲不振等流感样症状起病" | **工作台上的病历**：提供具体措辞素材和事实细节 |

KV 信号让模型在生成时**倾向于**提及发热相关概念（attention weight 向相关语义偏移），
Prompt 文本提供具体的措辞和事实依据。两者语义相关但措辞不同，
在 attention 空间中不形成位置冲突——协同而非竞争。

**一句话总结**：KV 将领域知识从"模型可见的文本输入"提升为"直接参与 attention softmax 的
预编译 KV 向量"，使知识信号绕过 token 级的间接理解，以向量级精度影响每个生成 token 的概率分布。
双通道设计确保 KV 传递结构性语义约束（三元组方向），Prompt 传递证据细节（原文内容），
两者在 attention 空间互补而非竞争。

#### 4.4.6 2026-03 运行时质量修复（已落地）

针对实验一中 "KV 注入可执行但 injected_answer 出现乱码/符号串" 的问题，做了两类工程修复：

1) **Cache 长度一致性修复（mask 对齐）**
- 文件：`src/runtime/hf_cache_prefix_injection.py`
- 现象：仅在前几层注入 prefix，其他层为 `None` 时，Transformers 5.x/Qwen2 可能用某一层的 `kv_len` 生成全局 mask，导致层间 `kv_len` 不一致，触发 attention 维度报错或异常生成。
- 修复：未注入层不再留空，改为填充与注入层同长度的零前缀 KV（zero K/V）。这样所有层 cache 长度一致，mask 计算稳定。

2) **Simple 流程双通道质量兜底（Dual-channel rescue）**
- 文件：`scripts/run_kvi2_runtime_test.py`
- 现象：即使不崩溃，纯 KV 通道有时会生成低质量文本（乱码、符号噪声）。
- 修复策略：
  - 主路径仍为 KV 注入生成（保持 KVI 机制）；
  - 增加低质量检测（异常符号密度、乱码字符 `�`、`<>` 模式、重复字符等）；
  - 若判定低质量，则自动启用 **Prompt 证据通道** 重生成（不附加 KV），并输出 `dual_channel_rescue_used=1` 与原因。
- 结果：实验一可稳定产出可读答案；在注入退化时自动切换到可解释的证据通道，避免输出污染。

### 4.5 Hybrid Retrieval

Graph retrieval alone has coverage blind spots (URLs, expert opinions, non-factoid content).
The system supplements graph evidence with keyword-based text search over raw `sentences.jsonl`:

```
Evidence = Graph Evidence (primary) ∪ Text Search (supplement)
         → deduplicate by text → rank by DRM score
```

URL-containing evidence is further separated as **verbatim evidence** — bypasses LLM generation
entirely and is appended to the output as-is (only when the query is reference-related).

### 4.6 Grounding Filter

Post-generation token-overlap check between LLM output sentences and:
- Evidence texts (all sources)
- Entity context
- KV triple texts

Sentences below the overlap threshold (0.10) are dropped from the final output.
This catches LLM hallucinations while allowing semantically correct but lexically
different summary sentences to pass.

---

## 5. Module Structure

```
src/graph/
  schema.py               # Triple, Entity, GraphNode, KnowledgeGraphIndex
                          #   + sentence_index, triple_sentence_index
  triple_extractor.py     # LLM-based (S,R,O) extraction with provenance
  knowledge_graph.py      # Graph build + entity/sentence index construction
  graph_retriever.py      # Entity recognition + multi-hop walk + evidence collection
  triple_kv_compiler.py   # Triple → KV bank (subject anchor + relation-layer KV)

scripts/
  extract_triples.py                                    # CLI: sentences → triples
  build_knowledge_graph.py                              # CLI: triples → graph_index
  build_evidence_blocks_from_raw_chunks_jsonl_deepseek.py  # CLI: PDF chunks → evidence
  run_graph_inference.py                                # CLI: query-time inference

src/llm_filter/
  extractive_evidence.py  # DeepSeek evidence extraction with noise filtering
  doc_meta_extractor.py   # Document metadata extraction (title, DOI, year)

authoring_app/
  server.py               # Web server: unified build pipeline + inference API
  static/app.js           # Frontend: Literature Import + Knowledge Authoring + Inference Debug
```

---

## 6. Evolution Summary

```
Stage 0: Tag routing + same-content KV/RAG
         ⚠ Context-unaware injection
         ⚠ Dual-channel interference → token corruption

Stage 1: Complementary injection (entity priming KV + RAG evidence)
         ✓ Eliminated dual-channel interference
         ⚠ Context-unaware injection still present
         ⚠ Depended on manual tag classification

Stage 2: GraphRAG (entity-anchored retrieval, no KV injection)
         ✓ Solved context-unaware problem
         ✓ Eliminated tag dependency
         ⚠ Lost attention-level knowledge guidance

Stage 3: Triple KVI (short triple KV + relation layer routing)
         ✓ Restored KV injection without corruption
         ⚠ Injected irrelevant triples (no DRM filtering)

Stage 4: DRM + Relation Gating + KV Budget
         ✓ Pre-injection filtering prevents irrelevant KV
         ✓ Evidence ranked by relevance

Stage 5: Hybrid Retrieval + Verbatim Evidence + Sentence Index
         ✓ Graph + text search for full coverage
         ✓ URL evidence bypasses LLM (no hallucination)
         ✓ (S,R,O,I) quadruple: triple structure + sentence provenance
         ✓ Strong noise filtering (section-aware + citation/DOI gate)
```

**Final architecture = Stage 5**: Entity-anchored GraphRAG with DRM-gated Triple KV Injection,
Hybrid Retrieval, and Sentence-indexed Knowledge Graph.

---

## 7. Comparison with Prior Paradigms

| Paradigm | Knowledge Level | Update Cost | Attention Dependency | Entity Gate | Noise Control |
|----------|----------------|-------------|---------------------|-------------|---------------|
| Prompt Engineering | Textual | Low | Weak | None | None |
| RAG | Textual | Medium | Weak–Implicit | None | Retrieval quality |
| Fine-tuning | Parameter | High | Strong but Static | None | Training data |
| KG-enhanced RAG | Textual + Graph | Medium | Weak | Entity match | Graph structure |
| **KVI (Ours)** | **Attention (KV) + Graph + Text** | **Low** | **Strong & Explicit** | **Entity-anchored** | **Multi-layer filtering** |

---

## 8. 核心贡献（Key Contributions）

### 8.1 注意力空间的知识注入（Attention-Space Knowledge Injection）

外部知识以预编译 KV 向量的形式直接参与 attention 的 softmax 权重计算，
而非仅作为文本上下文拼入 prompt。

**区别于 RAG 的本质差异**：RAG 中，知识以 token 序列的形式进入 prompt，
模型需要通过自身的语言理解能力去"读懂"并决定是否使用这些文本——模型可以忽略、
误解、或超越 evidence 进行幻觉。KVI 中，知识已经被编码为与模型内部 KV 数学等价的向量，
直接参与每个生成 token 的概率决策，模型在计算层面无法忽略这些信号。

这使得外部知识从"可选的文本提示"升级为推理过程中的"一等公民"。

### 8.2 互补双通道设计（Complementary Dual-Channel Design）

KV 通道承载极短的结构性语义约束（三元组方向，如"SFTSV导致发热"），
Prompt 通道承载详细的证据内容（原文素材，如"患者通常以急性发热、乏力...起病"）。

**消除干扰的机制**：早期方案将相同 evidence 同时注入 KV 和 Prompt，
导致 attention head 在两个位置看到相同语义 → 权重分裂 → token 选择不稳定
（实测表现为"血小板"→"血板"等 token 腐蚀现象）。
当前设计中，两个通道的内容语义相关但措辞不同，在 attention 空间中形成互补而非竞争：
KV 提供注意力偏移方向（"指南针"），Prompt 提供事实细节素材（"病历"）。

### 8.3 实体锚定检索 + 关系类型层段路由（Entity-Anchored Retrieval + Relation Layer Routing）

知识图谱提供结构化检索路径（实体识别 → 图谱遍历 → 关系类型边），
关系类型同时控制两个维度：
- **检索方向**：relation type 决定 graph walk 沿哪些边遍历（如 symptom 查询只沿 causes/manifests_as 边）
- **注入层段**：relation type 决定三元组 KV 注入 Transformer 的哪些层
  （定义→浅层，因果→中层，治疗→中高层）

这一设计基于 Transformer 不同层段的语义分工假设，将不同类型的知识信号路由到
对应的"认知层段"，避免异质信号在同一层竞争 attention 权重。

### 8.4 DRM 门控注入管线（DRM-Gated Injection Pipeline）

文档相关性模型（DRM）评分 → 关系门控（Relation Gating）→ KV 预算控制，
确保只有与查询相关的知识进入 attention 机制。

**解决的核心问题**：图谱遍历会返回目标实体的所有关联三元组（如查症状时也会遍历到
located_in、treats 等无关关系），若不经过滤直接注入，无关 KV 信号会干扰生成。
DRM 三级过滤链（triple 级评分 → relation 组级门控 → 全局预算）逐层收窄，
将注入的 KV 严格限制在与查询高度相关的少数三元组上。

### 8.5 `(S,R,O,I)` 知识表示（Quadruple Knowledge Representation）

三元组 `(S,R,O)` 提供结构化语义用于图谱遍历；句子索引 `I` 提供可追溯的证据链接
回到原始文本。

**解决的核心问题**：纯三元组过于精简，无法覆盖原始句子的完整语义（如一条句子可能
包含多个事实维度），导致三元组"散"而"薄"。加入句子索引后，三元组负责结构化检索路由，
句子负责承载完整语义——图谱遍历找到三元组后，通过 `triple_sentence_index` 反查原始句子，
作为 Prompt 通道的 evidence 输入 LLM。两者结合了结构化知识和非结构化知识的优势。

### 8.6 零参数修改（Zero Parameter Modification）

Base LLM 完全冻结，不做任何参数修改。领域知识维护在外部 KV Bank 中，
可独立更新、替换或扩展，无需重新训练模型。

**实际意义**：同一个 base LLM 可以通过切换不同的 KV Bank 服务于不同的领域
（如 SFTSV、肿瘤、心血管等），且知识更新只需重新编译 KV Bank（分钟级），
而非重新微调模型（小时-天级）。

---

## 9. Evaluation Protocol for Multi-hop Long-context QA (HotpotQA)

### 9.1 Why this protocol

KVI is designed for **long-context, multi-hop reasoning**, where evidence is distributed across
multiple documents and must be integrated through structured links. Therefore, evaluation should
preserve HotpotQA's original context structure rather than collapsing data into short QA templates.

### 9.2 Paper-track experiment line (recommended)

1. **Build multi-hop assets from original Hotpot context**
   - Use original `context` paragraphs and `supporting_facts` to construct:
     - eval set with `gold_supporting_sentences`
     - sentence pool for retrieval
     - triple source for graph/KV build
   - Script:
     - `experiments/exp01_main_qa/code/prepare_hotpot_multihop_assets.py`

2. **Compile retrieval/injection artifacts from the multi-hop assets**
   - Graph index: `scripts/build_knowledge_graph.py`
   - Triple KV bank: `src/graph/triple_kv_compiler.py`
   - ANN side: `scripts/annotate_sentences_semantic_tags.py` + `scripts/build_kvbank_from_blocks_jsonl.py`

3. **Run Exp01 with aligned comparison**
   - Methods: `RAG`, `GraphRAG`, `KVI`
   - Same dataset, same model, same EM mode, same run budget
   - Evaluate EM + CI95 + paired permutation significance

4. **Case-study extraction for paper writing**
   - Extract cases where KVI succeeds while GraphRAG/RAG fail:
     - `experiments/exp01_main_qa/code/collect_kvi_win_cases.py`
   - Produce markdown-ready qualitative analysis tables.

### 9.3 Anti-bias notes

- Do not compare against external reported SOTA numbers unless retrieval corpus, split,
  evaluation script, and decoding protocol are fully aligned.
- If KVI and GraphRAG tie on aggregate EM, inspect per-example divergence and failure modes;
  equal aggregate score does not imply equivalent behavior.
- Track both quantitative metrics and qualitative cases (especially multi-hop comparison and
  cross-document bridging questions).

---

## 10. Is KVI suitable for CL-bench?

### 10.1 Short answer

**Partially yes, but not as-is for the full benchmark.**

KVI is naturally strong when tasks require:
- large context digestion,
- multi-step dependency over long text,
- stable reuse of structured facts/rules during generation.

These properties overlap with CL-bench's goals (context learning, sequence dependency,
multi-angle verification). However, CL-bench is explicitly **self-contained** and does not rely on
external retrieval in its canonical setup, while current KVI pipeline is retrieval-centric.

### 10.2 Fit map

| CL-bench characteristic | KVI fit | Comment |
|---|---|---|
| Self-contained context (no hidden external knowledge) | Medium | KVI can be repurposed to compile KV from the provided context only |
| Long, complex, multi-hop reasoning | High | KVI's structured KV guidance is designed for this |
| Multi-turn sequence dependency | Medium-High | Requires session-level incremental KV updates |
| Fine-grained, multi-criterion scoring | Medium | Need task-specific output schema + deterministic checking |
| Data contamination resistance | High | KVI does not depend on memorized pretraining facts if context-only compilation is enforced |

### 10.3 Required adaptation before claiming CL-bench suitability

1. **Context-only KV compilation**
   - Build triples/KV strictly from the current CL-bench context.
   - Disable any external corpus retrieval.

2. **Session/incremental KV memory**
   - Support turn-by-turn updates so later tasks can depend on prior interaction outputs.

3. **Rule/procedure-aware relation schema**
   - Extend relation predicates beyond factual triples to include rule constraints,
     procedural steps, and state transitions.

4. **Verifier-compatible output interface**
   - Map generation outputs to CL-bench's multi-criteria validators (structured outputs,
     deterministic fields, traceable evidence).

### 10.4 Research positioning recommendation

For paper positioning, present KVI on CL-bench as:
- **Context-to-KV structured execution aid** (not external-memory retrieval),
- evaluated under strict context-only constraints,
- compared against prompt-only and retrieval-disabled baselines.

This framing aligns KVI's mechanism with CL-bench's "learning from current context" principle.
