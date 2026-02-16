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

### 4.4 Dual-Channel Generation

| Channel | Content | Token Budget | Role |
|---------|---------|-------------|------|
| **KV prefix** (`past_key_values`) | Subject anchor + Triple KV | ~50-100 tokens total | Attention structure constraint: anchor topic, establish concept connections |
| **Prompt** (text input) | Entity context + Evidence sentences (DRM-ranked) + Query + Instructions | ~200-500 tokens | Detailed factual content: evidence details, generation instructions |

**Complementarity rule**: KV provides directional guidance ("SFTSV导致发热"), Prompt provides
detail ("患者通常以急性发热、乏力、食欲不振等流感样症状起病"). They are semantically related
but lexically distinct — never duplicated.

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

## 8. Key Contributions

1. **Attention-space knowledge injection**: External knowledge participates directly in attention
   computation as KV prefix memory, not merely as textual context — making it a first-class
   citizen in the reasoning process.

2. **Complementary dual-channel design**: KV channel carries short, focused attention structure
   constraints; Prompt channel carries detailed evidence content. This eliminates the dual-channel
   interference that plagued earlier approaches.

3. **Entity-anchored retrieval with relation-layer routing**: Knowledge graph provides structured
   retrieval (entity recognition → graph walk → relation-typed edges); relation types simultaneously
   control both retrieval direction and KV injection layer placement.

4. **DRM-gated injection pipeline**: Document Relevance Model scoring → Relation Gating → KV Budget
   ensures only query-relevant knowledge enters the attention mechanism.

5. **`(S,R,O,I)` knowledge representation**: Triples provide structural semantics for graph
   traversal; sentence index provides traceable, grounded evidence links — combining the
   strengths of structured and unstructured knowledge.

6. **Zero parameter modification**: The base LLM remains completely frozen. Domain knowledge is
   maintained in external KV banks that can be updated, swapped, or extended without retraining.
