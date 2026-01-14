# RIM v0.4 —— KVI 2.0 with Pattern-first / Semantic-second / Introspection-gated Retrieve

## 0. 设计目标（更新）

在 **不修改 Base LLM 权重**、**不修改既有 KVI / KV Bank 实现** 的前提下，
将当前系统的「一次性、query 静态对齐的 KV 注入」升级为：

> **Pattern-first → Semantic-second → Introspection-gated**
> 的推理感知型知识对齐流程。

核心目标：

1. 允许 **推理中期（reasoning state）重新对齐外部知识**
2. 在不生成 token 的前提下，引入一个 **Introspection Gate**
3. 支持 **KV 不相关 → 自动更换新一批 KV（≤ 2 轮）**
4. 显式区分：
   - **形式/模式型知识（pattern, low-entropy）**
   - **语义/证据型知识（semantic, high-entropy）**

---

## 1. 系统硬约束（不可违背）

- Base LLM：冻结参数
- Attention forward：不可修改
- KVI 注入方式：prefix past_key_values（保持现状）
- KV Bank：仍为 (K_ext, V_ext) embedding 索引
- 不引入文本级 RAG
- 不强制 chain-of-thought 输出

---

## 2. 从 KVI 1.x 到 KVI 2.0 的关键变化

### 2.1 KVI 1.x（当前隐含假设）

query → embedding → ANN → KV → 注入 → 完整推理

缺陷：
- retrieve 与推理阶段解耦
- 所有知识统一走“语义相似性”
- 无法处理中期语义转向

---

### 2.2 KVI 2.0（本次升级）

nput / Reasoning State
↓
[ Pattern-first Retrieve ]
↓
[ Semantic-second Retrieve ]
↓
[ Introspection Gate ]
↓
[ Incremental KVI Injection ]

---

## 3. Pattern-first Retrieve（新增，必须实现）

### 3.1 设计目的

用于处理 **低熵、形式稳定、不应进入语义博弈的知识**：

- 缩写 ↔ 全称（SFTSV）
- 固定命名实体
- schema cue（传播途径 / 临床表现 / 定义）
- 标准术语模板

这些内容 **不得通过 ANN embedding 检索完成**。

---

### 3.2 Pattern-first 输入与输出

**输入（允许任一）：**
- token suffix（n-gram）
- 原始 query
- reasoning state 的 shallow cue（不需要 embedding）

**输出（必须结构化）：**

```json
{
  "source": "pattern",
  "pattern_type": "abbreviation_expansion | schema_trigger | fixed_entity",
  "confidence": 0.0-1.0,
  "payload": {...}
}

Pattern-first 的结果：

不直接注入 KV

作为 Semantic retrieve 的约束 / 引导

作为 Introspection Gate 的先验

4. Semantic-second Retrieve（沿用现有 KVI）
4.1 触发条件

Semantic retrieve 不得作为默认第一步，仅在以下情况下触发：

Pattern-first 无法覆盖 query

Pattern-first 引入了新的 slot / 概念

RIM Introspection Gate 判定需要证据支持

4.2 检索方式（保持不变）

使用 query embedding（初始或 reasoning-derived）

ANN / FAISS search

返回 KVItem(K_ext, V_ext, meta, score)

Semantic retrieve 的定位是：

候选证据池（evidence candidates），而非直接答案

5. RIM = Introspection Gate（核心升级）

RIM 不生成 token，仅作为 控制与裁决模块。

5.1 RIM 的观察输入

hidden states（decoder 中后层）

最近 N token

Pattern-first candidates

Semantic retrieve candidates

已注入 KV 的 meta（block_id）

5.2 Reasoning State 表示
H_i ∈ R^{N × d_model}
q'_t = Pool(H_i)

允许：

Mean pooling

Attention pooling

Last-token projection

5.3 Introspection Gate 的三类判定（必须实现）
Q1：是否发生 reasoning shift？
cosine_distance(q_0, q'_t) > τ

Q2：现有 KV 是否与当前 reasoning 不相关？

（通过 logit delta / entropy / heuristic）

Q3：是否应拒绝全部外部 KV，重新检索？

输出：{
  "retrieve_more": true | false,
  "reject_current_kv": true | false,
  "confidence": 0.0-1.0,
  "rationale": "pattern-mismatch | semantic-shift | low-impact"
}

6. KV Re-alignment（推理感知）

当 Introspection Gate 判定 retrieve_more == true：

使用 q'_t 构造新的 query embedding

在 KV Bank 中执行 Semantic-second retrieve

选取 未注入过 的 KV blocks

注入为增量 prefix

7. KV 不相关 → 自动更换批次（≤ 2 轮）
7.1 执行逻辑

一次检索 oversample：top_k * (rounds + 1)

每轮取一批 top_k

去重（block_id / chunk_id）

注入后做 relevance test：Δ = mean(|logits_injected - logits_zero_prefix|)
或
Δ < kv_irrelevant_logit_delta_threshold
→ 判定该批 KV 无效，切换下一批（≤ 2 轮）

8. RIM 接口（保持，但语义升级）
class RIM:
    def observe(self, hidden_states, generated_tokens, pattern_hits, semantic_hits): ...
    def should_realign(self) -> bool: ...
    def build_reasoning_query(self) -> Tensor: ...
    def retrieve_additional_kv(self, query_vec) -> KVBlocks: ...
    def inject(self, kv_blocks, past_key_values): ...

9. 解码循环（更新版伪代码）
if step == 0:
    pattern_hits = PatternRetriever.run(query)
    semantic_hits = None

for t in decoding_steps:
    logits, hidden_states = model.step(...)

    RIM.observe(hidden_states, generated_tokens, pattern_hits, semantic_hits)

    if RIM.should_realign() and not RIM.exceeded_budget():
        q_t_prime = RIM.build_reasoning_query()
        semantic_hits = RIM.retrieve_additional_kv(q_t_prime)
        past_key_values = KVI.inject_incremental(
            past_key_values, semantic_hits
        )

    token = sample(logits)
10. 一句话范式总结（用于论文 / README）

KVI 2.0 将知识注入从“query 驱动的静态相似性检索”，
升级为“pattern 先验锚定 → semantic 证据补充 → introspection 裁决”的
推理感知型条件记忆访问流程。

11. 明确禁止事项（再次强调）

❌ 修改 attention forward
❌ 修改 Base LLM 权重
❌ 文本级 RAG 替代 KV
❌ 强制 CoT 输出

12. 核心定位（必须保持）

KVI：决定 模型能看到哪些记忆

RIM / Introspection Gate：决定 模型在什么时候、是否、以及如何重新找记忆

Pattern-first：冻结问题空间

Semantic-second：提供证据

Gate：拥有最终否决权

---

### 供参考 `PatternRetriever` 的最小 Python 接口（可空实现）**
PatternRetriever
├── LiteralPatternOperator
│   ├── ExactMatch
│   ├── AliasMap
│   └── HashNgram (Engram-like)   ← 可选
├── StructurePatternOperator
└── BlockPatternOperator

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# =========================
# Pattern-level Data Types
# =========================

@dataclass
class PatternHit:
    block_id: str
    hit_types: List[str]          # e.g. ["literal", "schema", "structure"]
    confidence: float             # pattern-level confidence
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatternRetrieveResult:
    pattern_hits: List[PatternHit]
    recall_size: int
    debug_info: Optional[Dict[str, Any]] = None


# =========================
# Pattern Retriever Interface
# =========================

class PatternRetriever:
    """
    Stage-1 Retriever in KVI 2.0:
    Pattern-first (non-semantic, structure-aware, fast recall)
    """

    def __init__(
        self,
        literal_index: Optional[Any] = None,
        structure_index: Optional[Any] = None,
        block_index: Optional[Any] = None,
    ):
        """
        All indices are optional to allow empty / incremental implementation.
        """
        self.literal_index = literal_index
        self.structure_index = structure_index
        self.block_index = block_index

    # -------- Entry Point --------

    def retrieve(self, query: str) -> PatternRetrieveResult:
        """
        Main entry for Pattern-first retrieval.
        """
        literal_hits = self._retrieve_literal(query)
        structure_hits = self._retrieve_structure(query)
        block_hits = self._retrieve_block_type(query)

        merged_hits = self._merge_hits(
            literal_hits,
            structure_hits,
            block_hits
        )

        return PatternRetrieveResult(
            pattern_hits=merged_hits,
            recall_size=len(merged_hits),
            debug_info={
                "literal_hits": len(literal_hits),
                "structure_hits": len(structure_hits),
                "block_hits": len(block_hits),
            }
        )

    # -------- Sub-retrievers (can be empty) --------

    def _retrieve_literal(self, query: str) -> List[PatternHit]:
        """
        Literal / token / n-gram based pattern matching.
        Engram-like capability lives here.
        """
        return []

    def _retrieve_structure(self, query: str) -> List[PatternHit]:
        """
        Structural / schema / slot-level pattern matching.
        """
        return []

    def _retrieve_block_type(self, query: str) -> List[PatternHit]:
        """
        Block-type or document-template level matching.
        """
        return []

    # -------- Merge & Normalize --------

    def _merge_hits(
        self,
        *hit_lists: List[PatternHit]
    ) -> List[PatternHit]:
        """
        Merge hits from different pattern channels.
        Deduplicate by block_id and aggregate hit_types.
        """
        hit_map: Dict[str, PatternHit] = {}

        for hits in hit_lists:
            for hit in hits:
                if hit.block_id not in hit_map:
                    hit_map[hit.block_id] = hit
                else:
                    existing = hit_map[hit.block_id]
                    existing.hit_types = list(
                        set(existing.hit_types + hit.hit_types)
                    )
                    existing.confidence = max(
                        existing.confidence,
                        hit.confidence
                    )

        return list(hit_map.values())



