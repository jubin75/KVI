# Mode B 设计要点与不变式

本文档定义 Mode B（Evidence Routing + Evidence Projection）的设计要点、输入输出契约与编译约束。Mode B 是 grounding_only 的证据投影通道，禁止任何形式的生成与推断。

---

## 1. Mode B 角色定位

- 模式定义：Evidence Routing + Evidence Projection（非生成）
- 角色轴（role_axis）：`grounding_only`
- 目标：以可审计、可复现的方式输出原始证据投影

Mode B 与 RAG 根本不同：不拼接证据进入生成路径，不产出自然语言总结。

---

## 2. 输入 / 输出契约（硬边界）

Input（来自 Evidence Routing System）：

- `query.text: string`
- `query.semantic_primary: Axis`
- `query.role_axis: "grounding_only"`
- `evidence_candidates: List[EvidenceUnit]`

EvidenceUnit：

- `id: string`
- `text: string`
- `semantic_primary: Axis`
- `quality_score: float`
- `source_ref: { type, name, section }`

Output（Evidence Projection Document）：

- `mode: "B"`
- `query`
- `evidence_projection: List[ProjectionBlock]`
- `scope_note`
- `generation_policy`

---

## 3. 编译不变式（硬约束）

以下任一违规必须编译失败（hard error）：

1. `LLM_GENERATION == FORBIDDEN`
2. `NO_TEXT_SYNTHESIS`
3. `NO_EVIDENCE_MERGE`
4. `NO_INFERENCE`
5. `NO_COMPLETION`

---

## 4. 编译流程（概览）

```
function compileEvidenceProjection(input):
  assert input.query.role_axis == "grounding_only"

  filtered = filterEvidence(input.evidence_candidates, input.query)
  grouped  = groupByProjectionType(filtered)
  projected_blocks = []

  for each group in grouped:
    block = compileProjectionBlock(group)
    projected_blocks.append(block)

  scope = computeScopeNote(input, projected_blocks)

  return assembleDocument(query=input.query, blocks=projected_blocks, scope=scope)
```

---

## 5. Evidence 过滤（只裁剪）

规则：不重排、不补全、不合并，只判断“是否可投影”。

- `ev.semantic_primary == query.semantic_primary`
- `ev.quality_score >= MIN_QUALITY_THRESHOLD`
- `ev.source_ref != null`

---

## 6. Projection Type 映射（纯映射）

```
symptom   -> clinical_manifestation
drug      -> treatment_recommendation
mechanism -> pathophysiology_statement
location  -> epidemiologic_distribution
```

不支持的 Axis 必须 hard fail。

---

## 7. Projection Block 编译（固定模板）

固定 statement（不可生成）：

- clinical_manifestation: "指南或权威资料中明确记载的临床表现包括："
- treatment_recommendation: "指南中列出的治疗或用药建议包括："
- pathophysiology_statement: "权威资料中描述的发病机制包括："
- epidemiologic_distribution: "文献中报道的流行病学分布包括："

---

## 8. Evidence Item 编译（逐条直出）

逐条输出，原文直出，不改一个字：

```
{
  evidence_id: ev.id,
  text: ev.text,
  source: {
    type: ev.source_ref.type,
    name: ev.source_ref.name,
    section: ev.source_ref.section
  }
}
```

禁止：paraphrase / summarize / enumerate / rewrite。

---

## 9. Scope Note（诚实声明）

- 若无有效 block：`completeness = "empty"`，原因说明未命中权威证据条目
- 否则：`completeness = "partial"`，声明未进行补全或推断

---

## 10. 最终组装（含铁律声明）

```
{
  mode: "B",
  query: {
    text,
    semantic_primary,
    role_axis: "grounding_only"
  },
  evidence_projection: blocks,
  scope_note,
  generation_policy: {
    llm_generation: "disabled",
    reason: "Mode B evidence projection only"
  }
}
```

---

## 11. 审计与合规要求

- 任一输出必须可追溯到 `evidence_id` 与原文
- 模式输出不可复用（Mode A ≠ Mode B）
- Mode B 失败必须显式返回（不可用推理掩盖）

---

> **Mode B 是系统级约束，不是建议实现。**
>  
> **任何与上述约束冲突的改动必须标注为 Breaking Change。**
