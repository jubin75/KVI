《KV Evidence 构造最小增强规范（冻结版）》

Version: v1.0 (Frozen)
Scope: Evidence Construction Layer
Non-goal: 不涉及 schema / gate / cleaner 的逻辑调整

1. 设计目标（Design Goals）

本规范用于在 KVBank 构建阶段，将原始文献 block 提升为可被下游结构化消费的 Evidence 单元，同时满足：

语义对齐优先于结构对齐

证据质量在早期判定，而非后期修补

不引入任务特定（task-specific）硬编码

允许未来 schema 扩展而无需重构 Evidence 层

2. 核心设计原则（Frozen Principles）
P1. Evidence ≠ Block

block 是文档切分产物

evidence 是语义可注入单元

一个 block 可以包含 0～N 个 evidence unit

P2. Evidence 单元必须具备“语义角色”

Evidence 的最小判定标准不是格式，而是：

该文本在知识层面“扮演什么角色”

P3. Evidence 层不做 schema 判断

Evidence 层：

❌ 不知道 clinical / chemical / epidemiology

✅ 只提供 语义角色 + 结构线索 + 注入潜力

3. KV Evidence 的最小数据结构（冻结）
3.1 顶层结构
{
  "block_id": "doc123#p5",
  "source_doc": "doc123",
  "evidence_units": [ ... ]
}
3.2 Evidence Unit（核心）
{
  "unit_id": "doc123#p5#u2",
  "text": "...",
  "semantic_role": "X",
  "structural_features": { ... },
  "injectability": { ... }
}
4. Semantic Role（语义角色，冻结接口）
4.1 定义

semantic_role 描述该 evidence 在知识抽象层面的功能，而非领域标签。

4.2 冻结的最小角色集合（v1）

⚠️ 这是开放集合，但接口冻结
enumerative_fact        // 枚举性事实（list 或并列句）
descriptive_statement   // 描述性陈述
relational_statement    // 实体-实体关系
procedural_note         // 方法/流程说明
contextual_background   // 背景/讨论
metadata_notice         // 版权、数据声明、补充说明
示例映射（非规则，仅示意）：
| 文本类型                                      | semantic_role    |
| ----------------------------------------- | ---------------- |
| “A, B, and C were observed”               | enumerative_fact |
| “Patients typically present with …”       | enumerative_fact |
| “Figure 2 shows …”                        | metadata_notice  |
| “We performed a retrospective analysis …” | procedural_note  |

5. Structural Features（结构线索，非决定性）
"structural_features": {
  "has_list_structure": true,
  "list_style": "bullet | numbered | inline",
  "list_item_count": 3,
  "sentence_parallelism": true
}
说明：

结构线索 只能作为候选信号

不可单独决定是否进入 schema

6. Injectability（注入潜力，核心增强点）
6.1 定义

injectability 是 Evidence 层对**“是否值得进入知识注入链路”**的最小判断。

6.2 冻结字段
"injectability": {
  "score": 0.0,
  "signals": [],
  "blocking_reasons": []
}

6.3 Signals（正向信号）

示例（非穷举）：
semantic_role_enumerative
explicit_entity_anchor
low_context_dependency
low_ambiguity
6.4 Blocking Reasons（否决信号）
metadata_only
procedural_only
requires_external_context
mixed_semantics_unresolved
❗ 只要存在 blocking_reasons，Evidence 不进入 schema 候选池

7. Evidence Unit 构造流程（冻结）
Document
 └─ Block
     ├─ Evidence Unit Detection
     │    ├─ Sentence / Sub-span 切分
     │    └─ 混杂语义分离
     ├─ Semantic Role Assignment
     ├─ Structural Feature Extraction
     └─ Injectability Scoring
8. 明确不做的事情（Non-Goals）

Evidence 层 不允许：

❌ 针对某个 schema 写规则

❌ 修复下游抽取失败

❌ 为“勉强可用”证据打补丁

❌ 引入领域私有 ontology

9. 与下游模块的冻结契约（Interface Contract）

Evidence 层 只保证：
- semantic_role
- structural_features
- injectability
下游模块（retrieval / schema / gate）：

❌ 不反向修改 Evidence

❌ 不重新解释文本角色

✅ 只消费这些字段

10. 为什么这个规范是“可扩展但可冻结”的

扩展点：

新 semantic_role

新 injectability signal

冻结点：

Evidence Unit 结构

责任边界

上下游契约

这保证了：系统能力增长 ≠ 架构复杂度增长