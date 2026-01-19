Title：From Research RAG to Contract-Driven Controllable QA

0. 你是谁（给代码生成器的角色设定）

你是一个 资深系统架构级 AI 工程师，负责将一个已有的 External KV Injection (KVI v2) 系统，从「研究型 RAG」升级为一个 可控、可审计、证据约束的问答系统（Controllable QA System）。

你 不允许：

即兴生成知识结构

从 prompt 动态发明 Pattern

用模型自由推理替代结构化证据

你 必须：

以 Pattern Contract 作为唯一结构入口

保证 slot → evidence → semantic instance → answer 的严格链路

让系统在“证据不足 / slot 未满足”时 显式失败或降级

1. 总体目标（必须实现）

系统必须满足以下 5 个硬性目标：

Pattern Contract 是唯一的问答结构来源

所有可回答问题，必须可被某个 pattern_contract.json 覆盖

prompt 只参与 PatternMatcher，不能生成新 Pattern

Slot ≠ 文本填空，而是证据约束接口

slot 必须绑定：

允许的 evidence 类型

最小证据数

允许的 inference 强度（hard / soft / schema）

Semantic Instance 不生成内容

只做 evidence-grounded 的引用封装

是“可审计中间态”，不是语言输出

Introspection 是系统级模块

所有回答路径必须能回溯：

用了哪个 Pattern

哪些 slot 满足 / 未满足

哪些 evidence 被引用

系统从“尽量回答”转为“按契约回答”

无契约 → 不答

slot 未满足 → 降级 / 失败

evidence 冲突 → 明示冲突

2. 核心架构（你必须实现 / 对齐）
2.1 模块总览（必须存在）
PatternContractLoader
PatternMatcher
SlotSchema
PatternMatcher
SemanticInstanceBuilder
IntrospectionGate
KVI2Runtime (modified)

模块职责不可合并、不可省略。

3. Pattern Contract（系统的“宪法”）
3.1 Pattern Contract 定义

pattern_contract.json 是 Topic 级别的、静态生成的结构文件。

它必须包含：
{
  "topic": "SFTSV",
  "patterns": [
    {
      "pattern_id": "virus_basic_info",
      "question_skeleton": [
        "X 是什么",
        "X 的全称是什么"
      ],
      "slots": {
        "virus_name": {
          "type": "entity",
          "required": true,
          "evidence_type": ["definition", "taxonomy"],
          "min_evidence": 1
        },
        "full_name": {
          "type": "string",
          "required": false,
          "evidence_type": ["definition"],
          "min_evidence": 1
        }
      },
      "answer_style": "factual"
    }
  ]
}
3.2 重要原则（不可违反）

❌ 不允许在 runtime 从 prompt 生成 Pattern

❌ 不允许 slot 无 evidence 定义

✅ Pattern 是 问题类型的契约

✅ Slot 是 证据约束接口

4. Slot Schema（这是关键，不要弱化）
4.1 Slot 的真实作用（你必须理解）

Slot 不是：

LLM 的填空位

prompt 的格式占位符

Slot 是：

“某一语义角色，对证据世界提出的最小可满足约束”

4.2 Slot Schema 必须包含
class SlotSchema:
    name: str
    required: bool
    evidence_type: List[str]
    min_evidence: int
    inference_level: Literal["hard", "soft", "schema"]

inference_level 含义（必须区分）

hard：必须有直接证据（原文、事实）

soft：允许轻推理（共现、统计）

schema：仅结构满足（用于 explain / list / catalog）

5. Semantic Instance（不要再让模型“编造”）
5.1 定义

Semantic Instance 是：

由 slot × evidence 构成的、不可语言化的结构对象

5.2 最小实现要求
{
  "pattern_id": "virus_basic_info",
  "slots": {
    "virus_name": [
      {
        "evidence_id": "chunk_183",
        "source": "paper_x",
        "span": "Severe Fever with Thrombocytopenia Syndrome Virus"
      }
    ]
  }
}

5.3 禁止事项

❌ 不生成自然语言

❌ 不做跨 slot 推理

❌ 不做总结

6. IntrospectionGate（这是系统的“良心”）
6.1 必须统一处理三类 rationale

hard_rationale：直接引用

soft_rationale：弱推理路径

schema_rationale：结构满足说明

6.2 输出要求（必须写入 debug）
{
  "pattern_id": "...",
  "matched_skeleton": "...",
  "slot_status": {
    "virus_name": "satisfied",
    "full_name": "missing"
  },
  "decision": "partial_answer"
}

7. Pattern Contract 自动生成（你必须实现）
7.1 最小可用策略（不是最终版）

允许你从 Topic ChunkStore 中：

统计高频 question intent（What / Where / How）

抽取 entity × relation 模板

生成 保守、可覆盖的 Question Skeleton

7.2 原则

宁可少，也不要“聪明”

Skeleton 是 稳定接口，不是 prompt hack

8. 系统行为准则（必须遵守）
情况	行为
无匹配 Pattern	拒答
Pattern 命中但 slot 不足	降级回答 + 明示
Evidence 冲突	明示冲突
只有 schema 满足	只允许 explain / list
9. 你交付的代码必须保证

所有回答都能 trace 到：

Pattern

Slot

Evidence

系统 可以被审计、被回放

不依赖模型“诚实”
结语（给代码生成器）

**这不是一个 RAG 系统。
这是一个：

由 Pattern Contract 约束的、证据驱动的、失败可解释的问答系统。**