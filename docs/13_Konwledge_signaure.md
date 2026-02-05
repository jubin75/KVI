Signature Schema · Minimal Sustainable Spec (v1)

目标：
为 Evidence Routing / Knowledge Injection 提供稳定、可维护、行为可控的语义签名结构，
支撑 Mode A（推理生成）与 Mode B（证据投影），
严格避免 topic / intent / role 的语义混叠。

1. 设计不变式（必须遵守）

signature 只描述“语义维度 + 使用权限”

不描述问题类型

不描述推理策略

不描述具体实体

signature 在编译期/Authoring Layer 确定

运行时禁止修改

KVI2 不得“自动推断 role”

role_axis 决定 Evidence 的“可用行为”

不是“内容是什么”

是“系统可以用它来做什么”

2. Signature 顶层结构（JSON）
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

3. primary_axis（语义主轴）

描述 Evidence 的语义空间位置
数量必须 ≤ 10，长期稳定

{
  "symptom",
  "mechanism",
  "drug",
  "location",
  "diagnosis",
  "prognosis",
  "prevention",
  "laboratory",
  "population",
  "guideline"
}


规则：

每条 Evidence 只能有一个 primary_axis

primary_axis ≠ topic

不允许派生子轴（禁止 symptom/early_symptom 这种）

4. role_axis（行为权限轴）

描述 系统如何使用这条 Evidence

{
  "reasoning_input",
  "free_generation",
  "grounding_only",
  "reference_only",
  "action_trigger"
}

语义解释（必须遵守）
role_axis	系统行为
reasoning_input	可参与推理链，但不可作为最终事实
free_generation	允许 LLM 综合发挥（高风险）
grounding_only	只能作为生成依据，不得扩写
reference_only	只能展示/引用，不参与推理
action_trigger	驱动确定性流程或工具
5. role → constraints 映射（不可覆盖）
{
  "reasoning_input": {
    "allow_generation": true,
    "allow_reasoning": true,
    "allow_projection": false
  },
  "free_generation": {
    "allow_generation": true,
    "allow_reasoning": true,
    "allow_projection": false
  },
  "grounding_only": {
    "allow_generation": false,
    "allow_reasoning": false,
    "allow_projection": true
  },
  "reference_only": {
    "allow_generation": false,
    "allow_reasoning": false,
    "allow_projection": true
  },
  "action_trigger": {
    "allow_generation": false,
    "allow_reasoning": false,
    "allow_projection": false
  }
}


❗ KVI Runtime 只能读取 constraints，不能修改

6. Mode A / Mode B 支持规则
Mode A：Evidence Routing + 自由推理

允许 role：

["reasoning_input", "free_generation"]

Mode B：Evidence Routing + 证据投影

允许 role：

["grounding_only", "reference_only"]


❗ Mode B 下，任何 free_generation Evidence 必须被拒绝

7. 校验规则（编译期强校验）
必须失败的情况

primary_axis ∉ allowed set

role_axis ∉ allowed set

role_axis 与 constraints 不匹配

同一 Evidence 同时标记 reasoning_input 与 grounding_only

拒绝策略（非 LLM）
{
  "status": "REJECTED",
  "reason": "ROLE_CONSTRAINT_VIOLATION",
  "detail": "Evidence marked as grounding_only cannot allow_generation"
}

8. 与 Evidence Unit 的关系

signature 不包含 sentence 内容

EvidenceUnit 在落库时：

绑定一个 signature_id

运行时只使用 signature 做路由与控制

{
  "evidence_id": "sent_xxx",
  "signature_id": "sig_symptom_grounding_v1",
  "text": "..."
}

9. 明确禁止的设计（防止系统退化）

❌ 在 signature 中引入 intent / task / question
❌ role_axis 与 primary_axis 语义重叠
❌ 运行时基于 evidence 内容推断 role
❌ 用 threshold / rerank 修补 role 冲突

10. 一句话总结（给 GPT 用）

Signature 是 “语义位置 + 行为许可” 的不可变声明，
用来约束系统如何使用 Evidence，
而不是帮助模型理解 Evidence 内容。