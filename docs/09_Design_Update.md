# KVI System Invariants（不可违反的系统不变式）

## 0. 目的（Purpose）

本不变式用于约束 KVI 在医学语义系统中的核心行为，确保：
- 推理（Judgement）与证据（Evidence）严格解耦
- 任何生成能力都不会污染证据系统
- 系统在临床、合规、审计层面长期可演进

任何实现、优化、实验均不得违反以下不变式。

---

## 1. 模式划分不变式（Mode Separation）

KVI 系统 **必须且仅支持两种显式运行模式**：

### Mode A：Evidence Routing + Free Reasoning
- 用途：临床诊断 / 判断辅助
- 特性：允许 LLM 自由推理、综合、归纳
- 输出：诊断结论或判断性结果

### Mode B：Evidence Routing + Evidence Projection
- 用途：诊断依据 / 临床指南支撑 / 精准动作执行
- 特性：**禁止任何形式的生成**
- 输出：被路由与命中的原始证据条目

**系统不得存在隐式、自动或 fallback 的模式切换。**

---

## 2. 输出通道不变式（Output Channel Isolation）

两种模式 **必须使用不同且不可复用的输出结构**：

- Mode A 输出：
  - `diagnosis_result`
  - `reasoning_trace`（可选）
- Mode B 输出：
  - `evidence_projection`
  - `evidence_ids`
  - `evidence_texts`

**任何模式不得向对方的输出通道写入数据。**

---

## 3. 证据纯净性不变式（Evidence Purity）

以下内容 **严禁进入 Evidence / KVBank / FAISS**：

- LLM 生成文本（任何模式）
- 推理中间结论
- 模式 A 的诊断结果
- 用户主观输入的判断性语言

Evidence 系统 **仅允许写入**：
- 外部权威文本（指南、文献、原始记录）
- 经人工或规则确认的原始证据片段

---

## 4. 模式 B 的零生成不变式（Zero-Generation Guarantee）

在 Mode B 中：

- LLM 不得：
  - 改写证据
  - 总结证据
  - 合并多个证据为新句子
  - 引入未明确存在于证据中的事实

Mode B 的输出必须满足：
> **任一输出文本，均可在 Evidence 中逐字定位到原始来源。**

---

## 5. 失败可见性不变式（Failure Visibility）

Mode B 中若出现以下情况：

- 无证据命中
- 证据不足
- 指南不支持当前语义动作

系统 **必须显式返回失败状态**，例如：
- `NO_EVIDENCE_FOUND`
- `INSUFFICIENT_GUIDELINE_SUPPORT`

**禁止使用 Mode A 的合理性推理结果掩盖 Mode B 的失败。**

---

## 6. 单向依赖不变式（Dependency Direction）

系统依赖方向 **只能是单向的**：

系统依赖方向 **只能是单向的**：

Evidence System
↓
Evidence Routing
↓
Mode A / Mode B

严禁：
- Mode A 结果反向写回 Evidence
- Mode A 结果被 Mode B 直接或间接引用

---

## 7. 审计与合规不变式（Auditability）

系统必须支持以下审计能力：

- 任一 Mode B 输出可被追溯至具体 Evidence ID
- Mode A 与 Mode B 的调用日志可区分、可回放
- 模式、证据、输出三者关系可重建

---

## 8. UI 表达不变式（User Interface Semantics）

UI 必须明确区分两种模式：

- Mode A：显示为“诊断结果 / 推理建议”
- Mode B：显示为“依据 / 指南原文 / 证据引用”

UI 不得：
- 将 Mode A 输出标注为“指南”
- 将 Mode B 输出包装为“结论”

---

## 9. 不变式优先级（Priority）

当以下目标发生冲突时，优先级如下：

1. 证据纯净性
2. 模式隔离
3. 失败可见性
4. 用户体验
5. 推理完整性

---

## 10. 变更规则（Change Policy）

任何违反上述不变式的改动，必须：
- 明确标注为 **Breaking Change**
- 经过架构评审与合规评估
- 不得以“实验 / 临时方案 / 性能优化”为由绕过

---

> **本不变式是系统级约束，而非实现建议。**
>  
> **违反不变式的系统，将不再被视为 KVI。**
