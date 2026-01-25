# Knowledge Authoring Layer × EvidenceUnit 统一设计规范（MVP · 修订版）

> 目标：将 Evidence.txt 升级为**可人工编写、审核、注入的 EvidenceUnit 体系**，  
> 面向临床指南 / 专家共识 / 文献类知识，  
> 以**语义单元而非问题意图**为中心，避免路由规则爆炸。

---

## 一、核心设计原则（冻结）

1. **Evidence 是“语义条目”，不是“问题答案”**
   - 不包含 intent / question / use-case
   - 不感知用户问题类型

2. **EvidenceUnit 只做三件事**
   - 声明自身的 `semantic_type`
   - 明确可填充的 `slot_projection`
   - 提供可追溯来源（provenance）

3. **Evidence 检索基于语义相似性**
   - 多条 Evidence 语义接近 → 允许全部注入
   - 不做 intent 分流、不做 topic 路由

4. **所有可注入知识必须来自 Authoring Layer**
   - 禁止 KVI2 自动生成 Evidence
   - 禁止 runtime 动态构造事实

---

## 二、EvidenceUnit 数据模型（JSON Schema）

### 2.1 EvidenceUnit（抽象语义级）

```json
{
  "evidence_id": "EVIDENCE_000001",
  "semantic_type": "drug | symptom | location | procedure | laboratory | population | outcome",
  "schema_id": "schema:*",
  "claim": "Textual statement of a single evidence unit.",
  "polarity": "positive | negative | neutral",
  "slot_projection": {
    "<slot_name>": ["value1", "value2"]
  },
  "status": "draft | reviewed | approved",
  "provenance": {
    "source_type": "guideline | consensus | review | original_study",
    "organization": "string | null",
    "document_title": "string",
    "publication_year": 2024,
    "page_range": "string | null"
  },
  "external_refs": {
    "document_id": "string | null",
    "pmid": "string | null",
    "orcid": "string | null",
    "title": "string | null",
    "abstract": "string | null",
    "authors": [],
    "published_at": "string | null"
  }
}

三、Authoring Layer 的职责边界
3.1 Authoring Layer 做什么

将 PDF / 文献内容人工拆分为语义 EvidenceUnit

明确：

semantic_type

schema_id

slot_projection

审核与发布 EvidenceUnit

3.2 Authoring Layer 不做什么

不识别用户 intent

不做 schema 路由

不做 Evidence 推断或补全

四、与 Evidence → Slot 的对接规范
4.1 注入前校验（硬约束）
assert evidence.status == "approved"
assert evidence.semantic_type == slot.semantic_type
assert evidence.schema_id == active_schema
失败 → 直接拒绝该 EvidenceUnit
五、simple pipeline 的最小改造

禁用 sentence-level 动态 Evidence 抽取

检索目标从 block → EvidenceUnit

semantic similarity → top-k EvidenceUnits

直接进入 SlotExtractor（无 Gate 路由）

六、前端 Authoring 页面（MVP）

Evidence 列表（semantic_type / schema / status）

Evidence 编辑（结构化表单 + 原文引用）

Evidence 审核（approve / reject）

版本记录（可选）

==================================
Evidence 拒绝 → 前端 Authoring UI 可理解审核反馈设计（冻结版）
一、设计目标（不偏航）

把“系统为什么不用这条 Evidence”说清楚，
而不是教系统怎么修这条 Evidence。

核心原则：

❌ 不自动修补

❌ 不提示“加规则”

✅ 只解释“为什么不能作为知识”

✅ 让人类决定“改文本 / 换 schema / 放弃”

二、Evidence 状态机（Authoring 视角）

每条 EvidenceUnit 必须有 显式状态：
EvidenceStatus =
  | "draft"        // 编辑中
  | "approved"     // 可注入
  | "rejected"     // 不可注入（有原因）
Rejected ≠ Bad
Rejected = “当前不符合注入契约”

三、拒绝原因模型（后端 → 前端）
1️⃣ 后端输出统一结构（冻结）
"rejection": {
  "code": "SEMANTIC_TYPE_MISMATCH",
  "message": "Evidence semantic_type does not match target slot.",
  "details": {
    "expected": "drug",
    "actual": "location"
  },
  "confidence": 0.93
}
允许的 rejection.code（有限集，防爆炸）
| code                   | 含义                 |
| ---------------------- | ------------------ |
| SEMANTIC_TYPE_MISMATCH | 语义类型不匹配            |
| SCHEMA_MISMATCH        | schema 不一致         |
| NON_ENUMERATIVE        | 非条目式知识             |
| MIXED_SEMANTICS        | 单条 Evidence 混合多类事实 |
| LOW_CONFIDENCE         | 相似度/置信度不足          |
| NOT_APPROVED           | 尚未审核通过             |

🚫 禁止出现：

topic_xxx

custom_rule_xxx

per-schema 特例 code
四、前端 UI 展示设计（重点）
Evidence 列表视图（核心）

每条 EvidenceUnit 显示为一张卡片：
[ REJECTED ❌ ] Evidence #12
---------------------------------
Text:
"reported in China and 23 provinces..."

Semantic Type: location
Target Slot: drug

Reason:
❌ Semantic type mismatch
This evidence describes a location, but the current question
requires a drug/treatment evidence.

What you can do:
• Change semantic type
• Move to a different schema
• Edit the evidence text
• Leave it rejected
✅ 重点：
前端只提供“操作建议”，不提供“系统建议规则”

五、Evidence 详情页（编辑视角）
必须可见的 4 件事

1️⃣ 原始文本（不可隐藏）
2️⃣ semantic_type（可修改）
3️⃣ schema_id（可切换）
4️⃣ 当前状态 + 拒绝原因（不可删除）

六、拒绝与系统行为的关系（给用户看的说明）

前端需有一条固定文案（非常重要）：

Rejected evidence will never be injected into answers.
This does not mean the evidence is wrong —
only that it is not suitable for the current knowledge slot.

这句话是**防止用户误解系统“坏了”**的关键。

七、LIST_ONLY / 输出层联动说明（隐藏但必要）

前端无需暴露 LIST_ONLY 细节，只需在 slot 级展示：
Slot: clinical_features
Status: ❌ Not satisfied
Reason: No approved evidence units
用户不需要知道：

gate

projection

cleaner

他们只需要知道：
“我没给你可用的知识”