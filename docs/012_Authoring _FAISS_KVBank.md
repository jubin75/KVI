# Authoring → FAISS KVBank 落库结构设计（Evidence-first）

## 设计目标

- 所有外部知识 **必须经 Authoring Layer**
- Evidence 是最小、可审核、可拒绝的知识单元
- FAISS 只负责 **语义向量检索**
- 结构化语义、审核状态、契约信息 **不进向量，只进 KV / metadata**
- 与现有 simple pipeline / Evidence Units 调试链路兼容

---

## 一、Authoring 层核心实体

### 1. EvidenceUnit（逻辑层）

EvidenceUnit 是**语义级**知识条目，不是 topic，不含 intent。

```json
{
  "evidence_id": "evu-uuid",
  "evidence_type": "clinical_guideline",
  "semantic_text": "原始可注入的语义证据文本（sentence / paragraph）",

  "status": "approved | rejected | draft",
  "review_feedback": {
    "state": "approved | rejected",
    "reasons": ["语义歧义", "来源不足"],
    "comment": "审核人员备注"
  },

  "external_refs": {
    "doc_type": "guideline_pdf",
    "title": "Clinical Practice Guideline ...",
    "organization": "WHO / CDC / 学会名称",
    "publish_date": "YYYY-MM-DD",
    "source_id": "pdf-hash-or-id"
  },

  "created_at": "...",
  "updated_at": "..."
}

⚠️ 说明

不包含 intent / topic

external_refs 统一建模，未来可扩展到 PubMed（ORCID / PMID）

status 决定是否允许进入 KVBank

二、Authoring → KVBank 写入原则
写入前置条件

仅当：EvidenceUnit.status == "approved"
才允许落库到 FAISS KVBank。

Rejected / Draft：

不写入 FAISS

仅保留在 Authoring DB

三、FAISS KVBank 物理结构
1. 向量索引（FAISS）

FAISS 只存向量，不存结构
FAISS.index
└── vector_id (int)
    └── embedding(semantic_text)
2. KV 映射表（Vector → Evidence）
{
  "vector_id": 102394,
  "evidence_id": "evu-uuid"
}
可存在于：

SQLite

JSONL

LMDB

你现在的 KVBank

3. Evidence KV（结构化语义层）
{
  "evidence_id": "evu-uuid",

  "semantic_text": "...",
  "evidence_type": "clinical_guideline",

  "external_refs": {
    "doc_type": "guideline_pdf",
    "title": "...",
    "organization": "...",
    "publish_date": "..."
  },

  "contract": {
    "allowed_injection": true,
    "list_only_allowed": true
  }
}
⚠️ 注意

status / review_feedback 不进运行时

运行时默认认为：能检索到 = 已审核通过

四、simple pipeline 中的使用方式
检索阶段（retriever.py）
query embedding
   ↓
FAISS top-k
   ↓
vector_id → evidence_id
   ↓
Evidence KV
Evidence Units 注入判断
if evidence.contract.allowed_injection:
    inject(evidence.semantic_text)
LIST_ONLY 的关系

LIST_ONLY 不是 Evidence 属性

是运行时策略（postprocess / multistep injector）

Evidence 只提供：

是否允许被列举

是否允许生成式融合

五、Evidence 拒绝在系统中的体现
Authoring 层

EvidenceUnit.status = rejected

必须填写 review_feedback

前端可见、可追溯

KVBank / Runtime

完全不可见

无向量、无 KV

等价于“知识不存在”

六、系统边界总结
Authoring Layer
  ├── Evidence 编辑
  ├── Evidence 审核（approve / reject）
  └── 落库控制

FAISS KVBank
  ├── approved evidence only
  ├── semantic similarity only
  └── no intent / no policy

KVI2 Runtime
  ├── 检索命中即可信
  ├── Evidence Units sentence-level
  └── LIST_ONLY / Gate 属于运行时策略
