【RAG + KV Bank + Action Router 的系统架构图 总体分层】
┌──────────────────────────────────────────────┐
│                  User Query                  │
└──────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────┐
│        Intent / Trigger Detection Layer      │
│  - Keyword / Regex                           │
│  - Intent Classifier (small model / rules)  │
└──────────────────────────────────────────────┘
           │                         │
           │ 命中可执行规则           │ 未命中
           ▼                         ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│     Action Router        │   │        RAG Pipeline       │
│  - Priority resolution  │   │  Retriever (ANN/BM25)    │
│  - Conflict handling    │   │  Raw Context Assembly    │
└──────────────────────────┘   └──────────────────────────┘
           │                         │
           ▼                         ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│        KV Bank           │   │           LLM            │
│  - Action Objects        │   │  Reasoning / Explanation │
│  - URL / API / Flow      │   └──────────────────────────┘
└──────────────────────────┘              │
           │                                ▼
           └──────────► Deterministic ◄────┘
                      Output Layer

【PRD设计稿】
# PRD：RAG + KV Bank + Action Router 知识执行系统（简版）

## 1. 背景与问题

在缺乏组织内部结构化知识时，通常采用 RAG（Retrieval-Augmented Generation）增强大模型能力。但在实践中发现：

- RAG + LLM **无法保证精准、确定性的输出**
- 对于 URL、API、流程等“必须命中即执行”的内容，存在：
  - 输出不稳定
  - 被改写、融合、幻觉
  - 不可审计、不可复现

因此，需要将“可生成知识”和“可执行知识”进行系统级解耦。

---

## 2. 目标（Goals）

构建一套：

- 能同时支持 **自然语言理解（RAG）**
- 又能保证 **确定性动作执行（URL / API / Workflow）**
- 且 **不将关键动作交给 LLM 生成**

的混合知识执行系统。

---

## 3. 非目标（Non-Goals）

- 不追求通过 prompt 或 temperature 控制来“逼迫 LLM 精确输出”
- 不将 URL / API 等关键动作注入 prompt 或 attention
- 不让 LLM 直接决定最终可执行动作

---

## 4. 核心设计原则

1. **LLM 负责“理解与解释”，不负责“执行”**
2. **凡是必须精准输出的内容，不进入 RAG、不进入 LLM**
3. **系统行为应是确定性、可审计、可回滚的**
4. **规则负责判断，知识负责承载**

---

## 5. 系统核心组件

### 5.1 Intent Classifier（意图识别）

- 输入：用户自然语言 Query
- 输出：离散的语义意图标签（intent label）
- 作用：
  - 将无限自然语言空间压缩为有限意图空间
  - 提升规则与路由层的可控性
- 说明：
  - Intent ≠ 关键词
  - Intent 作为 Router 的输入信号之一

---

### 5.2 KV Bank（可执行知识库）

#### 定位
- 承载 **可执行知识（Executable Knowledge）**
- 与 RAG 文本知识严格区分

#### 数据形态
- 源形态：JSON / YAML（人类可维护）
- 运行形态：只读 KV Map（机器执行）

#### 内容示例
```json
{
  "id": "SFTSV_GUIDE_NHC_202406",
  "triggers": {
    "intent": "public_health_guidance",
    "disease": "SFTSV"
  },
  "action": {
    "type": "url",
    "value": "https://xxx.gov.cn/sftsv_2024.pdf"
  },
  "metadata": {
    "source": "NHC",
    "version": "2024-06"
  },
  "priority": 10
}
