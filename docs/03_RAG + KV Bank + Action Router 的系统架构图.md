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

---

## 6. 从 KVI 到“可执行知识系统”：IM 内省驱动的 SOP（再编译为 Tool 协议）

> 本节回答一个更强的问题：我们是否能把“知识增强问答（KVI）”升级为“可靠 SOP 输出”，并进一步让 SOP 变成可调用外部 tools 的协议？
>
> 结论：**可行，但必须走“分层 + 编译”的工程路线**，而不是让 LLM 直接决定执行逻辑。

### 6.1 目标：把“回答”升级为“可靠 SOP”

传统 RAG/问答的输出是“解释文本”，可靠性依赖模型自律；而 SOP 的目标是：
- **可操作**：每一步都能被执行器理解（输入/条件/动作/输出）
- **可审计**：每一步都能追溯到证据（Evidence Units / block ids / 来源）
- **可回滚**：执行失败时能有确定性处理（停止/回滚/人工确认）

KVI 的价值在于：外部证据以 `K_ext/V_ext` 的形式进入 Attention，使“证据”更可能成为计算的一部分；而 RIM/Contract/IM 的价值在于：在生成前/生成后以**非生成**方式做控制与验收，避免“看起来像 SOP 但其实在编造”的风险。

### 6.2 架构形态：SOP 作为中间表示（IR），Tool 调用作为编译目标

推荐的在线推理形态（高层）：

1) **KVI 检索与注入（知识增强）**
   - 从 KVBank 检索候选 evidence blocks（或 sentence-level evidence units）
   - 注入到模型 cache（past_key_values prefix）

2) **IM / RIM 内省（非生成 Gate）**
   - 判断：当前证据是否满足 pattern contract / 信息密度 / 语义一致性
   - 判断：注入是否有效（例如：低影响则换一批 KV；或直接 fail-closed）
   - 产出：允许生成 SOP / 需要补证据 / 直接回退（base LLM）/ 拒绝执行

3) **生成结构化 SOP（受约束的输出协议）**
   - 输出不是自由段落，而是字段化 SOP（见 6.3），并在关键步骤附带 evidence 引用
   - 这是“可执行知识系统”的核心：**把 LLM 的自由度限制在一个可检查的 IR 内**

4) **SOP → Tool Plan 编译（确定性）**
   - 把 SOP 编译为严格 JSON：`tool_name/args/precondition/postcondition/rollback`
   - 编译是确定性规则（不是 LLM 生成），确保可复现、可审计

5) **Tool Executor 执行（确定性）**
   - 只执行编译后的计划，不允许 LLM 临时改写调用参数
   - 记录执行日志，并可将结果作为新 evidence 进入下一轮（可选）

### 6.3 SOP 的最小输出协议（建议）

最小 SOP JSON（示意）：
- `goal`: 目标（用户问题的“操作化”表达）
- `scope`: 适用范围（适用人群/场景/前提）
- `inputs`: 必要输入（用户需提供的信息，或可由系统观测到的变量）
- `steps[]`:
  - `id`
  - `action`: 自然语言动作描述（可被编译器映射到工具）
  - `preconditions[]`: 可判定条件（必须是可观测变量/结构化信号）
  - `outputs[]`: 预期产物/状态变化
  - `evidence[]`: 证据引用（block_id / sentence unit id / source）
  - `risk_level`: low/medium/high
  - `requires_human_confirmation`: bool（高风险步骤强制）
- `stop_conditions[]`: 必须停止/转人工的条件
- `audit`: 生成时的路由/证据摘要（用于审计）

### 6.4 与当前仓库实现的对应关系（“已有 vs 缺口”）

**已有（可以复用的能力）**：
- **RIM 是非生成控制模块**：`src/rim.py`（MUST NOT generate tokens）
- **Contract 验证是 metadata-only**：`src/pattern_contract.py`（No LLM / No embeddings / No text RAG）
- **KVI2 runtime 已有 pattern-first + semantic-second + gate**：`src/runtime/kvi2_runtime.py`
- **Evidence policy / fallback 已在 postprocess**：`src/runtime/postprocess.py`（evidence_expected_slots / fallback）
- **sentence-level Evidence Units（simple pipeline 已接入）**：`scripts/run_kvi2_runtime_test.py`（`--simple_use_evidence_units`）

**缺口（需要新增的模块）**：
- SOP 的结构化 schema（IR）与解析/校验器
- IM 针对 SOP 的“可执行性检查项”（前置条件、分支可判定、风险/确认、证据绑定）
- SOP→ToolPlan 的确定性编译器
- Tool Executor（以及可审计/可回滚的执行日志）

### 6.5 为什么这条路线有产品价值（而非“简单 Action Router”）

“输入 query → 命中 action object → 输出 URL/API”只解决“确定性选择”，但不解决：
- **用户真正要的流程**（多步、带条件、带风险控制）
- **证据与步骤绑定**（高风险领域需要）
- **执行前验收/执行后审计**（可回滚、可复现）

而“KVI → IM → SOP → 编译执行”可以把“知识增强”变成“可靠流程”，并把 LLM 的不确定性限制在可验收的 IR 里，最终由确定性编译与执行兜底。
