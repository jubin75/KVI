# Pattern-first → Candidate Schema Scoring  
## 最小改造规范（冻结版 · Codegen Prompt）

### 目标
将现有 **Pattern-first 的“单一命中即裁决”机制**，改造为  
**“Candidate Schema Scoring（多候选 + 竞争选择）”机制**，  
避免低信息 schema（如 abbr）抢占高信息查询（如 地理分布 / 流行病学）。

本规范 **不要求新增 schema，不要求穷举触发词，不要求重构下游模块**。

---

### 核心设计原则（必须遵守）

#### 原则 1：Pattern-first 不再做“排他裁决”
- Pattern-first **只能生成候选 schema 集合**
- 不得在此阶段：
  - 决定最终 schema
  - 触发 contract
  - 过滤 evidence
- 单一 pattern 命中 **不得直接决定 gate.pattern_id**

---

#### 原则 2：Schema 之间是“竞争关系”，不是“先到先得”
- 一个 query **允许同时命中多个 schema**
- 每个 schema 必须被赋予一个 **schema_intent_score**
- 最终 schema 由 **score 排序 + gate 决策**确定

---

#### 原则 3：低信息密度 schema 永远不能抢占高信息 schema
- 以下类型 schema 必须被视为 **低信息 / 辅助 schema**：
  - abbr / full_name
  - definition-only
- 当 query 同时显式包含：
  - 时间范围
  - 空间/分布谓词
  - 枚举/统计意图  
  → 低信息 schema **自动降权或仅作为兜底**

---

#### 原则 4：Schema scoring 基于“问题结构”，不是关键词枚举
Schema intent scoring **不得依赖硬编码触发词列表**，  
而应基于以下抽象信号（实现方式不限制）：
- 是否包含时间范围
- 是否包含空间/地域约束
- 是否是枚举型问题（“有哪些”）
- 是否是定义型问题（“是什么”）
- 是否请求事实分布 / 比例 / 区域

---

#### 原则 5：Gate 只接收“已排序的候选 schema”
- Gate 输入必须变为：
  - `candidate_schemas: [{schema_id, score, rationale}]`
- Gate 负责：
  - 选择最终 schema
  - 或拒答（若所有 schema 不可实例化）
- Pattern-first **不得直接写 gate.pattern_id**

---

### 强制行为约束（验收标准）

- ❌ 不允许出现：  
  “命中 abbr → 自动忽略 geographic_distribution / epidemiology / temporal schema”
- ❌ 不允许通过“不断加触发词”解决路由问题
- ✅ 必须允许以下行为自然发生：  
  > Query 同时命中 abbr + geographic_distribution，  
  > 但最终选择 geographic_distribution

---

### 允许的最小实现方式（不限制）
以下任选其一或组合，**不要求全部实现**：
- score-based ranking
- priority class（high / medium / low）
- soft-penalty（对 abbr 降权）
- late binding（gate 决策）

---

### 不在本规范范围内（明确排除）
- 不修改 Evidence Unit 抽取
- 不修改 SlotExtractor / Cleaner
- 不修改 KVBank
- 不新增 schema 类型
- 不要求重跑 embedding / index

---

### 最终验收现象（必须可观测）
对于以下问题类型：

> “SFTSV 在 2009–2014 年我国的主要发病地区有哪些？”

系统应满足：
- abbr 不作为最终 schema
- geographic / distribution / epidemiology schema 获胜
- 能进入 evidence 检索与 slot 填充路径
- 不因 abbr contract 过滤掉分布证据

---

### 一句话冻结结论
> Pattern-first 负责“提出可能性”，  
> Schema scoring 负责“决定优先级”，  
> Gate 负责“是否允许回答”。

以上职责 **不得混用**。
