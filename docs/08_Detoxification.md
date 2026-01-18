# Pattern Contract Prompt — KVI 2.0（用于 GPT 代码生成器）

> **目的**  
> 将现有的 Pattern-first（non-semantic）机制，从“关键词 / 规则触发器”升级为  
> **Pattern Contract（信息契约声明器）**，用于约束后续 Semantic Retrieval 与 KV 注入，  
> 防止低信息、泛背景 Evidence 对 Base LLM 造成“注入毒性”。

---

## 一、角色与前提（必须遵守）

你是一个 **系统架构代码生成器**，目标是在 **不修改 Base LLM 权重、不改 Attention 公式、不引入文本级 RAG** 的前提下：

- 为 KVI 2.0 体系新增一个 **Pattern Contract 子系统**
- Pattern Contract **不负责检索答案**
- Pattern Contract **不进行语义 embedding 相似性计算**
- Pattern Contract **不调用 LLM 进行判断**
- Pattern Contract **只声明与校验“信息形态是否被兑现”**

---

## 二、Pattern Contract 的核心定义（不可更改）

> **Pattern Contract = Pattern 对后续 Evidence 提出的「结构化信息承诺」**

Pattern Contract 的职责只有三点：

1. 声明 **期望注入的知识类型**
2. 声明 **Evidence 必须满足的结构性信号**
3. 为 RIM / KVI 提供 **可程序校验的 gate 标准**

---

## 三、必须生成的核心数据结构

### 3.1 PatternContract（必须完整实现）

生成以下 Python dataclass（或等价结构）：

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PatternContract:
    pattern_id: str

    # 信息需求声明（WHAT）
    expected_information: Dict[str, Any]
    # 示例：
    # {
    #   "entity_types": ["drug"],
    #   "relation_types": ["approved_for", "studied_for", "repurposed_for"],
    #   "domain": "biomedical",
    #   "disease": "SFTSV"
    # }

    # 结构性验收标准（HOW）
    required_signals: Dict[str, List[str]]
    # 示例：
    # {
    #   "must_contain": ["drug_name", "study_context"],
    #   "must_not_be_only": ["general_fda_background", "virology_overview"]
    # }

    # 最低信息密度要求（0~1）
    min_information_density: float

四、Pattern-first 的新输出协议（强制）

所有 Pattern-first 命中，必须返回 PatternContract 列表，而不是仅返回 block_id / keyword：
def run_pattern_first(query_text: str) -> List[PatternContract]:
    ...
禁止返回以下内容作为最终产物：

❌ 纯关键词列表

❌ block_id（除非作为 pattern_id 的一部分）

❌ 语义 embedding

五、Contract Validation（去毒素核心模块）
5.1 Validator 接口（必须生成）
class PatternContractValidator:
    def validate(
        self,
        contract: PatternContract,
        evidence_blocks: List["KVItem"]
    ) -> dict:
        """
        返回：
        {
          "fulfilled": bool,
          "score": float,          # 0~1
          "violations": List[str]  # 未满足的契约条款
        }
        """
5.2 校验原则（不可违背）

不调用 LLM

不生成文本

不比较 Evidence 与 Base LLM 内置知识

只基于：

Evidence metadata

chunk tags / section 信息

规则化关键词 / 实体标注（若已存在）

六、Pattern Contract 与 RIM 的协作方式
6.1 在 RIM 中新增 gate（必须）

在 RIM 的 decision 阶段加入：
if not validator.validate(contract, evidence).fulfilled:
    reject_current_kv = True
    retrieve_more = True
6.2 语义检索（ANN）位置说明

ANN 检索 仍然存在

Pattern Contract 不替代 ANN

Pattern Contract 只对 ANN 结果做 验收 / 拒绝

---

## 七、关键修正（必读）

### 改动1：Pattern 契约不应作为“强一致性”全盘硬约束
过去把 Pattern 契约当作强一致性约束，会与下游现实产生冲突：

- RIM / KV Bank 的 block 粒度较粗
- schema 标注方式不稳定
- FDA 等弱上下文实体无法稳定命中

结果是 Introspection Gate 被迫持续触发：
retrieve_more → reject_current_kv 的死循环。

**结论**：Pattern 契约应被分级（Hard / Soft），只让 Hard 触发 reject。

### 改动2：Hard / Soft 是 topic 级配置（不是 prompt 级）

- Hard / Soft Pattern 来自该 topic 下 evidence 语料的结构统计
- 专家少量修正后冻结
- 描述的是“在这个知识子空间中，不可避免的信息结构”
- Prompt 不参与 Pattern 生成，只影响生成与排序

**工程实现建议**：
- Pattern 生成与配置绑定 topic（而非每次解析 prompt）
- CLI 仅作为 override/调试开关

七、针对上述设计，以下是示例：SFTSV + FDA 药物 Pattern Contract
PatternContract(
    pattern_id="schema:sftsv_fda_drug_research",
    expected_information={
        "entity_types": ["drug"],
        "relation_types": ["approved_for", "studied_for", "repurposed_for"],
        "domain": "biomedical",
        "disease": "SFTSV"
    },
    required_signals={
        "must_contain": [
            "drug_name",
            "study_context"
        ],
        "must_not_be_only": [
            "general_fda_description",
            "background_virology"
        ]
    },
    min_information_density=0.6
)

八、非目标（严禁生成）

❌ 不要让 Base LLM 判断 Evidence 是否正确
❌ 不要用 embedding 相似性作为契约满足条件
❌ 不要将 Pattern Contract 退化为 prompt 文本
❌ 不要做文本级 RAG

九、一句话原则（写在代码注释中）

Pattern 声明“我需要什么样的知识”，
而不是“我已经知道答案是什么”。

十、最终交付物清单（必须全部生成）

PatternContract 数据结构

Pattern-first → PatternContract 输出函数

PatternContractValidator 最小可运行实现

RIM gate 接入示例
十、最终交付物清单（必须全部生成）
PatternContract 数据结构
Pattern-first → PatternContract 输出函数
PatternContractValidator 最小可运行实现
RIM gate 接入示例
至少 1 个医学领域 Pattern Contract 示例（如 SFTSV）供测试使用
