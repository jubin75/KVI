《SchemaValueCleaner · 最小可扩展规则系统》MD Prompt
目标

实现一个 SchemaValueCleaner（或 SlotValueCleaner），
用于对 SemanticInstanceBuilder 输出的 slot values 做统一、可控、可审计的清洗与归一。

重要约束（必须遵守）

❌ 不允许为具体 schema（如 symptom / drug）写专用 Cleaner 类

❌ 不允许在代码中硬编码领域规则

✅ 所有规则必须数据驱动（YAML / JSON）

✅ Cleaner 只处理“值”，不参与 Gate / Pattern / Runtime 决策

一、模块位置与文件

新增文件：
external_kv_injection/src/semantic/schema_value_cleaner.py
external_kv_injection/config/value_cleaning_rules/
    ├── symptom.yaml
    ├── drug.yaml
    ├── generic.yaml

二、核心职责（必须只做这些）

SchemaValueCleaner 只做 三件事：

normalize：同义归一

split：拆分粘连短语

filter：过滤不属于该 semantic_type 的值

不允许做：
推理
补全
改写事实
访问证据原文全文

三、最小 Python 接口（必须严格一致）
class SchemaValueCleaner:
    def __init__(self, rule_dir: str):
        """
        rule_dir: 包含 <semantic_type>.yaml 的目录
        """

    def clean(
        self,
        values: list[str],
        semantic_type: str,
        evidence_ids: list[str] | None = None
    ) -> dict:
        """
        输入：
          values: 原始 slot 值列表
          semantic_type: 如 'symptom', 'drug', 'gene'
          evidence_ids: 可选，仅用于 debug 追踪

        输出（必须是结构化 dict）：
        {
          "cleaned_values": [...],
          "removed_values": [...],
          "normalization_map": {raw: normalized},
          "split_map": {raw: [parts]},
          "debug": {...}
        }
        """
四、规则系统（必须数据驱动）
规则加载逻辑

根据 semantic_type 加载：

generic.yaml

<semantic_type>.yaml（若存在）

同名规则 semantic_type 覆盖 generic

规则文件格式（最小规范）
示例：symptom.yaml
semantic_type: symptom

normalize:
  thrombocytopenia:
    - platelet counts
    - low platelet
    - decreased platelets

split:
  connectors:
    - " of "
    - " and "
    - " with "

filter:
  deny_terms:
    - clear mind
    - conscious
    - alert

  deny_semantic_classes:
    - mental_state
    - assessment
示例：generic.yaml
normalize: {}

split:
  connectors:
    - ","
    - ";"

filter:
  deny_terms: []
五、Cleaner 执行顺序（不可改变）
raw values
   ↓
normalize
   ↓
split
   ↓
filter
   ↓
cleaned_values
六、与现有系统的唯一接入点
接入位置（唯一合法位置）

在 SemanticInstanceBuilder 完成 slot value 收集之后：
cleaned = cleaner.clean(
    values=slot_values,
    semantic_type=slot.semantic_type,
    evidence_ids=slot.evidence_ids
)

❌ 不允许在 Gate 中调用
❌ 不允许在 PatternMatcher 中调用
❌ 不允许在 Runtime 控制流中调用

七、审计与调试要求（必须实现）
每个 SemanticInstance 中必须保留：
"value_cleaning": {
  "semantic_type": "symptom",
  "raw_values": [...],
  "cleaned_values": [...],
  "removed_values": [...],
  "normalization_map": {...},
  "split_map": {...}
}
八、设计铁律（必须遵守）
永远只有一个 Cleaner
规则 ≠ 代码
Cleaner 不知道 schema 名字，只知道 semantic_type
Cleaner 不产生新信息
Cleaner 的输出必须可回溯到原值

九、成功标准（验收）
新增 schema 只需加 YAML，不改代码
LIST_ONLY 输出中的值 全部来自 cleaned_values
无证据 → cleaned_values 为空 → Gate 控制拒答

十、禁止事项（强约束）

❌ 在 Cleaner 中写 if semantic_type == "symptom"

❌ 在 Cleaner 中访问 evidence 原文全文

❌ 在 Cleaner 中生成新医学概念

❌ 因 Cleaner 失败而放宽 Gate