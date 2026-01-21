《Evidence → List-like Feature 抽取规则 · 最小系统》MD Prompt
目标

实现一个 EvidenceListFeatureExtractor，
从 blocks.enriched.jsonl 中抽取 list-like 结构信号，
为后续 LIST_ONLY / schema slot 提供 可消费的列表候选。

一、模块定位（必须遵守）

新增模块：
external_kv_injection/src/evidence/list_feature_extractor.py
external_kv_injection/config/list_feature_rules/
    ├── generic.yaml
    ├── symptom.yaml
    ├── drug.yaml

执行时机（唯一合法）

在 blocks.enriched.jsonl 生成之后

在 KVBank 构建 之前或并行 sidecar

❌ 不在 Runtime / Gate 中运行

二、最小 Python 接口（严格一致）
class EvidenceListFeatureExtractor:
    def __init__(self, rule_dir: str):
        """
        rule_dir: list_feature_rules 目录
        """

    def extract(self, block: dict) -> dict:
        """
        输入：单条 enriched block
        输出：仅追加字段，不修改原内容
        {
          "list_features": {
            "is_list_like": bool,
            "list_items": [str],
            "signals": [str],
            "confidence": float
          }
        }
        """
三、抽取规则来源（数据驱动）
规则加载顺序

generic.yaml

<semantic_type>.yaml（若 block 可判定）

同名规则 semantic_type 覆盖 generic
四、规则文件最小格式
generic.yaml
signals:
  bullets:
    - "-"
    - "*"
    - "•"
  numbering_regex:
    - "^[0-9]+[.)]"
    - "^[a-zA-Z][.)]"

confidence:
  bullet: 0.6
  numbering: 0.6

symptom.yaml
semantic_type: symptom

signals:
  trigger_phrases:
    - "symptoms include"
    - "clinical features include"
    - "patients present with"
    - "manifested by"

split:
  delimiters:
    - ","
    - ";"
    - " and "

confidence:
  trigger_phrase: 0.7

五、最小抽取逻辑（不可扩权）

对每个 block：

检测 list signal

bullet / numbering

trigger phrase

若命中

标记 is_list_like = true

按 delimiter 拆出候选项

否则

is_list_like = false

不生成 list_items

❌ 不做语义判断
❌ 不做标准化
❌ 不做过滤

六、输出字段（必须写入 block）
"list_features": {
  "is_list_like": true,
  "list_items": [
    "fever",
    "thrombocytopenia",
    "leukopenia"
  ],
  "signals": [
    "trigger_phrase:symptoms include",
    "delimiter:comma"
  ],
  "confidence": 0.82
}
七、与下游系统的唯一关系

SemanticInstanceBuilder

只读取 list_features.list_items

SchemaValueCleaner

只清洗这些 list_items

Gate

只看 slot 是否 satisfied

不看抽取规则

八、设计铁律（必须遵守）

抽取 ≠ 理解

规则 ≠ 代码

list_features 只来源于 evidence

无 list_features → LIST_ONLY 不生成

confidence 只用于 debug，不做 hard gate

九、成功验收标准

KV / sidecar 中可看到 list_features

list_like_candidate_count > 0

LIST_ONLY 输出完全来自 list_items

无 evidence → 无 list_items → 拒答

十、禁止事项（强约束）

❌ 在抽取阶段写医学同义词

❌ 在抽取阶段过滤“是否像症状”

❌ 在抽取阶段生成新词

❌ 因抽取失败而 fallback 到自由生成

