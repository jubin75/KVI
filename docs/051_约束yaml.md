# Semantic-Type YAML Linter（工程门禁）

## 目标
校验任何新的 **semantic_type 规则 YAML** 是否安全、可扩展，并遵守：

- 只在**值域级清洗**（item/list-item），不做推断/路由
- 规则增长只能发生在 **semantic_type 级**，禁止 topic/问题级补丁
- 用失败指标驱动：宁可降级拒答/解释，也不要“硬凑规则”

---

## 输入
- 一个 YAML 文件路径（`*.yaml`）或规则目录路径
- 上下文：
  - 允许的 semantic_type 白名单：`location / symptom / drug / date / other`（`generic` 仅用于默认兜底）
  - 规则分两类：
    - `config/value_cleaning_rules/*.yaml`：值清洗（normalize/split/filter）
    - `config/list_feature_rules/*.yaml`：list-like 识别（signals/split/confidence）

---

## 校验规则（必须全通过）

1. **semantic_type 范围**
   - `semantic_type` 必须在白名单内
   - 不允许出现新的 semantic_type 除非通过准入门槛（见 Rule #5）

2. **允许操作**
   - 只允许：
     - split（按 delimiters/connectors 拆分值）
     - normalize（同义词归一/大小写统一/regex 去噪）
     - filter（去掉噪声或低质量值）
   - 不允许：
     - infer / generate / route / promote / 填补缺失值

3. **信号定义**
   - trigger_phrases / bullets / numbering_regex / paren_cases_regex 等可以存在
   - 只作为**识别或加权**，不可作为推断依据
   - confidence 分数可存在，但只能影响 list-like 识别强度（不得改变 routing/gate 决策）

4. **输出边界**
   - YAML 不能改变 semantic_type 或创建新的 slot
   - YAML 只能作用于值域级对象（单个 item 或 list item）
   - 不得改变 schema / pattern / question routing

5. **准入门槛（新增 semantic_type）**
   - 只有当该 semantic_type 在至少 N 个 schema 中复用
   - 并且已有 EvidenceUnit 标注该 semantic_type
   - 且当前失败仅由拆分/规范化导致
   - 才允许新增 YAML 文件

---

## 输出
使用脚本：`scripts/lint_semantic_type_yaml.py`

- 校验 value cleaning 规则目录：

```bash
python scripts/lint_semantic_type_yaml.py --mode value_cleaning --path config/value_cleaning_rules
```

- 校验 list feature 规则目录：

```bash
python scripts/lint_semantic_type_yaml.py --mode list_feature --path config/list_feature_rules
```

- 校验单个文件：

```bash
python scripts/lint_semantic_type_yaml.py --mode value_cleaning --path config/value_cleaning_rules/location.yaml
```

输出为 JSON：
- `pass=true` 表示合规
- `pass=false` 时 `errors[]` 会给出违规项（如：非法 semantic_type / 非法字段 / 含 forbidden 关键词）

---

## 失败指标驱动（防止无限修补）
当规则修改导致下面任一指标恶化时，应该**降级为拒答/解释**，而不是继续加规则硬凑：
- `cleaned_values` 为空的比例升高（抽取“被清洗掉了”）
- `removed_values` 比例异常升高（可能误删）
- LIST_ONLY 输出条目数在同一 query 分布上显著下降

强制原则：**YAML 只做清洗与分拆，不做任何推断或路由。**