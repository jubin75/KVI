# KVI 模块化 Checklist（落到现有代码结构）

> 目标：把“我们要做的事情”拆成可执行 checklist，并且每一条都能在当前仓库里找到对应模块/入口/日志字段。  
> 说明：本文先把 **现有 KVI2 / simple pipeline** 的结构讲清楚，再标注“若升级为 SOP→Tool 协议”应该插在哪些点（不要求重写主链路）。

---

## 0. 一句话总览：你们现在“已经有什么”

- **外部知识存储 + 检索**：FAISS KVBank（存 `K_ext/V_ext + meta`）→ `src/kv_bank.py` / `src/vector_store/faiss_kv_bank.py`
- **在线检索封装**：`src/retriever.py`
- **KVI2 运行时主链路**：Pattern-first → Gate → Semantic-second → 注入生成 →（可选）LIST_ONLY deterministic → `src/runtime/kvi2_runtime.py`
- **多步注入器（stop policy + 注入有效性信号）**：`src/runtime/multistep_injector.py`
- **非生成控制模块（RIM）**：`src/rim.py`
- **Detox/契约验证（metadata-only，无 LLM）**：`src/pattern_contract.py`
- **证据策略与输出后处理**：`src/runtime/postprocess.py`
- **Evidence Units（sentence-level）调试链路**：`scripts/run_kvi2_runtime_test.py --pipeline simple --simple_use_evidence_units`

---

## 1) Offline（建库侧）Checklist：从文本到可检索/可注入

> 这部分的“可复制命令”已经写在 `docs/90_experiment_runbook_linux.md`（包含 evidence.txt mini pipeline）。

- **blocks 构建**（raw/evidence → 256-token blocks）
  - 脚本：`scripts/build_blocks_from_raw_text.py`（mini evidence.txt）或 PDF 主线脚本
  - 验收：`blocks.jsonl` 非空、`text` 字段合理、无大片乱码

- **blocks.enriched 增强（pattern/list_features 等元信息）**
  - 脚本：`scripts/build_pattern_index_from_blocks_v2.py`
  - 产物：`blocks.enriched.jsonl` + pattern sidecar 文件（落在 `--pattern_out_dir`）

- **pattern_contract 生成（topic 级契约）**
  - 脚本：`scripts/pattern_contract_autogen.py`
  - 产物：`pattern_contract.json`
  - 作用：推理期 contract-driven（fail-closed），用于证据过滤/拒答/slot 规划（而不是让 LLM 自己猜）

- **KVBank 构建（向量检索 + K_ext/V_ext 可注入）**
  - 脚本：`scripts/build_kvbank_from_blocks_jsonl.py`
  - 产物：`kvbank_blocks/manifest.json` + shards（或非分片）
  - 关键 meta：`block_id`、`schema_slots`、`list_signals/list_type`（若启用 list_feature 排序）

---

## 2) Online（推理期）Checklist：当前 KVI2 主链路（对应代码）

### 2.1 入口与调试脚本

- **CLI Harness**：`scripts/run_kvi2_runtime_test.py`
  - `--pipeline kvi2`：走 `KVI2Runtime.run_ab(...)`（完整 pattern+gate+semantic-second+注入）
  - `--pipeline simple`：架构调试旁路（相似检索 + Evidence Units + 注入生成；不走 LIST_ONLY 投影）

### 2.2 Pattern-first（非语义、topic 级先验）

- **加载契约**：`src/runtime/kvi2_runtime.py`
  - `PatternContractLoader().infer_topic_dir_from_kv_dir(...)`
  - `PatternContractLoader().load(...)`
- **pattern 检索与匹配**：
  - `PatternRetriever` / `PatternMatcher`（见 `src/pattern_pipeline.py`、`src/pattern_retriever.py`）
- **schema 候选打分（防“低信息 schema 劫持高信息问题”）**
  - `score_candidate_schemas(...)`：`src/pattern_pipeline.py`
  - 你们已有：drug/treatment intent 的 schema boost，定义/枚举/时空意图的惩罚等

### 2.3 IM / Gate（非生成）

- **RIM 控制器**：`src/rim.py`
  - 约束：MUST NOT generate tokens（只做控制/门禁/refresh 逻辑）
- **IntrospectionGate**：`src/pattern_pipeline.py`
  - 输入：pattern_id / slot_status / contract missing / （可选）kv relevance delta 等
  - 输出：是否 retrieve_more / 是否 REFUSE / 选择 answer_style（如 LIST_ONLY）

#### 2.3.1（已知缺陷/风险记录）Gate 的 Q1/Q2/Q3 在设计上的不稳定性

> 目的：把“我们已经讨论过的问题”写进 checklist，后续再修代码时作为明确的改造目标与验收点。

**现状（代码事实）**：
- `retrieve_more` 的核心信号来自 `RIM.introspection_gate(...)`（Q1/Q2/Q3）：`src/rim.py`
  - **Q1 semantic-shift**：比较 `q0` 与 `q'` 的 cosine distance（`tau_cos_dist`）
  - **Q2 low-impact**：注入后 `logit_delta_vs_zero_prefix < kv_irrelevant_logit_delta_threshold`
  - **Q3 pattern-mismatch**：`missing_hard`（hard contract violations）触发 refresh
- 在 `KVI2Runtime.run_ab` 中第一次 gate（semantic-second 之前）使用：
  - `q0 = enc(prompt)`
  - `q' = enc(prompt + baseline)`（baseline 是 base LLM 的生成文本）

**我们认为的潜在问题（讨论结论）**：
- **Q1（semantic-shift）在第一次 gate 尤其不稳**：
  - `q'` 由 `prompt + baseline` 组成，baseline 是 LLM 生成文本，包含风格/幻觉/模板噪声；
  - DomainEncoder 对“生成风格噪声”敏感，cos_dist 可能反映的是表达变化而非语义漂移；
  - 这会导致“在没有任何外部证据之前，就用不稳定信号决定要不要检索”，属于高风险决策点。
- **Q2（low-impact）不等价于“证据不相关”**：
  - 注入影响小可能是“模型本来就会/答案短/层选择不敏感”，并不代表证据无价值（尤其对审计与引用仍有价值）；
  - 阈值强依赖模型、注入层、注入 token 数、prompt 长度，跨任务泛化差；
  - 更适合作为“注入管线健康度/强度监控”，而不是相关性判定。
- **Q3（pattern-mismatch = missing_hard）可能把“契约/抽取问题”误当成“证据问题”**：
  - hard missing 可能来自 contract 设计过硬、metadata 抽取不足、cleaner 过严等；
  - 若直接驱动 refresh，可能产生“无效换 KV 循环”，而不是引导补信息/降级/转人工。

**建议的未来修复方向（先记录，不改代码）**：
- **第一次 gate 降级为“弱门禁”**：默认进入 semantic-second（至少进行一小步检索），第一次 gate 只决定检索预算而不是做强拒绝/强刷新决策。
- **把是否需要更多证据改为“契约缺口/slot 缺口驱动”**：对 evidence-expected intents（诊断/治疗/风险/预后等）默认检索；对纯解释性问题降低强制检索。
- **把 Q2 从相关性判定降级为健康监控**：用于识别注入无效/注入过强/需要降噪，而不是直接判“证据不相关”。
- **Q3 优先导向“缺口解释/追问/转人工”**：missing_hard 更像“缺变量/缺证据/指南不覆盖”的提示，应优先 fail-closed 并给出缺口清单，而非盲目 refresh。

### 2.4 Semantic-second 检索 + 契约过滤（Detox）

- **检索**：`Retriever.search(...)` → `KVBank.search(...)`
  - `src/retriever.py`
  - `src/vector_store/faiss_kv_bank.py`
- **契约过滤（metadata-only）**：`filter_evidence_by_contracts(...)`
  - `src/pattern_contract.py`
  - 关键：不靠 LLM 判定证据是否满足契约（可审计、可复现）

### 2.5 注入与生成（K_ext/V_ext prefix）

- **KV prefix 构建与注入**：
  - `src/runtime/hf_cache_prefix_injection.py`
  - `src/runtime/kvi2_runtime.py` 里对 `build_past_key_values_prefix(...)` 的调用
- **多步注入器（可选）**：
  - `src/runtime/multistep_injector.py`
  - 你们已有 stop policy：`logit_delta_vs_zero_prefix`、冗余检测、token cap 等

### 2.6 输出后处理 / 证据策略（fail-closed）

- **输出清洗、结构校验、证据策略**：`src/runtime/postprocess.py`
  - `validate_answer_structure(...)`
  - `enforce_evidence_policy(...)`：对部分意图要求 evidence，否则返回标准 fallback（或 evidence retry）

---

## 3) Online（推理期）Checklist：simple pipeline（Evidence Units 调试闭环）

> 适用：你想验证“检索是否命中 + evidence units 是否抽到 + 注入后输出是否更可靠”，并且暂时绕过 LIST_ONLY 投影。

- **入口**：`scripts/run_kvi2_runtime_test.py --pipeline simple`
- **Evidence Units（sentence-level only）**：
  - `_extract_sentence_units_only(...)`：脚本内实现（调用 `EvidenceUnitExtractor`）
  - 语义路由：`_infer_target_semantic_type_for_query(...)`
  - 输出日志字段：`steps[].selected_unit_counts`、`steps[].evidence_units_shown`、`semantic_type_router`
- **注入输出**：
  - 仍然使用 KVI 注入（past_key_values prefix）+ 生成
  - 通过 `answer_postprocess` 做“引用/删除 unsupported 片段”的保守后处理（脚本内）

---

## 4) 从 “KVI 问答” 升级到 “SOP → Tool 协议” ：怎么落在现有代码结构里（不改主链路的插槽）

> 目标：让系统输出 **结构化 SOP（中间表示 IR）**，再由确定性编译器生成 Tool Plan 并执行；IM/RIM 负责“能不能生成 SOP / 能不能执行”。

### 4.1 建议新增 3 个模块（最小集合）

1) `src/sop/schema.py`
   - 定义 SOP JSON schema（字段、枚举、必填项）
   - 提供 `validate_sop(sop_json) -> (ok, errors)`

2) `src/sop/planner.py`
   - 输入：用户 query +（可选）Evidence Units +（可选）注入后的生成
   - 输出：结构化 SOP JSON（关键步骤带 evidence ids）
   - 注意：planner 可以调用 LLM 生成，但输出必须严格按 schema

3) `src/sop/compiler.py`
   - SOP JSON → Tool Plan JSON（确定性映射）
   - 失败则 fail-closed：不执行、返回原因、要求补输入

（可选）`src/sop/executor.py`
 - 执行 tool plan，并记录可审计日志（inputs/outputs/errors）

### 4.2 插入点：KVI2Runtime.run_ab（推荐）

在 `src/runtime/kvi2_runtime.py` 里，完成以下内容后插入 SOP：
- 已有：pattern-first + contract 选择 + semantic-second 检索 + filter_evidence_by_contracts
- 已有：gate_after_validation（可以作为 “是否允许进入 SOP 模式” 的信号）

插入逻辑建议（高层）：
- 若 `gate_after_validation` 表示证据不足/拒绝：直接 fail-closed（或回退 base LLM）
- 若证据足够：
  - 使用 Evidence Units（可从 blocks 文本中抽取）作为 SOP 的 grounding
  - 调用 `SOPPlanner` 生成 SOP JSON
  - 用 `RIM/IM` 的扩展检查项验证 SOP（见 4.4）
  - 通过 `SOPCompiler` 编译成 tool plan
  - （可选）执行器执行

### 4.3 插入点：simple pipeline（用于 SOP 模式的快速验收）

`scripts/run_kvi2_runtime_test.py --pipeline simple` 适合做 SOP 的最小闭环：
- 保留现有 Evidence Units 抽取与语义路由
- 把 “injected_answer” 替换为 “sop_json + tool_plan_json”（debug 输出）
- 用同样的 `answer_postprocess` 思路做 SOP 级的“证据绑定检查”

### 4.4 IM 检查项（SOP 专用，必须非生成）

建议把 SOP 的 IM 检查项做成纯函数（非生成），并在 `RIM` 或 `runtime/postprocess.py` 风格的位置调用：
- **可执行性**：每一步是否有 action、是否声明 inputs/outputs
- **可判定性**：preconditions 是否可观测（不能是“如果感觉严重”这种不可判定条件）
- **证据绑定**：高风险步骤是否引用 evidence（没有证据则必须 `requires_human_confirmation=true` 或拒绝）
- **风险控制**：是否有 stop_conditions（何时停止、转人工、报警）
- **跨意图污染**：SOP 是否只覆盖当前 intent/slot（复用你们现有 semantic_type_router 的思想）

---

## 5) 最小验收（你应该在日志里看到什么）

### 5.1 KVI2（完整链路）
- `pattern_first`：命中 patterns / skeleton
- `gate` / `gate_after_validation`：是否 retrieve_more、是否 REFUSE、最终 answer_style
- `retrieval.contract_validation`：hard/soft/schema_missing 是否合理
- `retrieval.final_rank`：list_feature 排序是否符合预期（尤其 location）

### 5.2 simple（Evidence Units）
- `selected_unit_counts` > 0（至少一个 block 有可注入 sentence_enumerative）
- `evidence_units_shown` 与 query 语义一致（symptom/location/drug 等）
- `answer_postprocess`：对 unsupported 句子有删除记录（fail-closed）

### 5.3 SOP（未来新增）
- `sop_validate.ok=true`
- `sop_im_gate.pass=true`（否则必须 fail-closed）
- `tool_plan_compile.ok=true`
- `executor`：完整执行日志可回放（可审计/可回滚）
