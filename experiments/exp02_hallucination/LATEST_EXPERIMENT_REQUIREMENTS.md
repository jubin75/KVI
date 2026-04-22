# Exp02 — 最新实验要求（滚动备忘）

本文档记录 **当前优先** 的实验目标与可复现约定，避免仅依赖对话记忆。**每次调整程序或改跑法时，在文件顶部追加一节**（日期 + 简短标题），旧内容保留作历史。

---

## 2026-04-20 — D v3：两阶段去冲突（放松阶段2约束，保留纠偏）

**动机（来自 D v2 全量回落）**

- 阶段2“单句 + 固定拒答句 + 48 token”约束过硬，压缩了可保真的事实表达，导致 KVI 在全量 MC1/MC2 回落。
- 保留两阶段框架不变，只做最小改动，目标是恢复 D v1 的有效增益同时继续抑制长尾噪声。

**v3 最小改动（`scripts/run_graph_inference.py`）**

- 阶段2指令从“**exactly one sentence + exact refusal**”调整为“**1-2 short sentences**，证据不足时一句话说明不足”。
- 阶段2 system 指令强调“**evidence consistency > draft wording**”，避免草稿措辞对定稿产生锚定冲突。
- 阶段2 `max_new_tokens` 从 `min(48, ...)` 放宽到 `min(72, ...)`。
- `_sanitize_two_stage_openqa_final` 放宽为：最多保留 **2 句**，`max_chars` 从 220 提到 320，仍保留去 `!` 和基础去噪。

**编排（`run_truthfulqa_kvi_D_v3_pre100_full.sh`）**

- 默认 **`WAIT_RESIDENT=1`**：开跑前轮询 `RESIDENT_URL/health`，未就绪则 **直接退出**（避免 `Connection refused` 后留下半截 `predictions.jsonl`）；健康后 `RESIDENT_READY_GRACE_SEC` 秒再启动 `run_exp01.py`。

---

## 2026-04-19 — D 两阶段：收紧阶段2定稿（全量归因后）

**归因结论（TruthfulQA D 全量 500）**：KVI 输题子集上常见 **更长定稿 + 感叹号噪声**；阶段2需强制短句与去 `!`，并限制 `max_new_tokens`。

**代码调整（`run_graph_inference.py`）**

- 阶段2 system/user 指令改为：**单句极短定稿**、禁止 markdown/URL/`!`；证据不足时输出固定拒答句。
- 阶段2 `max_new_tokens` 上限降为 **`min(48, args.max_new_tokens)`**。
- 新增 `_sanitize_two_stage_openqa_final`：去 `!`、收束首句、长度上限，空输出则回退到清洗后的阶段1草稿。

---

## 2026-04-17 — D 轮方案：两阶段（KV 草稿 → evidence 校正定稿）

**用户确认方案（用于定位“注入信息如何被生成器利用”）**

- **阶段1（KV 草稿）**：`openqa + KVI` 下仅给 `Question`，启用三元组 KV 注入，不给 evidence 文本，先生成 1-2 句草稿。
- **阶段2（evidence 校正）**：把“阶段1草稿 + prompt evidence”送入模型做校正与定稿；阶段2不注入 `past_key_values`。
- 目标：先让 KV 尖峰信号定锚，再由 evidence 通道做纠偏，降低双通道同轮竞争。

**实现开关**

- `run_graph_inference.py` 新增：`--kvi_two_stage_kv_then_evidence`
- `run_exp01.py` 新增同名透传开关（仅 KVI 路径生效）

---

## 2026-04-16 — 执行约束（防回退 / 防走回头路）

**新增硬性约束（从本节开始默认执行）**

- **GPU 常驻优先**：Exp02 运行前先确保仅有一个健康的 resident（`127.0.0.1:18888`）；若发现重复 resident 或其他重 GPU 进程挤占，先清理再启动，不在异常资源状态下盲跑。
- **不回退到低价值规模**：TruthfulQA 已有 `n=25` 冒烟基线后，后续排查默认继续 `n=25`（可 `--resume`），除非明确为了复现某个崩溃而临时降样本并在日志中注明原因。
- **串行优先，避免互相干扰**：同一时段只跑一条 `run_exp01.py` 主任务（或一条优化脚本主任务），避免并发导致 resident/显存/IO 干扰，把问题误判为算法退化。
- **失败先看阻塞点，不先改规模**：优先检查 resident 健康、超时位置、OOM、进程退出码；确认根因后再做最小调参（如 `timeout_s`、恢复 `--resume`），不先改成 `n=10` 重跑。

**本节意图**

- 将“先稳服务与资源，再跑既定规模”的流程固化为默认策略；
- 避免因临时调试把实验口径和结论可比性打散。

---

## 2026-04-16 — TruthfulQA: A→B→C 夜间串行优化（用于定位 KVI vs GraphRAG 差距）

**目标**：把 “KVI 相对 GraphRAG 的差距” 分解成可归因的三步实验，避免凭感觉改多处导致结论不可复现。

- **A（消融：禁用 KV 注入）**：KVI **照常组装 KV**，但在生成时 **不注入 `past_key_values`**，用于隔离 “KV 注入本身” 对 MC2 的影响。参数：`--kvi_disable_kv_injection`（透传至 `run_graph_inference.py`）。
- **B（提升相关性：更严格 KV 过滤）**：关闭 tuned overrides（`--no-truthfulqa_kvi_tuned_overrides`），显式提高 `kvi_drm_threshold`、收紧 `kvi_top_k_relations`、减少 `kvi_max_kv_triples`，目标是 **宁缺毋滥**，降低噪声 triples。
- **C（输出形态：更短更收敛）**：在 B 基础上启用 `--kvi_minimal_prompt` 并降低 `--truthfulqa_kvi_max_new_tokens`，减少长尾/跑题/hedging，优先拉升 MC2 proxy。

**运行脚本**：`experiments/exp02_hallucination/code/run_truthfulqa_kvi_abc_sequential.sh`（串行，默认 n=25，依赖单一健康 `18888` resident）。

---

## 2026-04-15 — 逐题 error bucket + MC 条件文本收敛（已启动新对照轮）

**已完成分析**

- 新增：`experiments/exp02_hallucination/code/analyze_truthfulqa_kvi_error_buckets.py`
- 基于 `truthfulqa_kvi_injection_levels_20260415_163012/level_6/predictions.jsonl` 输出：
  - `results/truthfulqa_kvi_error_buckets_level6.md`
  - `results/truthfulqa_kvi_error_buckets_level6.json`
- 结论：KVI 落后 GraphRAG 的样本里，`has_many_exclaim` 与 hedging 语气较常见，支持继续收敛条件文本与输出清洗。

**已做代码收敛（通用，不特例化）**

- `run_exp01.py::_kvi_pred_for_truthfulqa_mc1_likelihood`
  - 增加通用去噪（URL/渲染残片/泛化 hedging 前缀）。
  - 多句候选改为“事实密度打分”选句，而非总取首句。

**已启动新一轮 n=25 后台对照**

- 时间戳：`20260415_220828`
- 主日志：`results/exp02_truthfulqa_kvi_optimize_20260415_220828.log`
- 输出目录：`results/truthfulqa_kvi_optimize_20260415_220828/`

---

## 2026-04-15 — 注入层级对比（n=25）用于验证 KVI 噪声来源

**背景判断**：在 retrieval/prompt 漏斗都命中的情况下，KVI 仍可能因三元图谱注入层级（`max_kv_triples`）导致生成偏移，进而拖累 `likelihood_proxy`。

**新增实验入口**

- `experiments/exp02_hallucination/code/run_truthfulqa_kvi_injection_levels_detached.sh`
- 默认比较层级：`KVI_LEVELS="0 2 4 6"`，固定 `n=25`、固定数据工件、固定 `kvi_drm_threshold=0.05`、`kvi_top_k_relations=4`。
- 关键：显式传 `--no-truthfulqa_kvi_tuned_overrides`，避免自动 tuned override 覆盖层级 ablation。

**输出**

- 根目录：`results/truthfulqa_kvi_injection_levels_<timestamp>/`
- 每层级一个子目录：`level_<k>/summary.json`
- 汇总：`injection_levels_summary.md` / `injection_levels_summary.csv`

---

## 2026-04-15 — 目标：KVI 在 TruthfulQA proxy 上应优于 GraphRAG 与 KV Prefix

**目标**：在同一 eval 子集上，KVI 的 `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`（likelihood_proxy，**越高越好**）应稳定高于 GraphRAG 与 KV Prefix。若要写「显著」，需配对检验或置信区间；扩样本前先逐题与 audit 归因。

**本轮代码**：TruthfulQA 收紧 KVI（DRM floor **0.046**、KV triples **4–5**）、默认 **minimal prompt** 与 **reconcile**（open-QA 用证据 token 重叠；修复无 DrugBank ID 时 reconcile 总丢弃 KV 解码的问题）；`run_graph_inference` 对英文 openqa+KVI 加强重复惩罚与解码前轻清洗；KVI 答案清洗削弱中英混杂。

---

## 2026-04-15 — 提升 KVI 在 Exp02 的表现（检索 + 模板）与后台跑法

**目标**

- 降低 TruthfulQA（及后续 FEVER）上 **KVI 相对偏高的幻觉 proxy**，优先从 **证据检索噪声** 与 **英文 open-QA 生成模板** 两侧改进（与 audit 漏斗诊断一致）。
- 在 **不改数据工件** 的前提下，让图侧默认行为更适合 **TruthfulQA（`--openqa_mode`，非 FEVER claim）**。

**代码改动摘要（2026-04-15）**

1. **`scripts/run_graph_inference.py`**
   - **混合检索**：在 `--openqa_mode` 且 **非 FEVER claim** 时，将 `sentences.jsonl` 文本检索的 `min_score` **抬到至少 0.11**（仍可用 `--text_search_min_score` 设更高下限），减少弱相关句灌进 prompt。
   - **证据条数**：对同一设定，在拼 prompt 前对 DRM 排序后的证据句 **默认最多保留 8 条**（`--max_openqa_evidence_sentences`，`0` 表示不截断），减轻证据稀释与长生成。
   - **模板**：TruthfulQA 分支下加强 **仅用证据、禁止编造 / markdown / 图链**，并在用户指令中要求证据不足时简短说明。
2. **后台脚本**：`experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh`  
   - `nohup` 跑 `run_exp01.py`（`graphrag,kv_prefix,kvi`），日志与 pid 写入 `results/`；可选 **常驻**、**弱 oracle**、**audit jsonl**（见脚本内注释与环境变量）。

**后台运行（SSH 断开后仍继续）**

```bash
cd /home/zd/dev/KVI
chmod +x experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh
nohup env TRUTHFULQA_LIMIT=25 WAIT_RESIDENT=1 RUN_AUDIT=1 \
  bash experiments/exp02_hallucination/code/run_truthfulqa_kvi_optimization_detached.sh \
  </dev/null >> experiments/exp02_hallucination/results/exp02_truthfulqa_kvi_opt_outer.log 2>&1 &
```

完成后查看：`results/exp02_truthfulqa_kvi_optimize_*.log`、`truthfulqa_kvi_optimize_*/summary.json`。

**已自动启动的一次跑（Agent，2026-04-15）**

- 命令：`run_truthfulqa_kvi_optimization_detached.sh`，`TRUTHFULQA_LIMIT=25`，`RUN_AUDIT=1`，`WAIT_RESIDENT=1`，常驻 `http://127.0.0.1:18888`。
- 主日志：`results/exp02_truthfulqa_kvi_optimize_20260415_093135.log`；输出目录：`results/truthfulqa_kvi_optimize_20260415_093135/`；子进程 pid：`results/exp02_truthfulqa_kvi_optimize_20260415_093135.pid`；外层：`results/exp02_truthfulqa_kvi_opt_outer.log`。
- 结束后可对该次 `graph_audit_jsonl` 跑 `analyze_graph_prompt_audit.py`。

**验证建议**

- 与改前同一 `limit` 对比 `summary.json` 中 KVI 的 `truthfulqa_mc1_proxy` / `truthfulqa_mc2_proxy`（及 GraphRAG 对照）。
- 若开 `RUN_AUDIT=1`：对产出的 `graph_audit_jsonl` 跑 `analyze_graph_prompt_audit.py`，看 `MISS_RETRIEVAL` / `MISS_DRM` / `MISS_PROMPT` 是否变化。

---

## 2026-04-15 — Audit 冒烟优先（dualchannelfix + 程序调整后验证）

**目标**

- 以 **带 audit 的 TruthfulQA 冒烟** 为主，验证 **图侧 / KVI 路径** 在改代码后仍端到端可跑，并产出可分析的 audit 轨迹。
- 与「无 audit 的 smoke50」对照：无 audit 的 `truthfulqa_dualchannelfix_smoke50_20260414` 已跑通；**audit 版** 曾在子进程 **600s 超时**（GraphRAG 首题或 KV Prefix 中途），需在同样设定下重跑或调参。

**「Audit 冒烟」指什么**

1. **全链路（含生成）**：`experiments/exp01_main_qa/code/run_exp01.py`，TruthfulQA，`--methods graphrag,kv_prefix,kvi`（与 dualchannelfix 子集一致），并打开：
   - `--graph_audit_jsonl` → 追加 retrieval / DRM / prompt 审计行（GraphRAG 与 KVI 子进程均传 `run_graph_inference.py` 的 audit 参数）。
   - `--graph_audit_oracle_jsonl`（可选但推荐）：弱 oracle 证据，用于 R@retrieval / R@drm 等；可用 `experiments/exp02_hallucination/code/build_weak_oracle_evidence.py` 从 `truthfulqa_eval.jsonl` 生成（历史上曾出现 **NO_ORACLE** 导致审计表全零，需确认 oracle 路径与 `id` 对齐）。
2. **仅审计（无完整生成）**：`experiments/exp02_hallucination/code/run_graph_prompt_audit_only.py`（内部 `--audit_only`），用于快速检查检索/DRM/提示是否进 prompt，负载低于全链路。

**KVI / TruthfulQA 评测侧（与近期 run 一致）**

- `--truthfulqa_kvi_mc1_answer grounded`
- `--truthfulqa_kvi_max_new_tokens 96`（或与 `summary.json` 里 `kvi_truthfulqa_runtime_overrides` 一致）
- 常驻推理（若使用）：`--inference_service_url http://127.0.0.1:18888`；ANN 侧按本机习惯 `--ann_force_cpu` 或 `--ann_via_resident`。

**超时与规模**

- 默认 `--timeout_s` 在 `run_exp01.py` 为 **300**，Exp02 主表常用 **600**；**audit 冒烟若仍超时，优先调到 1200–1800**，或先 **warm resident** 再跑、或 **`--limit` 先 3→10** 逐步加。
- 输出目录建议带日期：`results/truthfulqa_dualchannelfix_smoke{N}_audit_YYYYMMDD/`，日志 `results/exp02_truthfulqa_dualchannelfix_smoke{N}_audit_YYYYMMDD.log`。

**结果分析**

- 汇总：`experiments/exp02_hallucination/code/analyze_graph_prompt_audit.py`（对 audit JSONL / 导出 md）。
- 历史上 `retrieval_drm_prompt_audit_smoke10_*_v2.md` 中 GraphRAG 曾出现 **MISS_DRM**、KVI 另一 breakdown；以 **最新一次** audit 输出为准解读。

**命令骨架（全链路 audit 冒烟，请按本机路径与 limit 修改）**

```bash
cd /home/zd/dev/KVI
# 可选：先生成弱 oracle（id 与 dataset 对齐，供 R@retrieval / R@drm）
KVI/bin/python3 experiments/exp02_hallucination/code/build_weak_oracle_evidence.py \
  --dataset_jsonl experiments/exp02_hallucination/data/truthfulqa_eval.jsonl \
  --sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --out_jsonl experiments/exp02_hallucination/results/truthfulqa_weak_oracle_evidence_smoke10.jsonl \
  --limit 10 --top_k 3

KVI/bin/python3 -u experiments/exp01_main_qa/code/run_exp01.py \
  --dataset experiments/exp02_hallucination/data/truthfulqa_eval.jsonl \
  --dataset_name TRUTHFULQA \
  --model models/Qwen2.5-7B-Instruct \
  --graph_index experiments/exp02_hallucination/artifacts/truthfulqa/graph_index.json \
  --triple_kvbank_dir experiments/exp02_hallucination/artifacts/truthfulqa/triple_kvbank \
  --graph_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --ann_kv_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences \
  --ann_sentences_jsonl experiments/exp02_hallucination/artifacts/truthfulqa/sentences.tagged.jsonl \
  --ann_semantic_type_specs experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar/semantic_type_specs.json \
  --ann_pattern_index_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar \
  --ann_sidecar_dir experiments/exp02_hallucination/artifacts/truthfulqa/kvbank_sentences/pattern_sidecar \
  --methods graphrag,kv_prefix,kvi \
  --limit 10 \
  --out_dir experiments/exp02_hallucination/results/truthfulqa_dualchannelfix_smoke10_audit_$(date +%Y%m%d) \
  --timeout_s 1800 \
  --inference_service_url http://127.0.0.1:18888 \
  --ann_inference_service_url "" \
  --ann_force_cpu \
  --truthfulqa_kvi_mc1_answer grounded \
  --truthfulqa_kvi_max_new_tokens 96 \
  --graph_audit_jsonl experiments/exp02_hallucination/results/truthfulqa_graph_prompt_audit_smoke10_$(date +%Y%m%d).jsonl \
  --graph_audit_oracle_jsonl experiments/exp02_hallucination/results/truthfulqa_weak_oracle_evidence_smoke10.jsonl \
  2>&1 | tee experiments/exp02_hallucination/results/exp02_truthfulqa_dualchannelfix_smoke10_audit_$(date +%Y%m%d).log
```

（若 `weak_oracle_evidence_smoke10.jsonl` 尚不存在，先对同一 `--limit` 用 `build_weak_oracle_evidence.py` 生成，或暂时去掉 `--graph_audit_oracle_jsonl` 仅看非 oracle 字段。）

---

## 维护说明

- **新增约定**：复制上一节模板，改日期、改「目标 / 参数 / 已知问题」。
- **Agent 提示**：处理 Exp02 冒烟或 audit 时，先读本文件最新一节，再查 `results/` 与 `run_exp01.py` 当前默认值。
