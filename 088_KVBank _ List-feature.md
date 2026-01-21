《KVBank 与 List-feature 对齐 · 离线物化版（高吞吐）MD Prompt》
角色

你是 检索系统工程师 + 数据管道工程师。

目标

在 离线阶段，将 blocks.enriched.jsonl 中的 list-like feature 信号
物化写入 KVBank，使后续 retrieval / semantic_instances 天然可见、可加权、可消费，无需 runtime 注入或 sidecar。

输入

blocks.jsonl（旧，KVBank 当前使用）

blocks.enriched.jsonl（新，包含 list_features / list_items / cue spans）

现有 KVBank 构建脚本（不可重构，只允许小改）

输出（新的 KVBank schema 约束）

对每个 block_id，KVBank 的 meta 必须包含：
meta: {
  "list_like": true | false,
  "list_feature_count": int,
  "list_features": [
    {
      "type": "symptom" | "clinical_feature" | "other",
      "items": [string],
      "source_span": string
    }
  ],
  "list_confidence": float
}
要求：

若 enriched 中不存在 list feature，则 list_like=false

不允许运行时再推断 list-like

离线对齐规则（必须实现）

block_id 对齐

以 block_id 为 join key

禁止 fuzzy / embedding 对齐

list_feature 注入

若 blocks.enriched[block_id].list_features 非空：

拷贝至 KVBank.meta.list_features

设 list_like=true

list_feature_count = len(list_features)

否则：

明确写入 list_like=false

list_confidence 计算（简单可解释）
list_confidence =
  min(1.0,
      0.3 * list_feature_count +
      0.1 * has_symptom_cue +
      0.1 * has_enumeration_marker)
KVBank 构建约束（非常重要）

❌ 不允许在 retrieval / runtime 再生成 list feature

❌ 不允许 schema slot 自己判断 list-like

✅ KVBank 是 唯一 list-like 信号源

验证要求（必须输出）

生成一个 kvbank_alignment_report.json，包含：
{
  "total_blocks": int,
  "list_like_blocks": int,
  "avg_list_feature_count": float,
  "blocks_missing_enriched": int,
  "sample_list_blocks": [
    {
      "block_id": "...",
      "items": [...]
    }
  ]
}
成功判定标准（Done Definition）

retrieval 日志中可见：meta.list_like == true

semantic_instances 中 list_like_candidate_count > 0

schema slot 的 slot_status_source == semantic_instances

LIST_ONLY 下 不再出现自由生成项

禁止事项（强约束）

不新增 runtime sidecar

不修改 gate 逻辑

不引入模型判断

不改变 schema 定义

《KVBank meta → Retrieval Score 加权公式 · 冻结版 MD Prompt》
角色

你是 检索排序（Retrieval Ranking）系统工程师。

目标

在 retrieval 阶段，将 KVBank 中已物化的 meta.list_like / list_features
以确定性、可解释、不可学习的方式 注入最终检索分数。

这是 最终公式，不是调参实验。

输入

每个候选 block 具备以下字段：
{
  "base_score": float,          // embedding / ANN 相似度（已归一化到 [0,1]）
  "meta": {
    "list_like": bool,
    "list_feature_count": int,
    "list_confidence": float,
    "list_features": [...]
  }
}
输出
{
  "final_score": float
}
冻结加权公式（必须严格实现）
Step 1 · Base score（不可改）
S_base = base_score

Step 2 · List-like boost（只来自 KVBank meta）
B_list =
  if meta.list_like == true:
      1.0
    + 0.15 * min(meta.list_feature_count, 3)
    + 0.20 * meta.list_confidence
  else:
      1.0
说明（不可写入代码注释，仅供你理解）

上限：list_feature_count 贡献最多 0.45

list_confidence 最多贡献 0.20

单 block 最大 boost ≈ 1.65

Step 3 · Schema-aware gating（硬规则）
if query.requires_list_like == true and meta.list_like == false:
    S_final = 0.0
else:
    S_final = S_base * B_list
⚠️ 这是关键冻结点：

不 list-like → 直接淘汰

不允许“弱相关兜底”

完整公式（合并表示）
S_final =
  if requires_list_like and not meta.list_like:
      0.0
  else:
      base_score *
      (
        1.0
        + 0.15 * min(list_feature_count, 3)
        + 0.20 * list_confidence
      )
强约束（禁止事项）

❌ 不引入模型打分

❌ 不使用 schema slot 状态参与排序

❌ 不使用 runtime 推断的 list-like

❌ 不允许根据实验再调权重

验证日志（必须输出）
{
  "block_id": "...",
  "base_score": 0.xxx,
  "list_like": true,
  "list_feature_count": n,
  "list_confidence": 0.xx,
  "boost": B_list,
  "final_score": S_final
}
Done Definition（冻结成功标准）

非 list-like block 在 list-query 下 完全不可见

Top-K 中 list-like block 占比 ≥ 80%

semantic_instances.list_like_candidate_count > 0

LIST_ONLY 不再出现“无证据条目”

《LIST_ONLY 输出一致性 · 最终封口 MD Prompt》
角色

你是 RIM 输出约束与一致性冻结工程师。

目标（封口目的）

在 LIST_ONLY 能力下，彻底消除模型自由生成空间，确保：

输出 = 检索结果的可追溯投影

不允许：

重新排序

合并/拆分语义

同义发挥

补充证据中不存在的条目

适用前提（必须同时满足）
gate.final_answer_style == LIST_ONLY
semantic_instances.list_like_candidate_count > 0
否则本规则 不生效。

核心封口规则（必须严格实现）
Rule 1 · 输出顺序冻结（最关键）
LIST_ONLY 输出顺序 = retrieval.final_rank 顺序

第 N 条输出 ← 第 N 个 block

不允许重排

不允许 rerank

不允许基于“重要性”调整

Rule 2 · 输出来源唯一

每一条 LIST item 只能来自一个 block：
{
  "item_text": "...",
  "source_block_id": "block_xxx",
  "source_span": "..."
}
❌ 禁止跨 block 合成

❌ 禁止跨句拼接

❌ 禁止总结式改写
Rule 3 · 文本变换白名单（硬限制）

仅允许以下 等价变换：

类型	允许
去冠词	a / the
单复数	symptom / symptoms
大小写	Fever / fever
标点	逗号 / 分号
❌ 禁止语义归纳
❌ 禁止同义替换
❌ 禁止医学标准化
标准化 / 归一化 只能发生在 SchemaValueCleaner
LIST_ONLY 阶段一律禁止

Rule 4 · 无证据即删除（不是拒答）
if list_item.source_span is None:
    drop item

不补

不猜

不兜底

不提示用户
Rule 5 · 零条目兜底行为（冻结）
if final_list_items == []:
    final_answer = "No list-like evidence found in retrieved sources."
❌ 不切换为 EXPLANATION

❌ 不请求澄清

❌ 不重新生成
运行期审计输出（必须打印）
{
  "list_only_audit": {
    "retrieval_rank": [...],
    "output_order": [...],
    "one_to_one_mapping": true,
    "dropped_items": n,
    "reason": "no_source_span | invalid_transform"
  }
}
禁止事项（最终冻结）

❌ 在 LIST_ONLY 中调用 LLM 生成新文本

❌ 使用 baseline_answer 补全

❌ 使用 schema slot 推断新条目

❌ 使用 prompt 中的示例诱导生成

最终封口声明（必须写入代码）
LIST_ONLY output is a deterministic projection of retrieval results.
No semantic invention is allowed beyond source spans.

Done Definition（系统封口完成标志）

LIST_ONLY 输出 100% 可回溯到 block_id

不再出现 evidence 中不存在的条目

baseline_answer 即使错误，也不影响最终输出

日志可证明：无重排、无生成、无补全