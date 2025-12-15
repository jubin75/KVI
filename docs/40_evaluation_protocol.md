# 评测协议：效果/延迟/稳定性/可解释性

本文档定义验收指标与实验分层，用于 AB 与回归测试（不包含代码）。

## 1) 端到端效果指标（领域 QA）
- 准确率（Accuracy / EM / F1）：按任务选用
- 引用正确率（Citation Correctness）：回答所引用 chunk 是否支持结论
- 幻觉率（Hallucination Rate）：回答中不可由 chunk 或常识支持的断言比例（需定义标注规则）

## 2) 检索指标（Retriever）
- Recall@k（k=8/16/32）
- MRR@k / nDCG@k（可选）
- 过滤/重排的收益（对比 retriever-only vs reranker）

## 3) 延迟指标（在线推理）
必须分别统计：
- 检索延迟（ANN + 过滤 + 重排）
- 注入额外开销（ms/层）
- 端到端 tokens/s（prefill 与 decode 分开统计）

## 4) 稳定性与回归
- 注入开关关闭：输出与原模型一致（允许随机性误差，需固定 seed）
- γ=0：等价于不注入
- γ 上限：超过上限不得放大（clamp 生效）
- 极端输入：长 query、空检索、低质量 chunk 等场景不崩溃

## 5) 可解释性输出（debug）
每次回答可选择输出：
- top-k chunk 列表（chunk_id、source_uri、page_range、score）
- γ（按层/按头聚合统计）
- 注意力分配（orig/ext 的聚合比例）


