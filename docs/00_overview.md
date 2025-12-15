# 概览：External KV Injection（HF Transformers 基线）

本文档解释系统端到端数据流与模块边界，用于指导实现与验收（不包含代码）。

## 端到端数据流
1. **离线**：PDF/文档 → 抽取/OCR → 结构化切分为 chunk → 清洗/去重/打分 → 形成 ChunkStore
2. **离线/准实时**：ChunkStore →（检索向量）DomainEncoder/pooled hidden →（注入向量）Projector 对齐到 `past_key_values` 空间 → 建立 KV Bank + ANN 索引
3. **在线推理**：Query（文本或 pooled hidden/Q）→ Retriever top-k（从 KV Bank 检索 chunk 级条目）→ 取回/生成 `K_ext/V_ext` → KVInjector 注入指定层 → 输出（可选：导出引用与统计）

## 关键约束
- **不训练主模型参数**（可训练外部模块：encoder/projector/gate）
- **维度不变**：`K_ext/V_ext` 的 `head_dim` 必须与目标模型 attention head_dim 一致
- **KV cache 兼容**：prefill/decode 阶段注入行为一致；外部 KV 通常视为“静态前缀 KV”（不写入 cache）


