# 注入策略：concat 与 gate

本文档定义注入策略的数学形式、推荐默认值与工程边界条件（不包含代码）。

## 1) concat 融合
```
K_fused = concat(K_orig, K_ext)
V_fused = concat(V_orig, V_ext)
Attention = softmax(Q · K_fused^T / sqrt(d))
```

### 优点
- 实现简单

### 风险/边界
- `ext_len` 增大会显著增加 attention 计算与显存压力
- 当外部 KV 噪声较大时，可能对输出有较强干扰

## 2) gate 融合（推荐）
```
Attn_orig = softmax(QK_orig^T / sqrt(d))
Attn_ext  = softmax(QK_ext^T  / sqrt(d))
Attn = (1−γ) * Attn_orig  +  γ * Attn_ext
```

### 推荐默认值
- `γ` 默认 0.05~0.1，且建议 clamp 到 `[0, 0.1]`
- `top_k` 默认 32（可按延迟与效果调参）

### γ 的来源（实现路线）
- P0：常数 γ（配置驱动）
- P1：按层 γ
- P2：按头/按 token γ（更复杂，但更精细）

## 统一边界条件（必须满足）
- 维度一致：`head_dim` 对齐
- `K_ext/V_ext` 必须携带可解释引用（chunk_id/source/page_range），便于 debug
- 注入开关关闭时，行为退化为原模型


