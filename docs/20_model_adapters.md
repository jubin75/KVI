# 模型适配协议：Qwen 优先、DeepSeek 可插拔（HF Transformers）

本文档定义 `ModelAdapter` 的职责与必须支持的行为，用于让不同模型族在同一注入框架下工作。

## 适配目标
- 在不修改主模型维度/结构的前提下，为 attention 注入 `K_ext/V_ext`
- 兼容 KV cache（prefill/decode）

## ModelAdapter：必须提供的语义能力

### 1) Attention 定位与绑定
- 能识别目标模型的 attention 模块位置（不同模型族路径可能不同）
- 能在 attention 计算 score 或融合输出前后挂载注入逻辑（由实现选择，但必须一致且可测试）

### 2) 注入接口语义（抽象）
- `inject_kv(layer_id, K_ext, V_ext, strategy, gamma, cache_mode)`
- 支持 `strategy = concat | gate`
- 支持 `gamma` 为常数或张量（按层/头/token）

### 3) KV cache 行为（必须一致）
- `prefill`：检索/准备 ext KV；并作为“静态前缀 KV”参与注意力
- `decode`：每步 decode ext KV 仍参与注意力；**ext KV 不写入 cache**（默认策略；若实现选择写入，需明确一致性与显存影响）

## QwenAdapter（优先实现）
- 默认注入层：前 2~4 层（可配置）
- 默认策略：gate
- 默认 γ：0.05~0.1（可配置，上限 clamp）

## DeepSeekAdapter（可插拔）
- 以相同注入语义为目标，避免与 MoE 路由强耦合
- 首要验收：注入开关可控、cache 一致、输出不崩溃


