"""
包：model_adapters

职责
- 放置不同模型族的适配器（Qwen 优先、DeepSeek 可插拔）。
- 对上层暴露统一的 `ModelAdapter` 抽象与工厂方法语义。

约束
- 适配器必须保证：注入开关可控、KV cache 行为一致、维度一致性、可观测 debug 输出。
"""


