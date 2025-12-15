"""
包：runtime

职责
- 放置“在线推理期”的 glue code：从 query 构造检索向量、调用 retriever、把外部 K/V 注入 HF 模型并执行生成。

demo 说明
- 当前实现以 HF Transformers 的 `past_key_values`（KV cache）作为注入载体：
  把外部 K_ext/V_ext 当作“静态前缀 KV”，从而避免直接改写 attention 代码。
"""


