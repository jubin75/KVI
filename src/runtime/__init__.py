"""
Package: runtime

Responsibilities
- Place "online inference" glue code: construct retrieval vectors from queries, call retriever, inject external K/V into HF models and perform generation.

Demo notes
- Current implementation uses HF Transformers' `past_key_values` (KV cache) as the injection carrier:
  treat external K_ext/V_ext as "static prefix KV", thereby avoiding direct modification of attention code.
"""


