"""
包：vector_store

职责
- 提供“向量库/索引后端”的实现与统一接口。
- demo 阶段（你当前选择）：本地 FAISS 文件落盘。

说明
- 本包的实现用于 External KV Injection 的“外部记忆检索”，不是 RAG 的“文本拼接”。
"""


