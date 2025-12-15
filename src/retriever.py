"""
模块：retriever（Demo 可运行实现）

定位
- 本模块负责“外部记忆检索”：输入 query 向量，输出可直接注入的 `K_ext/V_ext` 条目。
- 这不是 RAG 的“把文本拼接到 prompt”，而是 External KV Injection 的“检索→注入”链路。

你当前 demo 选型
- 向量索引：本地 FAISS
- KV Bank：直接存 K_ext/V_ext（检索命中后直接返回）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .kv_bank import KVBank, KVItem


@dataclass(frozen=True)
class RetrieverResult:
    items: List[KVItem]
    debug: Dict[str, Any]


class Retriever:
    """
    Demo Retriever：对 KVBank.search 做薄封装。

    注意
    - 本 retriever 假设上游已经把用户 query / pooled hidden / pooled Q 转成 query 向量（np.ndarray）。
    - 文本 query -> embedding 的编码器（DomainEncoder）不在此处实现。
    """

    def __init__(self, kv_bank: KVBank) -> None:
        self.kv_bank = kv_bank

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 32,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrieverResult:
        items, debug = self.kv_bank.search(query_vec, top_k=top_k, filters=filters)
        return RetrieverResult(items=items, debug=debug)



