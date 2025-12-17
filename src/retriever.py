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
        query_text: Optional[str] = None,
    ) -> RetrieverResult:
        items, debug = self.kv_bank.search(query_vec, top_k=top_k, filters=filters)
        return RetrieverResult(items=items, debug=debug)


@dataclass(frozen=True)
class RoutedRetrieverConfig:
    """
    Retrieval-side routing for "keep + split tables".

    Idea:
    - Always search the main text KVBank.
    - Only search the tables KVBank for table-like / numeric / comparison-heavy queries.
    - Merge results, with a small quota for tables (avoid drowning normal prose).
    """

    enable_table_routing: bool = True
    table_top_k: int = 4
    table_score_scale: float = 1.0
    force_use_tables: bool = False
    # Heuristic threshold for "table-like block" (only used in the KVBank split stage; kept here for debug symmetry).
    # Query classifier keywords (EN + ZH) tuned for medical/stat tables.
    query_keywords: Tuple[str, ...] = (
        "auc",
        "or",
        "hr",
        "ci",
        "95%",
        "p-value",
        "p value",
        "cutoff",
        "threshold",
        "sensitivity",
        "specificity",
        "multivariable",
        "univariable",
        "regression",
        "baseline",
        "cohort",
        "odds ratio",
        "hazard ratio",
        "confidence interval",
        "阈值",
        "cut-off",
        "敏感度",
        "特异度",
        "回归",
        "多变量",
        "单变量",
        "基线",
        "对照",
        "比较",
        "风险因素",
        "危险因素",
        "发生率",
        "死亡",
        "死亡率",
        "表1",
        "表 1",
        "table 1",
        "table",
    )


def _should_query_tables(query_text: Optional[str], cfg: RoutedRetrieverConfig) -> bool:
    if cfg.force_use_tables:
        return True
    if not cfg.enable_table_routing:
        return False
    if not query_text:
        return False

    q = query_text.strip().lower()
    if not q:
        return False

    # numeric heavy (units, percentages, comparisons)
    import re

    if re.search(r"\b\d+(\.\d+)?\b", q):
        # digits alone can appear in many prompts; require a hint of comparison/stat terms
        if any(kw in q for kw in cfg.query_keywords):
            return True
        # percent / CI / p-value patterns
        if "%" in q or "95" in q or "p<" in q or "p =" in q or "p=" in q:
            return True

    # keyword-based
    if any(kw in q for kw in cfg.query_keywords):
        return True

    return False


class RoutedRetriever:
    """
    Retriever that searches 1~2 KVBanks and merges results.

    - Primary bank: prose + non-table knowledge (`kvbank_blocks`)
    - Tables bank: table-like blocks (`kvbank_tables`)
    """

    def __init__(self, *, kv_bank: KVBank, table_kv_bank: Optional[KVBank], cfg: RoutedRetrieverConfig) -> None:
        self.kv_bank = kv_bank
        self.table_kv_bank = table_kv_bank
        self.cfg = cfg

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 32,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> RetrieverResult:
        # Always search main
        main_items, main_dbg = self.kv_bank.search(query_vec, top_k=top_k, filters=filters)

        use_tables = bool(self.table_kv_bank is not None) and _should_query_tables(query_text, self.cfg)
        tbl_items: List[KVItem] = []
        tbl_dbg: Dict[str, Any] = {}
        if use_tables and self.table_kv_bank is not None and int(self.cfg.table_top_k) > 0:
            tbl_items, tbl_dbg = self.table_kv_bank.search(
                query_vec, top_k=int(self.cfg.table_top_k), filters=filters
            )

        # Merge by (optionally scaled) score; ensure stable de-dupe by block_id.
        merged: List[KVItem] = []
        seen: set[str] = set()

        def _push(it: KVItem, *, source: str, scale: float = 1.0) -> None:
            meta = dict(it.meta or {})
            bid = str(meta.get("block_id") or meta.get("chunk_id") or "")
            if bid and bid in seen:
                return
            if bid:
                seen.add(bid)
            meta["retrieval_source"] = source
            merged.append(KVItem(score=float(it.score) * float(scale), meta=meta, K_ext=it.K_ext, V_ext=it.V_ext))

        for it in main_items:
            _push(it, source="text", scale=1.0)
        for it in tbl_items:
            _push(it, source="tables", scale=float(self.cfg.table_score_scale))

        merged.sort(key=lambda x: float(x.score), reverse=True)
        merged = merged[: min(int(top_k), len(merged))]

        debug = {
            "top_k": int(top_k),
            "use_tables": bool(use_tables),
            "table_top_k": int(self.cfg.table_top_k),
            "table_score_scale": float(self.cfg.table_score_scale),
            "main": main_dbg,
            "tables": tbl_dbg if use_tables else None,
        }
        return RetrieverResult(items=merged, debug=debug)



