"""
Demo: FAISS KV Bank (build → save → load → search)

Description:
- This is a minimal runnable example to verify:
  1) Small number of entries can be written to a local KV Bank
  2) Top-k retrieval works from a query vector
  3) Corresponding K_ext/V_ext can be retrieved for injection (here only prints shape/metadata)
"""

from typing import Any


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore


def main() -> None:
    out_dir = Path("external_kv_injection/_demo_kvbank")
    out_dir.mkdir(parents=True, exist_ok=True)

    # demo 数据：3 条知识条目
    # retrieval_keys：用于 ANN 检索的向量（shape [N, d]）
    N = 3
    d = 8
    retrieval_keys = np.random.randn(N, d).astype(np.float32)

    # 直接存 K_ext/V_ext：shape [N, heads, ext_len, head_dim]
    heads = 2
    ext_len = 4
    head_dim = 16
    k_ext = np.random.randn(N, heads, ext_len, head_dim).astype(np.float32)
    v_ext = np.random.randn(N, heads, ext_len, head_dim).astype(np.float32)

    metas = [
        {"chunk_id": "c0", "citation": "demo.pdf:1-1", "lang": "zh"},
        {"chunk_id": "c1", "citation": "demo.pdf:2-2", "lang": "zh"},
        {"chunk_id": "c2", "citation": "demo.pdf:3-3", "lang": "en"},
    ]

    bank = FaissKVBank.build(
        retrieval_keys=retrieval_keys,
        k_ext=k_ext,
        v_ext=v_ext,
        metas=metas,
        normalize=True,
        metric="ip",
    )
    bank.save(out_dir)

    bank2 = FaissKVBank.load(out_dir)

    # query：这里直接用一条随机向量；真实系统里来自文本 query 或 pooled hidden/Q
    query_vec = np.random.randn(d).astype(np.float32)
    items, debug = bank2.search(query_vec, top_k=2, filters={"lang": ["zh", "en"]})

    print("debug:", debug)
    for it in items:
        print(
            "hit:",
            {"score": it.score, "chunk_id": it.meta.get("chunk_id"), "citation": it.meta.get("citation")},
            "K_ext.shape=",
            tuple[Any, ...](it.K_ext.shape),
            "V_ext.shape=",
            tuple(it.V_ext.shape),
        )


if __name__ == "__main__":
    main()


