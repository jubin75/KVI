"""
模块：kv_bank

你当前的 demo 选择
- 向量索引：本地 FAISS（文件落盘）
- 存储内容：直接存 `K_ext/V_ext`（检索命中后可直接注入）

本文件的定位
- 对上层提供“KV Bank”统一入口（build/load/search），内部可切换不同后端实现。
- demo 阶段默认使用 `vector_store.faiss_kv_bank.FaissKVBank`。

生产演进方向（后续再做）
- 分片/冷热分层/增量更新/压缩量化/过滤预索引等
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .vector_store.faiss_kv_bank import FaissKVBank, KVItem


KVBank = FaissKVBank


__all__ = [
    "KVItem",
    "KVBank",
    "FaissKVBank",
]



