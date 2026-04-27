"""
Module: kv_bank

Current demo choices
- Vector index: local FAISS (persisted to file)
- Storage: directly stores K_ext/V_ext (can be injected directly when retrieved)

Purpose of this file
- Provide a unified "KV Bank" entry point for the upper layer (build/load/search), with switchable backend implementations.
- During demo phase, default to vector_store.faiss_kv_bank.FaissKVBank.

Production evolution direction (TBD)
- Sharding / hot-cold tiering / incremental updates / compression-quantization / filter pre-indexing, etc.
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



