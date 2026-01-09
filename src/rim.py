"""
RIM (Reasoning Introspection Module) - engineering implementation scaffold.

Design goals (see docs/06_KVI_IM.md):
- Do NOT modify base LLM weights
- Do NOT modify KV Bank implementation
- Provide reasoning-aware re-alignment:
  - detect reasoning shift (representation drift)
  - build a reasoning query vector q'_t
  - retrieve additional KV from KV Bank
  - inject incrementally via past_key_values prefix
  - optionally: if injected KV is judged irrelevant, refresh (retrieve a new batch) for up to N rounds

This module is intentionally model-agnostic; model forward for relevance checks lives in runtime utils.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .retriever import Retriever


@dataclass(frozen=True)
class RIMConfig:
    # Shift detection
    tau_cos_dist: float = 0.35

    # Retrieval
    top_k: int = 8
    oversample_factor: int = 3  # used when we need multiple batches / refresh rounds

    # Budget
    max_realign: int = 1

    # "KV irrelevant -> refresh" (2 rounds means: initial batch + up to 2 refresh attempts)
    kv_refresh_rounds: int = 2


@dataclass
class RIMState:
    q0: Optional[np.ndarray] = None
    num_realign: int = 0
    used_ids: Set[str] = field(default_factory=set)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(1.0 - float(np.dot(a, b)) / (na * nb))


def _kv_id(meta: Dict[str, Any]) -> str:
    return str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")


class RIM:
    """
    Minimal, reusable RIM controller.

    Notes:
    - This class does not assume how q'_t is constructed; caller can pass query_vec directly.
    - Retrieval uses `Retriever.search(query_vec, ...)`.
    - De-dupe is done by block_id/chunk_id/id in KVItem.meta.
    """

    def __init__(self, *, retriever: Retriever, cfg: RIMConfig) -> None:
        self.retriever = retriever
        self.cfg = cfg
        self.state = RIMState()

    def observe_query_vec(self, query_vec: np.ndarray, *, is_step0: bool) -> None:
        """
        Store q0 (step0) for drift comparison. Caller provides the query vec used for retrieval.
        """
        if is_step0 or self.state.q0 is None:
            self.state.q0 = np.asarray(query_vec, dtype=np.float32).reshape(-1)

    def should_realign(self, query_vec: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Decide whether to realign (trigger additional retrieval) based on representation drift.
        """
        if self.state.q0 is None:
            # If caller didn't call observe, treat as "no basis to drift-detect".
            return False, {"reason": "missing_q0"}
        if int(self.state.num_realign) >= int(self.cfg.max_realign):
            return False, {"reason": "budget_exceeded", "num_realign": int(self.state.num_realign)}

        dist = _cosine_distance(self.state.q0, np.asarray(query_vec, dtype=np.float32))
        ok = bool(dist > float(self.cfg.tau_cos_dist))
        return ok, {"cos_dist": float(dist), "tau": float(self.cfg.tau_cos_dist)}

    def retrieve_additional_kv(
        self,
        *,
        query_vec: np.ndarray,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Retrieve top-k *new* items (de-duped by id). Returns KVItem list and debug dict.
        """
        top_k = int(self.cfg.top_k)
        oversample = max(top_k, top_k * max(1, int(self.cfg.oversample_factor)))
        rr = self.retriever.search(query_vec, top_k=int(oversample), filters=filters, query_text=query_text)

        picked: List[Any] = []
        for it in rr.items:
            bid = _kv_id(getattr(it, "meta", None) or {})
            if not bid:
                continue
            if bid in self.state.used_ids:
                continue
            picked.append(it)
            self.state.used_ids.add(bid)
            if len(picked) >= top_k:
                break

        dbg = dict(rr.debug or {})
        dbg.update({"picked": len(picked), "top_k": top_k, "oversample": int(oversample)})
        return picked, dbg

    def mark_realign_used(self) -> None:
        self.state.num_realign += 1

