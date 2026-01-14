"""
RIM (Reasoning Introspection Module) - v0.4 (KVI 2.0) controller.

Spec: docs/07_KVI_RMI.md
- Pattern-first -> Semantic-second -> Introspection-gated
- RIM MUST NOT generate tokens; it is a control module / gate.
- Must support:
  - reasoning shift detection (cosine drift)
  - KV relevance judgement (e.g., logit delta / entropy / heuristic)
  - KV reject + refresh (<= 2 rounds)
  - incremental retrieval & injection (prefix past_key_values; do NOT modify attention forward)

Note:
- This file provides the controller/state machine and helper pooling logic.
- Model-based relevance tests are implemented in runtime utilities (e.g., runtime/kv_relevance.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

    # "KV irrelevant -> refresh" (<=2 per spec; interpreted as number of refresh attempts after the first batch)
    kv_refresh_rounds: int = 2

    # KV irrelevance threshold (default aligned with demo; interpreted by runtime relevance tests)
    kv_irrelevant_logit_delta_threshold: float = 0.05

    # Reasoning query pooling window (tokens)
    reasoning_window: int = 32


@dataclass
class RIMState:
    q0: Optional[np.ndarray] = None
    num_realign: int = 0
    used_ids: Set[str] = field(default_factory=set)
    last_gate: Optional[Dict[str, Any]] = None
    last_pattern_hits: Any = None
    last_semantic_hits: Any = None
    last_hidden: Any = None
    last_generated: Any = None


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
    Minimal, reusable RIM controller (v0.4 interface).

    Notes:
    - RIM.observe(...) stores current state (hidden, tokens, pattern hits, semantic hits).
    - RIM.should_realign() uses the last observed state + configured signals.
    - build_reasoning_query() returns a pooled representation from hidden states (q'_t in model space).
      Converting q'_t -> retrieval embedding is caller's responsibility (projector/encoder),
      because KV bank key space is not guaranteed to match base LLM hidden space.
    """

    def __init__(self, *, retriever: Retriever, cfg: RIMConfig) -> None:
        self.retriever = retriever
        self.cfg = cfg
        self.state = RIMState()

    # -------------------------
    # v0.4 required interface
    # -------------------------

    def observe(self, hidden_states: Any, generated_tokens: Any, pattern_hits: Any, semantic_hits: Any) -> None:
        self.state.last_hidden = hidden_states
        self.state.last_generated = generated_tokens
        self.state.last_pattern_hits = pattern_hits
        self.state.last_semantic_hits = semantic_hits

    def exceeded_budget(self) -> bool:
        return bool(int(self.state.num_realign) >= int(self.cfg.max_realign))

    def should_realign(self) -> bool:
        gate = self.state.last_gate or {}
        return bool(gate.get("retrieve_more") is True) and not self.exceeded_budget()

    def build_reasoning_query(self) -> np.ndarray:
        """
        Build q'_t from last observed hidden states via mean pooling.
        Returns a vector in model hidden space (not necessarily in KV bank embedding space).
        """
        hs = self.state.last_hidden
        if hs is None:
            raise ValueError("RIM.build_reasoning_query called before observe(hidden_states, ...)")

        # HF convention: hidden_states can be a tuple(list) of layers; take the last layer.
        h = hs[-1] if isinstance(hs, (list, tuple)) else hs
        # Expect shape [B, T, H] for last layer
        arr = np.asarray(getattr(h, "detach", lambda: h)())
        # torch->numpy path
        try:
            import torch  # local import to avoid hard dependency for pure numpy users

            if isinstance(h, torch.Tensor):
                t = h
                # last N tokens
                n = max(1, int(self.cfg.reasoning_window))
                t = t[:, -n:, :]
                pooled = t.mean(dim=1)
                return pooled[0].to(dtype=torch.float32).detach().cpu().numpy()
        except Exception:
            pass

        # numpy fallback (best-effort)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected hidden state shape for pooling: ndim={arr.ndim}")
        n = max(1, int(self.cfg.reasoning_window))
        arr = arr[:, -n:, :]
        pooled = arr.mean(axis=1)
        return pooled[0].astype(np.float32, copy=False)

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

    # -------------------------
    # v0.4 gate logic helpers
    # -------------------------

    def set_q0(self, q0_vec: np.ndarray) -> None:
        """
        Store q0 for drift detection. Caller decides what representation to use (e.g., retrieval embedding).
        """
        self.state.q0 = np.asarray(q0_vec, dtype=np.float32).reshape(-1)

    def introspection_gate(
        self,
        *,
        q_prime_vec: np.ndarray,
        kv_relevance_delta: Optional[float] = None,
        pattern_mismatch: bool = False,
    ) -> Dict[str, Any]:
        """
        Implements required Q1/Q2/Q3 outputs (docs/07_KVI_RMI.md §5.3).

        Inputs:
        - q_prime_vec: a vector representation for drift detection (must align with q0_vec space set via set_q0)
        - kv_relevance_delta: model-based relevance signal (e.g., logit delta vs zero prefix)
        - pattern_mismatch: optional signal when pattern priors conflict with retrieved semantics
        """
        retrieve_more = False
        reject_current_kv = False
        rationale = "none"
        conf = 0.5

        # Q1: reasoning shift
        if self.state.q0 is not None:
            dist = _cosine_distance(self.state.q0, np.asarray(q_prime_vec, dtype=np.float32))
            if dist > float(self.cfg.tau_cos_dist):
                retrieve_more = True
                rationale = "semantic-shift"
                conf = max(conf, 0.7)
        else:
            dist = None

        # Q2: KV relevance (low impact)
        if kv_relevance_delta is not None and kv_relevance_delta < float(self.cfg.kv_irrelevant_logit_delta_threshold):
            reject_current_kv = True
            retrieve_more = True
            rationale = "low-impact"
            conf = max(conf, 0.75)

        # Q3: reject + refresh based on pattern mismatch (optional)
        if pattern_mismatch:
            reject_current_kv = True
            retrieve_more = True
            rationale = "pattern-mismatch"
            conf = max(conf, 0.65)

        out = {
            "retrieve_more": bool(retrieve_more),
            "reject_current_kv": bool(reject_current_kv),
            "confidence": float(max(0.0, min(1.0, conf))),
            "rationale": str(rationale),
        }
        if dist is not None:
            out["cos_dist"] = float(dist)
            out["tau"] = float(self.cfg.tau_cos_dist)

        self.state.last_gate = out
        return out

    def mark_realign_used(self) -> None:
        self.state.num_realign += 1

