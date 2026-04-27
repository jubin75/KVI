"""
reranker: optional candidate re-ranking (minimal implementation)

Notes
- Production-grade can plug in cross-encoder reranker;
- Here we provide a "runnable" reranker: re-sort by (retrieval_score + alpha * length_prior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class RerankConfig:
    alpha: float = 0.0  # length prior weight


def rerank(
    *,
    query: Any,
    candidates: Sequence[Dict[str, Any]],
    cfg: RerankConfig = RerankConfig(),
) -> List[Dict[str, Any]]:
    def _score(c: Dict[str, Any]) -> float:
        base = float(c.get("score", 0.0))
        txt = c.get("text_snippet") or c.get("text") or ""
        return base + cfg.alpha * float(len(str(txt)))

    return sorted(list(candidates), key=_score, reverse=True)



