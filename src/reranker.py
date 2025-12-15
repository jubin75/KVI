"""
reranker: 可选的候选重排（最小实现）

说明
- 生产级可接 cross-encoder reranker；
- 这里提供一个“可运行”的 reranker：按 (retrieval_score + alpha * length_prior) 重新排序。
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



