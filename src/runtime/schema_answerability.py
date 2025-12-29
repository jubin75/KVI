from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import re


@dataclass(frozen=True)
class SchemaAnswerabilityConfig:
    """
    Heuristic schema answerability selector (NOT similarity ranking).

    We may still use a recall-stage candidate pool (e.g. ANN top_k), but we must NOT treat
    ANN score as the decision criterion. This selector chooses an answerable schema from
    candidate schema texts using lightweight lexical overlap with the query.
    """

    max_selected: int = 1
    min_overlap: int = 2
    max_query_terms: int = 32


_STOPWORDS_EN = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
}


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def _query_terms(q: str, *, max_terms: int) -> List[str]:
    q = (q or "").strip().lower()
    if not q:
        return []
    if _has_cjk(q):
        # character bigrams for CJK (cheap + robust to tokenization)
        chars = [c for c in q if re.match(r"[\u4e00-\u9fff]", c)]
        bigrams = ["".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))]
        out = [b for b in bigrams if b]
        return out[: max_terms]
    # english-ish: word tokens
    toks = re.findall(r"[a-z0-9][a-z0-9\-\_]{1,30}", q)
    toks = [t for t in toks if t not in _STOPWORDS_EN]
    return toks[: max_terms]


def choose_answerable_schema(
    *,
    query_text: str,
    candidate_ids: Sequence[str],
    candidate_texts: Sequence[str],
    cfg: Optional[SchemaAnswerabilityConfig] = None,
) -> Tuple[List[int], dict]:
    """
    Returns selected indices into candidate_* arrays, plus a small debug dict.
    """
    if cfg is None:
        cfg = SchemaAnswerabilityConfig()
    max_selected = max(1, int(cfg.max_selected))

    terms = _query_terms(query_text, max_terms=int(cfg.max_query_terms))
    scored: List[Tuple[int, int]] = []  # (overlap, idx)
    for i, txt in enumerate(candidate_texts):
        tl = (txt or "").lower()
        if not tl:
            scored.append((0, i))
            continue
        overlap = 0
        for t in terms:
            if t and t in tl:
                overlap += 1
        scored.append((overlap, i))

    # Choose answerable schemas by overlap threshold; fall back to the best-overlap candidate.
    scored_sorted = sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True)
    selected = [idx for ov, idx in scored_sorted if ov >= int(cfg.min_overlap)][:max_selected]
    if not selected and scored_sorted:
        selected = [scored_sorted[0][1]]

    dbg = {
        "selector": "lexical_overlap",
        "max_selected": int(cfg.max_selected),
        "min_overlap": int(cfg.min_overlap),
        "query_terms": terms[:10],
        "candidates": int(len(candidate_texts)),
        "selected_ids": [str(candidate_ids[i]) for i in selected] if selected else [],
        "selected_overlaps": [int(dict(scored).get(i, 0)) for i in selected] if selected else [],
    }
    return selected, dbg


