from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, TypedDict
from typing import Literal

import re


"""
Schema blocks JSON spec (blocks.schema.jsonl) — minimal required fields:

- block_id: string
- text: string
- slots: list[string]  # fixed enum, see SCHEMA_SLOT_ENUM

Example entry:
{
  "block_id": "10.1186_s12985-024-02387-x::schema",
  "text": "Confirmed evidence slots ...",
  "slots": ["transmission", "pathogenesis"],
  "doc_id": "10.1186_s12985-024-02387-x",
  "source_uri": "/path/to/pdf",
  "token_count": 123,
  "metadata": {"schema_version": "v1"}
}
"""

# Fixed enum of slot identifiers (extend cautiously; keep stable for evaluation).
SCHEMA_SLOT_ENUM: Tuple[str, ...] = (
    "transmission",
    "pathogenesis",
    "clinical_features",
    "diagnosis",
    "treatment",
)

SchemaSlot = Literal[
    "transmission",
    "pathogenesis",
    "clinical_features",
    "diagnosis",
    "treatment",
]


class SchemaBlock(TypedDict, total=False):
    block_id: str
    text: str
    slots: List[SchemaSlot]
    doc_id: str
    source_uri: str
    token_count: int
    metadata: dict


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
    candidate_slots: Optional[Sequence[Sequence[str]]] = None,
    answered_slots: Optional[Set[str]] = None,
    required_slots: Optional[Set[str]] = None,
    cfg: Optional[SchemaAnswerabilityConfig] = None,
) -> Tuple[List[int], dict]:
    """
    Returns selected indices into candidate_* arrays, plus a small debug dict.
    """
    if cfg is None:
        cfg = SchemaAnswerabilityConfig()
    max_selected = max(1, int(cfg.max_selected))
    answered_slots = set(answered_slots or set())
    required_slots = set(required_slots or set())
    uncovered = set(required_slots) - set(answered_slots)

    terms = _query_terms(query_text, max_terms=int(cfg.max_query_terms))
    scored: List[Tuple[int, int]] = []  # (overlap, idx) — overlap is secondary, never uses ANN score.
    rejected_due_to_slot_overlap: List[str] = []
    covered_by_candidate: List[Set[str]] = []
    for i, txt in enumerate(candidate_texts):
        tl = (txt or "").lower()
        slots_i: Set[str] = set()
        if candidate_slots is not None and i < len(candidate_slots):
            slots_i = {str(s) for s in (candidate_slots[i] or []) if str(s)}
        covered_by_candidate.append(slots_i)

        # Slot-gating (STRICT):
        # A schema is selectable ONLY IF it covers at least one still-unanswered required slot.
        if uncovered:
            if not (slots_i & uncovered):
                rejected_due_to_slot_overlap.append(str(candidate_ids[i]))
                scored.append((-1, i))
                continue
        if not tl:
            scored.append((0, i))
            continue
        overlap = 0
        for t in terms:
            if t and t in tl:
                overlap += 1
        scored.append((overlap, i))

    # If we are in slot-aware mode (required_slots provided), and no candidate passed slot gating -> return empty.
    if required_slots and uncovered:
        if all(ov < 0 for ov, _ in scored):
            dbg = {
                "selector": "slot_gating_then_lexical_overlap",
                "required_slots": sorted(required_slots),
                "answered_slots": sorted(answered_slots),
                "uncovered_slots": sorted(uncovered),
                "rejected_due_to_slot_overlap": rejected_due_to_slot_overlap[:50],
                "candidates": int(len(candidate_texts)),
                "selected_ids": [],
                "covered_slots": [],
                "newly_answered_slots": [],
            }
            return [], dbg

    # Secondary filter: lexical overlap (never ANN score). Only consider ov>=0 candidates.
    scored_sorted = sorted([x for x in scored if x[0] >= 0], key=lambda x: (x[0], -x[1]), reverse=True)
    selected = [idx for ov, idx in scored_sorted if ov >= int(cfg.min_overlap)][:max_selected]
    # Fallback: pick best lexical overlap among eligible candidates (still ignores ANN).
    if not selected and scored_sorted:
        selected = [scored_sorted[0][1]]

    covered_slots_sel: List[List[str]] = []
    newly_answered: Set[str] = set()
    for i in selected:
        cov = covered_by_candidate[i]
        covered_slots_sel.append(sorted(cov))
        newly_answered |= (cov & uncovered) if uncovered else set()

    dbg = {
        "selector": "slot_gating_then_lexical_overlap" if required_slots else "lexical_overlap",
        "max_selected": int(cfg.max_selected),
        "min_overlap": int(cfg.min_overlap),
        "query_terms": terms[:10],
        "candidates": int(len(candidate_texts)),
        "selected_ids": [str(candidate_ids[i]) for i in selected] if selected else [],
        "selected_overlaps": [int(scored[i][0]) if i < len(scored) else 0 for i in selected] if selected else [],
        "required_slots": sorted(required_slots),
        "answered_slots": sorted(answered_slots),
        "uncovered_slots": sorted(uncovered),
        "covered_slots": covered_slots_sel,
        "newly_answered_slots": sorted(newly_answered),
        "rejected_due_to_slot_overlap": rejected_due_to_slot_overlap[:50],
    }
    return selected, dbg


