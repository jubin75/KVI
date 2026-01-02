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
    # answerable_slots: subset of slots that the evidence can SUBSTANTIVELY answer.
    # If absent, falls back to slots. Only answerable_slots satisfy required slots.
    answerable_slots: List[SchemaSlot]
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


# Fallback regex patterns (used when dynamic extraction yields nothing).
_SLOT_PATTERNS_FALLBACK: Tuple[Tuple[str, re.Pattern], ...] = (
    ("transmission", re.compile(r"(transmit|传播|spread|route|vector|tick|蜱)", re.I)),
    ("pathogenesis", re.compile(r"(pathogen|mechanism|发病机制|致病|cytokine|immun|inflam)", re.I)),
    ("clinical_features", re.compile(r"(symptom|症状|clinical|manifest|fever|sign)", re.I)),
    ("diagnosis", re.compile(r"(diagnos|检测|诊断|test|pcr|elisa|assay)", re.I)),
    ("treatment", re.compile(r"(treat|治疗|therap|drug|antivir|cure|药物)", re.I)),
    ("epidemiology", re.compile(r"(epidem|流行病|incidence|prevalence|outbreak)", re.I)),
    # New adjudicable slots (low-risk, stable) — enables L0 adjudication for common questions
    ("disease_full_name", re.compile(r"(全称|全名|缩写|英文全称|stand\s*for|full\s*name)", re.I)),
    ("geographic_distribution", re.compile(r"(地区分布|分布在|哪些地方|哪里|在哪些省|china|province|geographic)", re.I)),
    ("prevention", re.compile(r"(prevent|预防|vaccine|vaccin|prophyla)", re.I)),
    ("prognosis", re.compile(r"(prognos|预后|outcome|mortality|survival)", re.I)),
)

# Dynamic slot extraction: question-word → slot-type mapping.
_QUESTION_SLOT_MAP: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(how|怎么|如何).*(transmit|传播|spread)", re.I), "transmission"),
    (re.compile(r"(what|什么).*(cause|原因|机制|mechan)", re.I), "pathogenesis"),
    (re.compile(r"(what|哪些).*(symptom|症状|表现)", re.I), "clinical_features"),
    (re.compile(r"(how|怎么|如何).*(diagnos|诊断|检测)", re.I), "diagnosis"),
    (re.compile(r"(how|怎么|如何).*(treat|治疗|cure)", re.I), "treatment"),
    (re.compile(r"(what|哪些).*(risk|危险因素|风险)", re.I), "risk_factors"),
    (re.compile(r"(what|哪些).*(complicat|并发症)", re.I), "complications"),
    (re.compile(r"(what|多少).*(mortality|死亡率|fatality)", re.I), "prognosis"),
    (re.compile(r"(how|怎么).*(prevent|预防)", re.I), "prevention"),
)

# Concept extraction patterns (noun phrases / key terms → dynamic slots).
_CONCEPT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(risk\s*factor|危险因素|风险因素)\b", re.I),
    re.compile(r"\b(complicat\w*|并发症)\b", re.I),
    re.compile(r"\b(incubation|潜伏期)\b", re.I),
    re.compile(r"\b(reservoir|宿主|动物宿主)\b", re.I),
    re.compile(r"\b(seroprevalence|血清阳性率)\b", re.I),
    re.compile(r"\b(case\s*fatality|病死率)\b", re.I),
    re.compile(r"\b(viral\s*load|病毒载量)\b", re.I),
    re.compile(r"\b(immune\s*evasion|免疫逃逸)\b", re.I),
    re.compile(r"\b(cytokine\s*storm|细胞因子风暴)\b", re.I),
    re.compile(r"\b(human-to-human|人传人)\b", re.I),
)


def _extract_dynamic_slots(query: str) -> Set[str]:
    """
    Extract slots dynamically from query using:
    1) Question-word → slot-type mapping
    2) Concept pattern matching → generate slot from matched concept
    Returns set of slot identifiers (arbitrary strings, not limited to enum).
    """
    q = (query or "").strip()
    if not q:
        return set()
    slots: Set[str] = set()
    # 1) Question-word patterns.
    for pat, slot_id in _QUESTION_SLOT_MAP:
        if pat.search(q):
            slots.add(slot_id)
    # 2) Concept patterns → derive slot name from matched text.
    for pat in _CONCEPT_PATTERNS:
        m = pat.search(q)
        if m:
            # Normalize matched concept to slot_id (lowercase, underscores).
            concept = m.group(0).lower().strip()
            concept = re.sub(r"\s+", "_", concept)
            concept = re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", concept)
            if concept:
                slots.add(concept)
    return slots


def infer_slots_from_query(query: str) -> Set[str]:
    """
    Infer required_slots from user query.
    Strategy: dynamic extraction PLUS fallback regex patterns (union).
    Rationale: multi-intent questions may match only partially via question-word patterns
    (e.g., "是什么？" can match pathogenesis while "传播/途径" should still be captured by fallback).
    Returns a set of slot identifiers (arbitrary, not limited to SCHEMA_SLOT_ENUM).
    """
    q = (query or "").strip()
    if not q:
        return set()
    # 1) Try dynamic extraction.
    slots = _extract_dynamic_slots(q)
    # 2) Always apply fallback: regex patterns (union, never early-return).
    for slot_id, pattern in _SLOT_PATTERNS_FALLBACK:
        if pattern.search(q):
            slots.add(slot_id)
    return slots


def _fuzzy_slot_match(candidate_slots: Set[str], required_slots: Set[str]) -> Set[str]:
    """
    Fuzzy slot matching: a candidate slot matches a required slot if:
    1) Exact match, OR
    2) One is substring of the other (e.g., "risk" matches "risk_factors"), OR
    3) Normalized forms match (underscores/spaces/case ignored).
    Returns the set of required_slots that are covered by candidate_slots.
    """
    matched: Set[str] = set()
    for req in required_slots:
        req_norm = req.lower().replace("_", "").replace(" ", "")
        for cand in candidate_slots:
            cand_norm = cand.lower().replace("_", "").replace(" ", "")
            # Exact or substring match.
            if req_norm == cand_norm or req_norm in cand_norm or cand_norm in req_norm:
                matched.add(req)
                break
    return matched


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
    Supports arbitrary slots via fuzzy matching (not limited to SCHEMA_SLOT_ENUM).
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

        # Slot-gating (STRICT) with fuzzy matching for dynamic slots:
        # A schema is selectable ONLY IF it covers at least one still-unanswered required slot.
        if uncovered:
            matched_slots = _fuzzy_slot_match(slots_i, uncovered)
            if not matched_slots:
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
        # Use fuzzy matching to determine newly answered slots.
        newly_answered |= _fuzzy_slot_match(cov, uncovered) if uncovered else set()

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


