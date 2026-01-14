"""
Pattern-first extraction utilities (KVI 2.0 / RIM v0.4).

Inputs: block text (+ optional metadata)
Outputs:
- abbreviation expansions (abbr <-> full name)
- lightweight entities (string candidates)
- schema slot cues (lexical, not embedding)

These signals are used for:
- building Pattern-first sidecar indices (alias_map, schema_triggers, fixed_entities)
- enriching evidence block metadata
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AbbreviationPair:
    abbr: str
    full: str
    confidence: float
    source: str  # regex name / heuristic channel


_PAREN_PAIRS = [
    ("(", ")"),
    ("（", "）"),
]


def _norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_abbreviation_pairs(text: str, *, max_pairs: int = 32) -> List[AbbreviationPair]:
    """
    Extract "Full Name (ABBR)" patterns (EN + ZH-friendly).
    Conservative heuristics to avoid swallowing long segments.
    """
    t = _norm_space(text)
    if not t:
        return []

    pairs: List[AbbreviationPair] = []

    # 1) Full (ABBR) patterns
    # English-ish full name: words/spaces/hyphens
    pat_en = re.compile(
        r"(?P<full>[A-Za-z][A-Za-z0-9 \-]{3,80})\s*[\(\（]\s*(?P<abbr>[A-Z][A-Z0-9\-]{1,12})\s*[\)\）]"
    )
    # Chinese/biomed full name with optional latin/digits
    pat_zh = re.compile(
        r"(?P<full>[\u4e00-\u9fffA-Za-z0-9·\-\s]{2,40})\s*[\(\（]\s*(?P<abbr>[A-Z][A-Z0-9\-]{1,12})\s*[\)\）]"
    )

    def _push(full: str, abbr: str, conf: float, source: str) -> None:
        full2 = _norm_space(full)
        abbr2 = _norm_space(abbr).upper()
        if not full2 or not abbr2:
            return
        # guardrails
        if len(abbr2) < 2 or len(abbr2) > 12:
            return
        if len(full2) < 2 or len(full2) > 120:
            return
        # avoid cases where full is mostly abbr
        if full2.upper() == abbr2:
            return
        pairs.append(AbbreviationPair(abbr=abbr2, full=full2, confidence=float(conf), source=str(source)))

    for m in pat_en.finditer(t):
        _push(m.group("full"), m.group("abbr"), 0.95, "full(abbr):en")
        if len(pairs) >= max_pairs:
            break
    if len(pairs) < max_pairs:
        for m in pat_zh.finditer(t):
            _push(m.group("full"), m.group("abbr"), 0.90, "full(abbr):zh")
            if len(pairs) >= max_pairs:
                break

    # 2) ABBR (Full) patterns (lower confidence; easy to over-capture)
    if len(pairs) < max_pairs:
        pat_rev = re.compile(
            r"(?P<abbr>[A-Z][A-Z0-9\-]{1,12})\s*[\(\（]\s*(?P<full>[^)\）]{2,60})\s*[\)\）]"
        )
        for m in pat_rev.finditer(t):
            full = _norm_space(m.group("full"))
            if len(full) > 60:
                continue
            # require full to have letters/cjk to avoid numeric ranges
            if not re.search(r"[A-Za-z\u4e00-\u9fff]", full):
                continue
            _push(full, m.group("abbr"), 0.55, "abbr(full):rev")
            if len(pairs) >= max_pairs:
                break

    # de-dupe (abbr, full)
    seen = set()
    out: List[AbbreviationPair] = []
    for p in pairs:
        key = (p.abbr, p.full)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out[:max_pairs]


def extract_entities(text: str, *, max_entities: int = 64) -> List[str]:
    """
    Very lightweight entity candidate extraction (no NER dependency).
    Intended for Pattern-first priors/debug, not as a semantic truth source.
    """
    t = _norm_space(text)
    if not t:
        return []

    ents: List[str] = []

    # English title-case / proper noun-ish phrases
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:[ \-][A-Z][a-z]{2,}){0,5}\b", t):
        s = m.group(0).strip()
        if 3 <= len(s) <= 80:
            ents.append(s)
        if len(ents) >= max_entities:
            break

    # CJK disease-ish terms: ...综合征/病毒/感染/发热/血小板减少/抗体...
    if len(ents) < max_entities:
        for m in re.finditer(r"[\u4e00-\u9fff]{2,20}(综合征|病毒|感染|发热|血小板减少|抗体|疫苗|治疗)", t):
            s = m.group(0).strip()
            if 2 <= len(s) <= 40:
                ents.append(s)
            if len(ents) >= max_entities:
                break

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for e in ents:
        k = e.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out[:max_entities]


def infer_schema_slots_from_text(text: str) -> List[str]:
    """
    Infer coarse schema slots from a block's text using existing heuristic slot inference.
    """
    try:
        from .runtime.schema_answerability import infer_slots_from_query  # type: ignore
        from .runtime.slot_registry import SLOT_REGISTRY  # type: ignore

        inferred = infer_slots_from_query(text or "")
        # keep stable, adjudicable slots first
        slots = [s for s in inferred if s in SLOT_REGISTRY]
        # stable ordering: registry order
        ordered = [s for s in SLOT_REGISTRY.keys() if s in set(slots)]
        return ordered
    except Exception:
        return []


def infer_block_type(*, text: str, metadata: Dict[str, Any]) -> str:
    """
    Coarse block type classification for metadata.
    """
    meta = metadata or {}
    if bool(meta.get("is_table")) or bool((meta.get("tables") or {}).get("table_ids")):
        return "table"
    slots = infer_schema_slots_from_text(text)
    if "treatment" in slots:
        return "treatment"
    if "diagnosis" in slots:
        return "diagnosis"
    if "transmission" in slots:
        return "transmission"
    if "pathogenesis" in slots or "mechanism" in slots:
        return "pathogenesis"
    if "clinical_features" in slots:
        return "clinical_features"
    if "prevention" in slots:
        return "prevention"
    return "general"

