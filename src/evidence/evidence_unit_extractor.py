"""
Evidence Unit Pipeline (Sentence-level Enumerative First).

Implements docs/078_Evidence_extract.md.
No heavy NLP libs; deterministic heuristics only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\。\！\？])\s+|\n+")
_ENUM_SIGNAL_RE = re.compile(
    # Enumeration signals (EN + ZH): commas/semicolons/conjunctions.
    r"(,|;|，|；|、|\band\b|\bor\b|和|或|以及|及)",
    flags=re.IGNORECASE,
)
_DISCOURSE_LIST_RE = re.compile(r"\b(in addition|furthermore|see also)\b", flags=re.IGNORECASE)
_BLOCKING_META_RE = re.compile(
    r"\b(statistical analysis|methods|materials and methods|copyright|license|p value|confidence interval)\b",
    flags=re.IGNORECASE,
)


@dataclass
class EvidenceUnitExtractor:
    def infer_section_type(self, *, text: str, metadata: Dict[str, Any]) -> str:
        """
        Minimal mapping to doc spec: paragraph | figure | table | supplementary | other
        """
        meta = metadata or {}
        bt = str(meta.get("block_type") or "").strip().lower()
        if bt in {"figure", "fig"}:
            return "figure"
        if bt in {"table"}:
            return "table"
        if bt in {"supplementary", "supplement", "supp"}:
            return "supplementary"
        # Heuristic flags propagated from pipelines
        if bool(meta.get("is_table")) or bool((meta.get("tables") or {}).get("table_ids")):
            return "table"
        t = str(text or "").lower()
        if "supplementary" in t:
            return "supplementary"
        if "figure" in t and ("fig." in t or "figure" in t):
            return "figure"
        if "table" in t and ("table " in t or "table:" in t):
            return "table"
        return "paragraph" if str(text or "").strip() else "other"

    def split_sentences(self, *, block_id: str, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        t = str(text or "")
        if not t.strip():
            return out
        # Approximate offset via incremental scan.
        cursor = 0
        parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
        for i, p in enumerate(parts):
            idx = t.find(p, cursor)
            if idx < 0:
                idx = cursor
            cursor = idx + len(p)
            out.append({"sentence_id": f"{block_id}#s{i}", "text": p, "offset": int(idx)})
        return out

    def extract_units(
        self,
        *,
        block_id: str,
        text: str,
        section_type: str,
        sentences: List[Dict[str, Any]],
        list_features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of evidence units (injectable or not), per spec.
        """
        block_id = str(block_id or "").strip()
        section_type = str(section_type or "").strip().lower()
        full_text = str(text or "")

        units: List[Dict[str, Any]] = []

        # STEP 1 — sentence-level enumerative (PRIMARY)
        # Always output sentence-derived units (injectable or not).
        sent_units: List[Dict[str, Any]] = []
        for s in sentences or []:
            sid = str(s.get("sentence_id") or "")
            stxt = str(s.get("text") or "").strip()
            if not stxt:
                continue
            unit = self._as_sentence_enumerative(block_id=block_id, sentence_id=sid, sentence_text=stxt)
            if unit is not None:
                sent_units.append(unit)
            else:
                # still output a non-injectable sentence fragment for traceability
                sent_units.append(self._unit(block_id, sid, "fragment", stxt, "non_fact", 0.2, ["non_enumerative"]))
        has_primary = any(u.get("unit_type") == "sentence_enumerative" for u in sent_units)
        if has_primary:
            # STEP 2 is skipped if any valid sentence-level enumerative unit exists in block.
            return sent_units

        # STEP 2 — list-like / fragment (SECONDARY, only if no valid sentence-level unit)
        blocking = self._blocking_reasons_for_block(section_type=section_type, text=full_text)
        if blocking:
            # output minimal blocked fragment unit (still traceable)
            return sent_units + [self._unit(block_id, None, "fragment", full_text, "non_fact", 0.1, blocking)]

        # Use list_features-derived items if present; otherwise fallback to a single fragment.
        items = self._list_items(list_features)
        if items:
            for idx, it in enumerate(items):
                itxt = str(it or "").strip()
                if not itxt:
                    continue
                role = self._semantic_role_list_item(itxt)
                br = self._blocking_reasons_for_item(itxt, role)
                conf = 0.65 if role == "enumerative_fact" and not br else 0.25
                units.append(self._unit(block_id, None, "list_item", itxt, role, conf, br))
            return sent_units + units

        # no list items; output a single fragment
        role = "descriptive_fact" if self._looks_factual(full_text) else "non_fact"
        br = ["insufficient_specificity"] if role != "descriptive_fact" else ["non_enumerative"]
        return sent_units + [self._unit(block_id, None, "fragment", full_text, role, 0.2, br)]

    # ---------- internals ----------

    @staticmethod
    def _unit(
        block_id: str,
        sentence_id: Optional[str],
        unit_type: str,
        text: str,
        semantic_role: str,
        confidence: float,
        blocking_reasons: List[str],
    ) -> Dict[str, Any]:
        sid = str(sentence_id) if sentence_id else None
        unit_id = f"{block_id}#u{unit_type}:{sid or 'null'}:{abs(hash(text)) % 100000}"
        allowed = bool(
            semantic_role == "enumerative_fact"
            and unit_type in {"sentence_enumerative", "list_item"}
            and not blocking_reasons
        )
        return {
            "unit_id": unit_id,
            "source": {"block_id": block_id, "sentence_id": sid},
            "unit_type": unit_type,
            "semantic_role": semantic_role,
            "text": str(text or ""),
            "confidence": float(confidence),
            "injectability": {"allowed": bool(allowed), "blocking_reasons": list(blocking_reasons or [])},
        }

    def _as_sentence_enumerative(
        self, *, block_id: str, sentence_id: str, sentence_text: str
    ) -> Optional[Dict[str, Any]]:
        t = str(sentence_text or "").strip()
        tl = t.lower()
        if not t:
            return None
        if _DISCOURSE_LIST_RE.search(tl):
            return None
        if _BLOCKING_META_RE.search(tl):
            return None
        if not _ENUM_SIGNAL_RE.search(t):
            return None

        items, cross_mixed = self._split_enumeration_items(t)
        if len(items) < 3:
            return None
        if cross_mixed:
            return self._unit(block_id, sentence_id, "fragment", t, "non_fact", 0.2, ["cross_semantic_mixed"])

        # self-contained heuristic: avoid "respectively"/"as follows" style references
        if any(k in tl for k in ("as follows", "respectively", "listed below")):
            return None

        return self._unit(block_id, sentence_id, "sentence_enumerative", t, "enumerative_fact", 0.85, [])

    @staticmethod
    def _split_enumeration_items(sentence: str) -> Tuple[List[str], bool]:
        """
        Very lightweight enumeration split.
        Returns (items, cross_semantic_mixed).
        """
        s = str(sentence)
        # normalize conjunctions (EN + ZH) into commas for a unified split.
        s2 = re.sub(r"\b(and|or)\b", ",", s, flags=re.IGNORECASE)
        # Chinese conjunctions (keep it conservative; no heavy NLP)
        s2 = re.sub(r"(以及|及|和|或)", "，", s2)
        # split on commas/semicolons (EN + ZH) and Chinese list delimiter
        raw = [p.strip() for p in re.split(r"[,;，；、]", s2) if p and p.strip()]
        # drop leading clause like "Patients present with" / "包括/表现为"
        if raw:
            raw0 = raw[0]
            raw0 = re.sub(
                r"^.*?\b(with|including|include|present with|characterized by)\b",
                "",
                raw0,
                flags=re.IGNORECASE,
            ).strip()
            raw0 = re.sub(r"^.*?(包括|主要包括|表现为|表现出|常见表现包括|临床表现包括|临床表现为)[:：]?", "", raw0).strip()
            raw[0] = raw0
        items = [r for r in raw if 2 <= len(r) <= 60]
        # cross semantic mixed heuristic: presence of many digits/units mixed with words
        digitish = sum(1 for it in items if re.search(r"\d", it))
        wordish = sum(1 for it in items if re.search(r"[A-Za-z\u4e00-\u9fff]", it))
        cross_mixed = bool(digitish >= 2 and wordish >= 2)
        return items, cross_mixed

    @staticmethod
    def _blocking_reasons_for_block(*, section_type: str, text: str) -> List[str]:
        st = str(section_type or "").lower()
        t = str(text or "").lower()
        if st in {"figure", "table", "supplementary"}:
            return [f"section_excluded:{st}"]
        if _DISCOURSE_LIST_RE.search(t):
            return ["discourse_list"]
        if _BLOCKING_META_RE.search(t):
            return ["procedural_or_metadata"]
        return []

    @staticmethod
    def _list_items(list_features: Dict[str, Any]) -> List[str]:
        if not isinstance(list_features, dict):
            return []
        items = list_features.get("list_items")
        if isinstance(items, list) and items:
            return [str(x) for x in items if str(x).strip()]
        items = list_features.get("list_like_items")
        if isinstance(items, list) and items:
            return [str(x) for x in items if str(x).strip()]
        return []

    @staticmethod
    def _semantic_role_list_item(text: str) -> str:
        t = str(text or "").strip()
        tl = t.lower()
        if not t:
            return "non_fact"
        if _DISCOURSE_LIST_RE.search(tl):
            return "non_fact"
        if _BLOCKING_META_RE.search(tl):
            return "non_fact"
        # list item: treat as enumerative_fact only if looks like a factual medical-ish phrase
        if re.search(r"[A-Za-z\u4e00-\u9fff]", t) and len(t) <= 120:
            return "enumerative_fact"
        return "non_fact"

    @staticmethod
    def _blocking_reasons_for_item(text: str, semantic_role: str) -> List[str]:
        tl = str(text or "").lower()
        if _DISCOURSE_LIST_RE.search(tl):
            return ["discourse_list"]
        if _BLOCKING_META_RE.search(tl):
            return ["procedural_or_metadata"]
        if semantic_role != "enumerative_fact":
            return ["non_enumerative"]
        if len(str(text or "").strip()) < 3:
            return ["insufficient_specificity"]
        return []

    @staticmethod
    def _looks_factual(text: str) -> bool:
        tl = str(text or "").lower()
        if any(k in tl for k in ("we ", "our ", "study", "analysis", "method", "supplementary")):
            return False
        return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", tl))

