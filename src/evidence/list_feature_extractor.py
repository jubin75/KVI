"""
EvidenceListFeatureExtractor: data-driven list-like feature extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML is required for list_feature_rules.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class EvidenceListFeatureExtractor:
    rule_dir: str

    def __init__(self, rule_dir: str) -> None:
        self.rule_dir = str(rule_dir)

    def extract(self, block: Dict[str, Any]) -> Dict[str, Any]:
        text = str(block.get("text") or "")
        if not text:
            return {"list_features": {"is_list_like": False, "list_items": [], "signals": [], "confidence": 0.0}}

        semantic_type = self._infer_semantic_type(block)
        rules = self._load_rules(semantic_type=semantic_type)
        signals: List[str] = []
        list_items: List[str] = []
        confidence = 0.0

        bullets = rules.get("signals", {}).get("bullets", []) if isinstance(rules.get("signals"), dict) else []
        numbering_regex = rules.get("signals", {}).get("numbering_regex", []) if isinstance(rules.get("signals"), dict) else []
        trigger_phrases = rules.get("signals", {}).get("trigger_phrases", []) if isinstance(rules.get("signals"), dict) else []
        paren_cases_regex = rules.get("signals", {}).get("paren_cases_regex") if isinstance(rules.get("signals"), dict) else None
        paren_cases_capture_regex = (
            rules.get("signals", {}).get("paren_cases_capture_regex") if isinstance(rules.get("signals"), dict) else None
        )
        delimiters = rules.get("split", {}).get("delimiters", []) if isinstance(rules.get("split"), dict) else []
        conf_rules = rules.get("confidence", {}) if isinstance(rules.get("confidence"), dict) else {}

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if any(ln.startswith(tuple(bullets)) for ln in lines if bullets):
            signals.append("bullet")
            confidence = max(confidence, float(conf_rules.get("bullet") or 0.0))
            list_items.extend(self._extract_bullets(lines, bullets))

        for rx in numbering_regex:
            try:
                if any(re.match(rx, ln) for ln in lines):
                    signals.append("numbering")
                    confidence = max(confidence, float(conf_rules.get("numbering") or 0.0))
                    list_items.extend(self._extract_numbered(lines, rx))
                    break
            except Exception:
                continue

        for phrase in trigger_phrases:
            if phrase and phrase.lower() in text.lower():
                signals.append(f"trigger_phrase:{phrase}")
                confidence = max(confidence, float(conf_rules.get("trigger_phrase") or 0.0))
                frag = self._after_phrase(text, phrase)
                list_items.extend(self._split_frag(frag, delimiters))

        # Location-style enumerations often appear as "X (70 cases), Y (3 cases), ...".
        # If configured, extract the *sentence containing* the first "(N cases)" match and split it.
        # Prefer capture regex when provided (more precise: returns only entity strings).
        if paren_cases_capture_regex:
            try:
                rx2 = re.compile(str(paren_cases_capture_regex), flags=re.IGNORECASE)
                caps = [c.strip() for c in rx2.findall(text) if str(c).strip()]
            except Exception:
                caps = []
            if caps:
                signals.append("paren_cases_capture")
                confidence = max(confidence, float(conf_rules.get("paren_cases") or 0.0))
                list_items.extend(caps)
        if paren_cases_regex:
            try:
                rx = re.compile(str(paren_cases_regex), flags=re.IGNORECASE)
                m = rx.search(text)
            except Exception:
                m = None
            if m is not None:
                sent = self._sentence_containing(text, m.start())
                if sent:
                    signals.append("paren_cases")
                    confidence = max(confidence, float(conf_rules.get("paren_cases") or 0.0))
                    list_items.extend(self._split_frag(sent, delimiters))

        list_items = list(dict.fromkeys([x for x in list_items if x]))
        return {
            "list_features": {
                "is_list_like": bool(signals),
                "list_items": list_items,
                "signals": signals,
                "confidence": float(confidence),
            }
        }

    def _load_rules(self, semantic_type: str) -> Dict[str, Any]:
        base = Path(self.rule_dir)
        generic = _load_yaml(base / "generic.yaml")
        specific = _load_yaml(base / f"{semantic_type}.yaml")
        return self._merge_rules(generic, specific)

    @staticmethod
    def _merge_rules(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a or {})
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                merged = dict(out.get(k) or {})
                merged.update(v)
                out[k] = merged
            elif isinstance(v, list) and isinstance(out.get(k), list):
                out[k] = list(out.get(k) or []) + list(v or [])
            else:
                out[k] = v
        return out

    @staticmethod
    def _infer_semantic_type(block: Dict[str, Any]) -> str:
        meta = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        pat = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
        slots = pat.get("schema_slots") if isinstance(pat.get("schema_slots"), list) else []
        slots_low = [str(s).lower() for s in slots if str(s).strip()]
        text = str(block.get("text") or "")
        text_low = text.lower()
        if any(
            ("geographic" in s)
            or ("distribution" in s)
            or ("region" in s)
            or ("location" in s)
            or ("area" in s)
            or ("epidemiolog" in s)
            for s in slots_low
        ):
            return "location"
        # Heuristic fallback (semantic_type-level, not topic-level):
        # Detect epidemiology-style enumerations even when schema_slots are missing.
        if re.search(r"\(\s*\d+\s+cases?\s*\)", text_low):
            return "location"
        if any(kw in text_low for kw in [" were reported in ", " was reported in ", "reported in ", " distributed in ", "occurred in "]):
            return "location"
        if any("clinical" in s or "symptom" in s for s in slots_low):
            return "symptom"
        if any("treatment" in s or "drug" in s for s in slots_low):
            return "drug"
        return "generic"

    @staticmethod
    def _extract_bullets(lines: List[str], bullets: List[str]) -> List[str]:
        out: List[str] = []
        for ln in lines:
            if ln.startswith(tuple(bullets)):
                out.append(ln.lstrip("".join(bullets)).strip())
        return out

    @staticmethod
    def _extract_numbered(lines: List[str], rx: str) -> List[str]:
        out: List[str] = []
        for ln in lines:
            if re.match(rx, ln):
                out.append(re.sub(rx, "", ln).strip())
        return out

    @staticmethod
    def _after_phrase(text: str, phrase: str) -> str:
        idx = text.lower().find(phrase.lower())
        if idx < 0:
            return ""
        frag = text[idx + len(phrase) :].strip()
        # Stop at the first sentence boundary to avoid pulling unrelated trailing clauses.
        for sep in [".", ";", "\n", "。", "；", "!", "！", "?", "？"]:
            j = frag.find(sep)
            if j > 0:
                frag = frag[:j].strip()
                break
        return frag

    @staticmethod
    def _sentence_containing(text: str, idx: int) -> str:
        """
        Return a rough sentence/segment containing character position idx.
        This is a lightweight heuristic for extracting one enumerative sentence.
        """
        if not text:
            return ""
        n = len(text)
        i = max(0, min(int(idx), n - 1))
        # Sentence boundary chars (both EN/ZH); also treat newlines as hard breaks.
        seps = set([".", ";", "\n", "。", "；", "!", "！", "?", "？"])
        l = i
        while l > 0 and text[l - 1] not in seps:
            l -= 1
        r = i
        while r < n and text[r] not in seps:
            r += 1
        frag = text[l:r].strip()
        return frag

    @staticmethod
    def _split_frag(frag: str, delimiters: List[str]) -> List[str]:
        if not frag:
            return []
        parts = [frag]
        for d in delimiters or []:
            new_parts: List[str] = []
            for p in parts:
                if d in p:
                    new_parts.extend([s.strip() for s in p.split(d) if s.strip()])
                else:
                    new_parts.append(p)
            parts = new_parts
        return parts
