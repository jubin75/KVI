"""
SchemaValueCleaner: data-driven normalization/split/filter for slot values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML is required for SchemaValueCleaner rules.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dedup(values: List[str]) -> List[str]:
    return list(dict.fromkeys([v for v in values if v]))


def _apply_regex_subs(value: str, subs: List[Dict[str, Any]]) -> str:
    """
    Apply regex substitutions described by:
      - pattern: regex pattern
      - repl: replacement string (default: "")
      - flags: optional list of strings (e.g., ["IGNORECASE"])
    This is intentionally generic and purely rule-driven (no per-slot logic).
    """
    import re

    out = str(value)
    for it in subs or []:
        if not isinstance(it, dict):
            continue
        pat = str(it.get("pattern") or "").strip()
        if not pat:
            continue
        repl = str(it.get("repl") or "")
        flags_list = it.get("flags") if isinstance(it.get("flags"), list) else []
        flags = 0
        for f in flags_list:
            f2 = str(f or "").upper()
            if f2 == "IGNORECASE":
                flags |= re.IGNORECASE
            elif f2 == "MULTILINE":
                flags |= re.MULTILINE
            elif f2 == "DOTALL":
                flags |= re.DOTALL
        try:
            out = re.sub(pat, repl, out, flags=flags)
        except Exception:
            continue
    return out


def _looks_like_location(value: str) -> bool:
    """
    Conservative semantic_type=location guardrail.

    Purpose: prevent non-location sentence fragments (methods/statistics/biology/tails) from
    entering LIST_ONLY projection. This is deletion-only (no inference).
    """
    import re

    v = str(value or "").strip()
    if not v:
        return False
    if len(v) > 64:
        return False

    vlow = v.lower()

    # Obvious non-entities / discourse / section markers.
    bad_exact = {
        "however",
        "moreover",
        "in addition",
        "since currently",
        "method",
        "methods",
        "results",
        "discussion",
        "abstract",
        "keywords",
        "introduction",
        "conclusion",
        "figure",
        "fig",
        "table",
        "supplementary",
        "copyright",
        "p.r.",  # common split artifact
    }
    if vlow in bad_exact:
        return False

    # Drop anything that still looks like a clause/sentence fragment.
    if any(ch in v for ch in [".", ":", ";", "?", "!", "／", "/", "\\", "http", "www"]):
        return False
    if re.search(r"\d", v):
        return False

    # Drop fragments starting with conjunctions/prepositions.
    if re.match(r"^(and|or|but|while|with|without|including|such as)\b", vlow):
        return False

    # Verb-heavy / analytic phrases are not locations.
    verb_noise = [
        " is ",
        " are ",
        " was ",
        " were ",
        " be ",
        " been ",
        " being ",
        " have ",
        " has ",
        " had ",
        " show",
        " indicate",
        " suggest",
        " report",
        " distribute",
        " occur",
        " detect",
        " classify",
        " carry",
        " transmit",
        " analyze",
        " investigation",
        " surveillance",
        " prevalence",
        " genotype",
        " virus",
        " tick",
        " antibody",
        " clinical",
        " symptom",
    ]
    if any(w in vlow for w in verb_noise):
        return False

    # If it contains CJK chars, accept if it looks like a place token.
    if re.search(r"[\u4e00-\u9fff]", v):
        # Common Chinese place suffixes.
        if re.search(r"(省|市|县|区|州|盟|旗|镇|乡|村|岛|山|湖|河)$", v):
            return True
        # Otherwise require it to be short (avoid long CJK sentences).
        return len(v) <= 12

    # For Latin script, require at least one uppercase letter (reject "longicornis").
    if not re.search(r"[A-Z]", v):
        return False

    # Accept common English location suffixes.
    if re.search(
        r"\b(province|city|county|district|state|region|prefecture|municipality|island|islands|mountain|mountains)\b",
        vlow,
    ):
        return True

    # Accept compact proper-name patterns: 1-4 capitalized words (e.g., "South Korea", "Cangzhou").
    if re.fullmatch(r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3}", v.strip()):
        return True

    return False


def _looks_like_symptom(value: str) -> bool:
    """
    Conservative semantic_type=symptom guardrail (deletion-only).

    We do NOT use any external medical dictionary. We only filter out obvious non-symptom
    fragments (locations, methods, taxonomy, discourse tails) and keep short symptom-like tokens.
    """
    import re

    v = str(value or "").strip()
    if not v:
        return False
    if len(v) > 80:
        return False

    vlow = v.lower()

    # Obvious non-symptom artifacts / discourse.
    if any(x in vlow for x in ["http", "www", "doi", "figure", "table", "supplementary", "copyright"]):
        return False
    if re.search(r"\d", v) and "covid" not in vlow:
        # most symptom tokens shouldn't carry digits (avoid years, case counts, p-values)
        return False
    if any(ch in v for ch in [":", ";"]):
        return False
    if re.search(r"\b(p\s*<|p\s*=|ci\b|confidence interval)\b", vlow):
        return False

    # Strong location cues -> not a symptom.
    if re.search(r"(reported in|distributed in|found in|occurred in)\b", vlow):
        return False
    if re.search(r"(地区|区域|地域|省|市|我国|中国|国内|分布|流行)", v):
        return False
    if re.search(r"\b(province|city|county|district|state|region|prefecture)\b", vlow):
        return False

    # Taxonomy / biology / methods noise.
    if re.search(r"\b(genotype|lineage|clade|family|phenuiviridae|bunyaviridae|nairoviridae)\b", vlow):
        return False
    if re.search(r"\b(pcr|elisa|assay|kit|diagnos|specimen|samples?)\b", vlow):
        return False

    # Keep common symptom morphology / minimal cue list (very small).
    symptom_cues = [
        "fever",
        "thrombocytopenia",
        "leukocytopenia",
        "bleeding",
        "hemorrhag",
        "vomit",
        "diarrhea",
        "fatigue",
        "headache",
        "nausea",
        "rash",
        "pain",
        "myalgia",
        "cough",
        "dyspnea",
        "neurolog",
        "gastro",
        "thrombo",
        "leuko",
    ]
    if any(c in vlow for c in symptom_cues):
        return True

    # Generic symptom suffix patterns (still conservative).
    if re.search(r"(itis|emia|osis|algia|uria|pathy|pnea|rrhea)$", vlow):
        return True

    # If it's a short phrase without verbs/punctuation, accept.
    if len(v.split()) <= 4 and not re.search(r"\b(is|are|was|were|be|been|being|have|has|had)\b", vlow):
        return True

    return False


def _looks_like_drug(value: str) -> bool:
    """
    Conservative semantic_type=drug guardrail (deletion-only).

    Keep plausible drug names; drop locations/symptoms/sentence fragments.
    """
    import re

    v = str(value or "").strip()
    if not v:
        return False
    if len(v) > 64:
        return False

    vlow = v.lower()
    if any(x in vlow for x in ["http", "www", "doi", "figure", "table", "supplementary"]):
        return False
    if any(ch in v for ch in [":", ";", "。", "，"]):
        return False
    if re.search(r"\b(reported in|distributed in|province|city|county|region)\b", vlow):
        return False
    if re.search(r"(地区|省|市|我国|中国|分布)", v):
        return False
    # Avoid symptom words in drug slot.
    if any(x in vlow for x in ["fever", "symptom", "thrombocytopenia", "bleeding", "diarrhea", "vomit"]):
        return False
    # Avoid generic therapy phrases.
    if re.search(r"\b(treatment|therapy|therapeutic|approved|approval|fda)\b", vlow):
        return False

    # Discourse / connector tokens that often appear after splitting, not drugs.
    stop = {
        "therefore",
        "however",
        "currently",
        "among",
        "of",
        "them",
        "including",
        "and",
        "or",
        "in",
        "this",
        "study",
        "the",
        "specimens",
        "oral",
        "sera",
    }
    if vlow in stop:
        return False

    # Multi-token phrases are almost never drug names; treat them as noise (deletion-only).
    if " " in v.strip():
        return False

    # Avoid lab/clinical abnormalities being misclassified as drugs.
    if re.search(r"(penia|cytopenia)$", vlow):
        return False

    # IFN-α / IFN-alpha etc. (very common therapeutic reference).
    if re.fullmatch(r"(ifn|interferon)[\-\s]?(alpha|beta|gamma|λ|lambda)?", vlow):
        return True
    if re.fullmatch(r"ifn[\-\s]?(α|β|γ|lambda|λ)", vlow):
        return True

    # Allow single-token drug-like strings (letters/digits/hyphen/plus/slash) with drug morphology.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\+/]*", v):
        # Strong suffix cues.
        if re.search(
            r"(vir|mab|nib|ciclovir|statin|mycin|pril|sartan|azole|caine|prazole|dine|dronate)$",
            vlow,
        ):
            return True
        # Small allowlist for common antivirals/therapeutics mentioned in papers (not topic-specific).
        allow = {
            "ribavirin",
            "favipiravir",
            "remdesivir",
            "oseltamivir",
            "baloxavir",
            "acyclovir",
        }
        if vlow in allow:
            return True
        return False

    return False


@dataclass
class SchemaValueCleaner:
    rule_dir: str

    def __init__(self, rule_dir: str) -> None:
        self.rule_dir = str(rule_dir)

    def clean(
        self,
        values: List[str],
        semantic_type: str,
        evidence_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        rules = self._load_rules(semantic_type=semantic_type)
        raw_values = [str(v).strip() for v in (values or []) if str(v).strip()]
        normalization_map: Dict[str, str] = {}
        split_map: Dict[str, List[str]] = {}
        removed_values: List[str] = []

        normalized = [self._normalize_value(v, rules, normalization_map) for v in raw_values]
        split_values = self._split_values(normalized, rules, split_map)
        cleaned = self._filter_values(split_values, rules, removed_values)
        # Semantic-type level guardrails (deletion-only).
        if str(semantic_type or "").strip().lower() == "location":
            kept: List[str] = []
            for v in cleaned:
                if _looks_like_location(v):
                    kept.append(v)
                else:
                    removed_values.append(v)
            cleaned = kept
        if str(semantic_type or "").strip().lower() == "symptom":
            kept = []
            for v in cleaned:
                if _looks_like_symptom(v):
                    kept.append(v)
                else:
                    removed_values.append(v)
            cleaned = kept
        if str(semantic_type or "").strip().lower() == "drug":
            kept = []
            for v in cleaned:
                if _looks_like_drug(v):
                    kept.append(v)
                else:
                    removed_values.append(v)
            cleaned = kept
        cleaned = _dedup(cleaned)

        return {
            "cleaned_values": cleaned,
            "removed_values": removed_values,
            "normalization_map": normalization_map,
            "split_map": split_map,
            "debug": {
                "semantic_type": semantic_type,
                "evidence_ids": list(evidence_ids or []),
            },
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
    def _normalize_value(
        value: str,
        rules: Dict[str, Any],
        normalization_map: Dict[str, str],
    ) -> str:
        norm_rules = rules.get("normalize") if isinstance(rules.get("normalize"), dict) else {}
        # Optional regex-based normalization (data-driven):
        # - normalize.regex_subs: list[{pattern,repl,flags}]
        # - normalize.strip_prefixes: list[regex]
        # - normalize.strip_suffixes: list[regex]
        # These help remove sentence-fragments/tails for LIST_ONLY projection without hardcoding slot logic.
        regex_subs = norm_rules.get("regex_subs") if isinstance(norm_rules.get("regex_subs"), list) else []
        strip_prefixes = norm_rules.get("strip_prefixes") if isinstance(norm_rules.get("strip_prefixes"), list) else []
        strip_suffixes = norm_rules.get("strip_suffixes") if isinstance(norm_rules.get("strip_suffixes"), list) else []

        v0 = str(value)
        if regex_subs:
            v0 = _apply_regex_subs(v0, regex_subs)
        if strip_prefixes:
            prefix_subs = [{"pattern": str(p), "repl": "", "flags": ["IGNORECASE"]} for p in strip_prefixes if str(p).strip()]
            v0 = _apply_regex_subs(v0, prefix_subs)
        if strip_suffixes:
            suffix_subs = [{"pattern": str(p), "repl": "", "flags": ["IGNORECASE"]} for p in strip_suffixes if str(p).strip()]
            v0 = _apply_regex_subs(v0, suffix_subs)
        v0 = v0.strip()

        # Exact-match normalization map (canonical/variants)
        vlow = v0.lower()
        for canonical, variants in norm_rules.items():
            # Skip meta keys used for regex normalization.
            if str(canonical) in {"regex_subs", "strip_prefixes", "strip_suffixes"}:
                continue
            cand = [str(canonical)] + [str(x) for x in (variants or [])]
            for it in cand:
                if vlow == str(it).lower():
                    normalization_map[value] = str(canonical)
                    return str(canonical)
        if v0 != value:
            normalization_map[value] = v0
        return v0

    @staticmethod
    def _split_values(
        values: List[str],
        rules: Dict[str, Any],
        split_map: Dict[str, List[str]],
    ) -> List[str]:
        split_rules = rules.get("split") if isinstance(rules.get("split"), dict) else {}
        connectors = split_rules.get("connectors") if isinstance(split_rules.get("connectors"), list) else []
        if not connectors:
            return values
        out: List[str] = []
        for v in values:
            parts = [v]
            for c in connectors:
                new_parts: List[str] = []
                for p in parts:
                    if c in p:
                        new_parts.extend([s.strip() for s in p.split(c) if s.strip()])
                    else:
                        new_parts.append(p)
                parts = new_parts
            if len(parts) > 1:
                split_map[v] = parts
            out.extend(parts)
        return out

    @staticmethod
    def _filter_values(
        values: List[str],
        rules: Dict[str, Any],
        removed_values: List[str],
    ) -> List[str]:
        flt = rules.get("filter") if isinstance(rules.get("filter"), dict) else {}
        deny_terms = [str(x).lower() for x in (flt.get("deny_terms") or [])]
        out: List[str] = []
        for v in values:
            vlow = v.lower()
            if any(t and t in vlow for t in deny_terms):
                removed_values.append(v)
                continue
            out.append(v)
        return out
