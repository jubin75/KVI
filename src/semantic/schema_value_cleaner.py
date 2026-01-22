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
