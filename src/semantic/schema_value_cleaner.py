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
        vlow = value.lower()
        for canonical, variants in norm_rules.items():
            cand = [str(canonical)] + [str(x) for x in (variants or [])]
            for it in cand:
                if vlow == str(it).lower():
                    normalization_map[value] = str(canonical)
                    return str(canonical)
        return value

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
