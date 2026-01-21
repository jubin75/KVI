"""
Evidence enhancement: semantic_role + structural_features + injectability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvidenceEnhancer:
    def build_units(
        self,
        *,
        block_id: str,
        text: str,
        list_features: Dict[str, Any],
        entities: List[str],
    ) -> List[Dict[str, Any]]:
        block_id = str(block_id or "").strip()
        text = str(text or "")
        list_items = self._list_items(list_features)
        structural = self._structural_features(list_features, list_items)

        units: List[Dict[str, Any]] = []
        if list_items:
            for idx, item in enumerate(list_items):
                unit_text = str(item or "").strip()
                if not unit_text:
                    continue
                role = self._semantic_role(unit_text, structural)
                injectability = self._injectability(role, unit_text, structural, entities)
                units.append(
                    {
                        "unit_id": f"{block_id}#u{idx}",
                        "text": unit_text,
                        "semantic_role": role,
                        "structural_features": dict(structural),
                        "injectability": injectability,
                    }
                )
        else:
            role = self._semantic_role(text, structural)
            injectability = self._injectability(role, text, structural, entities)
            units.append(
                {
                    "unit_id": f"{block_id}#u0",
                    "text": text,
                    "semantic_role": role,
                    "structural_features": dict(structural),
                    "injectability": injectability,
                }
            )
        return units

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
    def _structural_features(list_features: Dict[str, Any], list_items: List[str]) -> Dict[str, Any]:
        signals = list_features.get("signals") if isinstance(list_features, dict) else []
        signals = signals if isinstance(signals, list) else []
        list_style = "inline"
        if any("bullet" in str(s) for s in signals):
            list_style = "bullet"
        elif any("numbering" in str(s) for s in signals):
            list_style = "numbered"
        has_list = bool(list_features.get("is_list_like")) or bool(list_items) or bool(signals)
        return {
            "has_list_structure": bool(has_list),
            "list_style": list_style if has_list else "",
            "list_item_count": int(len(list_items)),
            "sentence_parallelism": bool(len(list_items) >= 2),
        }

    @staticmethod
    def _semantic_role(text: str, structural: Dict[str, Any]) -> str:
        t = str(text or "")
        tl = t.lower()
        if any(k in tl for k in ("supplementary", "table", "figure", "copyright", "license")):
            return "metadata_notice"
        if any(
            k in tl
            for k in (
                "methods",
                "materials and methods",
                "statistical analysis",
                "we performed",
                "we used",
                "p value",
                "confidence interval",
            )
        ):
            return "procedural_note"
        if bool(structural.get("has_list_structure")) or any(
            k in tl for k in ("including", "such as", "present with", "characterized by")
        ):
            return "enumerative_fact"
        if any(k in tl for k in ("associated with", "correlated with", "linked to", "risk factor", "caused by")):
            return "relational_statement"
        if any(k in tl for k in ("background", "introduction", "in this study", "we investigated")):
            return "contextual_background"
        return "descriptive_statement"

    @staticmethod
    def _injectability(
        semantic_role: str,
        text: str,
        structural: Dict[str, Any],
        entities: List[str],
    ) -> Dict[str, Any]:
        signals: List[str] = []
        blocking: List[str] = []
        score = 0.0

        if semantic_role == "enumerative_fact":
            signals.append("semantic_role_enumerative")
            score += 0.4
        if int(structural.get("list_item_count") or 0) >= 2:
            signals.append("list_structure")
            score += 0.2
        if entities:
            signals.append("explicit_entity_anchor")
            score += 0.2
        if len(str(text or "")) <= 320:
            signals.append("low_context_dependency")
            score += 0.1

        if semantic_role == "metadata_notice":
            blocking.append("metadata_only")
        elif semantic_role == "procedural_note":
            blocking.append("procedural_only")

        if blocking:
            score = 0.0
        return {"score": float(min(score, 1.0)), "signals": signals, "blocking_reasons": blocking}
