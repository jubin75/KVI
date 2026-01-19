"""
Sidecar: slot coverage aggregation (no inference, no semantics).
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_slot_coverage(
    semantic_instances: list,
    retrieved_block_ids: list,
    block_facets: dict,
) -> dict:
    """
    Returns:
    {
      slot_name: {
        "evidence_count": int,
        "covered": bool
      }
    }
    """
    slots: List[str] = []
    for inst in semantic_instances or []:
        if isinstance(inst, dict) and isinstance(inst.get("slots"), dict):
            slots.extend(list(inst["slots"].keys()))
    slots = list(dict.fromkeys([str(s) for s in slots if str(s).strip()]))

    out: Dict[str, Dict[str, Any]] = {s: {"evidence_count": 0, "covered": False} for s in slots}
    for bid in retrieved_block_ids or []:
        facets = block_facets.get(bid) if isinstance(block_facets, dict) else None
        if not isinstance(facets, dict):
            continue
        for s in slots:
            if s in facets:
                out[s]["evidence_count"] += 1
                out[s]["covered"] = True
                continue
            # Also allow list-valued facets to mark coverage by slot name.
            for v in facets.values():
                if isinstance(v, list) and s in v:
                    out[s]["evidence_count"] += 1
                    out[s]["covered"] = True
                    break
    return out
