"""
Evidence Sidecar Builder (minimal, offline).

Input: blocks.enriched.jsonl
Output: evidence.sidecar.jsonl

Rules are defined in docs/085_问题修复.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def _get_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rec.get("metadata"), dict):
        return rec["metadata"]
    if isinstance(rec.get("meta"), dict):
        meta = rec["meta"]
        if isinstance(meta.get("metadata"), dict):
            return meta["metadata"]
        return meta
    return {}


def _get_pattern_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    pat = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
    return pat or {}


def _normalize_text(text: str, max_len: int = 320) -> str:
    t = " ".join(str(text or "").strip().split())
    if len(t) > max_len:
        t = t[: max_len - 3].rstrip() + "..."
    return t


def _primary_entity(
    abbr_full: Optional[str],
    schema_primary: Optional[str],
    entities: List[str],
) -> str:
    if abbr_full:
        return abbr_full
    if schema_primary:
        return schema_primary
    if entities:
        return entities[0]
    return ""


def _emit(
    items: List[Dict[str, Any]],
    seen: Set[Tuple[str, str, str, str]],
    *,
    evidence_id: str,
    ev_type: str,
    subject: str,
    predicate: str,
    obj: str,
    source_block_id: str,
    confidence: float,
) -> None:
    key = (ev_type, subject, predicate, obj)
    if key in seen:
        return
    seen.add(key)
    items.append(
        {
            "evidence_id": evidence_id,
            "type": ev_type,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source_block_id": source_block_id,
            "confidence": float(confidence),
            "provenance": "blocks.enriched",
        }
    )


def _extract_evidence(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = _get_meta(rec)
    pat = _get_pattern_meta(meta)
    block_id = str(rec.get("block_id") or meta.get("block_id") or meta.get("id") or "")
    text = str(rec.get("text") or "")
    block_type = str(meta.get("block_type") or "").strip().lower()

    abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
    entities = pat.get("entities") if isinstance(pat.get("entities"), list) else []
    entities = [str(e) for e in entities if str(e).strip()]

    schema_slots = pat.get("schema_slots")
    schema_primary = None
    definition_like = False
    slot_fills: Dict[str, Any] = {}
    if isinstance(schema_slots, dict):
        schema_primary = str(schema_slots.get("primary_entity") or "").strip() or None
        definition_like = bool(schema_slots.get("definition_like") is True)
        for k, v in schema_slots.items():
            if k in {"primary_entity", "definition_like"}:
                continue
            if isinstance(v, (str, int, float)) and str(v).strip():
                slot_fills[str(k)] = str(v)
            elif isinstance(v, list) and v:
                slot_fills[str(k)] = ",".join(str(x) for x in v if str(x).strip())
    elif isinstance(schema_slots, list):
        # Only slot names; no values -> no slot_fill items
        schema_slots = [str(s) for s in schema_slots if str(s).strip()]

    items: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    # R1 Abbreviation
    for ap in abbr_pairs:
        if not isinstance(ap, dict):
            continue
        abbr = str(ap.get("abbr") or "").strip()
        full = str(ap.get("full") or "").strip()
        if not abbr or not full:
            continue
        _emit(
            items,
            seen,
            evidence_id=f"{block_id}::abbr::{abbr}",
            ev_type="abbreviation",
            subject=abbr,
            predicate="stands_for",
            obj=full,
            source_block_id=block_id,
            confidence=0.9,
        )

    # R2 Definition
    if block_type == "definition" or bool(definition_like):
        abbr_full = None
        if abbr_pairs:
            for ap in abbr_pairs:
                if isinstance(ap, dict) and str(ap.get("full") or "").strip():
                    abbr_full = str(ap.get("full") or "").strip()
                    break
        subject = _primary_entity(abbr_full, schema_primary, entities)
        if subject:
            _emit(
                items,
                seen,
                evidence_id=f"{block_id}::def::{subject}",
                ev_type="definition",
                subject=subject,
                predicate="is_defined_as",
                obj=_normalize_text(text),
                source_block_id=block_id,
                confidence=0.7,
            )

    # R3 Slot Fill
    if slot_fills:
        abbr_full = None
        if abbr_pairs:
            for ap in abbr_pairs:
                if isinstance(ap, dict) and str(ap.get("full") or "").strip():
                    abbr_full = str(ap.get("full") or "").strip()
                    break
        subject = _primary_entity(abbr_full, schema_primary, entities)
        for slot_name, slot_val in slot_fills.items():
            if not subject or not slot_val:
                continue
            _emit(
                items,
                seen,
                evidence_id=f"{block_id}::slot::{slot_name}",
                ev_type="slot_fill",
                subject=subject,
                predicate=f"has_{slot_name}",
                obj=str(slot_val),
                source_block_id=block_id,
                confidence=0.6,
            )

    return items


def main() -> None:
    p = argparse.ArgumentParser(description="Build evidence.sidecar.jsonl from blocks.enriched.jsonl")
    p.add_argument("--blocks_jsonl", required=True, help="Path to blocks.enriched.jsonl")
    p.add_argument("--out", required=True, help="Output evidence.sidecar.jsonl")
    args = p.parse_args()

    in_path = Path(str(args.blocks_jsonl))
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    seen_global: Set[Tuple[str, str, str, str]] = set()
    for rec in _read_jsonl(in_path):
        for it in _extract_evidence(rec):
            key = (it["type"], it["subject"], it["predicate"], it["object"])
            if key in seen_global:
                continue
            seen_global.add(key)
            items.append(it)

    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
