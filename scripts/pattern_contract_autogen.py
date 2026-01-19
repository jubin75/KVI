"""
Minimal, offline pattern contract auto-generator for KVI 2.0.

Purpose:
- Scan blocks.enriched.jsonl
- Collect high-frequency symbolic patterns
- Emit a draft pattern_contract.json for human review

This script intentionally DOES NOT:
- assign final hard/soft levels
- generate slot instantiation rules
- interpret prompt semantics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _get_pattern_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    # blocks.jsonl may store metadata at rec["metadata"] or rec["meta"]["metadata"].
    if isinstance(rec.get("metadata"), dict):
        meta = rec["metadata"]
    elif isinstance(rec.get("meta"), dict) and isinstance(rec["meta"].get("metadata"), dict):
        meta = rec["meta"]["metadata"]
    else:
        meta = {}
    pat = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
    return pat or {}


def _infer_topic_from_path(path: Path) -> Optional[str]:
    # Try /topics/{topic}/work/blocks.enriched.jsonl
    parts = [p for p in path.parts if p]
    if "topics" in parts:
        idx = parts.index("topics")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Try parent/work
    if path.parent.name == "work":
        return path.parent.parent.name or None
    return None


def _build_patterns(
    *,
    abbr_counts: Counter[str],
    slot_counts: Counter[str],
    min_abbr_count: int,
    min_slot_count: int,
    max_abbr: int,
    max_slots: int,
) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []

    for abbr, cnt in abbr_counts.most_common():
        if cnt < min_abbr_count:
            continue
        patterns.append(
            {
                "pattern_id": f"abbr:{abbr}",
                "question_skeleton": {
                    "intent": "ask_definition",
                    "surface_forms": [
                        "X 是什么",
                        "X 的全称是什么",
                        "What is X",
                        "What does X stand for",
                    ],
                },
                "slots": {
                    "abbr": {
                        "type": "string",
                        "required": True,
                        "evidence_type": ["abbreviation"],
                        "min_evidence": 1,
                        "inference_level": "hard",
                    },
                    "full_name": {
                        "type": "string",
                        "required": False,
                        "evidence_type": ["definition", "abbreviation"],
                        "min_evidence": 1,
                        "inference_level": "soft",
                    },
                },
                "answer_style": "factual",
                "count": int(cnt),
            }
        )
        if max_abbr and len(patterns) >= max_abbr:
            break

    slot_added = 0
    for slot, cnt in slot_counts.most_common():
        if cnt < min_slot_count:
            continue
        patterns.append(
            {
                "pattern_id": f"schema:{slot}",
                "question_skeleton": {
                    "intent": "ask_schema_list",
                    "surface_forms": [f"X 的{slot}是什么", f"X 有哪些{slot}"],
                },
                "slots": {
                    slot: {
                        "type": "string",
                        "required": False,
                        "evidence_type": ["schema"],
                        "min_evidence": 1,
                        "inference_level": "schema",
                    }
                },
                "answer_style": "factual",
                "count": int(cnt),
            }
        )
        slot_added += 1
        if max_slots and slot_added >= max_slots:
            break

    return patterns


def _collect_counts(recs: Iterable[Dict[str, Any]]) -> Tuple[Counter[str], Counter[str], int]:
    abbr_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    total = 0
    for rec in recs:
        total += 1
        pat = _get_pattern_meta(rec)
        abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
        for ap in abbr_pairs:
            if not isinstance(ap, dict):
                continue
            abbr = str(ap.get("abbr") or "").strip().upper()
            if abbr:
                abbr_counts[abbr] += 1
        slots = pat.get("schema_slots") if isinstance(pat.get("schema_slots"), list) else []
        for s in slots:
            s2 = str(s or "").strip()
            if s2:
                slot_counts[s2] += 1
    return abbr_counts, slot_counts, total


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-generate draft pattern_contract.json from blocks.enriched.jsonl")
    p.add_argument("--blocks_jsonl_in", required=True, help="Path to blocks.enriched.jsonl")
    p.add_argument("--out", required=True, help="Output pattern_contract.json path")
    p.add_argument("--topic", default="", help="Topic name (optional; inferred from path if empty)")
    p.add_argument("--min_abbr_count", type=int, default=2)
    p.add_argument("--min_slot_count", type=int, default=2)
    p.add_argument("--max_abbr", type=int, default=50)
    p.add_argument("--max_slots", type=int, default=50)
    args = p.parse_args()

    in_path = Path(str(args.blocks_jsonl_in))
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    topic = str(args.topic or "").strip() or _infer_topic_from_path(in_path) or ""

    abbr_counts, slot_counts, total_blocks = _collect_counts(_read_jsonl(in_path))
    patterns = _build_patterns(
        abbr_counts=abbr_counts,
        slot_counts=slot_counts,
        min_abbr_count=int(args.min_abbr_count),
        min_slot_count=int(args.min_slot_count),
        max_abbr=int(args.max_abbr),
        max_slots=int(args.max_slots),
    )

    payload = {
        "topic": topic,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": str(in_path),
        "patterns": patterns,
        "stats": {
            "total_blocks": int(total_blocks),
            "unique_abbr": int(len(abbr_counts)),
            "unique_slots": int(len(slot_counts)),
        },
    }

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
