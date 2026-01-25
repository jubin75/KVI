from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .models import EvidenceUnit, RuntimeEvidenceRecord, to_runtime_record


@dataclass(frozen=True)
class AuthoringLoadStats:
    read_lines: int
    parsed: int
    skipped: int
    errors: int


def read_evidence_units_jsonl(path: Path) -> Tuple[List[EvidenceUnit], AuthoringLoadStats]:
    units: List[EvidenceUnit] = []
    read_lines = 0
    parsed = 0
    skipped = 0
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            read_lines += 1
            s = line.strip()
            if not s:
                skipped += 1
                continue
            try:
                obj = json.loads(s)
            except Exception:
                errors += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            try:
                eu = EvidenceUnit.from_dict(obj)
            except Exception:
                errors += 1
                continue
            if not str(eu.evidence_id or "").strip():
                # allow authoring drafts but skip malformed ids in loader output
                skipped += 1
                continue
            parsed += 1
            units.append(eu)
    return units, AuthoringLoadStats(read_lines=read_lines, parsed=parsed, skipped=skipped, errors=errors)


def iter_approved_runtime_records(
    units: Iterable[EvidenceUnit],
    *,
    allowed_injection: bool = True,
    list_only_allowed: bool = True,
) -> Iterator[RuntimeEvidenceRecord]:
    for eu in units:
        if not isinstance(eu, EvidenceUnit):
            continue
        if not eu.is_approved():
            continue
        txt = eu.normalized_text()
        if not txt:
            continue
        yield to_runtime_record(eu, allowed_injection=allowed_injection, list_only_allowed=list_only_allowed)


def write_runtime_records_jsonl(path: Path, records: Iterable[RuntimeEvidenceRecord]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if not isinstance(r, RuntimeEvidenceRecord):
                continue
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return {"written": int(written), "out": str(path)}

