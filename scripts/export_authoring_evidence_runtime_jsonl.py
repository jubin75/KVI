"""
Export Authoring EvidenceUnit JSONL to runtime-safe approved-only JSONL.

This enforces the boundary:
- draft/rejected/reviewed stay in Authoring DB
- only approved records are exported for KVBank building/runtime usage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root, repo_root.parent]
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_repo_root_on_syspath()

try:
    from external_kv_injection.src.authoring.jsonl_store import (  # type: ignore
        iter_approved_runtime_records,
        read_evidence_units_jsonl,
        write_runtime_records_jsonl,
    )
except ModuleNotFoundError:
    from src.authoring.jsonl_store import iter_approved_runtime_records, read_evidence_units_jsonl, write_runtime_records_jsonl  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description="Export approved EvidenceUnits to runtime JSONL.")
    p.add_argument("--authoring_jsonl", required=True, help="Authoring evidence_units.jsonl (contains status).")
    p.add_argument("--out", required=True, help="Output runtime evidence jsonl (approved-only).")
    args = p.parse_args()

    units, stats = read_evidence_units_jsonl(Path(str(args.authoring_jsonl)))
    records = list(iter_approved_runtime_records(units))
    out_stats = write_runtime_records_jsonl(Path(str(args.out)), records)

    print(
        json.dumps(
            {
                "load": stats.__dict__,
                "approved_exported": int(len(records)),
                "out": out_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

