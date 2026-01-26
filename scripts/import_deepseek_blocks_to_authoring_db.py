"""
Import DeepSeek extractive evidence blocks into Authoring DB as draft EvidenceUnits.

This keeps DeepSeek capability, but enforces the boundary:
- DeepSeek output is *suggestions* (draft)
- runtime KVBank must be built from *approved* Authoring EvidenceUnits only
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
    from external_kv_injection.src.authoring.importers import (  # type: ignore
        import_deepseek_blocks_evidence_jsonl_to_authoring_db,
    )
except ModuleNotFoundError:
    from src.authoring.importers import import_deepseek_blocks_evidence_jsonl_to_authoring_db  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_evidence_jsonl", required=True, help="Path to blocks.evidence.jsonl (DeepSeek extractive output).")
    p.add_argument("--authoring_db_jsonl", required=True, help="Authoring DB evidence_units.jsonl (append).")
    p.add_argument("--schema_id", required=True, help="schema_id to assign to imported drafts (e.g. schema:infectious_disease).")
    p.add_argument("--default_semantic_type", default="generic")
    p.add_argument("--evidence_type", default="extractive_suggestion")
    args = p.parse_args()

    stats = import_deepseek_blocks_evidence_jsonl_to_authoring_db(
        blocks_evidence_jsonl=Path(str(args.blocks_evidence_jsonl)),
        authoring_db_jsonl=Path(str(args.authoring_db_jsonl)),
        schema_id=str(args.schema_id),
        default_semantic_type=str(args.default_semantic_type),
        evidence_type=str(args.evidence_type),
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

