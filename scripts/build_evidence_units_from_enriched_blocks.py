"""
Build evidence_units.jsonl from blocks.enriched.jsonl.

This is an offline, deterministic pipeline per docs/078_Evidence_extract.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root, repo_root.parent]
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_repo_root_on_syspath()

try:
    from external_kv_injection.src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore
except ModuleNotFoundError:
    from src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_enriched", required=True, help="blocks.enriched.jsonl")
    p.add_argument("--out", required=True, help="evidence_units.jsonl output")
    p.add_argument("--max_blocks", type=int, default=0)
    args = p.parse_args()

    extractor = EvidenceUnitExtractor()
    in_path = Path(str(args.blocks_enriched))
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    processed = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in _read_jsonl(in_path):
            processed += 1
            if int(args.max_blocks) > 0 and processed > int(args.max_blocks):
                break
            bid = str(rec.get("block_id") or rec.get("id") or rec.get("chunk_id") or "")
            text = str(rec.get("text") or "")
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            section_type = extractor.infer_section_type(text=text, metadata=meta)
            sentences = extractor.split_sentences(block_id=bid, text=text)
            pat = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
            list_features = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
            units = extractor.extract_units(
                block_id=bid,
                text=text,
                section_type=section_type,
                sentences=sentences,
                list_features=list_features,
            )
            for u in units:
                out.write(json.dumps(u, ensure_ascii=False) + "\n")
                written += 1

    print(json.dumps({"processed_blocks": processed, "written_units": written, "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

