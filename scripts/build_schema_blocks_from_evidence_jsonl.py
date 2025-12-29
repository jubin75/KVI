"""
Build schema blocks (`blocks.schema.jsonl`) from extractive evidence blocks (`blocks.evidence.jsonl`).

Design constraints (schema-first runtime):
- Schema is an intermediate constraint representation (NOT evidence quotes).
- Schema blocks will be *forwarded* through base LLM to obtain KV cache, then injected.
- Evidence/raw blocks are retrieval-only for grounding/citation/fallback; NEVER injected.

Implementation:
- Group evidence sentences by doc_id (default) and compile to a compact schema text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from external_kv_injection.scripts.rebuild_topic_kvbank_from_config import _approx_token_count, _safe_json_loads  # type: ignore
from external_kv_injection.src.runtime.struct_slots import build_schema_from_evidence_texts, schema_to_injection_text  # type: ignore


def build_schema_blocks_from_evidence_jsonl(
    *,
    blocks_evidence_jsonl: Path,
    out_jsonl: Path,
    group_by: str = "doc_id",
    max_docs: int = 0,
    max_evidence_per_doc: int = 200,
) -> Dict[str, Any]:
    if group_by not in {"doc_id"}:
        raise ValueError("Only group_by='doc_id' is supported in this demo.")

    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_in = 0
    with blocks_evidence_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _safe_json_loads(line)
            if not rec:
                continue
            total_in += 1
            doc_id = str(rec.get("doc_id") or "").strip()
            if not doc_id:
                continue
            if int(max_docs) > 0 and len(by_doc) >= int(max_docs) and doc_id not in by_doc:
                continue
            if len(by_doc[doc_id]) >= int(max_evidence_per_doc):
                continue
            by_doc[doc_id].append(rec)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_out = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for doc_id, recs in by_doc.items():
            ev_texts = [str(r.get("text") or "").strip() for r in recs if str(r.get("text") or "").strip()]
            if not ev_texts:
                continue
            schema = build_schema_from_evidence_texts(ev_texts)
            schema_text = schema_to_injection_text(schema, max_items_per_field=3).strip()
            if not schema_text:
                continue

            # Keep stable linkage for traceability.
            ev_block_ids = [str(r.get("block_id") or "") for r in recs if str(r.get("block_id") or "")]
            source_uri = recs[0].get("source_uri", None)
            lang = recs[0].get("lang", None)
            block_id = f"{doc_id}::schema"

            out_rec = {
                "block_id": block_id,
                "doc_id": doc_id,
                "source_uri": source_uri,
                "lang": lang,
                "text": schema_text,
                "token_count": int(_approx_token_count(schema_text)),
                "metadata": {
                    "schema_version": "v1",
                    "group_by": group_by,
                    "from_evidence_block_ids": ev_block_ids[:1000],
                    "num_evidence_blocks": int(len(ev_block_ids)),
                },
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            total_out += 1

    return {
        "in_evidence_blocks": int(total_in),
        "out_schema_blocks": int(total_out),
        "docs": int(len(by_doc)),
        "out_jsonl": str(out_jsonl),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl_evidence", required=True)
    p.add_argument("--out_jsonl", required=True, help="Output blocks.schema.jsonl")
    p.add_argument("--group_by", default="doc_id", choices=["doc_id"])
    p.add_argument("--max_docs", type=int, default=0)
    p.add_argument("--max_evidence_per_doc", type=int, default=200)
    args = p.parse_args()

    stats = build_schema_blocks_from_evidence_jsonl(
        blocks_evidence_jsonl=Path(str(args.blocks_jsonl_evidence)),
        out_jsonl=Path(str(args.out_jsonl)),
        group_by=str(args.group_by),
        max_docs=int(args.max_docs),
        max_evidence_per_doc=int(args.max_evidence_per_doc),
    )
    print(f"[schema_blocks] done stats={stats}", flush=True)


if __name__ == "__main__":
    main()


