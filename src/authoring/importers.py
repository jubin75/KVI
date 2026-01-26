from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import EvidenceUnit, ExternalRefs, Provenance


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = _safe_json_loads(s)
            if isinstance(obj, dict):
                yield obj


def _stable_ds_evidence_id(*, doc_id: str, source_uri: str, quote: str) -> str:
    key = f"{doc_id}|{source_uri}|{quote}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"EVIDENCE_DS_{digest}"


@dataclass(frozen=True)
class ImportDeepSeekBlocksStats:
    read_blocks: int
    created: int
    dedup_skipped: int
    errors: int
    out_db: str


def import_deepseek_blocks_evidence_jsonl_to_authoring_db(
    *,
    blocks_evidence_jsonl: Path,
    authoring_db_jsonl: Path,
    schema_id: str,
    default_semantic_type: str = "generic",
    evidence_type: str = "extractive_suggestion",
) -> ImportDeepSeekBlocksStats:
    """
    Convert DeepSeek extractive `blocks.evidence.jsonl` into Authoring EvidenceUnit drafts.

    Guarantees:
    - status is always "draft"
    - stable dedup by deterministic evidence_id derived from (doc_id, source_uri, quote)
    - no auto-approval (human must review)
    """
    authoring_db_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not authoring_db_jsonl.exists():
        authoring_db_jsonl.write_text("", encoding="utf-8")

    existing_ids: set[str] = set()
    try:
        for rec in _read_jsonl(authoring_db_jsonl):
            eid = str(rec.get("evidence_id") or rec.get("id") or "").strip()
            if eid:
                existing_ids.add(eid)
    except Exception:
        # tolerate unreadable db; we still append
        existing_ids = set()

    read_blocks = 0
    created = 0
    dedup_skipped = 0
    errors = 0

    schema_id = str(schema_id or "").strip()
    if not schema_id:
        raise ValueError("schema_id is required for importing DeepSeek suggestions")

    default_semantic_type = str(default_semantic_type or "generic").strip().lower()

    with authoring_db_jsonl.open("a", encoding="utf-8") as out:
        for b in _read_jsonl(blocks_evidence_jsonl):
            read_blocks += 1
            try:
                quote = str(b.get("text") or "").strip()
                if not quote:
                    continue
                doc_id = str(b.get("doc_id") or "").strip()
                source_uri = str(b.get("source_uri") or "").strip()
                eid = _stable_ds_evidence_id(doc_id=doc_id, source_uri=source_uri, quote=quote)
                if eid in existing_ids:
                    dedup_skipped += 1
                    continue
                existing_ids.add(eid)

                meta = b.get("metadata") if isinstance(b.get("metadata"), dict) else {}
                # Keep DS extraction hints in review_feedback for auditors (NOT for runtime).
                review_feedback = {
                    "state": "draft",
                    "reasons": ["deepseek_extractive_suggestion"],
                    "comment": "Imported from DeepSeek extractive evidence. Requires human review before approval.",
                    "deepseek": {
                        "relevance": meta.get("relevance"),
                        "claim": meta.get("claim"),
                        "span": meta.get("span"),
                        "source_level": ("raw_chunks" if "from_raw_chunk_id" in meta else "blocks"),
                        "from_raw_block_id": meta.get("from_raw_block_id"),
                        "from_raw_chunk_id": meta.get("from_raw_chunk_id"),
                        "paragraph_index": meta.get("paragraph_index"),
                    },
                }

                eu = EvidenceUnit(
                    evidence_id=eid,
                    semantic_type=default_semantic_type,  # type: ignore[arg-type]
                    schema_id=schema_id,
                    claim=quote,
                    polarity="neutral",  # type: ignore[arg-type]
                    slot_projection={},
                    status="draft",  # type: ignore[arg-type]
                    provenance=Provenance(
                        source_type="review",
                        organization=None,
                        document_title=str(doc_id or ""),
                        publication_year=None,
                        page_range=None,
                    ),
                    external_refs=ExternalRefs(
                        document_id=(doc_id or None),
                        title=(doc_id or None),
                        source_uri=(source_uri or None),
                        url=(source_uri or None),
                    ),
                    evidence_type=str(evidence_type or "extractive_suggestion"),
                    review_feedback=review_feedback,
                )

                out.write(json.dumps(eu.to_dict(), ensure_ascii=False) + "\n")
                created += 1
            except Exception:
                errors += 1
                continue

    return ImportDeepSeekBlocksStats(
        read_blocks=int(read_blocks),
        created=int(created),
        dedup_skipped=int(dedup_skipped),
        errors=int(errors),
        out_db=str(authoring_db_jsonl),
    )

