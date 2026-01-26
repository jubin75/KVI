"""
End-to-end demo: Authoring EvidenceUnits -> approved-only runtime export -> evidence KVBank build -> query search.

This script is the "t5" runnable path.

It does NOT require you to run the UI, but it works with the same Authoring DB JSONL that the UI edits.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    from external_kv_injection.src.pipelines.evidence_units_to_kvbank import (  # type: ignore
        build_kvbank_from_authoring_evidence_jsonl,
    )
    from external_kv_injection.src.vector_store.faiss_kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.authoring.jsonl_store import iter_approved_runtime_records, read_evidence_units_jsonl, write_runtime_records_jsonl  # type: ignore
    from src.pipelines.evidence_units_to_kvbank import build_kvbank_from_authoring_evidence_jsonl  # type: ignore
    from src.vector_store.faiss_kv_bank import FaissKVBank  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


def _print_block(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _count_status(units: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for u in units:
        s = str(getattr(u, "status", "") or "").strip().lower() or "unknown"
        out[s] = int(out.get(s, 0)) + 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Authoring E2E demo: export->build->search (approved evidence only).")
    p.add_argument(
        "--work_dir",
        required=True,
        help="Work directory for outputs (runtime jsonl + kvbank).",
    )
    p.add_argument(
        "--authoring_db_jsonl",
        required=True,
        help="Authoring DB jsonl (evidence_units.jsonl) edited by the Authoring UI.",
    )
    p.add_argument("--base_llm_name_or_path", required=True, help="Base causal LM to generate past_key_values (KV).")
    p.add_argument("--retrieval_encoder_model", required=True, help="Sentence embedding model for retrieval keys.")
    p.add_argument("--layers", default="0,1,2,3", help="Comma-separated layer ids for KV.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--max_items", type=int, default=0, help="Cap number of approved items to build (0=all).")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--query", required=True, help="Query text to retrieve evidence for.")
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    work_dir = Path(str(args.work_dir))
    work_dir.mkdir(parents=True, exist_ok=True)

    authoring_db = Path(str(args.authoring_db_jsonl))
    runtime_jsonl = work_dir / "authoring.evidence.runtime.jsonl"
    kvbank_dir = work_dir / "kvbank_evidence_authoring"

    layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]

    _print_block("STEP 0 — Load Authoring DB")
    units, load_stats = read_evidence_units_jsonl(authoring_db)
    print(json.dumps({"db": str(authoring_db), "load": asdict(load_stats), "status_counts": _count_status(units)}, ensure_ascii=False, indent=2))

    _print_block("STEP 1 — Export approved-only runtime evidence JSONL")
    records = list(iter_approved_runtime_records(units))
    out_stats = write_runtime_records_jsonl(runtime_jsonl, records)
    print(json.dumps({"runtime_jsonl": str(runtime_jsonl), "approved_exported": int(len(records)), "out": out_stats}, ensure_ascii=False, indent=2))
    if not records:
        raise SystemExit("No approved evidence found. Use the Authoring UI to approve at least 1 evidence unit.")

    _print_block("STEP 2 — Build evidence KVBank (base LLM forward -> K/V)")
    stats = build_kvbank_from_authoring_evidence_jsonl(
        evidence_jsonl=runtime_jsonl,
        out_dir=kvbank_dir,
        base_llm_name_or_path=str(args.base_llm_name_or_path),
        retrieval_encoder_model=str(args.retrieval_encoder_model),
        layers=tuple(layers),
        max_tokens=int(args.max_tokens),
        max_items=(int(args.max_items) if int(args.max_items) > 0 else None),
        device=(str(args.device) if args.device else None),
        dtype=(str(args.dtype) if args.dtype else None),
        trust_remote_code=True,
    )
    print(json.dumps({"kvbank_dir": str(kvbank_dir), "build_stats": asdict(stats)}, ensure_ascii=False, indent=2))

    _print_block("STEP 3 — Query search (FAISS top-k) + readable print")
    enc = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=str(args.retrieval_encoder_model), max_length=int(args.max_tokens), normalize=True))
    qv = enc.encode(str(args.query))[0]
    bank = FaissKVBank.load(kvbank_dir)
    items, dbg = bank.search(qv, top_k=int(args.top_k), filters=None)
    print(json.dumps({"query": str(args.query), "top_k": int(args.top_k), "debug": dbg}, ensure_ascii=False, indent=2))
    for i, it in enumerate(items, start=1):
        meta = getattr(it, "meta", None) or {}
        eid = meta.get("evidence_id") or meta.get("block_id") or meta.get("id")
        print("\n" + "-" * 78)
        print(f"[{i:02d}] score={float(getattr(it, 'score', 0.0)):.4f} evidence_id={eid}")
        print(f"     semantic_type={meta.get('semantic_type')} schema_id={meta.get('schema_id')} polarity={meta.get('polarity')}")
        if meta.get("doc_id") or meta.get("source_uri"):
            print(f"     doc_id={meta.get('doc_id')} source_uri={meta.get('source_uri')}")
        txt = str(meta.get("semantic_text") or "")
        if txt:
            print("     text:")
            print("     " + txt.replace("\n", "\n     "))

    _print_block("DONE")
    print(
        json.dumps(
            {
                "authoring_db_jsonl": str(authoring_db),
                "runtime_jsonl": str(runtime_jsonl),
                "kvbank_dir": str(kvbank_dir),
                "note": "Only approved evidence was exported/built. Draft/rejected remain authoring-only.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

