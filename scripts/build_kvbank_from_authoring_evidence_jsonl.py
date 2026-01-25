"""
Build an evidence KVBank from Authoring Layer JSONL.

Usage (example):
  python -u external_kv_injection/scripts/build_kvbank_from_authoring_evidence_jsonl.py \
    --evidence_jsonl path/to/authoring.evidence.runtime.jsonl \
    --out_dir path/to/kvbank_evidence_authoring \
    --base_llm_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --retrieval_encoder_model sentence-transformers/all-MiniLM-L6-v2
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
    from external_kv_injection.src.pipelines.evidence_units_to_kvbank import (  # type: ignore
        build_kvbank_from_authoring_evidence_jsonl,
    )
except ModuleNotFoundError:
    from src.pipelines.evidence_units_to_kvbank import build_kvbank_from_authoring_evidence_jsonl  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description="Build evidence KVBank from Authoring runtime JSONL (approved-only).")
    p.add_argument("--evidence_jsonl", required=True, help="Runtime evidence jsonl (approved-only projection recommended).")
    p.add_argument("--out_dir", required=True, help="Output KVBank dir (FAISS).")
    p.add_argument("--base_llm_name_or_path", required=True, help="Base causal LM to generate past_key_values.")
    p.add_argument("--retrieval_encoder_model", required=True, help="Sentence embedding model for retrieval keys.")
    p.add_argument("--layers", default="0,1,2,3", help="Comma-separated layer ids to store KV for.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    args = p.parse_args()

    layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
    stats = build_kvbank_from_authoring_evidence_jsonl(
        evidence_jsonl=Path(str(args.evidence_jsonl)),
        out_dir=Path(str(args.out_dir)),
        base_llm_name_or_path=str(args.base_llm_name_or_path),
        retrieval_encoder_model=str(args.retrieval_encoder_model),
        layers=tuple(layers),
        max_tokens=int(args.max_tokens),
        max_items=(int(args.max_items) if int(args.max_items) > 0 else None),
        device=(str(args.device) if args.device else None),
        dtype=(str(args.dtype) if args.dtype else None),
        trust_remote_code=True,
    )

    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

