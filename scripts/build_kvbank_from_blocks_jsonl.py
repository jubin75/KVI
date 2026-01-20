"""
CLI wrapper: blocks.jsonl -> KVBank (FAISS) using the pipeline implementation.

This exists to make runbook commands simple and stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl", required=True)
    p.add_argument(
        "--disable_enriched",
        action="store_true",
        help="Do not auto-switch to blocks.enriched.jsonl when available",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_llm", required=True)
    p.add_argument("--domain_encoder_model", required=True)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--split_tables", action="store_true")
    p.add_argument("--out_dir_tables", default=None)
    p.add_argument("--shard_size", type=int, default=1024)
    p.add_argument("--max_blocks", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda/cpu; default auto")
    p.add_argument("--dtype", default=None, help="torch dtype name; default auto (bf16 on cuda)")
    args = p.parse_args()

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""]
    max_blocks = int(args.max_blocks) if int(args.max_blocks) > 0 else None
    shard_size = int(args.shard_size) if int(args.shard_size) > 0 else None

    blocks_path = Path(str(args.blocks_jsonl))
    if not bool(args.disable_enriched):
        if blocks_path.name == "blocks.jsonl":
            enriched = blocks_path.with_name("blocks.enriched.jsonl")
            if enriched.exists():
                print(f"[build_kvbank_from_blocks_jsonl] using_enriched={enriched}", flush=True)
                blocks_path = enriched

    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=blocks_path,
        out_dir=Path(str(args.out_dir)),
        split_tables=bool(args.split_tables),
        out_dir_tables=(Path(str(args.out_dir_tables)) if args.out_dir_tables else None),
        base_llm_name_or_path=str(args.base_llm),
        retrieval_encoder_model=str(args.domain_encoder_model),
        layers=layers,
        block_tokens=int(args.block_tokens),
        max_blocks=max_blocks,
        device=(str(args.device) if args.device else None),
        dtype=(str(args.dtype) if args.dtype else None),
        shard_size=shard_size,
    )
    print(f"[build_kvbank_from_blocks_jsonl] done stats={stats}", flush=True)


if __name__ == "__main__":
    main()


