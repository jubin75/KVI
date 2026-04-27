"""
CLI: memory blocks JSONL → KVBank (FAISS)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    p.add_argument("--blocks", required=True, help="blocks.jsonl from build_blocks_from_raw_text.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_tables", action="store_true", help="Split table-like blocks into a separate KVBank.")
    p.add_argument("--out_dir_tables", default=None, help="Output dir for tables KVBank (required if --split_tables).")
    p.add_argument("--base_llm", required=True, help="Base LLM for extracting past_key_values")
    p.add_argument("--retrieval_encoder_model", required=True, help="DomainEncoder for retrieval_keys")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Plan A: shard KVBank. Write N blocks per shard (0=disabled, recommended 512~2048 depending on VRAM/RAM).",
    )
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=Path(args.blocks),
        out_dir=Path(args.out_dir),
        split_tables=bool(args.split_tables),
        out_dir_tables=(Path(args.out_dir_tables) if args.out_dir_tables else None),
        base_llm_name_or_path=args.base_llm,
        retrieval_encoder_model=args.retrieval_encoder_model,
        layers=layer_ids,
        block_tokens=args.block_tokens,
        max_blocks=args.max_blocks,
        shard_size=(int(args.shard_size) if int(args.shard_size) > 0 else None),
    )
    print("BuildBlocksKVBankStats:", stats)
    print(f"Saved KVBank to: {args.out_dir}")


if __name__ == "__main__":
    main()


