"""
CLI：Raw Context chunks（raw_chunks.jsonl）→ Memory blocks（blocks.jsonl）
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
    from external_kv_injection.src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_chunks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument(
        "--block_overlap_tokens",
        type=int,
        default=64,
        help="Overlap between consecutive blocks (reduces sentence/table-row truncation). Must be < block_tokens.",
    )
    p.add_argument("--keep_last_incomplete_block", action="store_true")
    args = p.parse_args()

    n = build_blocks_from_raw_chunks(
        raw_chunks_jsonl=Path(args.raw_chunks),
        out_blocks_jsonl=Path(args.out),
        tokenizer_name_or_path=args.tokenizer,
        block_tokens=args.block_tokens,
        block_overlap_tokens=int(args.block_overlap_tokens),
        drop_last_incomplete_block=not args.keep_last_incomplete_block,
    )
    print(f"Wrote {n} blocks to {args.out}")


if __name__ == "__main__":
    main()


