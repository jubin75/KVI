"""
CLI：Raw Context chunks（raw_chunks.jsonl）→ Memory blocks（blocks.jsonl）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_chunks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--keep_last_incomplete_block", action="store_true")
    args = p.parse_args()

    n = build_blocks_from_raw_chunks(
        raw_chunks_jsonl=Path(args.raw_chunks),
        out_blocks_jsonl=Path(args.out),
        tokenizer_name_or_path=args.tokenizer,
        block_tokens=args.block_tokens,
        drop_last_incomplete_block=not args.keep_last_incomplete_block,
    )
    print(f"Wrote {n} blocks to {args.out}")


if __name__ == "__main__":
    main()


