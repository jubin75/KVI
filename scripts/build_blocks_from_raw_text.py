"""
CLI：Raw Context(text file) → 4096-token chunks → 256-token memory blocks（JSONL）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.raw_context_to_blocks import RawToBlocksConfig, build_memory_blocks_from_raw_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_text", required=True, help="Path to a raw text file (utf-8)")
    p.add_argument("--out", required=True, help="Output JSONL of memory blocks")
    p.add_argument("--tokenizer", required=True, help="Tokenizer name/path (should match base LLM tokenizer)")
    p.add_argument("--chunk_tokens", type=int, default=4096)
    p.add_argument("--chunk_overlap", type=int, default=256)
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--keep_last_incomplete_block", action="store_true")
    args = p.parse_args()

    raw_text = Path(args.raw_text).read_text(encoding="utf-8", errors="ignore")
    cfg = RawToBlocksConfig(
        tokenizer_name_or_path=args.tokenizer,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        block_tokens=args.block_tokens,
        drop_last_incomplete_block=not args.keep_last_incomplete_block,
    )

    n = build_memory_blocks_from_raw_text(
        raw_text=raw_text,
        source_id=Path(args.raw_text).stem,
        out_jsonl=Path(args.out),
        cfg=cfg,
    )
    print(f"Wrote {n} memory blocks to {args.out}")


if __name__ == "__main__":
    main()


