"""
CLI：PDF → ChunkStore（JSONL）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.pdf_to_chunkstore import build_chunkstore_from_pdfs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", required=True)
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--dataset_version", default="v0")
    p.add_argument("--target_tokens", type=int, default=350)
    p.add_argument("--max_tokens", type=int, default=900)
    p.add_argument("--overlap_ratio", type=float, default=0.15)
    args = p.parse_args()

    n = build_chunkstore_from_pdfs(
        pdf_dir=Path(args.pdf_dir),
        output_jsonl=Path(args.out),
        dataset_version=args.dataset_version,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
    )
    print(f"Wrote {n} chunks to {args.out}")


if __name__ == "__main__":
    main()


