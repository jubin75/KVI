"""
CLI：Raw Context(text file) → blocks.jsonl → KVBank（FAISS）

严格遵循 PRD/多步注入的工程实现.md：
- Raw context 只用于建库
- 4096-token chunks（overlap=256）
- 256-token memory blocks
- KVBank 存 blocks 的 K/V（layers 0..3），检索用 DomainEncoder embedding

注意（避免误解）
- 该脚本是“从 raw text 文件快速演示”的兜底路径。
- 生产级推荐走：PDF→raw_chunks(4096)→blocks(256)→KVBank
  对应脚本：build_raw_context_from_pdfs.py / build_blocks_from_raw_chunks.py / build_kvbank_from_blocks.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.raw_context_to_blocks import RawToBlocksConfig, build_memory_blocks_from_raw_text
from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_text", required=True, help="Path to raw text file")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--base_llm", required=True)
    p.add_argument("--retrieval_encoder_model", required=True)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--chunk_tokens", type=int, default=4096)
    p.add_argument("--chunk_overlap", type=int, default=256)
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--max_blocks", type=int, default=None)
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    blocks_jsonl = work / "blocks.jsonl"
    kv_dir = work / "kvbank_blocks"

    raw_text = Path(args.raw_text).read_text(encoding="utf-8", errors="ignore")
    cfg = RawToBlocksConfig(
        tokenizer_name_or_path=args.base_llm,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        block_tokens=args.block_tokens,
        drop_last_incomplete_block=True,
    )
    n = build_memory_blocks_from_raw_text(raw_text=raw_text, source_id=Path(args.raw_text).stem, out_jsonl=blocks_jsonl, cfg=cfg)
    print(f"Wrote {n} blocks to {blocks_jsonl}")

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=blocks_jsonl,
        out_dir=kv_dir,
        base_llm_name_or_path=args.base_llm,
        retrieval_encoder_model=args.retrieval_encoder_model,
        layers=layer_ids,
        block_tokens=args.block_tokens,
        max_blocks=args.max_blocks,
    )
    print("BuildBlocksKVBankStats:", stats)
    print(f"Saved KVBank to: {kv_dir}")


if __name__ == "__main__":
    main()


