"""
一键：PDF目录 → raw_chunks(4096) → blocks(256) → KVBank（FAISS）

对齐 PRD/raw context构建流程.md：
- 保留 raw context（4096-token chunks）作为离线存储层产物
- KVBank 只存 memory blocks 的 embedding + K/V + metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir
from external_kv_injection.src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks
from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--base_llm", required=True)
    p.add_argument("--retrieval_encoder_model", required=True)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--chunk_tokens", type=int, default=4096)
    p.add_argument("--chunk_overlap", type=int, default=256)
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--ocr", default="auto", choices=["off", "auto", "on"])
    p.add_argument("--no_tables", action="store_true")
    p.add_argument("--knowledge_filter", action="store_true")
    p.add_argument("--deepseek_base_url", default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--strict_drop_uncertain", action="store_true")
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    raw_chunks = work / "raw_chunks.jsonl"
    blocks = work / "blocks.jsonl"
    kv_dir = work / "kvbank_blocks"

    n_chunks = build_raw_context_chunks_from_pdf_dir(
        pdf_dir=Path(args.pdf_dir),
        out_jsonl=raw_chunks,
        cfg=RawChunkConfig(
            tokenizer_name_or_path=args.base_llm,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            ocr=args.ocr,
            extract_tables=not args.no_tables,
            knowledge_filter=bool(args.knowledge_filter),
            deepseek_base_url=args.deepseek_base_url,
            deepseek_model=args.deepseek_model,
            deepseek_api_key_env=args.deepseek_api_key_env,
            strict_drop_uncertain=bool(args.strict_drop_uncertain),
        ),
    )
    print(f"Wrote {n_chunks} raw chunks to {raw_chunks}")

    n_blocks = build_blocks_from_raw_chunks(
        raw_chunks_jsonl=raw_chunks,
        out_blocks_jsonl=blocks,
        tokenizer_name_or_path=args.base_llm,
        block_tokens=args.block_tokens,
        drop_last_incomplete_block=True,
    )
    print(f"Wrote {n_blocks} blocks to {blocks}")

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=blocks,
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


