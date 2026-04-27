"""
CLI: PDF directory → Raw Context chunks (4096-token) JSONL
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
    from external_kv_injection.src.pipelines.pdf_to_raw_context_chunks import (  # type: ignore
        RawChunkConfig,
        build_raw_context_chunks_from_pdf_dir,
    )
except ModuleNotFoundError:
    from src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", required=True)
    p.add_argument("--out", required=True, help="Output raw_chunks.jsonl")
    p.add_argument("--tokenizer", required=True, help="Tokenizer should match base LLM tokenizer")
    p.add_argument("--chunk_tokens", type=int, default=4096)
    p.add_argument("--chunk_overlap", type=int, default=256)
    p.add_argument("--ocr", default="auto", choices=["off", "auto", "on"])
    p.add_argument("--no_tables", action="store_true", help="Disable table extraction (not recommended for medical PDFs)")
    p.add_argument("--knowledge_filter", action="store_true", help="Use DeepSeek to drop low-knowledge paragraphs")
    p.add_argument("--deepseek_base_url", default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--strict_drop_uncertain", action="store_true", help="Drop UNCERTAIN paragraphs (more aggressive)")
    args = p.parse_args()

    n = build_raw_context_chunks_from_pdf_dir(
        pdf_dir=Path(args.pdf_dir),
        out_jsonl=Path(args.out),
        cfg=RawChunkConfig(
            tokenizer_name_or_path=args.tokenizer,
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
    print(f"Wrote {n} raw chunks to {args.out}")


if __name__ == "__main__":
    main()


