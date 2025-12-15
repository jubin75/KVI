"""
CLI：ChunkStore → KVBank（用训练好的 KVProjector 生成 K/V）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from external_kv_injection.src.pipelines.chunkstore_to_kvbank_with_projector import build_faiss_kvbank_with_projector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunkstore", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--projector_ckpt", required=True, help="Path to projector_kv.pt")
    p.add_argument("--max_kv_tokens", type=int, default=256)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--retrieval_encoder_model", default=None, help="Optional HF encoder for retrieval_keys (DomainEncoder)")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    args = p.parse_args()

    build_faiss_kvbank_with_projector(
        chunkstore_jsonl=Path(args.chunkstore),
        out_dir=Path(args.out_dir),
        base_model_name_or_path=args.base_model,
        projector_ckpt_path=Path(args.projector_ckpt),
        max_kv_tokens=args.max_kv_tokens,
        max_chunks=args.max_chunks,
        retrieval_encoder_model=args.retrieval_encoder_model,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"KVBank saved to: {args.out_dir}")


if __name__ == "__main__":
    main()


