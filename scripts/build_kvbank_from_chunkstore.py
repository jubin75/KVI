"""
CLI：ChunkStore(JSONL) → FAISS KVBank（存真实 past_key_values）
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
    from external_kv_injection.src.pipelines.chunkstore_to_kvbank import build_faiss_kvbank_from_chunkstore  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.chunkstore_to_kvbank import build_faiss_kvbank_from_chunkstore  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunkstore", required=True, help="ChunkStore JSONL path")
    p.add_argument("--out_dir", required=True, help="Output KVBank directory")
    p.add_argument("--model", required=True, help="HF base model name/path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--layers", default="0,1,2,3", help="Comma-separated inject layer ids")
    p.add_argument("--max_kv_tokens", type=int, default=128)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--retrieval_encoder_model", default=None, help="Optional HF encoder for retrieval_keys (DomainEncoder)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default=None, help="float16|bfloat16|float32 (default: auto)")
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    _, stats = build_faiss_kvbank_from_chunkstore(
        chunkstore_jsonl=Path(args.chunkstore),
        out_dir=Path(args.out_dir),
        model_name_or_path=args.model,
        inject_layers=layer_ids,
        max_kv_tokens=args.max_kv_tokens,
        max_chunks=args.max_chunks,
        retrieval_encoder_model=args.retrieval_encoder_model,
        device=args.device,
        dtype=args.dtype,
    )
    print("BuildStats:", stats)
    print(f"KVBank saved to: {args.out_dir}")


if __name__ == "__main__":
    main()


