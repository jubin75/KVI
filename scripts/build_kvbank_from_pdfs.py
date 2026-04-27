"""
CLI: PDF directory → ChunkStore → KVBank (end-to-end)
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
    from external_kv_injection.src.pipelines.pdf_to_chunkstore import build_chunkstore_from_pdfs  # type: ignore
    from external_kv_injection.src.pipelines.chunkstore_to_kvbank import build_faiss_kvbank_from_chunkstore  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.pdf_to_chunkstore import build_chunkstore_from_pdfs  # type: ignore
    from src.pipelines.chunkstore_to_kvbank import build_faiss_kvbank_from_chunkstore  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", required=True)
    p.add_argument("--work_dir", required=True, help="Directory to write chunkstore + kvbank")
    p.add_argument("--model", required=True, help="HF base model name/path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--dataset_version", default="v0")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--max_kv_tokens", type=int, default=128)
    p.add_argument("--max_chunks", type=int, default=200)
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    chunkstore = work / "chunkstore.jsonl"
    kv_dir = work / "kvbank"

    n = build_chunkstore_from_pdfs(
        pdf_dir=Path(args.pdf_dir),
        output_jsonl=chunkstore,
        dataset_version=args.dataset_version,
    )
    print(f"Wrote {n} chunks to {chunkstore}")

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    _, stats = build_faiss_kvbank_from_chunkstore(
        chunkstore_jsonl=chunkstore,
        out_dir=kv_dir,
        model_name_or_path=args.model,
        inject_layers=layer_ids,
        max_kv_tokens=args.max_kv_tokens,
        max_chunks=args.max_chunks,
    )
    print("BuildStats:", stats)
    print(f"KVBank saved to: {kv_dir}")


if __name__ == "__main__":
    main()


