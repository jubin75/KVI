"""
CLI: Build teacher KV dataset (past_key_values supervision) from ChunkStore
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
    from external_kv_injection.src.training.teacher_kv_dataset import (  # type: ignore
        BuildTeacherKVDatasetConfig,
        build_teacher_kv_dataset,
    )
except ModuleNotFoundError:
    from src.training.teacher_kv_dataset import BuildTeacherKVDatasetConfig, build_teacher_kv_dataset  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunkstore", required=True)
    p.add_argument("--out", required=True, help="Output .pt path")
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--max_kv_tokens", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    cfg = BuildTeacherKVDatasetConfig(
        model_name_or_path=args.model,
        layers=layer_ids,
        max_kv_tokens=args.max_kv_tokens,
        max_samples=args.max_samples,
        device=args.device,
        dtype=args.dtype,
    )
    stats = build_teacher_kv_dataset(
        chunkstore_jsonl=Path(args.chunkstore),
        out_path=Path(args.out),
        cfg=cfg,
    )
    print("BuildTeacherKVDatasetStats:", stats)
    print(f"Saved dataset to: {args.out}")


if __name__ == "__main__":
    main()


