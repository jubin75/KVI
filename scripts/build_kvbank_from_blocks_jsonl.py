"""
CLI wrapper: blocks.jsonl -> KVBank (FAISS) using the pipeline implementation.

This exists to make runbook commands simple and stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore


def _build_alignment_report(blocks_path: Path) -> dict:
    total = 0
    list_like = 0
    missing = 0
    list_counts = []
    sample_list_blocks = []
    try:
        with blocks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                rec = json.loads(line)
                meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                pat = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
                lf = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
                if not lf:
                    missing += 1
                    continue
                items = lf.get("list_items") if isinstance(lf.get("list_items"), list) else []
                if not items:
                    items = lf.get("list_like_items") if isinstance(lf.get("list_like_items"), list) else []
                if items:
                    list_like += 1
                    list_counts.append(len(items))
                    if len(sample_list_blocks) < 5:
                        sample_list_blocks.append(
                            {"block_id": rec.get("block_id"), "items": items[:8]}
                        )
    except Exception:
        pass
    avg_count = float(sum(list_counts) / max(1, len(list_counts))) if list_counts else 0.0
    return {
        "total_blocks": int(total),
        "list_like_blocks": int(list_like),
        "avg_list_feature_count": float(avg_count),
        "blocks_missing_enriched": int(missing),
        "sample_list_blocks": sample_list_blocks,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl", required=True)
    p.add_argument(
        "--disable_enriched",
        action="store_true",
        help="Do not auto-switch to blocks.enriched.jsonl when available",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_llm", required=True)
    p.add_argument("--domain_encoder_model", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    p.add_argument("--block_tokens", type=int, default=256)
    p.add_argument("--split_tables", action="store_true")
    p.add_argument("--out_dir_tables", default=None)
    p.add_argument("--shard_size", type=int, default=1024)
    p.add_argument("--max_blocks", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda/cpu; default auto")
    p.add_argument("--dtype", default=None, help="torch dtype name; default auto (bf16 on cuda)")
    args = p.parse_args()

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""]
    max_blocks = int(args.max_blocks) if int(args.max_blocks) > 0 else None
    shard_size = int(args.shard_size) if int(args.shard_size) > 0 else None

    blocks_path = Path(str(args.blocks_jsonl))
    if not bool(args.disable_enriched):
        if blocks_path.name == "blocks.jsonl":
            enriched = blocks_path.with_name("blocks.enriched.jsonl")
            if enriched.exists():
                print(f"[build_kvbank_from_blocks_jsonl] using_enriched={enriched}", flush=True)
                blocks_path = enriched

    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=blocks_path,
        out_dir=Path(str(args.out_dir)),
        split_tables=bool(args.split_tables),
        out_dir_tables=(Path(str(args.out_dir_tables)) if args.out_dir_tables else None),
        base_llm_name_or_path=str(args.base_llm),
        retrieval_encoder_model=str(args.domain_encoder_model),
        layers=layers,
        block_tokens=int(args.block_tokens),
        max_blocks=max_blocks,
        device=(str(args.device) if args.device else None),
        dtype=(str(args.dtype) if args.dtype else None),
        shard_size=shard_size,
    )
    report = _build_alignment_report(blocks_path)
    report_path = Path(str(args.out_dir)) / "kvbank_alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_kvbank_from_blocks_jsonl] done stats={stats}", flush=True)
    print(f"[build_kvbank_from_blocks_jsonl] alignment_report={report_path}", flush=True)


if __name__ == "__main__":
    main()

