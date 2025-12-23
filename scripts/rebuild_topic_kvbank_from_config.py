"""
One-command rebuild for a topic KVBank (config-driven).

Workflow:
1) Doc-level DS filter by ABSTRACT only (keeps only PDFs whose abstract matches the user-provided goal)
2) Build pipeline on the kept PDFs:
   PDFs -> raw_chunks -> blocks -> KVBank (optionally split tables)

This script intentionally has no frontend/UI: you edit the per-topic config.json and rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir  # type: ignore
    from external_kv_injection.src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore
    from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
    from external_kv_injection.scripts.build_topic_pdf_subset_deepseek import main as _doc_filter_main  # type: ignore
except ModuleNotFoundError:
    from src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir  # type: ignore
    from src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore
    from src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
    from scripts.build_topic_pdf_subset_deepseek import main as _doc_filter_main  # type: ignore


def _load_config(path: Path) -> Dict[str, Any]:
    # Tolerate repo layout differences (monorepo vs flat KVI root).
    p = path
    if not p.is_absolute():
        p1 = (Path.cwd() / p).resolve()
        if p1.exists():
            p = p1
        else:
            p2 = (_REPO_ROOT / p).resolve()
            if p2.exists():
                p = p2
    if not p.exists():
        parts = list(p.parts)
        if "external_kv_injection" in parts:
            i = parts.index("external_kv_injection")
            alt_rel = Path(*parts[i + 1 :])
            alt = (_REPO_ROOT / alt_rel).resolve()
            if alt.exists():
                p = alt
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("config must be a JSON object")
    return obj


def _as_layers(x: Any) -> List[int]:
    if isinstance(x, list):
        return [int(i) for i in x]
    if isinstance(x, str):
        return [int(s.strip()) for s in x.split(",") if s.strip()]
    raise ValueError("layers must be list[int] or comma string")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Topic config JSON (see external_kv_injection/config/topics/*/config.json)")
    p.add_argument("--skip_doc_filter", action="store_true", help="If set, skip doc-level DS filtering and build directly from out_pdf_dir.")
    args = p.parse_args()

    cfg_path = Path(str(args.config))
    cfg = _load_config(cfg_path)

    topic_name = str(cfg.get("topic_name") or cfg_path.parent.name)
    goal = str(cfg.get("goal") or "").strip()
    if not goal:
        raise SystemExit("config.goal is required (专题库目标)")

    source_pdf_dir = Path(str(cfg.get("source_pdf_dir") or ""))
    out_pdf_dir = Path(str(cfg.get("out_pdf_dir") or ""))
    results_jsonl = Path(str(cfg.get("results_jsonl") or (out_pdf_dir / "doc_filter_results.jsonl")))
    if not str(source_pdf_dir):
        raise SystemExit("config.source_pdf_dir is required")
    if not str(out_pdf_dir):
        raise SystemExit("config.out_pdf_dir is required")

    ocr = str(cfg.get("ocr") or "auto")
    extract_tables = bool(cfg.get("extract_tables", True))
    split_tables = bool(cfg.get("split_tables", True))

    # 1) Doc-level filter (abstract-only)
    if not bool(args.skip_doc_filter):
        argv = [
            "build_topic_pdf_subset_deepseek.py",
            "--config",
            str(cfg_path),
            "--pdf_dir",
            str(source_pdf_dir),
            "--out_pdf_dir",
            str(out_pdf_dir),
            "--results_jsonl",
            str(results_jsonl),
        ]
        # run as "embedded CLI" to reuse existing logic without another subprocess
        old_argv = sys.argv
        try:
            sys.argv = argv
            _doc_filter_main()
        finally:
            sys.argv = old_argv

    # 2) Build pipeline on kept PDFs
    build_cfg = cfg.get("build") if isinstance(cfg.get("build"), dict) else {}
    if not isinstance(build_cfg, dict):
        raise SystemExit("config.build must be an object")

    work_dir = Path(str(build_cfg.get("work_dir") or ""))
    if not str(work_dir):
        raise SystemExit("config.build.work_dir is required")
    work_dir.mkdir(parents=True, exist_ok=True)

    base_llm = str(build_cfg.get("base_llm") or build_cfg.get("base_model") or "")
    retrieval_encoder_model = str(build_cfg.get("retrieval_encoder_model") or "")
    if not base_llm:
        raise SystemExit("config.build.base_llm is required")
    if not retrieval_encoder_model:
        raise SystemExit("config.build.retrieval_encoder_model is required")

    layers = _as_layers(build_cfg.get("layers", [0, 1, 2, 3]))
    chunk_tokens = int(build_cfg.get("chunk_tokens", 4096))
    chunk_overlap = int(build_cfg.get("chunk_overlap", 256))
    block_tokens = int(build_cfg.get("block_tokens", 256))
    block_overlap_tokens = int(build_cfg.get("block_overlap_tokens", 64))
    keep_last = bool(build_cfg.get("keep_last_incomplete_block", True))
    knowledge_filter = bool(build_cfg.get("knowledge_filter", True))
    shard_size = int(build_cfg.get("shard_size", 1024))

    deepseek_base_url = str(cfg.get("deepseek_base_url", "https://api.deepseek.com"))
    deepseek_model = str(cfg.get("deepseek_model", "deepseek-chat"))
    deepseek_api_key_env = str(cfg.get("deepseek_api_key_env", "DEEPSEEK_API_KEY"))
    strict_drop_uncertain = bool(cfg.get("strict_drop_uncertain", True))

    raw_chunks = work_dir / "raw_chunks.jsonl"
    blocks = work_dir / "blocks.jsonl"
    kv_dir = work_dir / "kvbank_blocks"
    kv_dir_tables = work_dir / "kvbank_tables"

    print(
        f"[rebuild_topic] topic={topic_name} work_dir={work_dir} pdfs_dir={out_pdf_dir} "
        f"extract_tables={extract_tables} split_tables={split_tables} ocr={ocr}",
        flush=True,
    )

    n_chunks = build_raw_context_chunks_from_pdf_dir(
        pdf_dir=out_pdf_dir,
        out_jsonl=raw_chunks,
        cfg=RawChunkConfig(
            tokenizer_name_or_path=base_llm,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            ocr=ocr,
            extract_tables=bool(extract_tables),
            knowledge_filter=bool(knowledge_filter),
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            deepseek_api_key_env=deepseek_api_key_env,
            strict_drop_uncertain=bool(strict_drop_uncertain),
        ),
    )
    print(f"[rebuild_topic] wrote_raw_chunks={n_chunks} path={raw_chunks}", flush=True)

    n_blocks = build_blocks_from_raw_chunks(
        raw_chunks_jsonl=raw_chunks,
        out_blocks_jsonl=blocks,
        tokenizer_name_or_path=base_llm,
        block_tokens=block_tokens,
        block_overlap_tokens=block_overlap_tokens,
        drop_last_incomplete_block=not bool(keep_last),
    )
    print(f"[rebuild_topic] wrote_blocks={n_blocks} path={blocks}", flush=True)

    stats = build_kvbank_from_blocks_jsonl(
        blocks_jsonl=blocks,
        out_dir=kv_dir,
        split_tables=bool(split_tables),
        out_dir_tables=(kv_dir_tables if bool(split_tables) else None),
        base_llm_name_or_path=base_llm,
        retrieval_encoder_model=retrieval_encoder_model,
        layers=layers,
        block_tokens=block_tokens,
        shard_size=(shard_size if shard_size > 0 else None),
    )
    print(f"[rebuild_topic] kvbank_done stats={stats}", flush=True)
    print(f"[rebuild_topic] kv_dir={kv_dir}", flush=True)
    if bool(split_tables):
        print(f"[rebuild_topic] kv_dir_tables={kv_dir_tables}", flush=True)


if __name__ == "__main__":
    main()


