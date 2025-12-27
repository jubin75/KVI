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
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir  # type: ignore
    from external_kv_injection.src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore
    from external_kv_injection.src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
    from external_kv_injection.scripts.build_topic_pdf_subset_deepseek import main as _doc_filter_main  # type: ignore
    from external_kv_injection.src.llm_filter.extractive_evidence import (  # type: ignore
        DeepSeekExtractiveEvidence,
        ExtractiveEvidenceConfig,
    )
except ModuleNotFoundError:
    from src.pipelines.pdf_to_raw_context_chunks import RawChunkConfig, build_raw_context_chunks_from_pdf_dir  # type: ignore
    from src.pipelines.raw_chunks_to_blocks import build_blocks_from_raw_chunks  # type: ignore
    from src.pipelines.blocks_to_kvbank import build_kvbank_from_blocks_jsonl  # type: ignore
    from scripts.build_topic_pdf_subset_deepseek import main as _doc_filter_main  # type: ignore
    from src.llm_filter.extractive_evidence import DeepSeekExtractiveEvidence, ExtractiveEvidenceConfig  # type: ignore


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _approx_token_count(text: str) -> int:
    """
    Cheap, tokenizer-free proxy for 'token_count' (for QA/debugging only).
    Counts alnum "words" and individual CJK characters as units.
    """
    import re

    t = str(text or "").strip()
    if not t:
        return 0
    units = re.findall(r"[A-Za-z0-9]+|[\u4E00-\u9FFF]", t)
    return int(len(units))


def build_evidence_blocks_from_blocks_jsonl(
    *,
    blocks_jsonl: Path,
    out_jsonl: Path,
    topic_goal: str,
    deepseek_base_url: str,
    deepseek_model: str,
    deepseek_api_key_env: str,
    max_sentences_per_block: int = 2,
    max_blocks: int = 0,
) -> Dict[str, Any]:
    """
    Build blocks.evidence.jsonl from raw blocks.jsonl using DeepSeek extractive evidence.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    extractor = DeepSeekExtractiveEvidence(
        ExtractiveEvidenceConfig(
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            api_key_env=deepseek_api_key_env,
            max_sentences=int(max_sentences_per_block),
        )
    )

    total_in = 0
    total_keep = 0
    total_out = 0
    with blocks_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = _safe_json_loads(line)
            if not rec:
                continue
            total_in += 1
            if int(max_blocks) > 0 and total_in > int(max_blocks):
                break
            raw_text = str(rec.get("text") or "")
            if not raw_text.strip():
                continue

            raw_block_id = str(rec.get("block_id") or "")
            doc_id = str(rec.get("doc_id") or "")
            source_uri = rec.get("source_uri", None)
            lang = rec.get("lang", None)

            res = extractor.extract(topic_goal=str(topic_goal), raw_block_text=raw_text)
            sents = res.get("evidence_sentences", []) if isinstance(res.get("evidence_sentences"), list) else []
            if sents:
                total_keep += 1
            for idx, it in enumerate(sents, start=1):
                quote = str(it.get("quote") or "").strip()
                if not quote:
                    continue
                span = it.get("span") if isinstance(it.get("span"), dict) else {}
                ev_block_id = f"{raw_block_id}::ev{idx}"
                out_rec = {
                    "block_id": ev_block_id,
                    "doc_id": doc_id,
                    "source_uri": source_uri,
                    "lang": lang,
                    "text": quote,
                    "token_count": int(_approx_token_count(quote)),
                    "metadata": {
                        "from_raw_block_id": raw_block_id,
                        "span": {"char_start": span.get("char_start"), "char_end": span.get("char_end")},
                        "relevance": it.get("relevance"),
                        "claim": it.get("claim"),
                    },
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                total_out += 1

    return {"in_blocks": total_in, "kept_blocks": total_keep, "out_evidence_blocks": total_out, "out_jsonl": str(out_jsonl)}


def build_evidence_blocks_from_raw_chunks_jsonl(
    *,
    raw_chunks_jsonl: Path,
    out_jsonl: Path,
    topic_goal: str,
    deepseek_base_url: str,
    deepseek_model: str,
    deepseek_api_key_env: str,
    max_sentences_per_paragraph: int = 2,
    max_paragraphs: int = 0,
) -> Dict[str, Any]:
    """
    Preferred evidence build: extract from raw_chunks (paragraph structure) to avoid 256-token block fragmentation.
    """
    import re

    def _split_paragraphs(text: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        return [p for p in parts if len(p) >= 30]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    extractor = DeepSeekExtractiveEvidence(
        ExtractiveEvidenceConfig(
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            api_key_env=deepseek_api_key_env,
            max_sentences=int(max_sentences_per_paragraph),
        )
    )

    chunks = 0
    paras = 0
    kept_paras = 0
    out_blocks = 0

    with raw_chunks_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = _safe_json_loads(line)
            if not rec:
                continue
            chunks += 1
            doc_id = str(rec.get("doc_id") or "")
            chunk_id = str(rec.get("chunk_id") or "")
            source_uri = rec.get("source_uri", None)
            lang = rec.get("lang", None)
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            txt = str(rec.get("text") or "")
            for p_idx, para in enumerate(_split_paragraphs(txt)):
                paras += 1
                if int(max_paragraphs) > 0 and paras > int(max_paragraphs):
                    break
                res = extractor.extract(topic_goal=str(topic_goal), raw_block_text=para)
                sents = res.get("evidence_sentences", []) if isinstance(res.get("evidence_sentences"), list) else []
                if not sents:
                    continue
                kept_paras += 1
                for s_idx, it in enumerate(sents, start=1):
                    quote = str(it.get("quote") or "").strip()
                    if not quote:
                        continue
                    span = it.get("span") if isinstance(it.get("span"), dict) else {}
                    ev_block_id = f"{chunk_id}_p{p_idx}::ev{s_idx}"
                    out_rec = {
                        "block_id": ev_block_id,
                        "doc_id": doc_id,
                        "source_uri": source_uri,
                        "lang": lang,
                        "text": quote,
                        "token_count": int(_approx_token_count(quote)),
                        "metadata": {
                            "from_raw_chunk_id": chunk_id,
                            "paragraph_index": int(p_idx),
                            "span": {"char_start": span.get("char_start"), "char_end": span.get("char_end")},
                            "relevance": it.get("relevance"),
                            "claim": it.get("claim"),
                            "raw_chunk_metadata": meta,
                        },
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    out_blocks += 1
            if int(max_paragraphs) > 0 and paras >= int(max_paragraphs):
                break

    return {
        "in_chunks": chunks,
        "in_paragraphs": paras,
        "kept_paragraphs": kept_paras,
        "out_evidence_blocks": out_blocks,
        "out_jsonl": str(out_jsonl),
    }


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
    blocks_evidence = work_dir / "blocks.evidence.jsonl"
    kv_dir = work_dir / "kvbank_blocks"
    kv_dir_tables = work_dir / "kvbank_tables"
    kv_dir_evidence = work_dir / "kvbank_evidence"

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

    # 2.5) Optional: build extractive evidence blocks and an evidence KVBank.
    evidence_cfg = build_cfg.get("evidence_build") if isinstance(build_cfg.get("evidence_build"), dict) else {}
    if not isinstance(evidence_cfg, dict):
        evidence_cfg = {}
    enable_evidence = bool(evidence_cfg.get("enabled", True))
    # preferred: raw_chunks -> paragraphs -> evidence sentences
    source_level = str(evidence_cfg.get("source_level", "raw_chunks")).strip().lower()
    max_sentences_per_paragraph = int(evidence_cfg.get("max_sentences_per_paragraph", evidence_cfg.get("max_sentences_per_block", 2)))
    max_paragraphs = int(evidence_cfg.get("max_paragraphs", 0))
    # legacy: blocks -> evidence sentences
    max_sentences_per_block = int(evidence_cfg.get("max_sentences_per_block", 2))
    max_blocks_evidence = int(evidence_cfg.get("max_blocks", 0))
    if enable_evidence:
        print(
            f"[rebuild_topic] build_evidence enabled source_level={source_level} "
            f"max_sentences_per_paragraph={max_sentences_per_paragraph} max_paragraphs={max_paragraphs} "
            f"(legacy max_sentences_per_block={max_sentences_per_block} max_blocks={max_blocks_evidence})",
            flush=True,
        )
        if source_level in {"raw_chunks", "raw", "chunks"}:
            ev_stats = build_evidence_blocks_from_raw_chunks_jsonl(
                raw_chunks_jsonl=raw_chunks,
                out_jsonl=blocks_evidence,
                topic_goal=goal,
                deepseek_base_url=deepseek_base_url,
                deepseek_model=deepseek_model,
                deepseek_api_key_env=deepseek_api_key_env,
                max_sentences_per_paragraph=max_sentences_per_paragraph,
                max_paragraphs=max_paragraphs,
            )
        else:
            ev_stats = build_evidence_blocks_from_blocks_jsonl(
                blocks_jsonl=blocks,
                out_jsonl=blocks_evidence,
                topic_goal=goal,
                deepseek_base_url=deepseek_base_url,
                deepseek_model=deepseek_model,
                deepseek_api_key_env=deepseek_api_key_env,
                max_sentences_per_block=max_sentences_per_block,
                max_blocks=max_blocks_evidence,
            )
        print(f"[rebuild_topic] wrote_evidence_blocks stats={ev_stats}", flush=True)

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

    if enable_evidence and blocks_evidence.exists():
        ev_kv_stats = build_kvbank_from_blocks_jsonl(
            blocks_jsonl=blocks_evidence,
            out_dir=kv_dir_evidence,
            split_tables=False,
            out_dir_tables=None,
            base_llm_name_or_path=base_llm,
            retrieval_encoder_model=retrieval_encoder_model,
            layers=layers,
            block_tokens=block_tokens,
            shard_size=(shard_size if shard_size > 0 else None),
        )
        print(f"[rebuild_topic] kvbank_evidence_done stats={ev_kv_stats}", flush=True)
        print(f"[rebuild_topic] kv_dir_evidence={kv_dir_evidence}", flush=True)


if __name__ == "__main__":
    main()


