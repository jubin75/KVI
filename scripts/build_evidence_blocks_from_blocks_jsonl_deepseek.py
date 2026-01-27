"""
Build extractive evidence blocks from raw blocks.jsonl using DeepSeek.

Input:  blocks.jsonl (each line contains at least: block_id, doc_id, text)
Output: blocks.evidence.jsonl (each line is a short evidence block; one input block can yield 0..N evidence blocks)

Design goals
- extractive-only quotes (verbatim) for verifiable provenance
- single-intent, short blocks to reduce KV injection noise
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.llm_filter.extractive_evidence import (  # type: ignore
        DeepSeekExtractiveEvidence,
        ExtractiveEvidenceConfig,
    )
except ModuleNotFoundError:
    from src.llm_filter.extractive_evidence import DeepSeekExtractiveEvidence, ExtractiveEvidenceConfig  # type: ignore

try:
    from external_kv_injection.src.llm_filter.doc_meta_extractor import (  # type: ignore
        DeepSeekDocMetaExtractor,
        DocMetaExtractorConfig,
    )
except ModuleNotFoundError:
    from src.llm_filter.doc_meta_extractor import DeepSeekDocMetaExtractor, DocMetaExtractorConfig  # type: ignore


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _approx_token_count(text: str) -> int:
    """
    Cheap, tokenizer-free proxy for 'token_count' used in QA/inspection.
    Counts alnum "words" and individual CJK characters as units.
    """
    t = str(text or "").strip()
    if not t:
        return 0
    units = re.findall(r"[A-Za-z0-9]+|[\u4E00-\u9FFF]", t)
    return int(len(units))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl", required=True, help="Input raw blocks.jsonl")
    p.add_argument("--out_jsonl", required=True, help="Output evidence blocks jsonl (e.g., blocks.evidence.jsonl)")
    p.add_argument("--out_docs_meta_jsonl", default="", help="Optional docs.meta.jsonl (doc-level metadata)")
    p.add_argument("--kb_id", default="", help="Optional kb_id (topic/knowledgebase id) attached to outputs")
    p.add_argument("--topic_goal", required=True, help="Topic goal text (used to guide extraction)")
    p.add_argument("--max_sentences_per_block", type=int, default=3)
    p.add_argument("--max_blocks", type=int, default=0, help="If >0, only process first N blocks (debug)")
    p.add_argument("--deepseek_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    args = p.parse_args()

    in_path = Path(str(args.blocks_jsonl))
    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = DeepSeekExtractiveEvidence(
        ExtractiveEvidenceConfig(
            deepseek_base_url=str(args.deepseek_base_url),
            deepseek_model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
            max_sentences=int(args.max_sentences_per_block),
        )
    )

    meta_extractor = DeepSeekDocMetaExtractor(
        DocMetaExtractorConfig(
            deepseek_base_url=str(args.deepseek_base_url),
            deepseek_model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
            max_chars=9000,
        )
    )

    docs_meta_path = Path(str(args.out_docs_meta_jsonl)) if str(args.out_docs_meta_jsonl or "").strip() else None
    docs_meta_f = docs_meta_path.open("w", encoding="utf-8") if docs_meta_path else None
    seen_docs: Set[str] = set()

    total_in = 0
    total_keep = 0
    total_out = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rec = _safe_json_loads(line)
            if not rec:
                continue
            total_in += 1
            if int(args.max_blocks) > 0 and total_in > int(args.max_blocks):
                break

            raw_text = str(rec.get("text") or "")
            if not raw_text.strip():
                continue
            raw_block_id = str(rec.get("block_id") or "")
            doc_id = str(rec.get("doc_id") or "")
            source_uri = rec.get("source_uri", None)
            lang = rec.get("lang", None)
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            para_type = str(meta.get("paragraph_type") or "").strip().lower()

            if docs_meta_f is not None and doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                snippet = str(rec.get("text") or "")[:9000]
                try:
                    doc_meta = meta_extractor.extract(doc_id=doc_id, source_uri=str(source_uri or ""), pdf_snippet=snippet)
                except Exception:
                    doc_meta = {"title": doc_id, "journal": None, "doi": None, "publication_year": None, "published_at": None, "authors": []}
                docs_meta_f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                            "source_uri": source_uri,
                            "meta": doc_meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            res = extractor.extract(topic_goal=str(args.topic_goal), raw_block_text=raw_text)
            keep = bool(res.get("keep", False))
            sents = res.get("evidence_sentences", []) if isinstance(res.get("evidence_sentences"), list) else []
            if keep and sents:
                total_keep += 1
            for idx, it in enumerate(sents, start=1):
                quote = str(it.get("quote") or "").strip()
                if not quote:
                    continue
                span = it.get("span") if isinstance(it.get("span"), dict) else {}
                ev_block_id = f"{raw_block_id}::ev{idx}"
                block_type = "paragraph_summary"
                if para_type == "abstract":
                    block_type = "abstract"
                out_rec = {
                    "block_id": ev_block_id,
                    "doc_id": doc_id,
                    "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                    "source_uri": source_uri,
                    "lang": lang,
                    "block_type": block_type,
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

            if total_in == 1 or total_in % 100 == 0:
                print(
                    f"[evidence_blocks] processed={total_in} kept_blocks={total_keep} out_evidence_blocks={total_out} "
                    f"line_no={line_no}",
                    flush=True,
                )

    print(
        f"[evidence_blocks] done in={total_in} kept_blocks={total_keep} out_evidence_blocks={total_out} out={out_path}",
        flush=True,
    )
    if docs_meta_f is not None:
        docs_meta_f.close()
        print(f"[evidence_blocks] wrote_docs_meta={docs_meta_path} docs={len(seen_docs)}", flush=True)


if __name__ == "__main__":
    main()


