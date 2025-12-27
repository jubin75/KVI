"""
Build extractive evidence blocks from raw_chunks.jsonl using DeepSeek.

Why raw_chunks (vs blocks)?
- blocks.jsonl is 256-token windowing, which can fragment sentences and mix intents.
- raw_chunks.jsonl preserves longer paragraph structure (after PDF cleanup), producing cleaner evidence sentences.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _split_paragraphs(text: str) -> List[str]:
    # Split on blank lines. Keep reasonably sized paragraphs.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return [p for p in parts if len(p) >= 30]


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
    p.add_argument("--raw_chunks_jsonl", required=True, help="Input raw_chunks.jsonl")
    p.add_argument("--out_jsonl", required=True, help="Output evidence blocks jsonl (blocks.evidence.jsonl)")
    p.add_argument("--topic_goal", required=True, help="Topic goal text (used to guide extraction)")
    p.add_argument("--max_sentences_per_paragraph", type=int, default=2)
    p.add_argument("--max_paragraphs", type=int, default=0, help="If >0, only process first N paragraphs (debug)")
    p.add_argument("--deepseek_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    args = p.parse_args()

    in_path = Path(str(args.raw_chunks_jsonl))
    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = DeepSeekExtractiveEvidence(
        ExtractiveEvidenceConfig(
            deepseek_base_url=str(args.deepseek_base_url),
            deepseek_model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
            max_sentences=int(args.max_sentences_per_paragraph),
        )
    )

    total_chunks = 0
    total_paras = 0
    kept_paras = 0
    out_blocks = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in _read_jsonl(in_path):
            total_chunks += 1
            doc_id = str(rec.get("doc_id") or "")
            chunk_id = str(rec.get("chunk_id") or "")
            source_uri = rec.get("source_uri", None)
            lang = rec.get("lang", None)
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}

            txt = str(rec.get("text") or "")
            paras = _split_paragraphs(txt)
            for p_idx, para in enumerate(paras):
                total_paras += 1
                if int(args.max_paragraphs) > 0 and total_paras > int(args.max_paragraphs):
                    break
                res = extractor.extract(topic_goal=str(args.topic_goal), raw_block_text=para)
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

                if total_paras == 1 or total_paras % 200 == 0:
                    print(
                        f"[evidence_from_raw_chunks] chunks={total_chunks} paras={total_paras} kept_paras={kept_paras} "
                        f"out_blocks={out_blocks}",
                        flush=True,
                    )
            if int(args.max_paragraphs) > 0 and total_paras >= int(args.max_paragraphs):
                break

    print(
        f"[evidence_from_raw_chunks] done chunks={total_chunks} paras={total_paras} kept_paras={kept_paras} "
        f"out_blocks={out_blocks} out={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()


