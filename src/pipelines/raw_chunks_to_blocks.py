"""
Pipeline：Raw Context chunks（4096-token）→ Memory blocks（256-token）JSONL

严格对齐 PRD/raw context构建流程.md：
- KV Bank 的最小单位是 256-token blocks（不是 raw text）
- blocks 必须携带 parent_chunk_id 与文档级 metadata（doc_id、段落类型、疾病、日期等）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_blocks_from_raw_chunks(
    *,
    raw_chunks_jsonl: Path,
    out_blocks_jsonl: Path,
    tokenizer_name_or_path: str,
    block_tokens: int = 256,
    block_overlap_tokens: int = 64,
    drop_last_incomplete_block: bool = True,
    trust_remote_code: bool = True,
) -> int:
    """
    对每条 raw chunk 的 text：tokenizer 级切分为 256-token blocks。
    """

    from transformers import AutoTokenizer  # type: ignore

    if block_overlap_tokens < 0:
        raise ValueError("block_overlap_tokens must be >= 0")
    if block_overlap_tokens >= block_tokens:
        raise ValueError("block_overlap_tokens must be < block_tokens")
    stride = int(block_tokens - block_overlap_tokens)

    print(
        f"[raw_chunks_to_blocks] Loading tokenizer: {tokenizer_name_or_path} "
        f"(block_tokens={block_tokens} block_overlap_tokens={block_overlap_tokens} stride={stride})",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, trust_remote_code=bool(trust_remote_code))
    out_blocks_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_blocks_jsonl.open("w", encoding="utf-8") as out:
        for idx, rec in enumerate(_read_jsonl(raw_chunks_jsonl), start=1):
            doc_id = rec.get("doc_id")
            chunk_id = rec.get("chunk_id")
            text = rec.get("text") or ""
            meta = rec.get("metadata") or {}
            lang = rec.get("lang")

            ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), stride):
                block_ids = ids[i : i + block_tokens]
                if drop_last_incomplete_block and len(block_ids) < block_tokens:
                    continue
                block_text = tok.decode(block_ids, skip_special_tokens=True)
                # block-level table ids: re-scan the block text (table markers are preserved in text)
                # Note: if a marker is split across blocks it may be missed; acceptable for now.
                import re
                table_ids = [
                    int(x)
                    for x in re.findall(
                        # Robust to tokenizer decode inserting spaces between punctuation
                        r"<\s*!\s*-\s*-\s*table\s*:\s*(\d+)\s*-\s*-\s*>",
                        block_text,
                        flags=re.IGNORECASE,
                    )
                ]
                # propagate + override
                meta2 = dict(meta)
                meta2["block_window_in_chunk_tokens"] = [int(i), int(i + len(block_ids))]
                meta2["block_overlap_tokens"] = int(block_overlap_tokens)
                meta2["block_stride_tokens"] = int(stride)
                if isinstance(meta2.get("tables"), dict):
                    t = dict(meta2["tables"])
                    t["table_ids"] = sorted(set((t.get("table_ids") or []) + table_ids))
                    meta2["tables"] = t
                else:
                    meta2["tables"] = {"table_ids": table_ids}
                out_rec = {
                    "block_id": f"{chunk_id}_t{i}-{i+len(block_ids)}",
                    "parent_chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "lang": lang,
                    "token_count": int(len(block_ids)),
                    "block_tokens": int(block_tokens),
                    "text": block_text,
                    "metadata": meta2,
                }
                out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1
            if idx % 50 == 0:
                print(f"[raw_chunks_to_blocks] processed_chunks={idx} written_blocks={written}", flush=True)

    print(f"[raw_chunks_to_blocks] done written_blocks={written} out={out_blocks_jsonl}", flush=True)
    return written


