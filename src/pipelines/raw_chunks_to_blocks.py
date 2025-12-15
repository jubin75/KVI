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
    drop_last_incomplete_block: bool = True,
) -> int:
    """
    对每条 raw chunk 的 text：tokenizer 级切分为 256-token blocks。
    """

    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    out_blocks_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_blocks_jsonl.open("w", encoding="utf-8") as out:
        for rec in _read_jsonl(raw_chunks_jsonl):
            doc_id = rec.get("doc_id")
            chunk_id = rec.get("chunk_id")
            text = rec.get("text") or ""
            meta = rec.get("metadata") or {}

            ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), block_tokens):
                block_ids = ids[i : i + block_tokens]
                if drop_last_incomplete_block and len(block_ids) < block_tokens:
                    continue
                block_text = tok.decode(block_ids, skip_special_tokens=True)
                # block-level table ids: re-scan the block text (table markers are preserved in text)
                # Note: if a marker is split across blocks it may be missed; acceptable for now.
                import re

                table_ids = [int(x) for x in re.findall(r"<!--\s*table:(\d+)\s*-->", block_text)]
                # propagate + override
                meta2 = dict(meta)
                if isinstance(meta2.get("tables"), dict):
                    t = dict(meta2["tables"])
                    t["table_ids"] = sorted(set((t.get("table_ids") or []) + table_ids))
                    meta2["tables"] = t
                else:
                    meta2["tables"] = {"table_ids": table_ids}
                out_rec = {
                    "block_id": f"{chunk_id}_block{i//block_tokens}",
                    "parent_chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "token_count": int(len(block_ids)),
                    "block_tokens": int(block_tokens),
                    "text": block_text,
                    "metadata": meta2,
                }
                out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1

    return written


