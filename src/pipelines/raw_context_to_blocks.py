"""
Pipeline：Raw Context(text) → 4096-token chunks → 256-token memory blocks

严格遵循 PRD/多步注入的工程实现.md：
- Raw context 仅用于构建 KV Bank，不直接参与 attention
- 4096-token chunks（可 overlap=256）
- 每个 chunk 再切 256-token memory blocks（KV Bank 存 blocks）

输出（JSONL）
每条记录是一条 memory block：
- block_id, parent_chunk_id
- token_count（必须=256，最后一块可 <256 但默认丢弃以满足严格 256；可配置）
- text（可选保留，用于 debug/引用，不用于直接注入）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RawToBlocksConfig:
    tokenizer_name_or_path: str
    chunk_tokens: int = 4096
    chunk_overlap: int = 256
    block_tokens: int = 256
    drop_last_incomplete_block: bool = True
    trust_remote_code: bool = True


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def build_memory_blocks_from_raw_text(
    *,
    raw_text: str,
    source_id: str,
    out_jsonl: Path,
    cfg: RawToBlocksConfig,
) -> int:
    """
    把 raw_text 切成 memory blocks 并写出 JSONL。
    """

    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True, trust_remote_code=bool(cfg.trust_remote_code))
    ids = tok(raw_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    n = len(ids)

    _ensure_dir(out_jsonl)
    written = 0

    # 4096-token chunks with overlap
    step = cfg.chunk_tokens - cfg.chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be < chunk_tokens")

    chunk_idx = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for start in range(0, n, step):
            end = min(n, start + cfg.chunk_tokens)
            chunk_ids = ids[start:end]
            if not chunk_ids:
                break
            parent_chunk_id = f"{source_id}_chunk{chunk_idx}_t{start}-{end}"
            chunk_idx += 1

            # split chunk into 256-token blocks (no overlap inside chunk by default)
            for bstart in range(0, len(chunk_ids), cfg.block_tokens):
                bend = min(len(chunk_ids), bstart + cfg.block_tokens)
                block_ids = chunk_ids[bstart:bend]
                if cfg.drop_last_incomplete_block and len(block_ids) < cfg.block_tokens:
                    continue
                block_id = f"{parent_chunk_id}_block{bstart//cfg.block_tokens}"
                block_text = tok.decode(block_ids, skip_special_tokens=True)
                rec = {
                    "block_id": block_id,
                    "parent_chunk_id": parent_chunk_id,
                    "source_id": source_id,
                    "token_count": int(len(block_ids)),
                    "block_tokens": int(cfg.block_tokens),
                    "text": block_text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if end >= n:
                break

    return written


