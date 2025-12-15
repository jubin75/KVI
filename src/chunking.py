"""
chunking: token-aware chunking 的最小可运行实现

说明
- 生产级 chunking 应结构感知（标题/表格/图注等），这里提供一个“足够工程化”的版本：
  - 输入文本 + tokenizer
  - 按 token 长度切分（支持 overlap）
  - 可用于 raw_chunks(4096) 或一般 chunks(200-500) 的构建
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass(frozen=True)
class TokenChunkConfig:
    target_tokens: int
    max_tokens: int
    overlap_tokens: int = 0
    add_special_tokens: bool = False


def chunk_text_by_tokenizer(
    *,
    text: str,
    tokenizer: Any,
    cfg: TokenChunkConfig,
) -> List[Tuple[int, int, str]]:
    """
    返回每个 chunk：(start_token_idx, end_token_idx, chunk_text)
    """

    enc = tokenizer(text, return_tensors=None, add_special_tokens=cfg.add_special_tokens)
    ids: List[int] = enc["input_ids"]
    n = len(ids)
    if n == 0:
        return []

    step = cfg.target_tokens - cfg.overlap_tokens
    if step <= 0:
        raise ValueError("overlap_tokens must be < target_tokens")

    out: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + cfg.max_tokens)
        # try to be near target_tokens if possible
        if end - start > cfg.target_tokens:
            end = min(n, start + cfg.target_tokens)
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        out.append((start, end, chunk_text))
        if end >= n:
            break
        start += step
    return out



