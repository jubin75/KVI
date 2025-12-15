"""
cleaning_and_dedupe: 清洗/语言检测/去重（可运行实现）

说明
- 这是一个不依赖额外库的“工程可用”版本：
  - normalize_text
  - detect_lang（粗略）
  - simhash64（用于去重键）
  - dedupe_by_hash（chunk 级去重）
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> str:
    # demo 级：中/英二分类
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _tokenize_for_hash(text: str) -> List[str]:
    # 简单分词：字母数字串 + 中文单字
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
    return [t.lower() for t in tokens if t]


def simhash64(text: str) -> str:
    """
    返回 64-bit simhash hex string。
    """

    tokens = _tokenize_for_hash(text)
    if not tokens:
        # fallback to sha
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little").to_bytes(8, "little").hex()

    v = [0] * 64
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).digest()
        x = int.from_bytes(h[:8], "little", signed=False)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1

    fp = 0
    for i in range(64):
        if v[i] > 0:
            fp |= 1 << i
    return fp.to_bytes(8, "little").hex()


def hamming_distance_hex64(a_hex: str, b_hex: str) -> int:
    a = int.from_bytes(bytes.fromhex(a_hex), "little", signed=False)
    b = int.from_bytes(bytes.fromhex(b_hex), "little", signed=False)
    x = a ^ b
    # popcount
    return int(x.bit_count())


def quality_score(text: str, *, ocr_confidence: Optional[float] = None) -> float:
    """
    0~1 质量分：极简启发式
    """

    t = normalize_text(text)
    if len(t) < 50:
        return 0.0
    # excessive repeated chars
    if re.search(r"(.)\1{8,}", t):
        return 0.2
    score = 1.0
    if ocr_confidence is not None:
        score *= max(0.0, min(1.0, float(ocr_confidence)))
    return float(score)


@dataclass
class DedupeStats:
    total_in: int
    total_out: int
    duplicates: int


def dedupe_by_hash(
    records: Iterable[Dict],
    *,
    text_key: str = "text",
    hash_key: str = "dedupe_hash",
    near_dup_hamming_threshold: Optional[int] = None,
) -> Tuple[List[Dict], DedupeStats]:
    seen: Dict[str, str] = {}
    out: List[Dict] = []
    total = 0
    dup = 0
    for r in records:
        total += 1
        text = normalize_text(str(r.get(text_key, "") or ""))
        h = r.get(hash_key) or simhash64(text)
        r[hash_key] = h
        if h in seen:
            dup += 1
            continue
        if near_dup_hamming_threshold is not None and seen:
            # demo 近重复：线性扫描已见集合（适合小规模）；生产级可换 LSH/分桶
            is_near = False
            for hh in seen.keys():
                if hamming_distance_hex64(h, hh) <= near_dup_hamming_threshold:
                    is_near = True
                    break
            if is_near:
                dup += 1
                continue
        seen[h] = str(r.get("chunk_id") or r.get("block_id") or total)
        r[text_key] = text
        out.append(r)
    return out, DedupeStats(total_in=total, total_out=len(out), duplicates=dup)



