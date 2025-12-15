"""
eval: 最小可运行评测工具（检索 Recall@k + 日志汇总）

说明
- 完整 QA 评测/引用正确率需要标注数据集，这里先提供：
  - retrieval_recall_at_k
  - parse_jsonl_logs + latency summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold = set(gold_ids)
    hit = any(x in gold for x in ranked_ids[:k])
    return 1.0 if hit else 0.0


def mean_recall_at_k(samples: Sequence[Tuple[Sequence[str], Sequence[str]]], k: int) -> float:
    if not samples:
        return 0.0
    return sum(recall_at_k(r, g, k) for r, g in samples) / len(samples)


@dataclass
class LatencySummary:
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p))
    return float(xs[idx])


def summarize_latency_ms(values: Sequence[float]) -> LatencySummary:
    xs = [float(v) for v in values]
    if not xs:
        return LatencySummary(count=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0)
    return LatencySummary(
        count=len(xs),
        avg_ms=sum(xs) / len(xs),
        p50_ms=_percentile(xs, 0.50),
        p95_ms=_percentile(xs, 0.95),
    )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



