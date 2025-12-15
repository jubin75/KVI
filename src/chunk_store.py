"""
chunk_store: ChunkStore 的最小可运行实现（JSONL）

目标
- 提供一个无额外依赖的 ChunkStore：读写 JSONL、字段校验、过滤迭代。
- 与 docs/10_data_spec.md 的字段语义对齐（字段可扩展，但必需字段必须存在）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


REQUIRED_FIELDS = ("doc_id", "chunk_id", "source_uri", "source_type", "text", "dataset_version")


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    source_uri: str
    source_type: str = "pdf"
    page_range: Optional[List[int]] = None
    section_path: Optional[List[str]] = None
    text: str = ""
    lang: Optional[str] = None
    quality_score: float = 1.0
    ocr_used: bool = False
    ocr_confidence: Optional[float] = None
    dedupe_hash: Optional[str] = None
    created_at: Optional[str] = None
    dataset_version: str = "v0"
    extra: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        for k in REQUIRED_FIELDS:
            v = getattr(self, k, None)
            if v is None or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"ChunkRecord missing required field: {k}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", None) or {}
        # keep top-level keys flat
        for k, v in extra.items():
            if k not in d:
                d[k] = v
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChunkRecord":
        known = {f.name for f in ChunkRecord.__dataclass_fields__.values()}  # type: ignore
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        base["extra"] = extra or None
        rec = ChunkRecord(**base)  # type: ignore[arg-type]
        rec.validate()
        return rec


def write_jsonl(path: Path, records: Iterable[ChunkRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            r.validate()
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[ChunkRecord]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield ChunkRecord.from_dict(json.loads(line))


def filter_chunks(
    chunks: Iterable[ChunkRecord],
    *,
    langs: Optional[Sequence[str]] = None,
    min_quality: float = 0.0,
    source_types: Optional[Sequence[str]] = None,
) -> Iterator[ChunkRecord]:
    langs_set = set(langs) if langs else None
    st_set = set(source_types) if source_types else None
    for c in chunks:
        if c.quality_score < min_quality:
            continue
        if langs_set is not None and c.lang is not None and c.lang not in langs_set:
            continue
        if st_set is not None and c.source_type not in st_set:
            continue
        yield c



