"""
Pipeline: PDF → ChunkStore (runnable implementation, demo-friendly)

Goal
- Input a PDF directory, extract text and split into chunks, write out JSONL ChunkStore.

Dependencies
- PyMuPDF (fitz)

Limitations
- This implementation primarily covers "selectable text PDFs". OCR for scanned PDFs is not implemented in this demo (production can integrate PaddleOCR).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _require_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyMuPDF is required. Install: pymupdf") from e
    return fitz


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _simple_lang_detect(text: str) -> str:
    # demo: rough detection
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _simhash_64(text: str) -> str:
    # demo: non-strict simhash, just a stable hash placeholder (production can use real simhash/minhash)
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False).__format__("016x")


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    返回 [(page_number_1based, text), ...]
    """

    fitz = _require_pymupdf()
    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_text_by_tokens(
    text: str,
    *,
    target_tokens: int = 350,
    max_tokens: int = 900,
    overlap_ratio: float = 0.15,
) -> List[str]:
    """
    Demo chunking:
    - Uses approximate token count = whitespace-tokenized word count (production should use tokenizer counting)
    """

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = text.split(" ")
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, int(target_tokens * (1 - overlap_ratio)))

    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        # Control target size: aim close to target_tokens, but not exceeding max_tokens
        if end - start > target_tokens:
            end = min(len(words), start + target_tokens)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    source_uri: str
    source_type: str
    page_range: Optional[Tuple[int, int]]
    section_path: List[str]
    text: str
    lang: str
    quality_score: float
    ocr_used: bool
    ocr_confidence: Optional[float]
    dedupe_hash: str
    created_at: str
    dataset_version: str

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "page_range": list(self.page_range) if self.page_range else None,
            "section_path": self.section_path,
            "text": self.text,
            "lang": self.lang,
            "quality_score": self.quality_score,
            "ocr_used": self.ocr_used,
            "ocr_confidence": self.ocr_confidence,
            "dedupe_hash": self.dedupe_hash,
            "created_at": self.created_at,
            "dataset_version": self.dataset_version,
        }


def build_chunkstore_from_pdfs(
    *,
    pdf_dir: Path,
    output_jsonl: Path,
    dataset_version: str,
    target_tokens: int = 350,
    max_tokens: int = 900,
    overlap_ratio: float = 0.15,
) -> int:
    pdfs = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            doc_id = pdf.stem
            pages = extract_pdf_pages(pdf)
            for page_no, page_text in pages:
                # Quality filter (demo)
                page_text = page_text.strip()
                if len(page_text) < 50:
                    continue
                chunks = chunk_text_by_tokens(
                    page_text,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_ratio=overlap_ratio,
                )
                for j, ch in enumerate(chunks):
                    lang = _simple_lang_detect(ch)
                    rec = ChunkRecord(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_p{page_no}_c{j}",
                        source_uri=str(pdf),
                        source_type="pdf",
                        page_range=(page_no, page_no),
                        section_path=[f"page_{page_no}"],
                        text=ch,
                        lang=lang,
                        quality_score=1.0,
                        ocr_used=False,
                        ocr_confidence=None,
                        dedupe_hash=_simhash_64(ch),
                        created_at=_now_iso(),
                        dataset_version=dataset_version,
                    )
                    out.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                    written += 1

    return written


