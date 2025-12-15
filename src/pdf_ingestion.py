"""
pdf_ingestion: PDF→按页文本抽取（可运行实现）

说明
- 依赖 PyMuPDF（fitz）
- OCR 在本仓库中暂不实现（可扩展），但会标记 ocr_used/ocr_confidence 字段
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


def _require_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyMuPDF is required. Install: pymupdf") from e
    return fitz


@dataclass(frozen=True)
class PdfPage:
    page_number: int  # 1-based
    text: str
    tables_markdown: Optional[str] = None
    tables_meta: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class PdfDocument:
    source_uri: str
    pages: List[PdfPage]
    ocr_used: bool = False
    ocr_confidence: Optional[float] = None


def _hash_str(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _tables_to_markdown_and_meta(tables: List[List[List[Optional[str]]]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    pdfplumber tables: list of rows(list of cells)
    输出为 markdown（多表拼接）+ tables_meta。
    """

    out: List[str] = []
    metas: List[Dict[str, Any]] = []
    for ti, table in enumerate(tables):
        if not table or len(table) < 2:
            continue
        # normalize cells
        norm_rows: List[List[str]] = []
        for row in table:
            norm_rows.append([("" if c is None else str(c).strip()) for c in row])
        header = norm_rows[0]
        # drop empty header table
        if not any(h for h in header):
            continue
        sep = ["---"] * len(header)
        rows = len(norm_rows)
        cols = len(header)
        header_join = " | ".join(header)
        metas.append(
            {
                "table_id": ti,
                "rows": rows,
                "cols": cols,
                "header": header,
                "header_hash": _hash_str(header_join),
            }
        )
        out.append(f"\n\n<!-- table:{ti} -->\n")
        out.append("| " + " | ".join(header) + " |")
        out.append("| " + " | ".join(sep) + " |")
        for r in norm_rows[1:]:
            # pad to header length
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            out.append("| " + " | ".join(r[: len(header)]) + " |")
    return ("\n".join(out).strip() or ""), metas


def ingest_pdf(
    pdf_path: Path,
    *,
    ocr: str = "off",  # off|auto|on
    extract_tables: bool = True,
) -> PdfDocument:
    fitz = _require_pymupdf()
    doc = fitz.open(pdf_path)
    pages: List[PdfPage] = []
    total_chars = 0
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        total_chars += len(text.strip())
        pages.append(PdfPage(page_number=i + 1, text=text, tables_markdown=None, tables_meta=None))
    doc.close()

    # naive scan detection
    avg_chars = (total_chars / max(1, len(pages))) if pages else 0.0
    likely_scanned = avg_chars < 50

    if ocr in {"auto", "on"} and likely_scanned:
        # 实际 OCR：使用 tesseract（通过 pytesseract）
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OCR requested but pytesseract/Pillow not available. "
                "Install: pip install pytesseract pillow, and install the system 'tesseract' binary."
            ) from e

        fitz = _require_pymupdf()
        doc = fitz.open(pdf_path)
        ocr_pages: List[PdfPage] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(img)
            ocr_pages.append(PdfPage(page_number=i + 1, text=text, tables_markdown=None, tables_meta=None))
        doc.close()
        return PdfDocument(source_uri=str(pdf_path), pages=ocr_pages, ocr_used=True, ocr_confidence=None)

    # table extraction for text PDFs (pdfplumber)
    if extract_tables:
        try:
            import pdfplumber  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("pdfplumber is required for table extraction. Install: pdfplumber") from e

        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                md, metas = _tables_to_markdown_and_meta(tables)
                if i < len(pages) and md:
                    pages[i] = PdfPage(
                        page_number=pages[i].page_number,
                        text=pages[i].text,
                        tables_markdown=md,
                        tables_meta=metas or None,
                    )

    return PdfDocument(source_uri=str(pdf_path), pages=pages, ocr_used=False, ocr_confidence=None)



