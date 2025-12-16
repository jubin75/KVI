"""
Pipeline：PDF 文献目录 → Raw Context（4096-token chunks）JSONL

严格对齐 PRD/raw context构建流程.md：
1) 原始文档处理：PDF→文本；清洗噪声；分章节/分段（本实现为 demo 级段落化）
2) Chunk 化：每篇文献切成 4096-token chunks（overlap 128–256）
   - 为每个 chunk 生成 metadata（文献ID、段落类型、疾病、日期等）

重要边界
- Raw context 仅用于建库（KV Bank 构建），不直接参与 attention 注入。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..cleaning_and_dedupe import normalize_text
from ..pdf_ingestion import ingest_pdf
from ..llm_filter.knowledge_filter import DeepSeekKnowledgeFilter, KnowledgeFilterConfig

def _require_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyMuPDF is required. Install: pymupdf") from e
    return fitz


def extract_pdf_text(pdf_path: Path) -> str:
    # backward-compat: keep for callers, but prefer ingest_pdf() below
    fitz = _require_pymupdf()
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        parts.append(txt)
    doc.close()
    return "\n".join(parts)


def clean_noise(text: str) -> str:
    """
    demo 级清洗：
    - 合并空白
    - 去掉明显的“References/参考文献”后的尾段（粗略）
    - 去掉图例/引用/公式等噪声（医疗场景通常不如表格重要）
    - 注意：表格内容由专用表格抽取提供，不依赖正文里的 “Table X” 行
    """

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # strip figure captions (keep table content elsewhere)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^(Figure|Fig\.|Table)\s*\d+", s, flags=re.IGNORECASE):
            # drop captions lines; real tables are extracted separately
            continue
        lines.append(s)
    text = "\n".join(lines)

    # truncate after references (rough)
    m = re.search(r"\n(References|参考文献)\n", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()]

    # remove common inline citations: [1], [1-3], (Smith et al., 2020)
    text = re.sub(r"\[[0-9,\-\s]+\]", "", text)
    text = re.sub(r"\(([A-Z][A-Za-z]+ et al\.,?\s*\d{4}[a-z]?)\)", "", text)

    # remove latex-like equations blocks (very rough): lines containing lots of math symbols
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        math_ratio = sum(ch in "=+*/^_{}[]<>" for ch in s) / max(1, len(s))
        if math_ratio > 0.12:
            continue
        kept.append(s)
    text = "\n".join(kept)

    # normalize blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    # 段落化：按空行切段，并剔除极短段
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paras if len(p) >= 30]


def _extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return m.group(1) if m else None


def _infer_disease(text: str) -> Optional[str]:
    # demo：关键词占位（生产级可用词典/NER）
    for kw in ["SFTS", "SFTSV", "severe fever with thrombocytopenia", "传染病", "流感", "COVID", "SARS"]:
        if kw.lower() in text.lower():
            return kw
    return None


def _infer_para_type(text: str) -> str:
    # demo：粗略段落类型
    t = text.lower()
    if "abstract" in t[:200]:
        return "abstract"
    if "introduction" in t[:200]:
        return "introduction"
    if "method" in t[:200]:
        return "methods"
    if "result" in t[:200]:
        return "results"
    if "conclusion" in t[:200]:
        return "conclusion"
    return "paragraph"


@dataclass(frozen=True)
class RawChunkConfig:
    tokenizer_name_or_path: str
    chunk_tokens: int = 4096
    chunk_overlap: int = 256
    ocr: str = "auto"  # off|auto|on
    extract_tables: bool = True
    trust_remote_code: bool = True
    verbose: bool = True
    # extraction guardrails
    fail_on_empty_extract: bool = True
    min_extracted_chars: int = 300
    # knowledge filter (DeepSeek)
    knowledge_filter: bool = False
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    strict_drop_uncertain: bool = False


def chunk_tokens_4096(
    *,
    token_ids: List[int],
    chunk_tokens: int,
    chunk_overlap: int,
) -> List[Tuple[int, int, List[int]]]:
    step = chunk_tokens - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be < chunk_tokens")
    out: List[Tuple[int, int, List[int]]] = []
    n = len(token_ids)
    for start in range(0, n, step):
        end = min(n, start + chunk_tokens)
        chunk = token_ids[start:end]
        if not chunk:
            break
        out.append((start, end, chunk))
        if end >= n:
            break
    return out


def build_raw_context_chunks_from_pdf_dir(
    *,
    pdf_dir: Path,
    out_jsonl: Path,
    cfg: RawChunkConfig,
) -> int:
    """
    遍历 pdf_dir 下所有 PDF，输出 raw_chunks.jsonl。
    每条记录是一条 4096-token chunk（来自段落串联后的全文 token stream）。
    """

    import time
    from transformers import AutoTokenizer  # type: ignore

    if cfg.verbose:
        print(f"[pdf_to_raw_chunks] Loading tokenizer: {cfg.tokenizer_name_or_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_or_path, use_fast=True, trust_remote_code=bool(cfg.trust_remote_code)
    )
    pdfs = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        raise RuntimeError(f"No .pdf files found under pdf_dir={pdf_dir}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        if cfg.verbose:
            print(f"[pdf_to_raw_chunks] Found {len(pdfs)} PDFs under {pdf_dir}", flush=True)
        for pdf in pdfs:
            t0 = time.time()
            doc_id = pdf.stem
            # 1) ingestion（支持 OCR）
            doc = ingest_pdf(pdf, ocr=cfg.ocr, extract_tables=cfg.extract_tables)
            doc_text_chars = int(sum(len((p.text or "").strip()) for p in doc.pages))
            doc_table_chars = int(sum(len((p.tables_markdown or "").strip()) for p in doc.pages))
            if cfg.verbose:
                print(
                    f"[pdf_to_raw_chunks] pdf={pdf.name} pages={len(doc.pages)} ocr_used={doc.ocr_used} "
                    f"text_chars={doc_text_chars} table_chars={doc_table_chars}",
                    flush=True,
                )
            if cfg.fail_on_empty_extract and (doc_text_chars + doc_table_chars) < int(cfg.min_extracted_chars):
                raise RuntimeError(
                    "PDF extraction produced near-empty text. "
                    f"pdf={pdf} pages={len(doc.pages)} ocr_used={doc.ocr_used} "
                    f"text_chars={doc_text_chars} table_chars={doc_table_chars}. "
                    "Common fixes: set --ocr on (or auto), ensure system 'tesseract' is installed for OCR, "
                    "and verify PyMuPDF can extract text from this PDF."
                )
            # 把表格（markdown）追加到每页末尾，确保表格信息进入 raw context
            page_texts: List[str] = []
            # doc-level table metadata（用于 chunk/block 的结构化 metadata）
            doc_tables: List[Dict[str, Any]] = []
            for p in doc.pages:
                t = p.text or ""
                if p.tables_markdown:
                    t = t + "\n\n" + p.tables_markdown
                if p.tables_meta:
                    for tm in p.tables_meta:
                        doc_tables.append(
                            {
                                "page": int(p.page_number),
                                **tm,
                            }
                        )
                page_texts.append(t)
            raw = "\n\n".join(page_texts)

            # 2) cleaning + paragraphing
            cleaned = clean_noise(raw)
            cleaned = normalize_text(cleaned)
            paras = split_paragraphs(cleaned)
            filter_stats = None
            if cfg.knowledge_filter:
                kf = DeepSeekKnowledgeFilter(
                    KnowledgeFilterConfig(
                        deepseek_base_url=cfg.deepseek_base_url,
                        deepseek_model=cfg.deepseek_model,
                        api_key_env=cfg.deepseek_api_key_env,
                        strict_drop_uncertain=cfg.strict_drop_uncertain,
                        verbose=bool(cfg.verbose),
                    )
                )
                paras, filter_stats = kf.filter_paragraphs(paras)
                if cfg.verbose:
                    kept = int(filter_stats.get("kept", 0)) if isinstance(filter_stats, dict) else -1
                    dropped = int(filter_stats.get("dropped", 0)) if isinstance(filter_stats, dict) else -1
                    print(f"[pdf_to_raw_chunks] knowledge_filter kept={kept} dropped={dropped}", flush=True)
            if cfg.fail_on_empty_extract and not paras:
                raise RuntimeError(
                    "No paragraphs remained after cleaning/filtering. "
                    f"pdf={pdf} ocr_used={doc.ocr_used} text_chars={doc_text_chars} table_chars={doc_table_chars}. "
                    "Common fixes: rerun without --knowledge_filter to validate extraction, or lower filtering aggressiveness."
                )
            full_text = "\n\n".join(paras)

            enc = tok(full_text, return_tensors=None, add_special_tokens=False)
            ids: List[int] = enc["input_ids"]

            chunks = chunk_tokens_4096(token_ids=ids, chunk_tokens=cfg.chunk_tokens, chunk_overlap=cfg.chunk_overlap)
            for i, (start, end, cids) in enumerate(chunks):
                text = tok.decode(cids, skip_special_tokens=True)
                # chunk-level table metadata：通过 table marker 识别本 chunk 引用了哪些 table_id
                table_ids = [int(x) for x in re.findall(r"<!--\s*table:(\d+)\s*-->", text)]
                # map to structured meta (page/rows/cols/header_hash)
                table_meta = [t for t in doc_tables if int(t.get("table_id", -1)) in set(table_ids)]
                rec = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk{i}_t{start}-{end}",
                    "source_uri": str(pdf),
                    "ocr_used": bool(doc.ocr_used),
                    "chunk_tokens": int(len(cids)),
                    "chunk_window": [int(start), int(end)],
                    "text": text,
                    "metadata": {
                        "extraction_stats": {
                            "pages": int(len(doc.pages)),
                            "text_chars": int(doc_text_chars),
                            "table_chars": int(doc_table_chars),
                            "ocr_used": bool(doc.ocr_used),
                        },
                        "paragraph_type": _infer_para_type(text),
                        "disease": _infer_disease(text),
                        "date": _extract_year(text),
                        "tables": {
                            "doc_table_count": int(len(doc_tables)),
                            "table_ids": table_ids,
                            "table_meta": table_meta,
                        },
                        "knowledge_filter": filter_stats,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            if cfg.verbose:
                dt = time.time() - t0
                print(f"[pdf_to_raw_chunks] wrote_chunks={len(chunks)} total_written={written} time_sec={dt:.1f}", flush=True)

    return written


