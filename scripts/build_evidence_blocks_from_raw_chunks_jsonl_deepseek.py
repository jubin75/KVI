"""
Build extractive evidence blocks from raw_chunks.jsonl using DeepSeek.

Why raw_chunks (vs blocks)?
- blocks.jsonl is 256-token windowing, which can fragment sentences and mix intents.
- raw_chunks.jsonl preserves longer paragraph structure (after PDF cleanup), producing cleaner evidence sentences.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.llm_filter.extractive_evidence import (  # type: ignore
        DeepSeekExtractiveEvidence,
        ExtractiveEvidenceConfig,
    )
except ModuleNotFoundError:
    from src.llm_filter.extractive_evidence import DeepSeekExtractiveEvidence, ExtractiveEvidenceConfig  # type: ignore

try:
    from external_kv_injection.src.llm_filter.doc_meta_extractor import (  # type: ignore
        DeepSeekDocMetaExtractor,
        DocMetaExtractorConfig,
    )
except ModuleNotFoundError:
    from src.llm_filter.doc_meta_extractor import DeepSeekDocMetaExtractor, DocMetaExtractorConfig  # type: ignore


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _split_paragraphs(text: str) -> List[str]:
    # Split on blank lines. Keep reasonably sized paragraphs.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return [p for p in parts if len(p) >= 30]

def _strip_table_markdown(text: str) -> str:
    """
    Remove table markdown blocks so DS doesn't extract from tables.
    """
    out_lines: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            out_lines.append("")
            continue
        if s.lower().startswith("<!-- table:"):
            continue
        # markdown table rows
        if s.startswith("|") and "|" in s:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def _approx_token_count(text: str) -> int:
    """
    Cheap, tokenizer-free proxy for 'token_count' used in QA/inspection.
    Counts alnum "words" and individual CJK characters as units.
    """
    t = str(text or "").strip()
    if not t:
        return 0
    units = re.findall(r"[A-Za-z0-9]+|[\u4E00-\u9FFF]", t)
    return int(len(units))


_REF_PREFIX_RE = re.compile(r"^\s*(references?|bibliography)\s*[:：]?\s*$", re.IGNORECASE)
_CITATION_RE = re.compile(r"\b[A-Z][A-Za-z\-']+\s+et al\.?,?\s*\(?\d{4}\)?")
_YEAR_PAREN_RE = re.compile(r"\(\d{4}\)")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.IGNORECASE)
_JOURNALISH_RE = re.compile(r"\b(vol\.?|issue|pages?|doi|journal|proceedings)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
#  Section detection — inline-aware, cross-chunk, handles numbered headings
# ---------------------------------------------------------------------------
#  PDF headings appear in many forms:
#   "Abstract"  /  "ABSTRACT"  /  "1. Introduction"  /  "3.1 Results"
#   "Abstract. This study..."  (inline, heading on same line as content)
#   "Abstract\nThis study..."  (heading on first line of paragraph)
#   "Results and Discussion"   (multi-word)
# ---------------------------------------------------------------------------

_SECTION_KEYWORDS: dict = {
    "abstract": "abstract",
    "summary": "abstract",        # some journals use "Summary" instead of "Abstract"
    "introduction": "introduction",
    "background": "introduction",
    "methods": "methods",
    "method": "methods",
    "materials": "methods",       # "Materials and Methods"
    "experimental": "methods",    # "Experimental Section"
    "results": "results",
    "result": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusions": "conclusion",
    "conclusion": "conclusion",
    "concluding": "conclusion",
    "references": "references",
    "bibliography": "references",
    "literature": "references",   # "Literature Cited"
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "acknowledgement": "acknowledgements",
    "acknowledgment": "acknowledgements",
    "funding": "acknowledgements",
    "conflicts": "acknowledgements",
    "competing": "acknowledgements",
    "declaration": "acknowledgements",
    "ethics": "acknowledgements",
    "data": "acknowledgements",   # "Data Availability"
    "supporting": "supplementary",
    "supplementary": "supplementary",
    "appendix": "supplementary",
}

# Matches section heading at the START of a line, with optional number prefix.
# Captures the first keyword for lookup in _SECTION_KEYWORDS.
# Works for both standalone headings ("Abstract") and inline ("Abstract. This study...")
_INLINE_HEADING_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+[\.\)]\s*)*"          # optional number/roman prefix: "1.", "3.1.", "IV)"
    r"([A-Za-z]+)"                              # first keyword (captured)
    r"(?:\s+(?:and\s+)?[A-Za-z]+)*"            # optional extra words: "and Methods", "and Discussion"
    r"\s*[:：.\n]",                              # followed by : or . or newline (signals end of heading)
    re.MULTILINE,
)

# Stricter standalone heading: heading is the ENTIRE line (possibly with number)
_STANDALONE_HEADING_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+[\.\)]\s*)*"
    r"([A-Za-z]+(?:\s+(?:and\s+)?[A-Za-z]+)*)"
    r"\s*[:：]?\s*$",
    re.MULTILINE,
)


def _match_heading_line(line: str) -> str:
    """If *line* is or starts with a section heading keyword, return section type; else ''."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return ""
    # Standalone heading (line IS the heading, e.g. "Abstract", "3.1 Results")
    if len(stripped) <= 60:
        m = _STANDALONE_HEADING_RE.match(stripped)
        if m:
            kw = m.group(1).split()[0].lower()
            sec = _SECTION_KEYWORDS.get(kw, "")
            if sec:
                return sec
    # Inline heading (heading + content, e.g. "Abstract. This study...")
    m2 = _INLINE_HEADING_RE.match(stripped)
    if m2:
        kw = m2.group(1).lower()
        sec = _SECTION_KEYWORDS.get(kw, "")
        if sec:
            return sec
    return ""


def _build_section_markers(full_text: str) -> list:
    """Scan ALL lines of *full_text* for section headings.

    Returns list of (char_offset, section_type) sorted by offset.
    """
    markers: list = []
    offset = 0
    for line in full_text.split("\n"):
        sec = _match_heading_line(line)
        if sec:
            markers.append((offset, sec))
        offset += len(line) + 1   # +1 for \n

    # Fallback for PDFs where heading is inline in one long line, e.g.:
    # "... Published online ... 2024 Abstract Severe fever with ..."
    # We only add this fallback for "abstract" to avoid over-triggering on
    # common words like "results" in body sentences.
    if not any(sec == "abstract" for _, sec in markers):
        m_abs = re.search(r"\babstract\b", full_text, flags=re.IGNORECASE)
        if m_abs:
            markers.append((int(m_abs.start()), "abstract"))

    markers.sort(key=lambda x: x[0])
    return markers


_INLINE_SECTION_ANYWHERE_RE = re.compile(
    r"\b("
    r"abstract|summary|"
    r"introduction|background|"
    r"methods?|materials?\s+and\s+methods?|experimental|"
    r"results?|discussion|conclusions?|"
    r"references?|bibliography|literature\s+cited|"
    r"funding|disclosure|author\s+contributions?|"
    r"conflicts?\s+of\s+interest|competing\s+interests?|"
    r"data\s+availability|ethics\s+statement|"
    r"acknowledg?ements?"
    r")\b[:\s]",
    flags=re.IGNORECASE,
)


def _section_from_keyword(keyword: str) -> str:
    k = re.sub(r"\s+", " ", str(keyword or "").strip().lower())
    if k in ("abstract", "summary"):
        return "abstract"
    if k in ("introduction", "background"):
        return "introduction"
    if k in ("method", "methods", "materials", "experimental", "materials and methods"):
        return "methods"
    if k in ("result", "results", "findings"):
        return "results"
    if k in ("discussion",):
        return "discussion"
    if k in ("conclusion", "conclusions", "concluding"):
        return "conclusion"
    if k in ("reference", "references", "bibliography", "literature cited"):
        return "references"
    if k in (
        "funding",
        "disclosure",
        "author contribution",
        "author contributions",
        "conflict of interest",
        "conflicts of interest",
        "competing interest",
        "competing interests",
        "data availability",
        "ethics statement",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
    ):
        return "acknowledgements"
    return ""


def _inline_section_and_content(para: str) -> tuple:
    """
    Detect section heading embedded inside a long paragraph and optionally
    return content trimmed from that heading onward.
    """
    p = str(para or "").strip()
    if not p:
        return "", p
    m = _INLINE_SECTION_ANYWHERE_RE.search(p)
    if not m:
        return "", p
    sec = _section_from_keyword(m.group(1))
    if not sec:
        return "", p
    # If heading appears early, trim preface metadata to improve extraction quality.
    # Keep original if trim would become too short.
    if m.start() <= 500:
        tail = p[m.end():].strip()
        if len(tail) >= 40:
            return sec, tail
    return sec, p


_ABSTRACT_END_RE = re.compile(
    r"\b("
    r"keywords?|introduction|background|methods?|materials?\s+and\s+methods?|results?|"
    r"funding|disclosure|author\s+contributions?|"
    r"conflicts?\s+of\s+interest|competing\s+interests?|"
    r"data\s+availability|ethics\s+statement|"
    r"acknowledg?ements?|references?|bibliography"
    r")\b",
    flags=re.IGNORECASE,
)
_METHODISH_SENT_RE = re.compile(
    r"\b("
    r"rna\s+was\s+extracted|dna\s+was\s+extracted|rt-?pcr|real-?time\s+pcr|"
    r"according\s+to\s+the\s+manufacturer|using\s+the\s+\w+\s+kit|"
    r"incubated|centrifuged|buffer|aliquot|assay|protocol|"
    r"mice\s+were\s+infected|sample[s]?\s+were\s+collected"
    r")\b",
    flags=re.IGNORECASE,
)

_ABSTRACT_BAD_RE = re.compile(
    r"\b("
    r"funding|disclosure|author\s+contributions?|"
    r"conflicts?\s+of\s+interest|competing\s+interests?|"
    r"data\s+availability|ethics\s+statement|"
    r"acknowledg?ements?|references?|bibliography|"
    r"received\s*:|accepted\s*:|published\s+online"
    r")\b",
    flags=re.IGNORECASE,
)


def _truncate_abstract_noise_tail(text: str) -> str:
    """Trim abstract text before trailing non-abstract sections if present."""
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return ""
    m = _ABSTRACT_END_RE.search(t)
    if m and m.start() > 120:
        t = t[: m.start()].strip()
    return t


def _extract_abstract_window(full_text: str) -> str:
    """Extract text window starting from the first 'Abstract' keyword.

    Designed for linearized PDF text where heading and body may appear inline:
    "... Published online ... Abstract Severe fever ... Keywords ..."
    """
    t = str(full_text or "")
    if not t.strip():
        return ""
    m = re.search(r"\babstract\b", t, flags=re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    tail = t[start:].strip()
    if len(tail) < 40:
        return ""
    # Cut before next major section marker if found.
    end_m = _ABSTRACT_END_RE.search(tail)
    if end_m and end_m.start() > 120:
        tail = tail[: end_m.start()]
    tail = re.sub(r"\s+", " ", tail).strip()
    return tail


def _split_sentences_simple(text: str) -> list:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？;；])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _pick_abstract_seed_sentence(window: str) -> str:
    """Pick one high-value abstract sentence for fallback abstract block."""
    sents = _split_sentences_simple(window)
    if not sents:
        return ""
    # Prefer non-method, complete, medium-length sentence.
    for s in sents:
        if len(s) < 45 or len(s) > 420:
            continue
        if _METHODISH_SENT_RE.search(s):
            continue
        if not re.search(r'[.!?。！？;；)\]）」\'\"]\s*$', s):
            continue
        return s
    # Fallback: first complete sentence
    for s in sents:
        if len(s) >= 40 and re.search(r'[.!?。！？;；)\]）」\'\"]\s*$', s):
            return s
    return sents[0][:420].strip()


def _pick_generic_seed_sentence(text: str) -> str:
    """Pick one generic high-level sentence from raw chunk text.

    Used when explicit 'Abstract' keyword is missing. We still prefer
    non-method, complete sentences and avoid very short title-like fragments.
    """
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return ""
    sents = _split_sentences_simple(t)
    for s in sents:
        if len(s) < 55 or len(s) > 420:
            continue
        if _METHODISH_SENT_RE.search(s):
            continue
        if _DOI_RE.search(s):
            continue
        # Skip mostly title-like fragments (few words, no verb-ish cues)
        if len(s.split()) < 8:
            continue
        return s
    for s in sents:
        if len(s) >= 45:
            return s[:420].strip()
    return ""


def _infer_sections_for_paragraphs(
    full_text: str,
    paras: list,
    initial_section: str = "",
) -> list:
    """Determine the section type for each paragraph by scanning ALL headings
    in the original chunk text (including short heading lines that were
    dropped by _split_paragraphs).

    Returns a list of section types parallel to *paras*.
    """
    markers = _build_section_markers(full_text)

    sections: list = []
    for para in paras:
        # Find paragraph position in full text
        snippet = para[:min(60, len(para))]
        pos = full_text.find(snippet)

        # Determine active section at this position
        current = initial_section
        if pos >= 0:
            for mpos, msec in markers:
                if mpos <= pos:
                    current = msec
                else:
                    break
        elif markers:
            # Couldn't locate paragraph; use last known heading
            current = markers[-1][1]

        sections.append(current)

    return sections


def _is_low_value_paragraph(para: str) -> bool:
    """Reject only the most obvious noise.  Everything else goes to DeepSeek.

    DeepSeek handles semantic summarisation — it can extract meaningful
    sentences from paragraphs that contain citations or methodology details.
    We only pre-filter content that is structurally unparseable by DeepSeek:
    pure reference entries, concatenated citation blocks, etc.
    """
    p = str(para or "").strip()
    if not p:
        return True
    # Explicit "References" / "Bibliography" section headers
    if _REF_PREFIX_RE.match(p):
        return True
    # Extremely short fragments (< 40 chars) — too short to be useful
    if len(p) < 40:
        return True
    # Dense reference block: many (year) citations packed together
    year_count = len(_YEAR_PAREN_RE.findall(p))
    if year_count >= 6:
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_chunks_jsonl", required=True, help="Input raw_chunks.jsonl")
    p.add_argument("--out_jsonl", required=True, help="Output evidence blocks jsonl (blocks.evidence.jsonl)")
    p.add_argument("--out_docs_meta_jsonl", default="", help="Optional docs.meta.jsonl (doc-level metadata)")
    p.add_argument("--kb_id", default="", help="Optional kb_id (topic/knowledgebase id) attached to outputs")
    p.add_argument("--topic_goal", required=True, help="Topic goal text (used to guide extraction)")
    p.add_argument("--max_sentences_per_paragraph", type=int, default=3)
    p.add_argument("--max_paragraphs", type=int, default=0, help="If >0, only process first N paragraphs (debug)")
    p.add_argument(
        "--allowed_paragraph_types",
        default="abstract,introduction,results,discussion,conclusion",
        help="Comma-separated allowlist of section types to keep",
    )
    p.add_argument(
        "--keep_figure_captions",
        action="store_true",
        help="Keep figure captions as evidence (default: drop)",
    )
    p.add_argument("--deepseek_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    args = p.parse_args()

    in_path = Path(str(args.raw_chunks_jsonl))
    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = DeepSeekExtractiveEvidence(
        ExtractiveEvidenceConfig(
            deepseek_base_url=str(args.deepseek_base_url),
            deepseek_model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
            max_sentences=int(args.max_sentences_per_paragraph),
            strict_noise_filter=True,
        )
    )

    meta_extractor = DeepSeekDocMetaExtractor(
        DocMetaExtractorConfig(
            deepseek_base_url=str(args.deepseek_base_url),
            deepseek_model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
            max_chars=9000,
        )
    )

    total_chunks = 0
    total_paras = 0
    kept_paras = 0
    out_blocks = 0
    filtered_by_section = 0
    filtered_by_noise = 0
    filtered_heading = 0

    docs_meta_path = Path(str(args.out_docs_meta_jsonl)) if str(args.out_docs_meta_jsonl or "").strip() else None
    docs_meta_f = docs_meta_path.open("w", encoding="utf-8") if docs_meta_path else None
    seen_docs: Set[str] = set()

    # Figure caption heuristic: keep as its own evidence block (no DS) when detected.
    _FIG_CAP_RE = re.compile(r"^(Figure|Fig\.)\s*\d+[\s:：\-]", flags=re.IGNORECASE)
    allowed_para_types = {
        x.strip().lower()
        for x in str(args.allowed_paragraph_types or "").split(",")
        if x.strip()
    }

    # Cross-chunk section tracking: inherit section label across chunks of the same document
    doc_section_state: Dict[str, str] = {}
    # Abstract quality/coverage tracking
    doc_has_abstract_block: Dict[str, bool] = {}
    doc_abstract_seed: Dict[str, Dict[str, Any]] = {}
    doc_generic_seed: Dict[str, Dict[str, Any]] = {}

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in _read_jsonl(in_path):
            total_chunks += 1
            doc_id = str(rec.get("doc_id") or "")
            chunk_id = str(rec.get("chunk_id") or "")
            source_uri = rec.get("source_uri", None)
            lang = rec.get("lang", None)
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            para_type = str(meta.get("paragraph_type") or "").strip().lower()

            if docs_meta_f is not None and doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                # Use the first chunk text as a snippet for doc-level metadata extraction.
                snippet = str(rec.get("text") or "")[:9000]
                try:
                    doc_meta = meta_extractor.extract(doc_id=doc_id, source_uri=str(source_uri or ""), pdf_snippet=snippet)
                except Exception as exc:
                    print(f"[meta_extract] FAILED doc={doc_id}: {exc}", file=sys.stderr)
                    fallback_title = Path(str(source_uri or "")).stem or doc_id
                    doc_meta = {"title": fallback_title, "journal": None, "doi": None, "publication_year": None, "published_at": None, "authors": []}
                docs_meta_f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                            "source_uri": source_uri,
                            "meta": doc_meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            txt = _strip_table_markdown(str(rec.get("text") or ""))
            # Prepare one fallback abstract window per document from raw PDF text.
            if doc_id and doc_id not in doc_abstract_seed:
                abs_window = _extract_abstract_window(txt)
                if abs_window:
                    window_text = _truncate_abstract_noise_tail(abs_window)
                    if len(window_text) >= 120:
                        doc_abstract_seed[doc_id] = {
                            "text": window_text[:2500],
                            "chunk_id": chunk_id,
                            "source_uri": source_uri,
                            "lang": lang,
                            "meta": meta,
                        }
            # Generic fallback seed for documents without explicit 'Abstract' heading.
            if doc_id and doc_id not in doc_generic_seed:
                seed_g = _pick_generic_seed_sentence(txt[:2500])
                if seed_g:
                    doc_generic_seed[doc_id] = {
                        "text": seed_g,
                        "chunk_id": chunk_id,
                        "source_uri": source_uri,
                        "lang": lang,
                        "meta": meta,
                    }
            paras = _split_paragraphs(txt)

            # --- Section inference: scan ALL lines of full chunk text ---
            # This catches short heading lines (e.g. "Abstract") that were
            # dropped by _split_paragraphs (requires ≥30 chars).
            inherited = doc_section_state.get(doc_id, "")
            if para_type:
                inferred_sections = [para_type] * len(paras)
            else:
                inferred_sections = _infer_sections_for_paragraphs(
                    txt, paras, initial_section=inherited,
                )
            # Update document section state with the last detected section in this chunk
            markers = _build_section_markers(txt)
            if markers:
                doc_section_state[doc_id] = markers[-1][1]
            elif inferred_sections:
                last_sec = inferred_sections[-1]
                if last_sec:
                    doc_section_state[doc_id] = last_sec

            for p_idx, para in enumerate(paras):
                total_paras += 1
                if int(args.max_paragraphs) > 0 and total_paras > int(args.max_paragraphs):
                    break

                # Effective section: metadata para_type > inferred from headings
                effective_type = para_type or inferred_sections[p_idx]
                para_for_extract = para

                # Skip standalone section heading paragraphs (short, only a heading)
                if not para_type and len(para.strip()) < 60 and _match_heading_line(para.strip()):
                    filtered_heading += 1
                    continue

                # Fallback for inline headings in long paragraphs:
                # "... 2024 Abstract Severe fever ..."
                inline_sec, para_inline = _inline_section_and_content(para)
                if inline_sec:
                    # Always allow inline heading to override inherited section.
                    # This prevents "abstract" state from leaking into
                    # Funding/Disclosure/Acknowledgements paragraphs.
                    effective_type = inline_sec
                if para_inline:
                    para_for_extract = para_inline

                # --- Section filter ---
                if allowed_para_types and effective_type and effective_type not in allowed_para_types:
                    filtered_by_section += 1
                    continue

                if _is_low_value_paragraph(para_for_extract):
                    filtered_by_noise += 1
                    continue

                # Abstract handling: keep full abstract paragraph/window directly.
                # Do NOT send abstract to DeepSeek sentence summarization.
                if effective_type == "abstract":
                    abs_text = _truncate_abstract_noise_tail(para_for_extract)
                    if len(abs_text) < 80:
                        continue
                    # If the paragraph still looks like non-abstract metadata,
                    # don't keep it as abstract; let normal DS flow handle it.
                    if _ABSTRACT_BAD_RE.search(abs_text):
                        effective_type = "paragraph"
                    else:
                        ev_block_id = f"{chunk_id}_p{p_idx}::abs"
                        out_rec = {
                            "block_id": ev_block_id,
                            "doc_id": doc_id,
                            "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                            "source_uri": source_uri,
                            "lang": lang,
                            "block_type": "abstract",
                            "text": abs_text,
                            "token_count": int(_approx_token_count(abs_text)),
                            "metadata": {
                                "from_raw_chunk_id": chunk_id,
                                "paragraph_index": int(p_idx),
                                "span": {"char_start": None, "char_end": None},
                                "relevance": None,
                                "claim": None,
                                "raw_chunk_metadata": meta,
                                "direct_abstract": True,
                            },
                        }
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        out_blocks += 1
                        kept_paras += 1
                        if doc_id:
                            doc_has_abstract_block[doc_id] = True
                        continue

                # Figure captions: drop by default (low value for knowledge extraction)
                if _FIG_CAP_RE.match(para_for_extract.strip()):
                    if not bool(args.keep_figure_captions):
                        continue
                    ev_block_id = f"{chunk_id}_p{p_idx}::cap"
                    out_rec = {
                        "block_id": ev_block_id,
                        "doc_id": doc_id,
                        "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                        "source_uri": source_uri,
                        "lang": lang,
                        "block_type": "figure_caption",
                        "text": para_for_extract.strip(),
                        "token_count": int(_approx_token_count(para_for_extract.strip())),
                        "metadata": {"from_raw_chunk_id": chunk_id, "paragraph_index": int(p_idx), "raw_chunk_metadata": meta},
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    out_blocks += 1
                    kept_paras += 1
                    continue

                res = extractor.extract(
                    topic_goal=str(args.topic_goal),
                    raw_block_text=para_for_extract,
                    section_hint=effective_type or "paragraph",
                )
                sents = res.get("evidence_sentences", []) if isinstance(res.get("evidence_sentences"), list) else []
                if not sents:
                    continue
                kept_paras += 1
                for s_idx, it in enumerate(sents, start=1):
                    quote = str(it.get("quote") or "").strip()
                    if not quote:
                        continue
                    # Reject truncated sentences: must end with sentence-ending punctuation
                    if not re.search(r'[.!?。！？;；)\]）」\'\"]\s*$', quote):
                        continue
                    span = it.get("span") if isinstance(it.get("span"), dict) else {}
                    ev_block_id = f"{chunk_id}_p{p_idx}::ev{s_idx}"
                    block_type = "paragraph_summary"
                    if effective_type == "abstract":
                        block_type = "abstract"
                    elif effective_type in ("results", "conclusion", "discussion", "introduction"):
                        block_type = effective_type
                    # Keep abstract blocks focused on consensus-level summary:
                    # method-like sentences are downgraded to paragraph_summary.
                    if block_type == "abstract" and _METHODISH_SENT_RE.search(quote):
                        block_type = "paragraph_summary"
                    out_rec = {
                        "block_id": ev_block_id,
                        "doc_id": doc_id,
                        "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                        "source_uri": source_uri,
                        "lang": lang,
                        "block_type": block_type,
                        "text": quote,
                        "token_count": int(_approx_token_count(quote)),
                        "metadata": {
                            "from_raw_chunk_id": chunk_id,
                            "paragraph_index": int(p_idx),
                            "span": {"char_start": span.get("char_start"), "char_end": span.get("char_end")},
                            "relevance": it.get("relevance"),
                            "claim": it.get("claim"),
                            "raw_chunk_metadata": meta,
                        },
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    out_blocks += 1
                    if block_type == "abstract" and doc_id:
                        doc_has_abstract_block[doc_id] = True

                if total_paras == 1 or total_paras % 200 == 0:
                    print(
                        f"[evidence_from_raw_chunks] chunks={total_chunks} paras={total_paras} kept_paras={kept_paras} "
                        f"out_blocks={out_blocks}",
                        flush=True,
                    )
            if int(args.max_paragraphs) > 0 and total_paras >= int(args.max_paragraphs):
                break

        # Ensure abstract coverage: for each seen doc without extracted abstract,
        # inject one synthetic abstract block from (1) abstract seed, else (2) generic seed.
        added_fallback_abstract = 0
        for did in sorted(seen_docs):
            if doc_has_abstract_block.get(did, False):
                continue
            seed = doc_abstract_seed.get(did) or doc_generic_seed.get(did)
            if not seed:
                continue
            txt_seed = str(seed.get("text") or "").strip()
            if not txt_seed:
                continue
            txt_seed = _truncate_abstract_noise_tail(txt_seed)
            if len(txt_seed) < 80:
                continue
            if _ABSTRACT_BAD_RE.search(txt_seed):
                continue
            block_id = f"{seed.get('chunk_id', 'chunk')}_fallback::abs"
            out_rec = {
                "block_id": block_id,
                "doc_id": did,
                "kb_id": (str(args.kb_id) if str(args.kb_id or "").strip() else None),
                "source_uri": seed.get("source_uri"),
                "lang": seed.get("lang"),
                "block_type": "abstract",
                "text": txt_seed,
                "token_count": int(_approx_token_count(txt_seed)),
                "metadata": {
                    "from_raw_chunk_id": seed.get("chunk_id"),
                    "paragraph_index": -1,
                    "span": {"char_start": None, "char_end": None},
                    "relevance": None,
                    "claim": None,
                    "raw_chunk_metadata": seed.get("meta") if isinstance(seed.get("meta"), dict) else {},
                    "synthetic_fallback": True,
                },
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_blocks += 1
            added_fallback_abstract += 1
            doc_has_abstract_block[did] = True

    print(
        f"[evidence_from_raw_chunks] done chunks={total_chunks} paras={total_paras} kept_paras={kept_paras} "
        f"out_blocks={out_blocks} out={out_path}",
        flush=True,
    )
    print(
        f"[evidence_from_raw_chunks] filter_breakdown: "
        f"heading_skipped={filtered_heading}  "
        f"section_rejected={filtered_by_section}  "
        f"noise_rejected={filtered_by_noise}  "
        f"allowed_sections={sorted(allowed_para_types) if allowed_para_types else 'all'}",
        flush=True,
    )
    print(
        f"[evidence_from_raw_chunks] abstract_coverage: "
        f"seed_docs={len(doc_abstract_seed)}  "
        f"generic_seed_docs={len(doc_generic_seed)}  "
        f"docs_with_abstract={sum(1 for _d, ok in doc_has_abstract_block.items() if ok)}  "
        f"fallback_added={added_fallback_abstract}",
        flush=True,
    )
    if docs_meta_f is not None:
        docs_meta_f.close()
        print(f"[evidence_from_raw_chunks] wrote_docs_meta={docs_meta_path} docs={len(seen_docs)}", flush=True)


if __name__ == "__main__":
    main()


