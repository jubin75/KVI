"""
DeepSeek doc-level metadata extraction.

Outputs a compact JSON object with bibliographic metadata:
- title, journal, doi, publication_year/published_at, authors (optional)

This is used to build `docs.meta.jsonl` for the new two-file evidence format:
- docs.meta.jsonl (doc-level)
- blocks.evidence.jsonl (evidence blocks referencing doc_id)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .deepseek_client import DeepSeekClient, DeepSeekClientConfig


def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _heuristic_meta(text: str, *, source_uri: str, doc_id: str) -> Dict[str, Any]:
    t = str(text or "")
    doi = None
    m = _DOI_RE.search(t)
    if m:
        doi = m.group(1)
    year = None
    my = _YEAR_RE.search(t[:4000])
    if my:
        year = int(my.group(1))
    # fallback title: prefer PDF basename (without suffix), then doc_id
    basename = str(source_uri).split("/")[-1] if str(source_uri or "").strip() else ""
    if basename.lower().endswith(".pdf"):
        basename = basename[:-4]
    title = basename.strip() or str(doc_id or "").strip()
    return {
        "title": title,
        "journal": None,
        "doi": doi,
        "publication_year": year,
        "published_at": None,
        "authors": [],
    }


SYSTEM_PROMPT = """You are a bibliographic metadata extraction engine for scientific PDFs.
You MUST output JSON only.
If a field is unknown, output null (or [] for arrays).
Do NOT hallucinate; only extract from the provided text snippet.
"""

USER_TEMPLATE = """Extract document-level metadata from the following PDF snippet.

Return JSON schema:
{
  "title": "string|null",
  "journal": "string|null",
  "doi": "string|null",
  "publication_year": 2024|null,
  "published_at": "YYYY-MM-DD|null",
  "authors": ["..."]
}

doc_id: {doc_id}
source_uri: {source_uri}

pdf_snippet:
<<<
{snippet}
>>>
"""


@dataclass(frozen=True)
class DocMetaExtractorConfig:
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_chars: int = 9000


class DeepSeekDocMetaExtractor:
    def __init__(self, cfg: DocMetaExtractorConfig) -> None:
        self.cfg = cfg
        self.client = DeepSeekClient(
            DeepSeekClientConfig(base_url=cfg.deepseek_base_url, api_key_env=cfg.api_key_env, model=cfg.deepseek_model)
        )

    def extract(self, *, doc_id: str, source_uri: str, pdf_snippet: str) -> Dict[str, Any]:
        snippet = str(pdf_snippet or "")[: int(self.cfg.max_chars)]
        user = USER_TEMPLATE.format(doc_id=str(doc_id), source_uri=str(source_uri), snippet=snippet)
        out = self.client.chat(system=SYSTEM_PROMPT, user=user, temperature=0.0)
        obj = _safe_parse_json(out)
        if not obj:
            return _heuristic_meta(snippet, source_uri=source_uri, doc_id=doc_id)
        # sanitize
        meta = {
            "title": obj.get("title", None),
            "journal": obj.get("journal", None),
            "doi": obj.get("doi", None),
            "publication_year": obj.get("publication_year", None),
            "published_at": obj.get("published_at", None),
            "authors": obj.get("authors", []) if isinstance(obj.get("authors"), list) else [],
        }
        # fill missing with heuristic (doi/year)
        h = _heuristic_meta(snippet, source_uri=source_uri, doc_id=doc_id)
        if not meta.get("doi"):
            meta["doi"] = h.get("doi")
        if not meta.get("publication_year"):
            meta["publication_year"] = h.get("publication_year")
        if not meta.get("title"):
            meta["title"] = h.get("title")
        return meta

