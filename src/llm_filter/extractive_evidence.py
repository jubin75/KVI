"""
Extractive evidence sentence selection (DeepSeek).

Purpose
- Convert noisy/long raw blocks into short, single-intent evidence sentences that are:
  - strictly copied from the input text (extractive-only)
  - span-located for verifiable provenance
  - suitable for building an "evidence KVBank" with much lower injection noise
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .deepseek_client import DeepSeekClient, DeepSeekClientConfig


def _normalize_ws(s: str) -> str:
    # Keep semantics but make matching easier.
    return re.sub(r"\s+", " ", (s or "")).strip()


def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


SYSTEM_PROMPT = """You are an information extraction engine for medical/scientific text.
You MUST output JSON only.
You MUST NOT paraphrase or summarize.
You MUST ONLY copy exact substrings from the provided raw_block_text into the 'quote' fields.
"""


USER_TEMPLATE = """topic_goal: {topic_goal}
task: Extract up to {max_sentences} COMPLETE evidence sentences for the topic_goal (HIGH RECALL).
section_hint: {section_hint}

Rules (MUST follow):
1) Output JSON only. No markdown, no explanation.
2) Extractive-only: each quote MUST be an exact substring of raw_block_text (verbatim; you may ignore line breaks when selecting the substring, but do not change words).
3) Provide span offsets (char_start, char_end) into the raw_block_text for each quote.
4) HIGH RECALL: If any sentence is relevant to the topic_goal, keep it (even if it is only "supporting" or "background").
   Prefer factual, reusable statements about: transmission routes/vectors, reservoirs/hosts, epidemiology, clinical features/outcomes,
   pathogenesis/mechanisms/immune response, diagnosis, prevention, treatment and guideline recommendations (if present in raw_block_text).
5) Only return keep=false if there is truly NOTHING relevant to the topic_goal in raw_block_text.
6) Do NOT create new medical advice; you may extract recommendations ONLY if they already appear in the raw_block_text.
7) Quotes MUST be complete sentences (not fragments). Prefer quotes that end with sentence punctuation: . ! ? 。 ！ ？
8) Do NOT output table rows or partial clauses.

Return JSON schema:
{{
  "keep": true|false,
  "evidence_sentences": [
    {{
      "quote": "verbatim sentence copied from raw_block_text",
      "relevance": "direct|supporting|background",
      "claim": "the atomic claim supported by the quote (no new facts)",
      "span": {{"char_start": 0, "char_end": 0}}
    }}
  ],
  "reject_reason": "string|null"
}}

raw_block_text:
<<<
{raw_block_text}
>>>
"""


@dataclass(frozen=True)
class ExtractiveEvidenceConfig:
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # Default to 1 to reduce fragmentation; callers can override for recall.
    max_sentences: int = 1
    # Allow a bit more context to increase recall (still bounded to avoid overly long requests).
    max_chars: int = 9000
    # When true, aggressively drop bibliography/header-like noise sentences.
    strict_noise_filter: bool = True


_CITATION_RE = re.compile(r"\b[A-Z][A-Za-z\-']+\s+et al\.?,?\s*\(?\d{4}\)?")
_YEAR_PAREN_RE = re.compile(r"\(\d{4}\)")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.IGNORECASE)
_JOURNALISH_RE = re.compile(
    r"\b(vol\.?|issue|pages?|doi|journal|proceedings|copyright)\b",
    re.IGNORECASE,
)
_REF_PREFIX_RE = re.compile(r"^\s*(references?|bibliography)\s*[:：]?\s*$", re.IGNORECASE)


def _is_low_value_noise_sentence(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    if _REF_PREFIX_RE.match(t):
        return True
    # Strong bibliography patterns
    if _DOI_RE.search(t):
        return True
    if _CITATION_RE.search(t):
        return True
    if _JOURNALISH_RE.search(t):
        return True
    # Citation-heavy lines are usually references/noise for graph extraction.
    years = len(_YEAR_PAREN_RE.findall(t))
    if years >= 3:
        return True
    # Author list style: many commas but little clinical predicate signal.
    comma_count = t.count(",") + t.count("，")
    if comma_count >= 6 and len(t) < 280:
        return True
    return False


class DeepSeekExtractiveEvidence:
    def __init__(self, cfg: ExtractiveEvidenceConfig) -> None:
        self.cfg = cfg
        self.client = DeepSeekClient(
            DeepSeekClientConfig(
                base_url=cfg.deepseek_base_url,
                api_key_env=cfg.api_key_env,
                model=cfg.deepseek_model,
            )
        )

    def extract(self, *, topic_goal: str, raw_block_text: str, section_hint: str = "") -> Dict[str, Any]:
        raw_block_text = str(raw_block_text or "")
        clipped = raw_block_text[: int(self.cfg.max_chars)]
        user = USER_TEMPLATE.format(
            topic_goal=str(topic_goal),
            max_sentences=int(self.cfg.max_sentences),
            section_hint=str(section_hint or "unknown"),
            raw_block_text=clipped,
        )
        out = self.client.chat(system=SYSTEM_PROMPT, user=user, temperature=0.0)
        obj = _safe_parse_json(out) or {"keep": False, "evidence_sentences": [], "reject_reason": "parse_failed"}

        keep = bool(obj.get("keep", False))
        sents = obj.get("evidence_sentences", [])
        if not isinstance(sents, list):
            sents = []

        cleaned: List[Dict[str, Any]] = []
        norm_raw = _normalize_ws(clipped)
        seen: set[str] = set()
        sent_end = re.compile(r"[\.!\?。！？]$")
        for it in sents[: int(self.cfg.max_sentences)]:
            if not isinstance(it, dict):
                continue
            quote = str(it.get("quote") or "").strip()
            if not quote:
                continue
            if self.cfg.strict_noise_filter and _is_low_value_noise_sentence(quote):
                continue
            # Enforce "sentence-like" shape to avoid fragmentary evidence blocks.
            if len(quote) < 20:
                continue
            if len(quote) > 800:
                continue
            if not sent_end.search(quote):
                # Allow rare cases where the PDF has no punctuation; keep only if it's long enough.
                if len(quote) < 80:
                    continue
            # Try to validate quote appears in raw text (whitespace-normalized fallback).
            if quote not in clipped:
                if _normalize_ws(quote) not in norm_raw:
                    # reject invalid quote (non-extractive hallucination)
                    continue
            # Dedupe within one call.
            k = _normalize_ws(quote).lower()
            if k in seen:
                continue
            seen.add(k)
            span = it.get("span") if isinstance(it.get("span"), dict) else {}
            cs = span.get("char_start")
            ce = span.get("char_end")
            cleaned.append(
                {
                    "quote": quote,
                    "relevance": str(it.get("relevance") or "supporting"),
                    "claim": str(it.get("claim") or ""),
                    "span": {"char_start": int(cs) if isinstance(cs, int) else None, "char_end": int(ce) if isinstance(ce, int) else None},
                }
            )

        if not cleaned:
            return {"keep": False, "evidence_sentences": [], "reject_reason": str(obj.get("reject_reason") or "no_valid_quotes")}
        return {"keep": keep or True, "evidence_sentences": cleaned, "reject_reason": None}


