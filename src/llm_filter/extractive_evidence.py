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
task: Extract up to {max_sentences} high-value, reusable evidence sentences for the topic_goal.

Rules (MUST follow):
1) Output JSON only. No markdown, no explanation.
2) Extractive-only: each quote MUST be an exact substring of raw_block_text (verbatim; you may ignore line breaks when selecting the substring, but do not change words).
3) Provide span offsets (char_start, char_end) into the raw_block_text for each quote.
4) Prefer sentences that are actionable or answerable: key findings, guideline recommendations, definitions, transmission routes, diagnosis criteria, treatment recommendations, contraindications, dosages, outcomes.
5) If nothing is reusable/high-value, return keep=false and a short reject_reason.
6) Do NOT create new medical advice; you may extract recommendations ONLY if they already appear in the raw_block_text.

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
    max_sentences: int = 3
    max_chars: int = 7000


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

    def extract(self, *, topic_goal: str, raw_block_text: str) -> Dict[str, Any]:
        raw_block_text = str(raw_block_text or "")
        clipped = raw_block_text[: int(self.cfg.max_chars)]
        user = USER_TEMPLATE.format(
            topic_goal=str(topic_goal),
            max_sentences=int(self.cfg.max_sentences),
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
        for it in sents[: int(self.cfg.max_sentences)]:
            if not isinstance(it, dict):
                continue
            quote = str(it.get("quote") or "").strip()
            if not quote:
                continue
            # Try to validate quote appears in raw text (whitespace-normalized fallback).
            if quote not in clipped:
                if _normalize_ws(quote) not in norm_raw:
                    # reject invalid quote (non-extractive hallucination)
                    continue
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


