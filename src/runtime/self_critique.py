"""
Self-critique utilities for RIM demos / runtime gating.

This module provides:
- A lightweight heuristic critique (non-generative)
- An optional LLM-generated JSON critique (generative, for demo convenience)

Important:
- In the RIM v0.3 ideal form, critique can be a non-generative classifier.
  The LLM-json option is only for a runnable demo, not a hard architecture requirement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .multistep_injector import MultiStepInjector


@dataclass(frozen=True)
class CritiqueResult:
    is_medical_fact: bool
    confidence: float
    reason: str


def extract_first_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def heuristic_self_critique(question: str, answer: str) -> CritiqueResult:
    q = (question or "").lower()
    a = (answer or "").lower()

    # Generic "medical factual" heuristic (topic-agnostic).
    med_keywords = [
        # EN
        "fda",
        "approved",
        "indication",
        "contraindication",
        "adverse",
        "side effect",
        "dosage",
        "drug",
        "medication",
        "treatment",
        "therapy",
        "clinical",
        "trial",
        # ZH
        "已批准",
        "适应症",
        "禁忌",
        "不良反应",
        "副作用",
        "剂量",
        "用法",
        "药物",
        "药品",
        "治疗",
        "临床",
        "试验",
    ]
    is_med = any((k in q) or (k in question) for k in med_keywords)

    hedges = [
        "不确定",
        "尚不清楚",
        "证据不足",
        "可能",
        "推测",
        "无法确认",
        "i am not sure",
        "i'm not sure",
        "unclear",
        "unknown",
        "no clear evidence",
    ]
    hedge_hits = sum(1 for h in hedges if h in a)

    short = len(answer.strip()) < 80
    conf = 0.85
    conf -= 0.20 * min(3, hedge_hits)
    if short:
        conf -= 0.25
    conf = float(max(0.0, min(1.0, conf)))
    reason = f"heuristic: is_medical_fact={is_med} hedge_hits={hedge_hits} short={short}"
    return CritiqueResult(is_medical_fact=is_med, confidence=conf, reason=reason)


def llm_json_self_critique(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    question: str,
    draft_answer: str,
    device: torch.device,
    past_key_values: Any = None,
    max_new_tokens: int = 128,
) -> Optional[CritiqueResult]:
    """
    Ask the base LLM to produce a strict JSON critique.
    `prompt_text` should be the exact formatted prompt used for the main answer (chat template already applied if needed).
    """
    instr = (
        "You are a strict self-critique module.\n"
        "Given a user question and a draft answer, decide whether the question is a medical factual query, "
        "and estimate confidence of the draft answer.\n"
        "Output ONLY valid JSON with keys: is_medical_fact (boolean), confidence (number 0..1), reason (string).\n"
    )
    payload = f"{instr}\nQuestion:\n{question}\n\nDraft answer:\n{draft_answer}\n"
    # Note: `prompt_text` is kept for API symmetry with callers that already have a formatted prompt,
    # but this critique prompt is always standalone and must not depend on the main answer prompt.
    txt = MultiStepInjector._greedy_generate_with_past_prefix(
        model=model,
        tokenizer=tokenizer,
        prompt=payload,
        device=device,
        past_key_values=past_key_values,
        max_new_tokens=int(max_new_tokens),
        no_repeat_ngram_size=0,
        repetition_penalty=1.05,
    )
    obj = extract_first_json_obj(txt)
    if not isinstance(obj, dict):
        return None
    is_med = bool(obj.get("is_medical_fact"))
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        return None
    conf = float(max(0.0, min(1.0, conf)))
    reason = str(obj.get("reason") or "").strip() or "llm_json"
    return CritiqueResult(is_medical_fact=is_med, confidence=conf, reason=reason)

