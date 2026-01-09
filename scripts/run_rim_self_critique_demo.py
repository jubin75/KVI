"""
RIM Demo: Self-Critique gated 2-pass generation (Base -> Critique -> Retrieve+Inject -> Regenerate)

Demo flow (topic-agnostic):
1) Base LLM generates an initial answer (no injection)
2) Self-Critique decides: (low confidence) AND (medical factual question) -> retrieve more
3) Retrieve from a KV Bank (any prebuilt literature/KB KV)
4) Inject KV prefix via past_key_values
5) Second-pass generation with injected prefix
6) Print comparison: without RIM vs with RIM

Notes:
- This demo intentionally uses token-generating self-critique when --self_critique_mode=llm_json.
  The core RIM v0.3 design can keep the critique as a non-generative classifier; this script is a
  runnable illustration and not a hard architectural requirement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import Retriever  # type: ignore
    from external_kv_injection.src.runtime.hf_cache_prefix_injection import (  # type: ignore
        build_past_key_values_prefix,
        stack_ext_kv_items_by_layer,
    )
    from external_kv_injection.src.runtime.multistep_injector import MultiStepInjector  # type: ignore
except ModuleNotFoundError:
    from src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import Retriever  # type: ignore
    from src.runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer  # type: ignore
    from src.runtime.multistep_injector import MultiStepInjector  # type: ignore


@dataclass(frozen=True)
class CritiqueResult:
    is_medical_fact: bool
    confidence: float
    reason: str


def _format_prompt(tokenizer: Any, user_text: str, *, use_chat_template: bool) -> str:
    txt = str(user_text or "").strip()
    if not use_chat_template:
        return txt
    fn = getattr(tokenizer, "apply_chat_template", None)
    if not callable(fn):
        return txt
    try:
        messages = [{"role": "user", "content": txt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[no-any-return]
    except Exception:
        return txt


def _extract_first_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    # best-effort: grab the first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _heuristic_self_critique(question: str, answer: str) -> CritiqueResult:
    q = (question or "").lower()
    a = (answer or "").lower()

    # Generic "medical factual" heuristic (do NOT hardcode any single disease/topic).
    # This is only used when --self_critique_mode=heuristic.
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
    # confidence heuristic: penalize hedging + overly short answers for factual/medical questions
    conf = 0.85
    conf -= 0.20 * min(3, hedge_hits)
    if short:
        conf -= 0.25
    conf = float(max(0.0, min(1.0, conf)))

    reason = f"heuristic: is_medical_fact={is_med} hedge_hits={hedge_hits} short={short}"
    return CritiqueResult(is_medical_fact=is_med, confidence=conf, reason=reason)


def _llm_json_self_critique(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    draft_answer: str,
    device: torch.device,
    use_chat_template: bool,
    max_new_tokens: int = 128,
) -> Optional[CritiqueResult]:
    instr = (
        "You are a strict self-critique module.\n"
        "Given a user question and a draft answer, decide whether the question is a medical factual query, "
        "and estimate confidence of the draft answer.\n"
        "Output ONLY valid JSON with keys: is_medical_fact (boolean), confidence (number 0..1), reason (string).\n"
    )
    payload = f"{instr}\nQuestion:\n{question}\n\nDraft answer:\n{draft_answer}\n"
    prompt = _format_prompt(tokenizer, payload, use_chat_template=use_chat_template)
    txt = MultiStepInjector._greedy_generate_with_past_prefix(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        past_key_values=None,
        max_new_tokens=int(max_new_tokens),
        no_repeat_ngram_size=0,
        repetition_penalty=1.05,
    )
    obj = _extract_first_json_obj(txt)
    if not isinstance(obj, dict):
        return None
    is_med = bool(obj.get("is_medical_fact"))
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        conf = float("nan")
    if conf != conf:  # NaN check
        return None
    conf = float(max(0.0, min(1.0, conf)))
    reason = str(obj.get("reason") or "").strip() or "llm_json"
    return CritiqueResult(is_medical_fact=is_med, confidence=conf, reason=reason)


def _resolve_topic_kv_dir(*, topic: str, topic_work_dir: str) -> Path:
    twd = Path(str(topic_work_dir))
    t = str(topic).strip()
    if not t:
        return twd / "UNKNOWN" / "kvbank_blocks"
    # Try common variants; do NOT hardcode specific topics.
    candidates = [
        twd / t,
        twd / t / "work",
        twd / t.lower(),
        twd / t.lower() / "work",
        twd / t.upper(),
        twd / t.upper() / "work",
    ]
    for b in candidates:
        for opt in [b / "kvbank_blocks", b / "kvbank_blocks_v2"]:
            if (opt / "manifest.json").exists():
                return opt
    # fall back to the first expected path; FaissKVBank.load will provide hints
    return candidates[0] / "kvbank_blocks"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompt", required=True, help="User question")
    p.add_argument("--use_chat_template", action="store_true", help="Use tokenizer.apply_chat_template if available")

    p.add_argument("--kv_dir", default=None, help="Path to KVBank dir (manifest.json etc.)")
    p.add_argument("--topic", default=None, help="Optional topic mode (any string; resolves <topic_work_dir>/<topic>/kvbank_blocks)")
    p.add_argument("--topic_work_dir", default=None, help="Topic work dir (used with --topic)")

    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model for retrieval query embedding")
    p.add_argument("--domain_encoder_max_length", type=int, default=256)

    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--layers", type=str, default="0,1,2,3")
    p.add_argument("--max_new_tokens_base", type=int, default=192)
    p.add_argument("--max_new_tokens_rim", type=int, default=192)

    p.add_argument("--self_critique_mode", choices=["heuristic", "llm_json"], default="heuristic")
    p.add_argument("--critique_max_new_tokens", type=int, default=128)
    p.add_argument("--critique_conf_threshold", type=float, default=0.55, help="Trigger retrieval if confidence < threshold")
    p.add_argument("--force_rim", action="store_true", help="Force retrieval+injection regardless of critique")
    args = p.parse_args()

    # Resolve KV bank path
    kv_dir: Optional[str] = args.kv_dir
    if kv_dir is None:
        if not args.topic or not args.topic_work_dir:
            raise SystemExit("Missing --kv_dir. Either pass --kv_dir or use --topic + --topic_work_dir.")
        kv_dir = str(_resolve_topic_kv_dir(topic=str(args.topic), topic_work_dir=str(args.topic_work_dir)))

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else None)
    model.to(device)
    model.eval()

    # --- Pass 1: Base LLM (no injection) ---
    user_prompt = str(args.prompt)
    prompt1 = _format_prompt(tok, user_prompt, use_chat_template=bool(args.use_chat_template))
    base_answer = MultiStepInjector._greedy_generate_with_past_prefix(
        model=model,
        tokenizer=tok,
        prompt=prompt1,
        device=device,
        past_key_values=None,
        max_new_tokens=int(args.max_new_tokens_base),
        no_repeat_ngram_size=12,
        repetition_penalty=1.08,
    )

    # --- Self-critique gate ---
    if str(args.self_critique_mode) == "llm_json":
        crit = _llm_json_self_critique(
            model=model,
            tokenizer=tok,
            question=user_prompt,
            draft_answer=base_answer,
            device=device,
            use_chat_template=bool(args.use_chat_template),
            max_new_tokens=int(args.critique_max_new_tokens),
        )
        if crit is None:
            crit = _heuristic_self_critique(user_prompt, base_answer)
            crit_mode = "heuristic_fallback"
        else:
            crit_mode = "llm_json"
    else:
        crit = _heuristic_self_critique(user_prompt, base_answer)
        crit_mode = "heuristic"

    trigger = bool(args.force_rim) or (crit.is_medical_fact and float(crit.confidence) < float(args.critique_conf_threshold))

    # --- Pass 2: Retrieve + Inject + Regenerate ---
    rim_answer = ""
    retrieved_debug: Dict[str, Any] = {}
    retrieved_ids: List[str] = []
    if trigger:
        bank = FaissKVBank.load(Path(str(kv_dir)))
        retriever = Retriever(bank)
        enc = DomainEncoder(
            DomainEncoderConfig(
                model_name_or_path=str(args.domain_encoder_model),
                max_length=int(args.domain_encoder_max_length),
                normalize=True,
                device=str(device),
            )
        )
        query_vec = enc.encode(user_prompt)[0]
        rr = retriever.search(query_vec, top_k=int(args.top_k), filters=None, query_text=user_prompt)
        retrieved_debug = dict(rr.debug or {})

        layer_ids = [int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""]
        dtype = next(model.parameters()).dtype
        ext_by_layer: Dict[int, Any] = {}
        for li in layer_ids:
            ext_by_layer[li] = stack_ext_kv_items_by_layer(
                items=rr.items,
                layer_id=int(li),
                batch_size=1,
                device=device,
                dtype=dtype,
            )
        past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
        retrieved_ids = [str(it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")) for it in rr.items]
        retrieved_ids = [x for x in retrieved_ids if x]

        rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tok,
            prompt=prompt1,
            device=device,
            past_key_values=past_key_values,
            max_new_tokens=int(args.max_new_tokens_rim),
            no_repeat_ngram_size=12,
            repetition_penalty=1.08,
        )

    # --- Print comparison ---
    print("\n=== 无 RIM（Base LLM）===\n")
    print(base_answer.strip())
    print("\n=== RIM Self-Critique ===\n")
    print(
        json.dumps(
            {
                "mode": crit_mode,
                "is_medical_fact": bool(crit.is_medical_fact),
                "confidence": float(crit.confidence),
                "threshold": float(args.critique_conf_threshold),
                "trigger_retrieve": bool(trigger),
                "reason": str(crit.reason),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if trigger:
        print("\n=== RIM 检索（KV Bank）===\n")
        print(json.dumps({"kv_dir": str(kv_dir), "top_k": int(args.top_k), "retriever_debug": retrieved_debug}, ensure_ascii=False, indent=2))
        if retrieved_ids:
            print("\nretrieved_ids(top_k):")
            for x in retrieved_ids[: int(args.top_k)]:
                print("-", x)
        print("\n=== 有 RIM（注入 KV → 第二轮生成）===\n")
        print(rim_answer.strip())
    else:
        print("\n=== 有 RIM ===\n")
        print("(本次未触发检索/注入：Self-Critique 认为置信度足够或非医学事实问题；可用 --force_rim 强制演示)")


if __name__ == "__main__":
    main()

