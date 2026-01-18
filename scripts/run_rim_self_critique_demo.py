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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.pattern_retriever import PatternRetriever  # type: ignore
    from external_kv_injection.src.retriever import Retriever  # type: ignore
    from external_kv_injection.src.rim import RIM, RIMConfig  # type: ignore
    from external_kv_injection.src.runtime.hf_cache_prefix_injection import (  # type: ignore
        build_past_key_values_prefix,
        stack_ext_kv_items_by_layer,
    )
    from external_kv_injection.src.runtime.kv_relevance import logit_delta_vs_zero_prefix  # type: ignore
    from external_kv_injection.src.runtime.kvi2_runtime import KVI2Config, KVI2Runtime  # type: ignore
    from external_kv_injection.src.runtime.multistep_injector import MultiStepInjector  # type: ignore
    from external_kv_injection.src.runtime.self_critique import (  # type: ignore
        CritiqueResult,
        heuristic_self_critique,
        llm_json_self_critique,
    )
except ModuleNotFoundError:
    from src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.pattern_retriever import PatternRetriever  # type: ignore
    from src.retriever import Retriever  # type: ignore
    from src.rim import RIM, RIMConfig  # type: ignore
    from src.runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer  # type: ignore
    from src.runtime.kv_relevance import logit_delta_vs_zero_prefix  # type: ignore
    from src.runtime.kvi2_runtime import KVI2Config, KVI2Runtime  # type: ignore
    from src.runtime.multistep_injector import MultiStepInjector  # type: ignore
    from src.runtime.self_critique import CritiqueResult, heuristic_self_critique, llm_json_self_critique  # type: ignore


def _kv_id(it: Any) -> str:
    meta = getattr(it, "meta", None) or {}
    return str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")


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


def _extract_keywords(text: str) -> List[str]:
    raw = str(text or "")
    kws: List[str] = []
    # English-like tokens
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-_/]{1,}", raw):
        if len(w) >= 3:
            kws.append(w)
    # CJK tokens: sliding window of length 2-4
    cjk = re.findall(r"[\u4e00-\u9fff]{2,4}", raw)
    kws.extend(cjk)
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _split_sentences(text: str) -> List[str]:
    t = str(text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _highlight_keywords(sentence: str, keywords: Sequence[str]) -> str:
    out = sentence
    for k in keywords:
        if not k:
            continue
        try:
            out = re.sub(re.escape(k), lambda m: f"[[{m.group(0)}]]", out)
        except Exception:
            continue
    return out


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
    p.add_argument(
        "--blocks_jsonl",
        default="",
        help="Optional blocks.jsonl path for debug: print evidence text for retrieved_ids.",
    )

    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model for retrieval query embedding")
    p.add_argument("--domain_encoder_max_length", type=int, default=256)

    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--layers", type=str, default="0,1,2,3")
    p.add_argument("--max_new_tokens_base", type=int, default=192)
    p.add_argument("--max_new_tokens_rim", type=int, default=192)
    p.add_argument("--pattern_index_dir", default="", help="Pattern-first sidecar index directory (KVI2 only)")
    p.add_argument(
        "--pattern_hard",
        default="",
        help="Comma-separated pattern_ids to treat as hard contracts (e.g., abbr:SFTSV).",
    )
    p.add_argument(
        "--pattern_soft",
        default="",
        help="Comma-separated pattern_ids to treat as soft contracts (override hard if both set).",
    )
    p.add_argument(
        "--structured_template",
        action="store_true",
        help="If set, append a structured answer template to the prompt (LLM still answers; no slot hardcoding).",
    )
    p.add_argument(
        "--structured_template_text",
        type=str,
        default="",
        help="Optional template text to append when --structured_template is set. If empty, use a safe default.",
    )

    p.add_argument(
        "--runtime_backend",
        choices=["kvi1", "kvi2"],
        default="kvi1",
        help="Which runtime path to use: kvi1=legacy semantic KV injection (no Pattern-first); "
        "kvi2=KVI2Runtime (Pattern-first + introspection-gated).",
    )
    p.add_argument(
        "--gate_mode",
        choices=["none", "introspection", "self_critique"],
        default="none",
        help="Gate strategy. none=always do single semantic retrieval+inject (KVI1.0, no gate); "
        "introspection=non-generative gate (cosine drift) without Pattern-first when runtime=kvi1; "
        "self_critique=legacy gate.",
    )
    p.add_argument("--self_critique_mode", choices=["heuristic", "llm_json"], default="heuristic")
    p.add_argument("--critique_max_new_tokens", type=int, default=128)
    p.add_argument("--critique_conf_threshold", type=float, default=0.55, help="Trigger retrieval if confidence < threshold")
    p.add_argument("--force_rim", action="store_true", help="Force retrieval+injection regardless of critique")
    p.add_argument(
        "--force_inject_on_unknown_delta",
        action="store_true",
        help="If set, still inject when KV impact delta is unavailable (None). Default is fail-closed to base answer.",
    )
    p.add_argument(
        "--kv_refresh_rounds",
        type=int,
        default=2,
        help="If injected KV is judged irrelevant for this query, re-select a new batch and retry up to N rounds.",
    )
    p.add_argument(
        "--kv_irrelevant_logit_delta_threshold",
        type=float,
        default=0.05,
        help="If mean|logits(injected)-logits(zero_prefix)| < threshold, treat current KV batch as irrelevant.",
    )
    args = p.parse_args()

    # Resolve KV bank path
    kv_dir: Optional[str] = args.kv_dir
    if kv_dir is None:
        if not args.topic or not args.topic_work_dir:
            raise SystemExit("Missing --kv_dir. Either pass --kv_dir or use --topic + --topic_work_dir.")
        kv_dir = str(_resolve_topic_kv_dir(topic=str(args.topic), topic_work_dir=str(args.topic_work_dir)))

    # Optional block text lookup for debug
    block_text_by_id: Dict[str, str] = {}
    blocks_jsonl = str(args.blocks_jsonl or "").strip()
    if blocks_jsonl:
        try:
            with Path(blocks_jsonl).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    bid = str(rec.get("block_id") or rec.get("id") or rec.get("chunk_id") or "")
                    txt = rec.get("text")
                    if bid and isinstance(txt, str) and txt.strip():
                        block_text_by_id[bid] = txt.strip()
        except Exception:
            block_text_by_id = {}

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else None)
    model.to(device)
    model.eval()

    # Fast path: v0.4 pipeline in reusable runtime (only when explicitly selected)
    if str(args.runtime_backend) == "kvi2" and str(args.gate_mode) == "introspection":
        rt = KVI2Runtime(
            cfg=KVI2Config(
                layers=tuple(int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""),
                top_k=int(args.top_k),
                max_new_tokens_base=int(args.max_new_tokens_base),
                max_new_tokens_rim=int(args.max_new_tokens_rim),
                kv_refresh_rounds=int(args.kv_refresh_rounds),
                kv_irrelevant_logit_delta_threshold=float(args.kv_irrelevant_logit_delta_threshold),
                pattern_index_dir=str(args.pattern_index_dir or ""),
                structured_answer_template=bool(args.structured_template),
                structured_template_text=(
                    str(args.structured_template_text).strip()
                    if str(args.structured_template_text or "").strip()
                    else "请按以下结构回答：\n- 结论：...\n- 证据依据：...\n- 不确定性/限制：..."
                ),
                pattern_hard=[x.strip() for x in str(args.pattern_hard or "").split(",") if x.strip()],
                pattern_soft=[x.strip() for x in str(args.pattern_soft or "").split(",") if x.strip()],
            ),
            domain_encoder_model=str(args.domain_encoder_model),
            domain_encoder_max_length=int(args.domain_encoder_max_length),
        )
        out = rt.run_ab(
            model=model,
            tokenizer=tok,
            prompt=str(args.prompt),
            kv_dir=str(kv_dir),
            device=device,
            use_chat_template=bool(args.use_chat_template),
            force_rim=bool(args.force_rim),
            block_text_lookup=block_text_by_id or None,
        )
        print("\n=== 无 RIM（Base LLM）===\n")
        print(str(out.get("baseline_answer", "")).strip())
        print("\n=== Pattern-first（non-semantic）===\n")
        print(json.dumps(out.get("pattern_first", {}), ensure_ascii=False, indent=2))
        print("\n=== Introspection Gate ===\n")
        print(json.dumps({"gate_mode": "introspection", "gate": out.get("gate", {}), "retrieve_more": out.get("retrieve_more")}, ensure_ascii=False, indent=2))
        if out.get("retrieve_more"):
            print("\n=== RIM 检索（KV Bank）===\n")
            print(json.dumps(out.get("retrieval", {}), ensure_ascii=False, indent=2))
            if block_text_by_id:
                print("\n=== Retrieved Evidence Text (debug) ===\n")
                retrieved_ids = (out.get("retrieval", {}) or {}).get("retrieved_ids", [])
                keywords = _extract_keywords(str(args.prompt))
                if keywords:
                    print("[evidence_keywords] " + ", ".join(keywords), flush=True)
                for rid in retrieved_ids or []:
                    text = block_text_by_id.get(str(rid), "")
                    if text:
                        sentences = _split_sentences(text)
                        matched: List[str] = []
                        for s in sentences:
                            s_norm = s.lower()
                            hit = any((k.lower() in s_norm) for k in keywords) if keywords else False
                            if hit:
                                matched.append(_highlight_keywords(s, keywords))
                            if len(matched) >= 3:
                                break
                        if not matched:
                            excerpt = text[:240].replace("\n", " ").strip()
                            matched = [_highlight_keywords(excerpt, keywords)]
                        print(f"- {rid}")
                        for s in matched:
                            print(f"  {s}")
                        print("")
            print("\n=== 有 RIM（注入 KV → 第二轮生成）===\n")
            print(str(out.get("rim_answer", "")).strip())
        else:
            print("\n=== 有 RIM ===\n")
            print("(本次未触发检索/注入；可用 --force_rim 强制演示)")
        return

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

    # --- Gate ---
    # Legacy path (kvi1): no Pattern-first; gate can be none/introspection/self_critique.
    gate_debug: Dict[str, Any] = {}
    trigger = False
    crit: Optional[CritiqueResult] = None
    crit_mode: Optional[str] = None

    if str(args.gate_mode) == "none":
        trigger = True
        gate_debug = {"gate_mode": "none", "retrieve_more": True, "note": "kvi1_single_pass_inject"}
    elif str(args.gate_mode) == "self_critique":
        if str(args.self_critique_mode) == "llm_json":
            crit = llm_json_self_critique(
                model=model,
                tokenizer=tok,
                prompt_text=prompt1,
                question=user_prompt,
                draft_answer=base_answer,
                device=device,
                max_new_tokens=int(args.critique_max_new_tokens),
            )
            if crit is None:
                crit = heuristic_self_critique(user_prompt, base_answer)
                crit_mode = "heuristic_fallback"
            else:
                crit_mode = "llm_json"
        else:
            crit = heuristic_self_critique(user_prompt, base_answer)
            crit_mode = "heuristic"

        trigger = bool(args.force_rim) or (crit.is_medical_fact and float(crit.confidence) < float(args.critique_conf_threshold))
        gate_debug = {
            "gate_mode": "self_critique",
            "mode": crit_mode,
            "is_medical_fact": bool(crit.is_medical_fact),
            "confidence": float(crit.confidence),
            "threshold": float(args.critique_conf_threshold),
            "trigger_retrieve": bool(trigger),
            "reason": str(crit.reason),
        }
    else:
        # introspection gate (legacy path): use retrieval-embedding drift as a proxy for q0 vs q'_t
        enc_gate = DomainEncoder(
            DomainEncoderConfig(
                model_name_or_path=str(args.domain_encoder_model),
                max_length=int(args.domain_encoder_max_length),
                normalize=True,
                device=str(device),
            )
        )
        q0 = enc_gate.encode(user_prompt)[0]
        qprime = enc_gate.encode(user_prompt + "\n" + base_answer)[0]

        rim_gate = RIM(
            retriever=Retriever(FaissKVBank.load(Path(str(kv_dir)))),
            cfg=RIMConfig(
                tau_cos_dist=0.35,
                top_k=int(args.top_k),
                kv_refresh_rounds=int(args.kv_refresh_rounds),
                kv_irrelevant_logit_delta_threshold=float(args.kv_irrelevant_logit_delta_threshold),
            ),
        )
        rim_gate.set_q0(q0)
        rim_gate.observe(hidden_states=None, generated_tokens=None, pattern_hits=None, semantic_hits=None)
        gate = rim_gate.introspection_gate(q_prime_vec=qprime, kv_relevance_delta=None, pattern_mismatch=False)
        trigger = bool(args.force_rim) or bool(gate.get("retrieve_more") is True)
        gate_debug = {"gate_mode": "introspection", "gate": gate, "retrieve_more": bool(trigger)}

    # --- Pass 2: Retrieve + Inject + Regenerate ---
    rim_answer = ""
    retrieved_debug: Dict[str, Any] = {}
    retrieved_ids: List[str] = []
    kv_refresh_debug: Dict[str, Any] = {}
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
        # Oversample once, then slice into multiple non-overlapping batches for "KV refresh" rounds.
        oversample_k = max(int(args.top_k), int(args.top_k) * max(1, int(args.kv_refresh_rounds) + 1))
        rr = retriever.search(query_vec, top_k=int(oversample_k), filters=None, query_text=user_prompt)
        retrieved_debug = dict(rr.debug or {})

        layer_ids = [int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""]
        dtype = next(model.parameters()).dtype

        seen_ids: set[str] = set()
        chosen_items: List[Any] = []
        chosen_pkv: Any = None
        chosen_delta: Optional[float] = None
        attempts = max(0, int(args.kv_refresh_rounds)) + 1
        for attempt in range(attempts):
            # pick next unseen batch
            batch: List[Any] = []
            for it in rr.items:
                bid = _kv_id(it)
                if not bid or bid in seen_ids:
                    continue
                batch.append(it)
                seen_ids.add(bid)
                if len(batch) >= int(args.top_k):
                    break
            if not batch:
                break

            ext_by_layer: Dict[int, Any] = {}
            ext_errors: List[str] = []
            for li in layer_ids:
                try:
                    ext_by_layer[li] = stack_ext_kv_items_by_layer(
                        items=batch,
                        layer_id=int(li),
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                except Exception as e:
                    ext_errors.append(f"layer={li} err={type(e).__name__}: {e}")
            if not ext_by_layer:
                chosen_delta = None
                chosen_pkv = None
                kv_refresh_debug = {
                    "oversample_top_k": int(oversample_k),
                    "attempts": int(attempts),
                    "final_logit_delta_vs_zero_prefix": None,
                    "threshold": float(args.kv_irrelevant_logit_delta_threshold),
                    "note": "ext_by_layer_empty",
                    "ext_errors": ext_errors[:5],
                }
                break
            try:
                pkv = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
            except Exception as e:
                chosen_delta = None
                chosen_pkv = None
                kv_refresh_debug = {
                    "oversample_top_k": int(oversample_k),
                    "attempts": int(attempts),
                    "final_logit_delta_vs_zero_prefix": None,
                    "threshold": float(args.kv_irrelevant_logit_delta_threshold),
                    "note": f"build_pkv_failed:{type(e).__name__}",
                }
                break
            delta = logit_delta_vs_zero_prefix(model=model, tokenizer=tok, prompt=prompt1, device=device, past_key_values=pkv)

            chosen_items = batch
            chosen_pkv = pkv
            chosen_delta = delta

            # If delta is measurable and above threshold -> accept; else retry with a fresh batch.
            th = float(args.kv_irrelevant_logit_delta_threshold)
            if delta is None or float(delta) >= th:
                break

        retrieved_ids = [_kv_id(it) for it in chosen_items]
        retrieved_ids = [x for x in retrieved_ids if x]
        kv_refresh_debug = {
            "oversample_top_k": int(oversample_k),
            "attempts": int(attempts),
            "final_logit_delta_vs_zero_prefix": chosen_delta,
            "threshold": float(args.kv_irrelevant_logit_delta_threshold),
        }

        # For kvi1 + gate_mode=none: always inject if we have a prefix (no relevance gating).
        if chosen_pkv is None:
            trigger = False
            rim_answer = base_answer
            gate_debug = {**gate_debug, "note": "no_past_key_values_built"}
        else:
            th = float(args.kv_irrelevant_logit_delta_threshold)
            if str(args.gate_mode) == "none":
                rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt1,
                    device=device,
                    past_key_values=chosen_pkv,
                    max_new_tokens=int(args.max_new_tokens_rim),
                    no_repeat_ngram_size=12,
                    repetition_penalty=1.08,
                )
            elif chosen_delta is not None and float(chosen_delta) < th:
                trigger = False
                rim_answer = base_answer
                gate_debug = {**gate_debug, "note": "kv_prefix_low_impact_fallback"}
            elif chosen_delta is None and not bool(args.force_inject_on_unknown_delta):
                trigger = False
                rim_answer = base_answer
                gate_debug = {**gate_debug, "note": "kv_delta_unknown_fallback"}
            else:
                rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt1,
                    device=device,
                    past_key_values=chosen_pkv,
                    max_new_tokens=int(args.max_new_tokens_rim),
                    no_repeat_ngram_size=12,
                    repetition_penalty=1.08,
                )

    # --- Print comparison ---
    print("\n=== 无 RIM（Base LLM）===\n")
    print(base_answer.strip())
    if str(args.gate_mode) != "none":
        print("\n=== Introspection Gate ===\n")
        print(json.dumps(gate_debug, ensure_ascii=False, indent=2))
    if trigger:
        print("\n=== RIM 检索（KV Bank）===\n")
        print(
            json.dumps(
                {"kv_dir": str(kv_dir), "top_k": int(args.top_k), "retriever_debug": retrieved_debug, "kv_refresh": kv_refresh_debug},
                ensure_ascii=False,
                indent=2,
            )
        )
        if retrieved_ids:
            print("\nretrieved_ids(top_k):")
            for x in retrieved_ids[: int(args.top_k)]:
                print("-", x)
        print("\n=== 有 RIM（注入 KV → 第二轮生成）===\n")
        print(rim_answer.strip())
    else:
        print("\n=== 有 RIM ===\n")
        if str(args.gate_mode) == "none":
            print("(本次未注入：未能构建可用 past_key_values 前缀或触发 fail-closed 回退)")
        else:
            print("(本次未触发检索/注入：Gate 未触发；可用 --force_rim 强制演示)")


if __name__ == "__main__":
    main()

