"""
KVI 2.0 Runtime (RIM v0.4): Pattern-first -> Semantic-second -> Introspection-gated -> Injection (+ refresh <= 2 rounds).

This module provides a reusable runtime pipeline (beyond a one-off demo script).

Hard constraints (docs/07_KVI_RMI.md):
- Do NOT modify base LLM weights
- Do NOT modify attention forward
- Injection via past_key_values prefix (existing)
- No text-level RAG
- RIM introspection gate MUST NOT generate tokens
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ..domain_encoder import DomainEncoder, DomainEncoderConfig
from ..kv_bank import FaissKVBank
from ..pattern_contract import PatternContractValidator, filter_evidence_by_contracts, run_pattern_first
from ..pattern_pipeline import (
    IntrospectionGate,
    PatternContractLoader,
    PatternMatcher,
    PatternSpec,
    SemanticInstanceBuilder,
    SlotSchema,
    compute_slot_status_from_instances,
    find_unconsumed_evidence_blocks,
)
from ..pattern_retriever import PatternRetriever, PatternRetrieveResult
from ..retriever import Retriever
from ..rim import RIM, RIMConfig
from .hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer
from .kv_relevance import logit_delta_vs_zero_prefix
from .multistep_injector import MultiStepInjector


@dataclass(frozen=True)
class KVI2Config:
    layers: Sequence[int] = (0, 1, 2, 3)
    top_k: int = 8
    max_new_tokens_base: int = 192
    max_new_tokens_rim: int = 192
    kv_refresh_rounds: int = 2
    kv_irrelevant_logit_delta_threshold: float = 0.05
    tau_cos_dist: float = 0.35
    pattern_index_dir: str = ""
    structured_answer_template: bool = False
    structured_template_text: str = ""
    # Pattern contract level config (ids are pattern_id, e.g., "abbr:SFTSV")
    pattern_hard: Sequence[str] = ()
    pattern_soft: Sequence[str] = ()


class KVI2Runtime:
    def __init__(
        self,
        *,
        cfg: KVI2Config,
        domain_encoder_model: str,
        domain_encoder_max_length: int = 256,
    ) -> None:
        self.cfg = cfg
        self.domain_encoder_model = str(domain_encoder_model)
        self.domain_encoder_max_length = int(domain_encoder_max_length)

    def run_ab(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
        kv_dir: str,
        device: torch.device,
        use_chat_template: bool = False,
        force_rim: bool = False,
        block_text_lookup: Optional[Dict[str, str]] = None,
        sidecar_dir: str = "",
    ) -> Dict[str, Any]:
        """
        Returns a dict containing:
        - baseline_answer
        - pattern_first
        - gate
        - retrieval (debug + selected ids)
        - rim_answer (optional)
        """
        user_prompt = str(prompt)
        formatted_prompt = self._format_prompt(tokenizer, user_prompt, use_chat_template=bool(use_chat_template))
        if bool(self.cfg.structured_answer_template):
            tmpl = (self.cfg.structured_template_text or "").strip()
            if tmpl:
                formatted_prompt = formatted_prompt + "\n\n" + tmpl

        # 1) base LLM first pass (no injection)
        baseline = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            device=device,
            past_key_values=None,
            max_new_tokens=int(self.cfg.max_new_tokens_base),
            no_repeat_ngram_size=12,
            repetition_penalty=1.08,
        )

        # 2) Pattern-first (non-semantic) + topic-level contracts (prompt-agnostic)
        pattern = (
            PatternRetriever.from_dir(str(self.cfg.pattern_index_dir))
            if str(getattr(self.cfg, "pattern_index_dir", "") or "").strip()
            else PatternRetriever()
        )
        loader = PatternContractLoader()
        topic_dir = loader.infer_topic_dir_from_kv_dir(str(kv_dir))
        loaded_contracts = loader.load(topic_dir=str(topic_dir) if topic_dir else None)

        pattern_res: PatternRetrieveResult
        matched_patterns: Sequence[PatternSpec]
        matched_skeletons: Dict[str, str]
        if loaded_contracts:
            matcher = PatternMatcher(loaded_contracts, retriever=pattern)
            pattern_res, matched_patterns, matched_skeletons = matcher.match(user_prompt)
        else:
            pattern_res = pattern.retrieve(user_prompt)
            matched_patterns = []
            matched_skeletons = {}

        # No contract => reject (contract-driven QA)
        if not loaded_contracts:
            out: Dict[str, Any] = {
                "baseline_answer": baseline.strip(),
                "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
                "gate": {"decision": "reject_no_contract", "pattern_id": "", "matched_skeleton": ""},
                "retrieve_more": False,
                "rim_answer": "",
                "retrieval": {},
            }
            return out

        if not matched_patterns:
            out = {
                "baseline_answer": baseline.strip(),
                "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
                "gate": {"decision": "reject_no_pattern", "pattern_id": "", "matched_skeleton": ""},
                "retrieve_more": False,
                "rim_answer": "",
                "retrieval": {},
            }
            return out

        # Use best pattern (highest match score)
        best_pattern = matched_patterns[0]
        slot_schema = SlotSchema.from_pattern(best_pattern)
        validator = PatternContractValidator()
        contracts = run_pattern_first(
            user_prompt,
            pattern_res,
            hard_pattern_ids=self.cfg.pattern_hard,
            soft_pattern_ids=self.cfg.pattern_soft,
        )

        # 3) Introspection gate (non-generative)
        bank = FaissKVBank.load(Path(str(kv_dir)))
        retriever = Retriever(bank)
        enc = DomainEncoder(
            DomainEncoderConfig(
                model_name_or_path=self.domain_encoder_model,
                max_length=int(self.domain_encoder_max_length),
                normalize=True,
                device=str(device),
            )
        )
        q0 = enc.encode(user_prompt)[0]
        qprime = enc.encode(user_prompt + "\n" + baseline)[0]

        rim = RIM(
            retriever=retriever,
            cfg=RIMConfig(
                tau_cos_dist=float(self.cfg.tau_cos_dist),
                top_k=int(self.cfg.top_k),
                kv_refresh_rounds=int(self.cfg.kv_refresh_rounds),
                kv_irrelevant_logit_delta_threshold=float(self.cfg.kv_irrelevant_logit_delta_threshold),
            ),
        )
        rim.set_q0(q0)
        rim.observe(hidden_states=None, generated_tokens=None, pattern_hits=pattern_res, semantic_hits=None)
        gate = IntrospectionGate(rim).evaluate(
            q_prime_vec=qprime,
            kv_relevance_delta=None,
            missing_hard=[],
            missing_soft=[],
            missing_schema=[],
            pattern_id=best_pattern.pattern_id,
            matched_skeleton=matched_skeletons.get(best_pattern.pattern_id, ""),
            slot_status=slot_schema.status([]),
            slot_schema=slot_schema,
            answer_style=best_pattern.answer_style,
            question_intent=best_pattern.question_intent,
            semantic_instances=[],
        )
        if gate.get("decision") == "REFUSE":
            if gate.get("allow_rim_retry"):
                # retrieve-only retry (no generation)
                oversample_top_k = max(int(self.cfg.top_k), int(self.cfg.top_k) * (int(self.cfg.kv_refresh_rounds) + 1))
                rr = retriever.search(q0, top_k=int(oversample_top_k), filters=None, query_text=user_prompt)
                rr_items = _boost_list_like_items(rr.items, slot_schema)
                batch: list[Any] = []
                seen_ids: set[str] = set()
                for it in rr_items:
                    meta = getattr(it, "meta", None) or {}
                    bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
                    if not bid or bid in seen_ids:
                        continue
                    batch.append(it)
                    seen_ids.add(bid)
                    if len(batch) >= int(self.cfg.top_k):
                        break
                temp_instances = SemanticInstanceBuilder().build(
                    pattern=best_pattern,
                    evidence_blocks=batch,
                    slot_schema=slot_schema,
                    block_text_lookup=block_text_lookup,
                )
                contract_validation = validator.validate_all(contracts, batch, block_text_lookup=block_text_lookup)
                slot_status = compute_slot_status_from_instances(slot_schema, temp_instances)
                hard_missing = list(contract_validation.get("hard_missing") or [])
                soft_missing = list(contract_validation.get("soft_missing") or [])
                schema_missing = list(contract_validation.get("schema_missing") or [])
                gate_retry = IntrospectionGate(rim).evaluate(
                    q_prime_vec=qprime,
                    kv_relevance_delta=None,
                    missing_hard=hard_missing,
                    missing_soft=soft_missing,
                    missing_schema=schema_missing,
                    pattern_id=best_pattern.pattern_id,
                    matched_skeleton=matched_skeletons.get(best_pattern.pattern_id, ""),
                    slot_status=slot_status,
                    slot_schema=slot_schema,
                    answer_style=best_pattern.answer_style,
                    question_intent=best_pattern.question_intent,
                    semantic_instances=temp_instances,
                )
                out: Dict[str, Any] = {
                    "baseline_answer": baseline.strip(),
                    "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
                    "gate": gate_retry,
                    "retrieve_more": False,
                    "rim_answer": "",
                    "retrieval": {
                        "kv_dir": str(kv_dir),
                        "top_k": int(self.cfg.top_k),
                        "oversample_top_k": int(oversample_top_k),
                        "retriever_debug": dict(rr.debug or {}),
                        "rim_retry": True,
                        "list_like_candidate_count": int(len(_collect_list_like_ids(rr_items))),
                        "list_like_candidate_ids": _collect_list_like_ids(rr_items)[: int(self.cfg.top_k)],
                        "contract_validation": dict(contract_validation or {}),
                        "semantic_instances": temp_instances,
                        "slot_status_source": "semantic_instances",
                        "slot_status_snapshot": dict(slot_status or {}),
                    },
                }
                if any(v == "missing" for v in (slot_status or {}).values()):
                    out["retrieval"]["unconsumed_evidence_blocks"] = find_unconsumed_evidence_blocks(
                        batch, slot_schema
                    )
                return out
            out = {
                "baseline_answer": baseline.strip(),
                "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
                "gate": gate,
                "retrieve_more": False,
                "rim_answer": "",
                "retrieval": {},
            }
            return out
        retrieve_more = bool(force_rim) or bool(gate.get("retrieve_more") is True)

        out: Dict[str, Any] = {
            "baseline_answer": baseline.strip(),
            "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
            "gate": gate,
            "retrieve_more": bool(retrieve_more),
        }

        if not retrieve_more:
            out["rim_answer"] = ""
            out["retrieval"] = {}
            return out

        # 4) Semantic-second retrieve + refresh (<=2 rounds)
        oversample_top_k = max(int(self.cfg.top_k), int(self.cfg.top_k) * (int(self.cfg.kv_refresh_rounds) + 1))
        rr = retriever.search(q0, top_k=int(oversample_top_k), filters=None, query_text=user_prompt)
        rr_items = _boost_list_like_items(rr.items, slot_schema)
        layer_ids = [int(x) for x in self.cfg.layers]
        dtype = next(model.parameters()).dtype

        seen_ids: set[str] = set()
        chosen_items = []
        chosen_pkv: Any = None
        chosen_delta: Optional[float] = None
        chosen_contract_validation: Optional[Dict[str, Any]] = None
        chosen_gate_after_validation: Optional[Dict[str, Any]] = None
        attempts = max(0, int(self.cfg.kv_refresh_rounds)) + 1
        for attempt_idx in range(attempts):
            batch = []
            for it in rr_items:
                meta = getattr(it, "meta", None) or {}
                bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
                if not bid or bid in seen_ids:
                    continue
                batch.append(it)
                seen_ids.add(bid)
                if len(batch) >= int(self.cfg.top_k):
                    break
            if not batch:
                break

            filtered_batch = filter_evidence_by_contracts(
                contracts,
                batch,
                block_text_lookup=block_text_lookup,
                pattern_id=best_pattern.pattern_id,
            )
            if not filtered_batch:
                if attempt_idx + 1 < attempts:
                    continue
                break

            ext_by_layer = {
                li: stack_ext_kv_items_by_layer(
                    items=filtered_batch,
                    layer_id=int(li),
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                for li in layer_ids
            }
            pkv = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
            delta = logit_delta_vs_zero_prefix(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                device=device,
                past_key_values=pkv,
            )

            contract_validation = validator.validate_all(contracts, batch, block_text_lookup=block_text_lookup)
            chosen_contract_validation = contract_validation
            temp_instances = SemanticInstanceBuilder().build(
                pattern=best_pattern,
                evidence_blocks=filtered_batch,
                slot_schema=slot_schema,
                block_text_lookup=block_text_lookup,
            )
            slot_status = compute_slot_status_from_instances(slot_schema, temp_instances)
            hard_missing = list(contract_validation.get("hard_missing") or [])
            soft_missing = list(contract_validation.get("soft_missing") or [])
            schema_missing = list(contract_validation.get("schema_missing") or [])
            pattern_mismatch = bool(len(hard_missing) > 0)
            gate_after_validation = IntrospectionGate(rim).evaluate(
                q_prime_vec=qprime,
                kv_relevance_delta=delta,
                missing_hard=hard_missing,
                missing_soft=soft_missing,
                missing_schema=schema_missing,
                pattern_id=best_pattern.pattern_id,
                matched_skeleton=matched_skeletons.get(best_pattern.pattern_id, ""),
                slot_status=slot_status,
                slot_schema=slot_schema,
                answer_style=best_pattern.answer_style,
                question_intent=best_pattern.question_intent,
                semantic_instances=temp_instances,
            )
            chosen_gate_after_validation = gate_after_validation

            if gate_after_validation.get("decision") == "REFUSE":
                chosen_items = []
                chosen_pkv = None
                chosen_delta = None
                break
            if pattern_mismatch:
                # Reject current KV batch; try refresh if budget remains.
                if attempt_idx + 1 < attempts:
                    continue
                chosen_items = []
                chosen_pkv = None
                chosen_delta = None
                break

            chosen_items = filtered_batch
            chosen_pkv = pkv
            chosen_delta = delta
            th = float(self.cfg.kv_irrelevant_logit_delta_threshold)
            if delta is None or float(delta) >= th:
                break

        semantic_instances = SemanticInstanceBuilder().build(
            pattern=best_pattern,
            evidence_blocks=chosen_items,
            slot_schema=slot_schema,
            block_text_lookup=block_text_lookup,
        )
        slot_status = compute_slot_status_from_instances(slot_schema, semantic_instances)
        chosen_gate_after_validation = _apply_sidecar_slot_guard(
            gate_after_validation=chosen_gate_after_validation,
            slot_schema=slot_schema,
            semantic_instances=semantic_instances,
            retrieved_items=chosen_items,
            kv_dir=str(kv_dir),
            sidecar_dir=str(sidecar_dir or "").strip(),
        )
        out["retrieval"] = {
            "kv_dir": str(kv_dir),
            "top_k": int(self.cfg.top_k),
            "oversample_top_k": int(oversample_top_k),
            "retriever_debug": dict(rr.debug or {}),
            "list_like_candidate_count": int(len(_collect_list_like_ids(rr_items))),
            "list_like_candidate_ids": _collect_list_like_ids(rr_items)[: int(self.cfg.top_k)],
            "kv_refresh": {
                "attempts": int(attempts),
                "final_logit_delta_vs_zero_prefix": chosen_delta,
                "threshold": float(self.cfg.kv_irrelevant_logit_delta_threshold),
            },
            "contract_validation": dict(chosen_contract_validation or {}),
            "gate_after_validation": dict(chosen_gate_after_validation or {}),
            "semantic_instances": semantic_instances,
            "slot_status_source": "semantic_instances",
            "slot_status_snapshot": dict(slot_status or {}),
        }
        if any(v == "missing" for v in (slot_status or {}).values()):
            out["retrieval"]["unconsumed_evidence_blocks"] = find_unconsumed_evidence_blocks(
                chosen_items, slot_schema
            )

        # If this is an abbreviation pattern and we can extract a valid expansion
        # from evidence metadata, return a deterministic answer to avoid hallucination.
        if str(best_pattern.pattern_id).startswith("abbr:"):
            abbr = str(best_pattern.pattern_id).split(":", 1)[1].strip()
            full_name = _extract_abbr_expansion_from_blocks(chosen_items, abbr, block_text_lookup=block_text_lookup)
            if full_name:
                out["rim_answer"] = f"{abbr} 的全称是 {full_name}。"
                return out
            out["rim_answer"] = "现有证据不足以回答该问题。"
            return out

        # 5) Second pass generation with injected prefix
        if chosen_pkv is None:
            out["rim_answer"] = ""
            return out
        guard_style = ""
        if isinstance(chosen_gate_after_validation, dict):
            guard_style = str(chosen_gate_after_validation.get("final_answer_style") or "")
        if not guard_style and isinstance(gate, dict):
            guard_style = str(gate.get("final_answer_style") or "")

        # LIST_ONLY: extract-only from evidence, no free generation.
        if str(guard_style or "").upper() == "LIST_ONLY":
            items, src_ids, fail_reason = _extract_list_items_from_instances(
                semantic_instances=semantic_instances,
                slot_schema=slot_schema,
            )
            out["retrieval"]["list_items_extracted"] = items
            out["retrieval"]["list_items_source_block_ids"] = src_ids
            if fail_reason:
                out["retrieval"]["list_extraction_failed_reason"] = fail_reason
            if not items:
                if isinstance(chosen_gate_after_validation, dict):
                    chosen_gate_after_validation["decision"] = "REFUSE"
                    chosen_gate_after_validation["decision_reason"] = "slot-satisfied-but-not-extractable"
                    chosen_gate_after_validation["final_answer_style"] = None
                    chosen_gate_after_validation["allowed_answer_capabilities"] = []
                out["gate"] = dict(chosen_gate_after_validation or {})
                out["rim_answer"] = ""
                return out
            out["rim_answer"] = "\n".join([f"- {x}" for x in items])
            out["gate"] = dict(chosen_gate_after_validation or {})
            return out
        guarded_prompt = _apply_answer_style_guard(formatted_prompt, guard_style)
        rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=guarded_prompt,
            device=device,
            past_key_values=chosen_pkv,
            max_new_tokens=int(self.cfg.max_new_tokens_rim),
            no_repeat_ngram_size=12,
            repetition_penalty=1.08,
        )
        out["rim_answer"] = rim_answer.strip()
        return out

    @staticmethod
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


def _pattern_to_json(
    r: PatternRetrieveResult, patterns: Sequence[PatternSpec], matched_skeletons: Dict[str, str]
) -> Dict[str, Any]:
    return {
        "recall_size": int(r.recall_size),
        "debug": r.debug_info,
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "question_skeleton": {
                    "intent": p.question_intent,
                    "surface_forms": list(p.question_surface_forms or []),
                },
                "slots": {k: v.to_dict() for k, v in (p.slots or {}).items()},
                "answer_style": p.answer_style,
                "matched_skeleton": matched_skeletons.get(p.pattern_id, ""),
            }
            for p in (patterns or [])
        ],
    }


def _apply_answer_style_guard(prompt: str, final_style: str) -> str:
    style = str(final_style or "").strip().upper()
    if style == "LIST_ONLY":
        guard = (
            "\n\n[STYLE_GUARD]\n"
            "Return ONLY a bullet list of categories or item types. "
            "Do NOT assert factual claims or name specific entities. "
            "If evidence is insufficient, return only: '现有证据不足以回答该问题。'\n"
        )
        return str(prompt) + guard
    if style == "EXPLANATION":
        guard = (
            "\n\n[STYLE_GUARD]\n"
            "Provide a structural explanation only (no factual assertions, "
            "no specific entity names). If evidence is insufficient, return only: "
            "'现有证据不足以回答该问题。'\n"
        )
        return str(prompt) + guard
    return str(prompt)


def _extract_abbr_expansion_from_blocks(
    evidence_blocks: Sequence[Any],
    abbr: str,
    *,
    block_text_lookup: Optional[Dict[str, str]] = None,
) -> str:
    abbr_up = str(abbr or "").strip().upper()
    if not abbr_up:
        return ""
    # 1) Prefer explicit patterns in text: "Full (ABBR)" or "ABBR (Full)"
    for it in evidence_blocks or []:
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        text = ""
        if block_text_lookup and bid:
            text = str(block_text_lookup.get(bid) or "")
        elif isinstance(getattr(it, "text", None), str):
            text = str(getattr(it, "text") or "")
        if not text:
            continue
        full = _find_abbr_expansion_in_text(text, abbr_up)
        if full:
            return full

    # 2) Fallback: abbreviation_pairs in metadata (filtered)
    for it in evidence_blocks or []:
        meta = getattr(it, "meta", None) or {}
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
        best_full = ""
        best_score = -1
        for ap in abbr_pairs:
            if not isinstance(ap, dict):
                continue
            ap_abbr = str(ap.get("abbr") or "").strip()
            full = str(ap.get("full") or "").strip()
            if not ap_abbr or not full:
                continue
            if ap_abbr.upper() != abbr_up:
                continue
            if len(ap_abbr) < 3 or len(full) < (len(ap_abbr) + 4):
                continue
            full_low = full.lower()
            if full_low.startswith(("abstract", "keywords", "introduction")):
                continue
            if " is " in full_low or " are " in full_low or " was " in full_low or " were " in full_low:
                continue
            if " caused by " in full_low:
                continue
            first_word = full_low.split()[0] if full_low.split() else ""
            if len(first_word) < 4:
                continue
            if "confidence" in ap:
                try:
                    conf = float(ap.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                if conf < 0.85:
                    continue
            score = _score_abbr_full(full)
            if score > best_score:
                best_score = score
                best_full = full
        if best_full:
            return best_full
    return ""


def _extract_list_items_from_instances(
    *,
    semantic_instances: Sequence[Dict[str, Any]],
    slot_schema: Optional[SlotSchema],
) -> Tuple[List[str], List[str], str]:
    """
    Extract list items from evidence spans only (no free generation).
    Returns (items, source_block_ids, fail_reason).
    """
    if slot_schema is None:
        return [], [], "no-slot-schema"
    items: List[str] = []
    src_ids: List[str] = []
    for inst in semantic_instances or []:
        slots = inst.get("slots") if isinstance(inst, dict) else None
        if not isinstance(slots, dict):
            continue
        for slot_name, ev_list in slots.items():
            if not isinstance(ev_list, list):
                continue
            for ev in ev_list:
                if not isinstance(ev, dict):
                    continue
                bid = str(ev.get("evidence_id") or "").strip()
                span = str(ev.get("span") or "").strip()
                list_items = ev.get("list_items") if isinstance(ev.get("list_items"), list) else []
                extracted = [str(x).strip() for x in list_items if str(x).strip()]
                if not extracted:
                    if not span:
                        continue
                    extracted = _extract_bullet_like_items(span)
                if extracted:
                    items.extend(extracted)
                    if bid:
                        src_ids.append(bid)
    # de-dup
    cleaned: List[str] = []
    for x in items:
        y = _normalize_list_item(x)
        if y:
            cleaned.append(y)
    items = list(dict.fromkeys(cleaned))
    src_ids = list(dict.fromkeys([x for x in src_ids if x]))
    if not items:
        return [], src_ids, "no-bullets-in-evidence"
    return items[:12], src_ids, ""


def _extract_bullet_like_items(text: str) -> List[str]:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        if ln.startswith(("-", "*", "•", "·")):
            out.append(ln.lstrip("-*•· ").strip())
            continue
        if ln[:2].isdigit() and (ln[2:3] in {".", ")", "、"}):
            out.append(ln[3:].strip())
            continue
        if ln.startswith(("（", "(", "【")) and len(ln) > 3:
            # e.g., （1）xx
            if "）" in ln[:4] or ")" in ln[:4] or "】" in ln[:4]:
                out.append(ln[3:].strip())
                continue
    return [x for x in out if x]


def _normalize_list_item(item: str) -> str:
    s = " ".join(str(item or "").split())
    if not s:
        return ""
    # Remove leading boilerplate phrases.
    s = re.sub(
        r"^(symptoms?|clinical features|clinical manifestations|patients typically present)\s+"
        r"(include|including|such as|with|are)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"^(表现为|症状包括|临床表现为|典型表现为|常见表现为)\s*", "", s)
    # Drop trailing citations / bracketed refs.
    s = re.sub(r"\[[^\]]+\]$", "", s).strip()
    s = re.sub(r"\([^)]+\)$", "", s).strip()
    # Fix common split medical terms.
    s = re.sub(r"\bthrombo\s+cytopenia\b", "thrombocytopenia", s, flags=re.IGNORECASE)
    s = re.sub(r"\bleuko\s+cytopenia\b", "leukocytopenia", s, flags=re.IGNORECASE)
    s = re.sub(r"\blympho\s+cytopenia\b", "lymphocytopenia", s, flags=re.IGNORECASE)
    # Drop overly long or sentence-like fragments.
    if len(s) < 2 or len(s) > 60:
        return ""
    if re.search(r"[.;:]", s):
        return ""
    if re.search(
        r"\b(symptoms?|include|including|such as|present|presented|characterized|caused)\b",
        s,
        flags=re.IGNORECASE,
    ):
        return ""
    return s.strip(" ,;:-")


def _boost_list_like_items(items: Sequence[Any], slot_schema: Optional[SlotSchema]) -> List[Any]:
    """
    Boost list-like evidence for schema queries (non-destructive reordering).
    """
    if not items:
        return list(items or [])
    has_schema = bool(slot_schema) and any(
        str(spec.inference_level).lower() == "schema" for spec in (slot_schema.slots or {}).values()
    )
    if not has_schema:
        return list(items or [])

    def _score(it: Any) -> float:
        base = float(getattr(it, "score", 0.0) or 0.0)
        meta = getattr(it, "meta", None) or {}
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        lf = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
        boost = 0.0
        if lf.get("has_bullets"):
            boost += 0.25
        if lf.get("has_enumeration"):
            boost += 0.2
        try:
            dens = float(lf.get("list_density") or 0.0)
        except Exception:
            dens = 0.0
        boost += min(0.3, max(0.0, dens))
        if isinstance(lf.get("list_like_items"), list) and lf.get("list_like_items"):
            boost += 0.2
        return base + boost

    return sorted(list(items), key=_score, reverse=True)


def _collect_list_like_ids(items: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for it in items or []:
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        if not bid:
            continue
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        lf = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
        has_list = bool(lf.get("has_bullets") or lf.get("has_enumeration") or lf.get("list_like_items"))
        if has_list:
            ids.append(bid)
    return list(dict.fromkeys(ids))


def _find_abbr_expansion_in_text(text: str, abbr_up: str) -> str:
    import re

    t = " ".join(str(text or "").split())
    if not t:
        return ""
    abbr_esc = re.escape(abbr_up)
    # Full (ABBR)
    m = re.search(rf"([A-Za-z][A-Za-z0-9\-\s]{{5,120}})\s*\(\s*{abbr_esc}\s*\)", t)
    if m:
        full = m.group(1).strip(" ,;:.-")
        if _is_plausible_full(full, abbr_up):
            return full
    # ABBR (Full)
    m = re.search(rf"\b{abbr_esc}\b\s*\(\s*([A-Za-z][A-Za-z0-9\-\s]{{5,120}})\)", t)
    if m:
        full = m.group(1).strip(" ,;:.-")
        if _is_plausible_full(full, abbr_up):
            return full
    return ""


def _is_plausible_full(full: str, abbr_up: str) -> bool:
    if not full:
        return False
    full_low = full.lower()
    if full_low.startswith(("abstract", "keywords", "introduction")):
        return False
    if " is " in full_low or " are " in full_low or " was " in full_low or " were " in full_low:
        return False
    if " caused by " in full_low:
        return False
    words = [w for w in full.split() if w.isalpha() or w.isalnum()]
    if len(words) < 2:
        return False
    if len(full) < len(abbr_up) + 4:
        return False
    return True


def _score_abbr_full(full: str) -> int:
    words = [w for w in full.split() if w]
    score = len(words) * 2
    if len(words) > 12:
        score -= 2
    return score


def _apply_sidecar_slot_guard(
    *,
    gate_after_validation: Optional[Dict[str, Any]],
    slot_schema: SlotSchema,
    semantic_instances: Sequence[Dict[str, Any]],
    retrieved_items: Sequence[Any],
    kv_dir: str,
    sidecar_dir: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(gate_after_validation, dict):
        return gate_after_validation
    if gate_after_validation.get("decision") != "ALLOW":
        return gate_after_validation
    if str(gate_after_validation.get("final_answer_style") or "") != "FACTUAL_ASSERTION":
        return gate_after_validation

    required_or_soft = [
        k
        for k, v in (slot_schema.slots or {}).items()
        if bool(v.required) or str(v.inference_level) == "soft"
    ]
    if not required_or_soft:
        return gate_after_validation

    block_ids: list[str] = []
    for it in retrieved_items or []:
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        if bid:
            block_ids.append(bid)

    evidence_items = _load_evidence_sidecar(kv_dir, sidecar_dir=sidecar_dir)
    if evidence_items:
        block_facets = _build_facets_from_evidence(evidence_items, slot_schema, block_ids)
    else:
        block_facets = _load_block_facets(kv_dir, sidecar_dir=sidecar_dir)

    coverage_fn = _load_slot_coverage_fn(sidecar_dir=sidecar_dir)
    if not block_facets or not coverage_fn:
        return gate_after_validation

    coverage = coverage_fn(semantic_instances, block_ids, block_facets)
    if not isinstance(coverage, dict):
        return gate_after_validation
    missing_all = True
    for s in required_or_soft:
        if isinstance(coverage.get(s), dict) and int(coverage[s].get("evidence_count") or 0) > 0:
            missing_all = False
            break

    if missing_all:
        gate_after_validation["final_answer_style"] = "EXPLANATION"
        gate_after_validation["allowed_answer_capabilities"] = ["EXPLANATION"]
        gate_after_validation["decision_reason"] = "slot-uncovered"
    return gate_after_validation


def _load_block_facets(kv_dir: str, *, sidecar_dir: str) -> Dict[str, Dict[str, Any]]:
    # Prefer topic sidecar: {topic}/sidecar/block_facets.jsonl
    topic_dir = Path(str(kv_dir)).resolve()
    if topic_dir.name in {"kvbank_blocks", "kvbank_blocks_v2"}:
        if topic_dir.parent.name == "work":
            topic_dir = topic_dir.parent.parent
        else:
            topic_dir = topic_dir.parent
    candidates = []
    if sidecar_dir:
        candidates.append(Path(sidecar_dir) / "block_facets.jsonl")
    candidates.extend(
        [
            topic_dir / "sidecar" / "block_facets.jsonl",
            Path(__file__).resolve().parents[1] / "sidecar" / "block_facets.jsonl",
        ]
    )
    facets: Dict[str, Dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rec = json.loads(s)
                    if not isinstance(rec, dict):
                        continue
                    bid = str(rec.get("block_id") or "").strip()
                    if not bid:
                        continue
                    fac = rec.get("facets") if isinstance(rec.get("facets"), dict) else {}
                    facets[bid] = fac
        except Exception:
            continue
        if facets:
            break
    return facets


def _load_evidence_sidecar(kv_dir: str, *, sidecar_dir: str) -> List[Dict[str, Any]]:
    topic_dir = Path(str(kv_dir)).resolve()
    if topic_dir.name in {"kvbank_blocks", "kvbank_blocks_v2"}:
        if topic_dir.parent.name == "work":
            topic_dir = topic_dir.parent.parent
        else:
            topic_dir = topic_dir.parent
    candidates = []
    if sidecar_dir:
        candidates.append(Path(sidecar_dir) / "evidence.sidecar.jsonl")
    candidates.extend(
        [
            topic_dir / "work" / "evidence.sidecar.jsonl",
            Path(__file__).resolve().parents[1] / "sidecar" / "evidence.sidecar.jsonl",
        ]
    )
    items: List[Dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rec = json.loads(s)
                    if isinstance(rec, dict):
                        items.append(rec)
        except Exception:
            continue
        if items:
            break
    return items


def _build_facets_from_evidence(
    evidence_items: Sequence[Dict[str, Any]],
    slot_schema: SlotSchema,
    block_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_by_slot: Dict[str, set[str]] = {}
    for name, spec in (slot_schema.slots or {}).items():
        allowed_by_slot[name] = {str(t).strip().lower() for t in (spec.evidence_type or []) if str(t).strip()}

    block_id_set = {str(b) for b in (block_ids or []) if str(b).strip()}
    facets: Dict[str, Dict[str, Any]] = {}
    for it in evidence_items or []:
        bid = str(it.get("source_block_id") or "").strip()
        if not bid or (block_id_set and bid not in block_id_set):
            continue
        ev_type = str(it.get("type") or "").strip().lower()
        if not ev_type:
            continue
        for slot_name, allowed in allowed_by_slot.items():
            if ev_type in allowed:
                facets.setdefault(bid, {})[slot_name] = True
    return facets


def _load_slot_coverage_fn(*, sidecar_dir: str):
    from importlib.util import module_from_spec, spec_from_file_location

    paths = []
    if sidecar_dir:
        paths.append(Path(sidecar_dir) / "slot_coverage.py")
    paths.append(Path(__file__).resolve().parents[1] / "sidecar" / "slot_coverage.py")
    path = None
    for p in paths:
        if p.exists():
            path = p
            break
    if path is None:
        return None
    try:
        spec = spec_from_file_location("slot_coverage", str(path))
        if spec is None or spec.loader is None:
            return None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "compute_slot_coverage", None)
    except Exception:
        return None

