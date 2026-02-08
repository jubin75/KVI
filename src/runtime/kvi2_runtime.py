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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..domain_encoder import DomainEncoder, DomainEncoderConfig
from ..kv_bank import FaissKVBank, KVItem
from ..pattern_contract import PatternContractValidator, filter_evidence_by_contracts, run_pattern_first
from ..pattern_pipeline import (
    IntrospectionGate,
    PatternContractLoader,
    PatternMatcher,
    PatternSpec,
    SemanticInstanceBuilder,
    SlotSchema,
    score_candidate_schemas,
    compute_slot_status_from_instances,
    find_unconsumed_evidence_blocks,
)
from ..pattern_retriever import PatternRetriever, PatternRetrieveResult
from ..retriever import Retriever, RetrieverResult
from ..rim import RIM, RIMConfig
from .hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer
from .kv_relevance import logit_delta_vs_zero_prefix
from .multistep_injector import MultiStepInjector


@dataclass(frozen=True)
class KVI2Config:
    layers: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7)
    top_k: int = 8
    max_new_tokens_base: int = 128
    max_new_tokens_rim: int = 128
    no_repeat_ngram_size: int = 6
    repetition_penalty: float = 1.15
    kv_refresh_rounds: int = 2
    kv_irrelevant_logit_delta_threshold: float = 0.05
    tau_cos_dist: float = 0.35
    pattern_index_dir: str = ""
    structured_answer_template: bool = False
    structured_template_text: str = ""
    debug_retrieved_ids: bool = False
    # How to render LIST_ONLY answers:
    # - "list_only": deterministic bullet list projection (safest)
    # - "narrative": constrained LLM summary (grounded) + bullets
    # - "llm": bypass LIST_ONLY projection and force KV-injected generative answering
    # - "llm_prose": like llm, but also forces prose (no bullets) via a style guard
    answer_mode: str = "llm_prose"
    # Pattern contract level config (ids are pattern_id, e.g., "abbr:entity_name")
    pattern_hard: Sequence[str] = ()
    pattern_soft: Sequence[str] = ()
    # Optional: restrict retrieval to these block_ids (routing-aligned)
    allowed_block_ids: Sequence[str] = ()


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
            no_repeat_ngram_size=int(self.cfg.no_repeat_ngram_size),
            repetition_penalty=float(self.cfg.repetition_penalty),
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

        # Candidate schema scoring (docs/079_pattern_roles.md):
        # Pattern-first proposes candidates; this layer ranks them so low-info schemas (abbr/definition)
        # cannot hijack high-info queries (time range / spatial / enumeration).
        candidate_schemas = score_candidate_schemas(user_prompt, matched_patterns)
        # IMPORTANT: PatternMatcher output order is NOT decisive; do NOT treat list order as semantics.
        # Late-bind the selected pattern only after scoring.
        best_pattern: PatternSpec = matched_patterns[0]
        if candidate_schemas:
            best_id = str(candidate_schemas[0].get("schema_id") or "").strip()
            if best_id:
                for p in matched_patterns:
                    if str(p.pattern_id) == best_id:
                        best_pattern = p
                        break
        slot_schema = SlotSchema.from_pattern(best_pattern)
        validator = PatternContractValidator()
        all_contracts = run_pattern_first(
            user_prompt,
            pattern_res,
            hard_pattern_ids=self.cfg.pattern_hard,
            soft_pattern_ids=self.cfg.pattern_soft,
        )
        # Late binding invariant: only the selected pattern's contract may filter evidence or drive rejection.
        contracts = [c for c in (all_contracts or []) if str(c.pattern_id) == str(best_pattern.pattern_id)]

        # 3) Introspection gate (non-generative)
        bank = FaissKVBank.load(Path(str(kv_dir)))
        retriever = Retriever(bank)
        allowed_ids = list(getattr(self.cfg, "allowed_block_ids", []) or [])

        def _retrieve_with_allowlist(query_vec: Any, *, top_k: int, query_text: str) -> RetrieverResult:
            if allowed_ids:
                items = _select_kv_items_by_ids(bank=bank, ids=allowed_ids)
                dbg = {"source": "allowed_block_ids", "requested": len(allowed_ids), "returned": len(items)}
                return RetrieverResult(items=items, debug=dbg)
            return retriever.search(query_vec, top_k=int(top_k), filters=None, query_text=query_text)
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
        gate["candidate_schemas"] = candidate_schemas
        gate["active_contract_ids"] = [str(c.pattern_id) for c in (contracts or [])]
        if gate.get("decision") == "REFUSE":
            if gate.get("allow_rim_retry"):
                # retrieve-only retry (no generation)
                oversample_top_k = max(int(self.cfg.top_k), int(self.cfg.top_k) * (int(self.cfg.kv_refresh_rounds) + 1))
                rr = _retrieve_with_allowlist(q0, top_k=int(oversample_top_k), query_text=user_prompt)
                rr_items = _filter_items_by_allowed(rr.items, self.cfg.allowed_block_ids)
                rr_items, rank_debug, final_rank = _apply_list_feature_ranking(rr_items, slot_schema)
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
                gate_retry["candidate_schemas"] = candidate_schemas
                gate_retry["active_contract_ids"] = [str(c.pattern_id) for c in (contracts or [])]

                # Deterministic output for retrieve-only retry:
                # If the gate selects LIST_ONLY, project list items from semantic_instances instead of generating.
                rim_answer_retry = ""
                guard_style_retry = str(gate_retry.get("final_answer_style") or "")
                if str(guard_style_retry).upper() == "LIST_ONLY":
                    items, mapping, audit = _project_list_only(
                        semantic_instances=temp_instances,
                        retrieval_rank=final_rank,
                    )
                    if items:
                        if str(getattr(self.cfg, "answer_mode", "list_only") or "").strip().lower() == "narrative":
                            narrative_prompt = _build_list_only_narrative_prompt(
                                user_prompt=user_prompt,
                                items=[m["item_text"] for m in mapping],
                                max_items=16,
                            )
                            # NOTE: retrieve-only retry does not allow generation by design; keep fail-closed.
                            # Fall back to deterministic list for this branch.
                            rim_answer_retry = "\n".join([f"- {m['item_text']}" for m in mapping])
                        else:
                            rim_answer_retry = "\n".join([f"- {m['item_text']}" for m in mapping])
                    else:
                        # Must never return empty output on LIST_ONLY.
                        rim_answer_retry = "现有证据不足以回答该问题。"
                else:
                    # No generation is allowed in retrieve-only retry; fail closed with a short refusal.
                    rim_answer_retry = "现有证据不足以回答该问题。"

                out: Dict[str, Any] = {
                    "baseline_answer": baseline.strip(),
                    "pattern_first": _pattern_to_json(pattern_res, matched_patterns, matched_skeletons),
                    "gate": gate_retry,
                    "retrieve_more": False,
                    "rim_answer": str(rim_answer_retry or "").strip(),
                    "retrieval": {
                        "kv_dir": str(kv_dir),
                        "top_k": int(self.cfg.top_k),
                        "oversample_top_k": int(oversample_top_k),
                        "retriever_debug": dict(rr.debug or {}),
                        "rim_retry": True,
                        "list_rank_debug": rank_debug[: int(self.cfg.top_k)],
                        "final_rank": final_rank[: int(self.cfg.top_k)],
                        "list_like_candidate_count": int(len(_collect_list_like_ids(rr_items))),
                        "list_like_candidate_ids": _collect_list_like_ids(rr_items)[: int(self.cfg.top_k)],
                        "contract_validation": dict(contract_validation or {}),
                        "semantic_instances": temp_instances,
                        "slot_status_source": "semantic_instances",
                        "slot_status_snapshot": dict(slot_status or {}),
                    },
                }
                # Attach LIST_ONLY audit when applicable (debug-only, no effect on downstream logic).
                if str(guard_style_retry).upper() == "LIST_ONLY":
                    items, mapping, audit = _project_list_only(
                        semantic_instances=temp_instances,
                        retrieval_rank=final_rank,
                    )
                    out["retrieval"]["list_items_extracted"] = items
                    out["retrieval"]["list_items_source_block_ids"] = [m["source_block_id"] for m in mapping]
                    out["retrieval"]["list_only_audit"] = audit
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
        rr = _retrieve_with_allowlist(q0, top_k=int(oversample_top_k), query_text=user_prompt)
        rr_items = _filter_items_by_allowed(rr.items, self.cfg.allowed_block_ids)
        rr_items, rank_debug, final_rank = _apply_list_feature_ranking(rr_items, slot_schema)
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
            "list_rank_debug": rank_debug[: int(self.cfg.top_k)],
            "final_rank": final_rank[: int(self.cfg.top_k)],
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
        if bool(self.cfg.debug_retrieved_ids):
            retrieved_ids = [_kv_id(it) for it in chosen_items]
            out["retrieval"]["retrieved_ids"] = [x for x in retrieved_ids if x]
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

        # LIST_ONLY: deterministic projection of retrieval results.
        if str(guard_style or "").upper() == "LIST_ONLY":
            # If user explicitly requests LLM narrative answering, bypass LIST_ONLY entirely.
            # This keeps injection behavior intact but trades determinism for fluency.
            am = str(getattr(self.cfg, "answer_mode", "llm_prose") or "").strip().lower()
            if am in {"llm", "llm_prose"}:
                # Still compute list items for debug/traceability, but do not use projection as final answer.
                items, mapping, audit = _project_list_only(
                    semantic_instances=semantic_instances,
                    retrieval_rank=final_rank,
                )
                out["retrieval"]["list_items_extracted"] = items
                out["retrieval"]["list_items_source_block_ids"] = [m["source_block_id"] for m in mapping]
                out["retrieval"]["list_only_audit"] = audit
                # Force prose generation downstream.
                guard_style = "PROSE" if am == "llm_prose" else ""
            else:
                items, mapping, audit = _project_list_only(
                    semantic_instances=semantic_instances,
                    retrieval_rank=final_rank,
                )
                out["retrieval"]["list_items_extracted"] = items
                out["retrieval"]["list_items_source_block_ids"] = [m["source_block_id"] for m in mapping]
                out["retrieval"]["list_only_audit"] = audit
                if not items:
                    out["rim_answer"] = "现有证据不足以回答该问题。"
                    out["gate"] = dict(chosen_gate_after_validation or {})
                    return out
                # Optional: narrative rendering (still grounded; items are the only allowed facts).
                if str(getattr(self.cfg, "answer_mode", "list_only") or "").strip().lower() == "narrative":
                    narrative_prompt = _build_list_only_narrative_prompt(
                        user_prompt=user_prompt,
                        items=[m["item_text"] for m in mapping],
                        max_items=16,
                    )
                    narrative_prompt = self._format_prompt(
                        tokenizer, narrative_prompt, use_chat_template=bool(use_chat_template)
                    )
                    rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=narrative_prompt,
                        device=device,
                        past_key_values=chosen_pkv,
                        max_new_tokens=int(self.cfg.max_new_tokens_rim),
                        no_repeat_ngram_size=int(self.cfg.no_repeat_ngram_size),
                        repetition_penalty=float(self.cfg.repetition_penalty),
                    )
                    summary = str(rim_answer or "").strip()
                    if not summary:
                        out["rim_answer"] = "\n".join([f"- {m['item_text']}" for m in mapping])
                    else:
                        out["rim_answer"] = (summary + "\n\n" + "\n".join([f"- {m['item_text']}" for m in mapping])).strip()
                else:
                    out["rim_answer"] = "\n".join([f"- {m['item_text']}" for m in mapping])
                out["gate"] = dict(chosen_gate_after_validation or {})
                return out
        am2 = str(getattr(self.cfg, "answer_mode", "list_only") or "").strip().lower()
        if am2 == "llm_prose":
            guard_style = "PROSE"
        guarded_prompt = _apply_answer_style_guard(formatted_prompt, guard_style)
        rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=guarded_prompt,
            device=device,
            past_key_values=chosen_pkv,
            max_new_tokens=int(self.cfg.max_new_tokens_rim),
            no_repeat_ngram_size=int(self.cfg.no_repeat_ngram_size),
            repetition_penalty=float(self.cfg.repetition_penalty),
        )
        ans = str(rim_answer or "").strip()
        # If model still outputs bullets in llm_prose mode, rewrite into deterministic prose using extracted items.
        if str(getattr(self.cfg, "answer_mode", "") or "").strip().lower() == "llm_prose":
            extracted = out.get("retrieval", {}).get("list_items_extracted") if isinstance(out.get("retrieval"), dict) else None
            extracted_items = [str(x).strip() for x in (extracted or []) if str(x).strip()] if isinstance(extracted, list) else []
            if _looks_like_bullet_list(ans) and extracted_items:
                ans = _rewrite_items_to_prose(user_prompt=user_prompt, items=extracted_items)
        out["rim_answer"] = ans
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
    if style == "PROSE":
        guard = (
            "\n\n[STYLE_GUARD]\n"
            "Answer in Chinese prose paragraphs (1–2 short paragraphs). "
            "DO NOT use bullet lists or numbered lists. "
            "Be grounded in the injected evidence; do not invent facts. "
            "If evidence is insufficient, return only: '现有证据不足以回答该问题。'\n"
        )
        return str(prompt) + guard
    return str(prompt)


def _looks_like_bullet_list(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    bullet = 0
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            bullet += 1
            continue
        if len(ln) >= 3 and ln[0].isdigit() and ln[1:2] in {".", ")"}:
            bullet += 1
            continue
    return bullet >= max(2, int(len(lines) * 0.5))


def _rewrite_items_to_prose(*, user_prompt: str, items: Sequence[str], max_items: int = 12) -> str:
    it = [str(x).strip() for x in (items or []) if str(x).strip()]
    it = it[: max(1, int(max_items))]
    joined = "、".join(it)
    if not joined:
        return "现有证据不足以回答该问题。"
    return f"根据检索到的证据，主要包括：{joined}。"


def _build_list_only_narrative_prompt(*, user_prompt: str, items: Sequence[str], max_items: int = 16) -> str:
    """
    Build a constrained prompt for narrative rendering of LIST_ONLY answers.
    The model is allowed to write fluent text, but MUST NOT introduce new facts beyond `items`.
    """
    it = [str(x).strip() for x in (items or []) if str(x).strip()]
    it = it[: max(1, int(max_items))]
    bullet = "\n".join([f"- {x}" for x in it]) if it else ""
    return (
        str(user_prompt).strip()
        + "\n\n[INJECTED_LIST_ITEMS]\n"
        + bullet
        + "\n\n[STYLE_GUARD]\n"
        + "Write ONLY 1–3 Chinese sentences as a concise summary using ONLY the items above as factual content. "
        + "Do NOT add any new symptoms, mechanisms, locations, drugs, numbers, or approvals not present in the list. "
        + "Do NOT output any bullet list (the system will attach the list separately). "
        + "If the list is empty, output only: '现有证据不足以回答该问题。'\n"
    )


def _kv_id(it: Any) -> str:
    meta = getattr(it, "meta", None) or {}
    return str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")


def _select_kv_items_by_ids(*, bank: Any, ids: Sequence[str]) -> List[Any]:
    target = [str(x) for x in (ids or []) if str(x)]
    if not target:
        return []
    target_set = set(target)
    by_id: Dict[str, Any] = {}

    def _scan(b: Any) -> None:
        metas = getattr(b, "metas", None)
        k_ext = getattr(b, "k_ext", None)
        v_ext = getattr(b, "v_ext", None)
        if not isinstance(metas, list) or k_ext is None or v_ext is None:
            return
        for i, meta in enumerate(metas):
            if not isinstance(meta, dict):
                continue
            meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
            bid = str(
                meta.get("block_id")
                or meta.get("chunk_id")
                or meta.get("id")
                or meta.get("evidence_id")
                or meta_payload.get("block_id")
                or meta_payload.get("evidence_id")
                or ""
            )
            if not bid or bid not in target_set:
                continue
            if bid in by_id:
                continue
            try:
                by_id[bid] = KVItem(score=0.0, meta=meta, K_ext=k_ext[int(i)], V_ext=v_ext[int(i)])
            except Exception:
                continue

    if hasattr(bank, "shards"):
        for s in getattr(bank, "shards") or []:
            _scan(s)
    else:
        _scan(bank)

    return [by_id[x] for x in target if x in by_id]

def _filter_items_by_allowed(items: Sequence[Any], allowed_ids: Sequence[str]) -> List[Any]:
    if not items:
        return list(items or [])
    allowed = {str(x) for x in (allowed_ids or []) if str(x)}
    if not allowed:
        return list(items or [])
    out: List[Any] = []
    for it in items or []:
        bid = _kv_id(it)
        if bid and bid in allowed:
            out.append(it)
    return out


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
            if len(ap_abbr) < 3:
                continue
            if not _is_plausible_full(full, abbr_up):
                continue
            # Extra strictness for machine-extracted pairs: first word must be >=4 chars
            full_low = full.lower()
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


def _project_list_only(
    *,
    semantic_instances: Sequence[Dict[str, Any]],
    retrieval_rank: Sequence[str],
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, Any]]:
    """
    LIST_ONLY output is a deterministic projection of retrieval results.
    """
    block_items: Dict[str, List[str]] = {}
    block_span: Dict[str, str] = {}
    for inst in semantic_instances or []:
        if not isinstance(inst, dict):
            continue
        by_block = inst.get("value_cleaning_by_block") if isinstance(inst.get("value_cleaning_by_block"), dict) else {}
        for _, blocks in by_block.items():
            if not isinstance(blocks, dict):
                continue
            for bid, payload in blocks.items():
                if not isinstance(payload, dict):
                    continue
                vals = payload.get("cleaned_values") if isinstance(payload.get("cleaned_values"), list) else []
                if vals:
                    block_items.setdefault(str(bid), []).extend([str(v).strip() for v in vals if str(v).strip()])
        slots = inst.get("slots") if isinstance(inst.get("slots"), dict) else {}
        for _, ev_list in slots.items():
            if not isinstance(ev_list, list):
                continue
            for ev in ev_list:
                if not isinstance(ev, dict):
                    continue
                bid = str(ev.get("evidence_id") or "").strip()
                span = str(ev.get("span") or "").strip()
                if bid and span and bid not in block_span:
                    block_span[bid] = span

    output_items: List[str] = []
    mapping: List[Dict[str, str]] = []
    dropped_blocks_no_items = 0
    dropped_duplicates = 0
    seen_items: set[str] = set()
    # Conservative caps to avoid giant lists, but scale with retrieval size (no per-domain hardcoding).
    per_block_cap = 4
    total_cap = max(8, int(len(list(retrieval_rank or [])) * 4))
    for bid in retrieval_rank or []:
        items = block_items.get(bid) or []
        if not items:
            dropped_blocks_no_items += 1
            continue
        kept_in_block = 0
        for item in items:
            it = str(item or "").strip()
            if not it:
                continue
            key = it.lower()
            if key in seen_items:
                dropped_duplicates += 1
                continue
            seen_items.add(key)
            mapping.append({"item_text": it, "source_block_id": str(bid), "source_span": block_span.get(str(bid), "")})
            output_items.append(it)
            kept_in_block += 1
            if kept_in_block >= int(per_block_cap):
                break
            if len(output_items) >= int(total_cap):
                break
        if len(output_items) >= int(total_cap):
            break

    audit = {
        "retrieval_rank": list(retrieval_rank or []),
        "output_order": [m["source_block_id"] for m in mapping],
        "one_to_one_mapping": False,
        "per_block_cap": int(per_block_cap),
        "total_cap": int(total_cap),
        "dropped_blocks_no_items": int(dropped_blocks_no_items),
        "dropped_duplicates": int(dropped_duplicates),
        "reason": "no_source_span" if any(not m.get("source_span") for m in mapping) else "ok",
    }
    return output_items, mapping, audit


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




def _apply_list_feature_ranking(
    items: Sequence[Any],
    slot_schema: Optional[SlotSchema],
) -> Tuple[List[Any], List[Dict[str, Any]], List[str]]:
    """
    Apply frozen list-like ranking formula using KVBank meta.
    """
    if not items:
        return list(items or []), [], []
    requires_list_like = bool(slot_schema) and any(
        str(spec.inference_level).lower() == "schema" for spec in (slot_schema.slots or {}).values()
    )
    # For location-like schema slots, list-like should mean *structural location enumerations*,
    # not generic "key points" bullets.
    requires_location_list = bool(slot_schema) and any(
        str(getattr(spec, "semantic_type", "") or "").lower() == "location" for spec in (slot_schema.slots or {}).values()
    )
    ranked: List[Tuple[float, Any, Dict[str, Any]]] = []
    debug: List[Dict[str, Any]] = []
    for it in items or []:
        base_raw = float(getattr(it, "score", 0.0) or 0.0)
        base_score = max(0.0, min(1.0, base_raw))
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        list_like = bool(meta.get("list_like"))
        lfc = int(meta.get("list_feature_count") or 0)
        lconf = float(meta.get("list_confidence") or 0.0)
        boost = 1.0 + 0.15 * min(lfc, 3) + 0.20 * lconf if list_like else 1.0
        # Location-aware shaping (semantic_type-level): prefer paren_cases_capture/paren_cases/trigger_phrase
        # and demote bullet/numbering noise.
        signals = meta.get("list_signals") if isinstance(meta.get("list_signals"), list) else []
        sigs = [str(s) for s in signals if str(s).strip()]
        if requires_location_list and list_like:
            if any(s == "paren_cases_capture" or s == "paren_cases" for s in sigs):
                boost *= 1.25
            if any(str(s).startswith("trigger_phrase:") for s in sigs):
                boost *= 1.10
            if any(s in {"bullet", "numbering"} for s in sigs):
                boost *= 0.80
        final_score = 0.0 if (requires_list_like and not list_like) else base_score * boost
        dbg = {
            "block_id": bid,
            "base_score": base_score,
            "list_like": list_like,
            "list_feature_count": lfc,
            "list_confidence": lconf,
            "list_signals": sigs,
            "boost": boost,
            "final_score": final_score,
        }
        debug.append(dbg)
        if final_score > 0.0:
            ranked.append((final_score, it, dbg))
    ranked.sort(key=lambda x: x[0], reverse=True)
    final_items = [it for _, it, _ in ranked]
    final_rank = [str(getattr(it, "meta", {}).get("block_id") or "") for it in final_items]
    return final_items, debug, [x for x in final_rank if x]


def _collect_list_like_ids(items: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for it in items or []:
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        if not bid:
            continue
        has_list = bool(meta.get("list_like"))
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

