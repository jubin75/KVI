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
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ..domain_encoder import DomainEncoder, DomainEncoderConfig
from ..kv_bank import FaissKVBank
from ..pattern_contract import PatternContract, PatternContractValidator, run_pattern_first
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

        # 2) Pattern-first (non-semantic)
        pattern = (
            PatternRetriever.from_dir(str(self.cfg.pattern_index_dir))
            if str(getattr(self.cfg, "pattern_index_dir", "") or "").strip()
            else PatternRetriever()
        )
        pattern_res = pattern.retrieve(user_prompt)
        pattern_contracts = run_pattern_first(
            user_prompt,
            pattern_res,
            hard_pattern_ids=list(self.cfg.pattern_hard or []),
            soft_pattern_ids=list(self.cfg.pattern_soft or []),
        )
        validator = PatternContractValidator()

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
        gate = rim.introspection_gate(q_prime_vec=qprime, kv_relevance_delta=None, pattern_mismatch=False)
        retrieve_more = bool(force_rim) or bool(gate.get("retrieve_more") is True)

        out: Dict[str, Any] = {
            "baseline_answer": baseline.strip(),
            "pattern_first": _pattern_to_json(pattern_res, pattern_contracts),
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
            for it in rr.items:
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

            ext_by_layer = {
                li: stack_ext_kv_items_by_layer(items=batch, layer_id=int(li), batch_size=1, device=device, dtype=dtype)
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

            contract_validation = validator.validate_all(pattern_contracts, batch, block_text_lookup=block_text_lookup)
            chosen_contract_validation = contract_validation
            hard_missing = contract_validation.get("hard_missing") or []
            soft_missing = contract_validation.get("soft_missing") or []
            schema_missing = contract_validation.get("schema_missing") or []
            pattern_mismatch = bool(len(hard_missing) > 0)
            gate_after_validation = rim.introspection_gate(
                q_prime_vec=qprime,
                kv_relevance_delta=delta,
                pattern_mismatch=pattern_mismatch,
            )
            if pattern_mismatch:
                gate_after_validation["rationale"] = "pattern-missing-hard"
            elif not gate_after_validation.get("reject_current_kv") and soft_missing:
                gate_after_validation["rationale"] = "pattern-missing-soft"
            elif not gate_after_validation.get("reject_current_kv") and schema_missing:
                gate_after_validation["rationale"] = "schema-not-yet-instantiated"
            chosen_gate_after_validation = gate_after_validation

            if pattern_mismatch:
                # Reject current KV batch; try refresh if budget remains.
                if attempt_idx + 1 < attempts:
                    continue
                chosen_items = []
                chosen_pkv = None
                chosen_delta = None
                break

            chosen_items = batch
            chosen_pkv = pkv
            chosen_delta = delta
            th = float(self.cfg.kv_irrelevant_logit_delta_threshold)
            if delta is None or float(delta) >= th:
                break

        out["retrieval"] = {
            "kv_dir": str(kv_dir),
            "top_k": int(self.cfg.top_k),
            "oversample_top_k": int(oversample_top_k),
            "retriever_debug": dict(rr.debug or {}),
            "kv_refresh": {
                "attempts": int(attempts),
                "final_logit_delta_vs_zero_prefix": chosen_delta,
                "threshold": float(self.cfg.kv_irrelevant_logit_delta_threshold),
            },
            "contract_validation": dict(chosen_contract_validation or {}),
            "gate_after_validation": dict(chosen_gate_after_validation or {}),
        }

        # 5) Second pass generation with injected prefix
        if chosen_pkv is None:
            out["rim_answer"] = ""
            return out
        rim_answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
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


def _pattern_to_json(r: PatternRetrieveResult, contracts: Sequence[PatternContract]) -> Dict[str, Any]:
    return {
        "recall_size": int(r.recall_size),
        "debug": r.debug_info,
        "contracts": [
            {
                "pattern_id": c.pattern_id,
                "expected_information": c.expected_information,
                "required_signals": c.required_signals,
                "min_information_density": c.min_information_density,
                "level": c.level,
                "kind": c.kind,
            }
            for c in (contracts or [])
        ],
    }

