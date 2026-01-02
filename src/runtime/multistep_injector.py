"""
Multi-step Injection Runtime

严格遵循 PRD/多步注入的工程实现.md：
- 检索是推理的一部分：每一步都基于当前状态重新检索
- 每步注入 external KV tokens ≤ 1024（推荐），绝对上限 2048
- 每步注入由多个 256-token memory blocks 组成（4~8 个）
- 只在前几层（默认 0..3）注入
- 必须实现 stopping policy（不能硬编码步数）：至少两类信号 + 安全上限

工程实现选择（HF Transformers 友好）
- 外部 KV 以 past_key_values 前缀注入（不改写 attention forward）
- hidden state 演化：每一步以当前 prompt/已生成文本的 last_hidden 构造 query embedding

================================================================================
SLOT BYPASS SEMANTICS (CRITICAL ARCHITECTURE CONTRACT)
================================================================================

Slots are *retrieval-side bypass signals* (交通信号灯), NOT a semantic protocol (答题卡).

What slots MAY do (and only here, before generation):
- Influence candidate filtering / selection priority
- Influence injection eligibility (should_inject)
- Provide coverage statistics / debug logging

What slots MUST NOT do (hard prohibition):
- MUST NOT enter the LLM prompt (no "slots", "required/uncovered", checklists)
- MUST NOT influence decoding (no constraints/penalties conditional on slots)
- MUST NOT influence stop conditions (no "answered", no "covered => stop")
- MUST NOT influence postprocess validation/completeness checks
- MUST NOT be interpreted as "question answered" or "grounded"

Schema-only injection:
- Injected schema text is a *structural hint*, not evidence and not "completion"
- Injecting schema MUST NOT mark anything as semantically answered
- If slot filtering yields no injectable schema blocks, generation proceeds as base LLM

If you change this file, keep the LLM behavior identical to a system where slots do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import re
import torch

from ..retriever import Retriever
from .hf_cache_prefix_injection import ExtKV, build_past_key_values_prefix, stack_ext_kv_items_by_layer
from .schema_answerability import infer_slots_from_query
from .slot_registry import (
    adjudicable_slots_for_query,
    classify_fact_types,
    domain_prior_allowed_for_fact_types,
    fact_types_need_schema_coverage,
    speculative_allowed_for_fact_types,
)


@dataclass(frozen=True)
class MultiStepConfig:
    inject_layers: Sequence[int] = (0, 1, 2, 3)
    block_tokens: int = 256
    max_step_tokens: int = 1024
    max_total_tokens: int = 2048
    max_steps: int = 8
    top_k_blocks: int = 8  # retrieve candidates
    max_blocks_per_step: int = 8  # cap selected blocks per step (RoPE safety: set to 1 if needed)
    min_kv_len_to_inject: int = 128
    allow_table_injection: bool = False

    # stopping thresholds
    min_logit_delta: float = 1e-3
    min_hidden_delta: float = 1e-3
    redundancy_sim_threshold: float = 0.95

    # attention convergence (external KV attention entropy)
    use_attention_entropy: bool = True
    entropy_window: int = 2  # require consecutive decrease over this many steps
    entropy_threshold: float = 0.35  # normalized entropy in [0,1], lower means more concentrated

    # debug
    debug_print_candidates_top_n: int = 0  # if >0, print top-N retrieved candidates each step (block_id + score)
    # struct-slot mode (Evidence -> Schema -> Injection/Generation)
    struct_slots_max_prefix_tokens: int = 192
    # Optional per-layer decay for struct-slot prefix (layer_id -> scale in (0,1]).
    # If empty/None, no decay is applied.
    struct_slots_decay: Optional[Dict[int, float]] = None
    # Schema selection: retrieval-side only (slots are bypass signals; not semantic "answeredness").
    schema_max_selected_per_step: int = 1
    # (compat) Optional slot signal override for retrieval-side filtering only.
    schema_required_slots: Optional[Sequence[str]] = None
    # Stop-rule epsilon for "injection has no effect" sanity check.
    stop_epsilon_logit_delta_vs_zero: float = 0.05

    # Grounding evidence rendering (pre-generation only; slot-agnostic)
    evidence_max_sentences: int = 3
    # If set, force evidence text to a single language by filtering (no translation).
    # Allowed values: "zh" | "en" | None
    evidence_target_lang: Optional[str] = None
    # Near-duplicate threshold for evidence sentence dedupe (SequenceMatcher-based, no embeddings).
    evidence_near_dup_threshold: float = 0.95

    # Decoding stability
    repetition_penalty: float = 1.08

    # Scheme A: evidence-aware slot gating at generation time.
    # If enabled, for required slots that are NOT covered by evidence, we emit fixed uncertainty templates
    # and do not call the model to elaborate those aspects.
    enable_evidence_slot_gating: bool = True

    # Three knowledge layers output (docs/74_three_knowledge_layers.md)
    # - Layer 0: Evidence-Bound (slot-gated, no expansion)
    # - Layer 1: Domain Prior (textbook-level explanation; does NOT require evidence coverage)
    # - Layer 2: Speculative/Open (optional; clearly marked; uncertainty wording)
    enable_three_knowledge_layers: bool = True
    enable_speculative_layer: bool = False
    layer1_max_new_tokens: int = 256
    layer2_max_new_tokens: int = 192
    # Deterministic layer compression (do not rely on prompt verbosity).
    # These are hard caps applied in code for L1/L2 to prevent redundancy/truncation.
    l1_max_sentences: int = 3
    l1_max_tokens: int = 80
    l2_max_sentences: int = 3
    l2_max_tokens: int = 100

    # Schema KV injection scale (applied to V only; <1.0 weakens bias, >1.0 strengthens).
    # NOTE: Schema KV comes from kvbank_schema (precomputed K_ext/V_ext). No schema text is tokenized.
    schema_kv_scale: float = 1.0


@dataclass
class StepDebug:
    step: int
    selected_block_ids: List[str]
    injected_tokens: int
    total_injected_tokens: int
    logit_delta: float
    hidden_delta: float
    # More meaningful for step=0: compare injected-prefix forward against a same-length ZERO prefix.
    # If these stay ~0, injection likely isn't affecting model computation (or prefix is being ignored).
    logit_delta_vs_zero_prefix: Optional[float] = None
    hidden_delta_vs_zero_prefix: Optional[float] = None
    # Defaults are required because these fields come after defaulted fields above (dataclasses constraint).
    redundancy_hits: int = 0
    ext_attn_entropy: Optional[float] = None
    retrieved_candidates: int = 0
    note: Optional[str] = None


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


class MultiStepInjector:
    def __init__(
        self,
        *,
        retriever: Retriever,
        cfg: MultiStepConfig,
        allowed_block_ids: Optional[Set[str]] = None,
        block_text_lookup: Optional[Callable[[str], Optional[str]]] = None,
        grounding_retriever: Optional[Any] = None,
    ) -> None:
        self.retriever = retriever
        self.grounding_retriever = grounding_retriever
        self.cfg = cfg
        self.allowed_block_ids = allowed_block_ids
        self.block_text_lookup = block_text_lookup
        self.used_block_ids: Set[str] = set()
        self.total_injected_tokens = 0

    def _select_schema_texts(
        self,
        items: Sequence[Any],
        *,
        query_text: str,
        slot_signal: Set[str],
    ) -> Tuple[List[Any], List[str], int, Dict[str, Any]]:
        """
        Select schema candidates (by ID de-dupe + allowlist) and return their TEXT (from block_text_lookup).

        IMPORTANT: Schema injection must NOT use evidence/raw KV. We only use schema texts here.

        Slot bypass invariants:
        - `slot_signal` is ONLY a pre-generation gating/prioritization signal.
        - This function MUST NOT create/update any semantic "answered" state.
        - This function MUST NOT imply semantic completeness or grounding.

        Returns: selected_items, selected_texts, redundancy_hits, selector_debug
        """
        # block_text_lookup is optional: schema texts must never enter the model token sequence.
        # We may still use schema text for retrieval-side lexical ranking when available.

        candidates: List[Any] = []
        candidate_ids: List[str] = []
        candidate_texts: List[str] = []
        redundancy_hits = 0

        for it in items:
            bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
            if not bid:
                continue
            if self.allowed_block_ids is not None and str(bid) not in self.allowed_block_ids:
                continue
            if str(bid) in self.used_block_ids:
                redundancy_hits += 1
                continue
            t = self.block_text_lookup(str(bid)) if self.block_text_lookup is not None else ""
            candidates.append(it)
            candidate_ids.append(str(bid))
            candidate_texts.append(t.strip() if isinstance(t, str) else "")
            if len(candidates) >= int(self.cfg.top_k_blocks):
                break

        # Slots are bypass-side signals only:
        # - used ONLY to decide whether a candidate block is eligible for injection (and to prioritize)
        # - MUST NOT be treated as semantic completion/answering
        cand_answerable_slots: List[List[str]] = []
        for it in candidates:
            meta = getattr(it, "meta", {}) or {}
            # Prefer answerable_slots over slots (answerable_slots = subset that evidence can substantively answer).
            ans_slots = meta.get("answerable_slots")
            if not isinstance(ans_slots, list):
                ans_slots = (meta.get("metadata") or {}).get("answerable_slots") if isinstance(meta.get("metadata"), dict) else None
            if isinstance(ans_slots, list) and ans_slots:
                cand_answerable_slots.append([str(s) for s in ans_slots if str(s).strip()])
            else:
                # Fallback: use slots (legacy schema blocks without explicit answerable_slots).
                slots = meta.get("slots")
                if not isinstance(slots, list):
                    slots = (meta.get("metadata") or {}).get("slots") if isinstance(meta.get("metadata"), dict) else []
                if not isinstance(slots, list):
                    slots = []
                cand_answerable_slots.append([str(s) for s in slots if str(s).strip()])

        # ---- bypass-only API: should we inject this block under the current slot signal? ----
        import re

        def _norm_slot(s: str) -> str:
            return (s or "").strip().lower().replace(" ", "").replace("_", "")

        slot_norm = {_norm_slot(s) for s in (slot_signal or set()) if _norm_slot(s)}

        def should_inject(*, block_slots: List[str]) -> bool:
            # If we have no slot signal, allow injection (pure bypass signal).
            if not slot_norm:
                return True
            bs = {_norm_slot(s) for s in (block_slots or []) if _norm_slot(s)}
            if not bs:
                return False
            for a in slot_norm:
                for b in bs:
                    if a == b or (a in b) or (b in a):
                        return True
            return False

        eligible = [i for i, s in enumerate(cand_answerable_slots) if should_inject(block_slots=s)]
        if not eligible:
            # No eligible schema blocks under current slot signal -> inject nothing; caller proceeds as base LLM.
            sel_dbg = {
                "selector": "slot_bypass_filter_then_lexical_overlap",
                "slot_signal": sorted(list(slot_signal or set())),
                "candidates": int(len(candidate_ids)),
                "eligible": 0,
                "selected_ids": [],
            }
            return [], [], redundancy_hits, sel_dbg

        # Rank eligible blocks by cheap lexical overlap (still ignores ANN score).
        def _has_cjk(s: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

        def _query_terms(q: str, max_terms: int = 32) -> List[str]:
            q = (q or "").strip().lower()
            if not q:
                return []
            if _has_cjk(q):
                chars = [c for c in q if re.match(r"[\u4e00-\u9fff]", c)]
                bigrams = ["".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))]
                return bigrams[:max_terms]
            toks = re.findall(r"[a-z0-9][a-z0-9\-\_]{1,30}", q)
            stop = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "of",
                "to",
                "in",
                "on",
                "for",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "as",
                "that",
                "this",
                "these",
                "those",
                "it",
                "its",
            }
            toks = [t for t in toks if t not in stop]
            return toks[:max_terms]

        terms = _query_terms(str(query_text or ""), max_terms=32)
        scored: List[Tuple[int, int]] = []
        for i in eligible:
            tl = (candidate_texts[i] or "").lower()
            ov = 0
            for t in terms:
                if t and t in tl:
                    ov += 1
            scored.append((ov, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        # If we have no schema text (all empty), just pick the first eligible blocks to keep behavior stable.
        if all(not (candidate_texts[i] or "").strip() for i in eligible):
            sel_idx = eligible[: max(1, int(self.cfg.schema_max_selected_per_step))]
        else:
            sel_idx = [i for _ov, i in scored[: max(1, int(self.cfg.schema_max_selected_per_step))]]
        sel_dbg = {
            "selector": "slot_bypass_filter_then_lexical_overlap",
            "slot_signal": sorted(list(slot_signal or set())),
            "candidates": int(len(candidate_ids)),
            "eligible": int(len(eligible)),
            "selected_ids": [candidate_ids[i] for i in sel_idx],
        }
        selected: List[Any] = []
        texts: List[str] = []
        for i in sel_idx[: max(1, int(self.cfg.schema_max_selected_per_step))]:
            selected.append(candidates[i])
            texts.append(candidate_texts[i])
            self.used_block_ids.add(candidate_ids[i])

        return selected, texts, redundancy_hits, sel_dbg

    def _schema_kv_prefix_from_items(
        self,
        *,
        items: Sequence[Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, ExtKV]:
        """
        Build schema KV prefix from retrieved schema KVItems (kvbank_schema) WITHOUT tokenizing schema text.

        This is the "real" schema KV injection:
        - Use stored K_ext/V_ext per block (multi-layer) and stack per injected layer.
        - Optionally scale V by cfg.schema_kv_scale.
        """
        if not items:
            return {}
        ext_by_layer: Dict[int, ExtKV] = {}
        for li in self.cfg.inject_layers:
            try:
                ext = stack_ext_kv_items_by_layer(
                    items=items,
                    layer_id=int(li),
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
            except Exception:
                continue
            scale = float(getattr(self.cfg, "schema_kv_scale", 1.0))
            if scale != 1.0:
                ext = ExtKV(K=ext.K, V=ext.V * scale)
            ext_by_layer[int(li)] = ext
        return ext_by_layer

    def run(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
        query_text: Optional[str] = None,
        device: torch.device,
        max_new_tokens: int = 128,
        query_embed_fn: Optional[Any] = None,
        no_repeat_ngram_size: int = 0,
        ground_with_selected_text: bool = False,
        grounding_instructions: Optional[str] = None,
        use_struct_slots: bool = False,
    ) -> Tuple[str, List[StepDebug]]:
        """
        执行多步注入推理：
        - 每一步：根据当前状态构造 query embedding → 检索 blocks → 注入 → 走一次 forward 更新 hidden/logits
        - 满足 stopping policy 则停止并进入最终 generate

        Slot bypass invariants (do not violate):
        - Slots MUST NOT influence prompt text, decoding, stop rules, or postprocess.
        - Slots may only influence whether a block is injected (retrieval-side bypass signal).
        """

        step_debugs: List[StepDebug] = []
        last_past_key_values: Any = None
        grounding_texts: List[str] = []
        last_query_vec: Optional[np.ndarray] = None

        if not bool(use_struct_slots):
            # Enforce the new contract: schema KV is the ONLY KV allowed to constrain generation.
            raise ValueError("This runtime enforces schema-only injection. Pass use_struct_slots=True.")

        # Slot signals are bypass-side only (retrieval/injection eligibility).
        # They MUST NOT influence decoding, stop rules, prompt contracts, or postprocess.
        qtxt0 = str(query_text or prompt)
        inferred_slots = (
            {str(s) for s in self.cfg.schema_required_slots if str(s).strip()}
            if self.cfg.schema_required_slots
            else set(infer_slots_from_query(qtxt0))
        )
        fact_types = set(classify_fact_types(qtxt0))
        adjudicable_slots = adjudicable_slots_for_query(inferred_slots=inferred_slots, fact_types=fact_types)
        need_cov = fact_types_need_schema_coverage(fact_types)
        injection_enabled = bool(adjudicable_slots)  # fail-closed: no adjudicable slots => no schema injection
        slot_signal = set(adjudicable_slots)

        # If we need attentions (entropy signal), force an attention implementation that supports it.
        # Some HF backends (sdpa/flash-attn) don't support output_attentions.
        if bool(self.cfg.use_attention_entropy) and hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")  # type: ignore[attr-defined]
            except Exception:
                pass

        # initial inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        prompt_len = int(input_ids.shape[1])

        prev_logits = None
        prev_hidden = None
        entropy_hist: List[float] = []

        if query_text is None:
            query_text = prompt

        # Fail-closed injection: if the query does not map to any adjudicable slots, do NOT inject schema KV.
        for step in range(int(self.cfg.max_steps) if bool(injection_enabled) else 0):
            # ---- build query embedding from current state ----
            if query_embed_fn is not None:
                query_vec = query_embed_fn(query_text)
            else:
                # fallback: use last_hidden mean pool from current prompt (cheap demo)
                with torch.no_grad():
                    out_q = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                last_hidden = out_q.hidden_states[-1]  # [1,T,H]
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                query_vec = pooled[0].to(torch.float32).cpu().numpy()
            last_query_vec = query_vec

            # ---- retrieve & select blocks ----
            result = self.retriever.search(query_vec, top_k=self.cfg.top_k_blocks, filters=None, query_text=query_text)
            if int(self.cfg.debug_print_candidates_top_n) > 0:
                topn = int(self.cfg.debug_print_candidates_top_n)
                cands = []
                for it in (result.items or [])[:topn]:
                    bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
                    src = (it.meta or {}).get("retrieval_source")
                    cands.append(f"{bid}@{float(getattr(it, 'score', 0.0)):.4f}" + (f"({src})" if src else ""))
                shown = min(int(topn), len(result.items or []))
                print(f"[retrieval] step={step} top{shown}=" + " | ".join(cands), flush=True)
            selected, schema_texts, redundancy_hits, sel_dbg = self._select_schema_texts(
                result.items,
                query_text=str(query_text or ""),
                slot_signal=set(slot_signal),
            )

            # Stop-rule (slot-agnostic):
            # 1) No selectable schema remains (retrieval exhausted / allowlist filtered / de-dupe).
            if not selected:
                step_debugs.append(
                    StepDebug(
                        step=step,
                        selected_block_ids=[],
                        injected_tokens=0,
                        total_injected_tokens=self.total_injected_tokens,
                        logit_delta=0.0,
                        hidden_delta=0.0,
                        redundancy_hits=int(redundancy_hits),
                        ext_attn_entropy=None,
                        retrieved_candidates=int(len(result.items)),
                        note=f"stop_reason=no_selectable_schema selector={sel_dbg}",
                    )
                )
                break

            # 2) redundancy_hits > 0 (de-dupe/empty-text indicates we're circling already-used blocks)
            if int(redundancy_hits) > 0:
                step_debugs.append(
                    StepDebug(
                        step=step,
                        selected_block_ids=[str(it.meta.get("block_id") or it.meta.get("chunk_id")) for it in selected],
                        injected_tokens=0,
                        total_injected_tokens=self.total_injected_tokens,
                        logit_delta=0.0,
                        hidden_delta=0.0,
                        redundancy_hits=int(redundancy_hits),
                        ext_attn_entropy=None,
                        retrieved_candidates=int(len(result.items)),
                        note=f"stop_reason=redundancy_hits selector={sel_dbg}",
                    )
                )
                break

            if not selected:
                # no new info -> stop (but still emit a debug record for observability)
                step_debugs.append(
                    StepDebug(
                        step=step,
                        selected_block_ids=[],
                        injected_tokens=0,
                        total_injected_tokens=self.total_injected_tokens,
                        logit_delta=0.0,
                        hidden_delta=0.0,
                        redundancy_hits=redundancy_hits,
                        ext_attn_entropy=None,
                        retrieved_candidates=int(len(result.items)),
                        note=(
                            "no_selected_blocks (common causes: kv_dir/DOMAIN_ENCODER mismatch, "
                            "allowed_langs+blocks_jsonl allowlist mismatch, top_k too small, or max_step_tokens too small)"
                        ),
                    )
                )
                break

            # ---- build prefix cache ----
            # Schema KV-only injection path (NO schema text tokens):
            # - Use retrieved schema KVItems' stored K_ext/V_ext (kvbank_schema)
            # - Stack them into a cache prefix per injected layer
            # - Inject via past_key_values prefix
            dtype = next(model.parameters()).dtype
            schema_block_ids = [str(it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")) for it in selected]
            schema_block_ids = [x for x in schema_block_ids if x]
            ext_by_layer = self._schema_kv_prefix_from_items(items=selected, device=device, dtype=dtype)
            if not ext_by_layer:
                # No injection; proceed with base LLM behavior (no suppression).
                step_debugs.append(
                    StepDebug(
                        step=step,
                        selected_block_ids=schema_block_ids,
                        injected_tokens=0,
                        total_injected_tokens=self.total_injected_tokens,
                        logit_delta=0.0,
                        hidden_delta=0.0,
                        redundancy_hits=int(redundancy_hits),
                        ext_attn_entropy=None,
                        retrieved_candidates=int(len(result.items)),
                        note=f"no_schema_kv_prefix_built selector={sel_dbg}",
                    )
                )
                break

            past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
            # prefix length is the ext_len of the stacked KV (token-invisible, cache-only)
            injected_tokens_effective = int(next(iter(ext_by_layer.values())).K.shape[-2])

            # update total injected token accounting (used by stopping policy + safety cap)
            self.total_injected_tokens += int(injected_tokens_effective)
            last_past_key_values = past_key_values

            # Build a same-shape ZERO prefix as a control for measuring injection effect.
            try:
                zero_by_layer = {li: ExtKV(K=torch.zeros_like(ext.K), V=torch.zeros_like(ext.V)) for li, ext in ext_by_layer.items()}
                past_key_values_zero = build_past_key_values_prefix(model=model, ext_kv_by_layer=zero_by_layer)
            except Exception:
                past_key_values_zero = None

            # ---- one forward step to update logits/hidden ----
            # IMPORTANT: when using an *external prefix cache*, many HF models expect the
            # attention_mask/position_ids/cache_position to account for the prefix length.
            prefix_len = int(injected_tokens_effective)
            prefix_mask = torch.ones((attention_mask.shape[0], prefix_len), dtype=attention_mask.dtype, device=device)
            attn = torch.cat([prefix_mask, attention_mask], dim=1)
            pos0 = torch.arange(prefix_len, prefix_len + prompt_len, device=device, dtype=torch.long).unsqueeze(0)
            cache_pos0 = torch.arange(prefix_len, prefix_len + prompt_len, device=device, dtype=torch.long)

            # Control forward: same position offset + same prefix length, but zero KV content.
            ctrl_logits = None
            ctrl_hidden = None
            if past_key_values_zero is not None:
                with torch.no_grad():
                    try:
                        out0 = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            position_ids=pos0,
                            cache_position=cache_pos0,
                            use_cache=True,
                            past_key_values=past_key_values_zero,
                            output_hidden_states=True,
                            output_attentions=False,
                            return_dict=True,
                        )
                    except TypeError:
                        out0 = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            position_ids=pos0,
                            use_cache=True,
                            past_key_values=past_key_values_zero,
                            output_hidden_states=True,
                            output_attentions=False,
                            return_dict=True,
                        )
                ctrl_logits = out0.logits[:, -1, :]
                ctrl_hidden = out0.hidden_states[-1][:, -1, :]

            with torch.no_grad():
                try:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        position_ids=pos0,
                        cache_position=cache_pos0,
                        use_cache=True,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        output_attentions=bool(self.cfg.use_attention_entropy),
                        return_dict=True,
                    )
                except TypeError:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        position_ids=pos0,
                        use_cache=True,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        output_attentions=bool(self.cfg.use_attention_entropy),
                        return_dict=True,
                    )
            logits = out.logits[:, -1, :]  # [1,V]
            hidden = out.hidden_states[-1][:, -1, :]  # [1,H]

            # ---- external KV attention entropy (normalized) ----
            ext_entropy: Optional[float] = None
            if self.cfg.use_attention_entropy and getattr(out, "attentions", None) is not None:
                # out.attentions: tuple(num_layers) of [B, heads, T, S]
                # For injected layers, S = ext_len + T (ext prefix first).
                entropies: List[float] = []
                for li in self.cfg.inject_layers:
                    if li < 0 or li >= len(out.attentions):
                        continue
                    attn = out.attentions[li]  # [B, heads, T, S]
                    if attn is None:
                        continue
                    # ext_len from any selected item (all selected are 256-token blocks, but kv_len may vary)
                    # In struct-slot mode, the prefix length is the schema prefix length, not sum(kv_len).
                    ext_len = int(prefix_len)
                    if ext_len <= 0:
                        continue
                    # take last token query row, restrict to external prefix positions [0:ext_len]
                    w = attn[0, :, -1, :ext_len]  # [heads, ext_len]
                    mass = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    p = w / mass
                    # entropy per head: -sum p log p
                    h = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)  # [heads]
                    # normalize by log(ext_len) to [0,1]
                    denom = float(np.log(max(ext_len, 2)))
                    h_norm = (h / denom).mean().item() if denom > 0 else float(h.mean().item())
                    entropies.append(float(h_norm))
                if entropies:
                    ext_entropy = float(sum(entropies) / len(entropies))
                    entropy_hist.append(ext_entropy)

            # stopping signals
            logit_delta = 0.0
            hidden_delta = 0.0
            if prev_logits is not None:
                logit_delta = float(torch.mean(torch.abs(logits - prev_logits)).item())
            if prev_hidden is not None:
                hidden_delta = float(torch.mean(torch.abs(hidden - prev_hidden)).item())

            logit_delta_vs_zero = None
            hidden_delta_vs_zero = None
            if ctrl_logits is not None:
                logit_delta_vs_zero = float(torch.mean(torch.abs(logits - ctrl_logits)).item())
            if ctrl_hidden is not None:
                hidden_delta_vs_zero = float(torch.mean(torch.abs(hidden - ctrl_hidden)).item())

            # 4) logit_delta_vs_zero_prefix < epsilon triggers STOP after this step.
            stop_reason = None
            if logit_delta_vs_zero is not None and float(logit_delta_vs_zero) < float(self.cfg.stop_epsilon_logit_delta_vs_zero):
                stop_reason = f"logit_delta_vs_zero<{float(self.cfg.stop_epsilon_logit_delta_vs_zero):.4f}"

            step_debugs.append(
                StepDebug(
                    step=step,
                    selected_block_ids=[str(it.meta.get("block_id") or it.meta.get("chunk_id")) for it in selected],
                    injected_tokens=injected_tokens_effective,
                    total_injected_tokens=self.total_injected_tokens,
                    logit_delta=logit_delta,
                    hidden_delta=hidden_delta,
                    logit_delta_vs_zero_prefix=logit_delta_vs_zero,
                    hidden_delta_vs_zero_prefix=hidden_delta_vs_zero,
                    redundancy_hits=redundancy_hits,
                    ext_attn_entropy=ext_entropy,
                    retrieved_candidates=int(len(result.items)),
                    note=(f"schema_selector={sel_dbg}" + (f" stop_reason={stop_reason}" if stop_reason else "")),
                )
            )

            prev_logits = logits
            prev_hidden = hidden

            # stopping policy (slot-agnostic): require >=2 signals among
            # - marginal gain small (logit+hidden deltas)
            # - redundancy observed
            # - safety caps (max_total_tokens)
            # - optional attention convergence
            signals = 0
            if logit_delta < self.cfg.min_logit_delta and hidden_delta < self.cfg.min_hidden_delta:
                signals += 1  # marginal gain small
            if redundancy_hits > 0:
                signals += 1  # redundancy observed
            if self.total_injected_tokens >= self.cfg.max_total_tokens:
                signals += 1  # safety cap

            # attention convergence: entropy decreases consecutively and below threshold
            if self.cfg.use_attention_entropy and len(entropy_hist) >= self.cfg.entropy_window:
                window = entropy_hist[-self.cfg.entropy_window :]
                if all(window[i] > window[i + 1] for i in range(len(window) - 1)) and window[-1] <= self.cfg.entropy_threshold:
                    signals += 1

            if stop_reason is not None:
                break
            if signals >= 2:
                break

        def _extract_evidence(
            texts: List[str],
            *,
            q: str,
            keywords: Optional[Sequence[str]] = None,
        ) -> str:
            """
            Build a short, high-signal evidence appendix to reduce hallucinations.
            We keep it heuristic and conservative: pick a few short sentences to append.

            IMPORTANT (design):
            - This function is ONLY about prompt-grounding for final decoding (when ground_with_selected_text=True).
              It is NOT the retrieval query rewrite.
            - By default, we avoid hardcoding "intent" rules: we simply pick a few sentences from the selected blocks.
            - If `keywords` is provided, we will prefer sentences containing those keywords (user-configurable hook).
            - MUST strip instruction/meta-language artifacts (e.g., "请回答", "依据", "Human:", "Assistant:").
            """
            if not texts:
                return ""
            kws = [str(x).strip().lower() for x in (keywords or []) if str(x).strip()]
            picked: List[str] = []
            import re
            from difflib import SequenceMatcher

            # Instruction/meta-language patterns to strip (case-insensitive).
            meta_patterns = [
                r"^(请|Please)\s*(回答|answer|respond)",
                r"^(依据|Evidence|Proof|证据)[：:]\s*",
                r"(Human|User|Assistant|System)\s*[：:]",
                r"^\[?(注|Note|提示|Hint)\]?[：:]\s*",
                r"^(主要|其他).*(途径|途经|方式)[：:]",  # instruction-like headers
                r"^证据原文[：:]",
                r"^回答要求",
                # Bibliography / reference noise (must NOT be treated as evidence)
                r"^(参考文献|references?)\s*[:：]?",
                r"\bet\s+al\.\b",
                r"\bdoi\s*[:：]",
                r"^\s*\[[0-9]{1,3}\]\s*",  # [1] ...
            ]
            meta_re = re.compile("|".join(meta_patterns), re.IGNORECASE)

            def _is_meta(s: str) -> bool:
                """Return True if sentence is instruction/meta-language (should be stripped)."""
                s_stripped = s.strip()
                if meta_re.search(s_stripped):
                    return True
                # Very short fragments that look like labels.
                if len(s_stripped) < 10 and (":" in s_stripped or "：" in s_stripped):
                    return True
                return False

            def _has_cjk(s: str) -> bool:
                return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

            def _sent_lang(s: str) -> str:
                # very lightweight: treat any CJK as "zh", otherwise "en"
                return "zh" if _has_cjk(s) else "en"

            def _norm_sent(s: str) -> str:
                s0 = (s or "").strip().lower()
                s0 = re.sub(r"\s+", " ", s0)
                # normalize quotes/punct a bit
                s0 = s0.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
                s0 = re.sub(r"[，,。\.；;：:！!？?\(\)\[\]\{\}<>\"']", "", s0)
                return s0

            def _near_dup(a: str, b: str, *, thr: float) -> bool:
                na = _norm_sent(a)
                nb = _norm_sent(b)
                if not na or not nb:
                    return False
                if na == nb:
                    return True
                if na in nb or nb in na:
                    return True
                if min(len(na), len(nb)) < 24:
                    return False
                return SequenceMatcher(a=na, b=nb).ratio() >= float(thr)

            # Language unification: prefer query language, and keep evidence in a single language.
            forced_lang = str(getattr(self.cfg, "evidence_target_lang", None) or "").strip().lower()
            forced_lang = forced_lang if forced_lang in {"zh", "en"} else ""
            preferred_lang = forced_lang or _sent_lang(q or "")
            candidates_by_lang: dict[str, list[str]] = {"zh": [], "en": []}

            for t in texts:
                # Split into rough sentences.
                sents = [s.strip() for s in re.split(r"(?<=[\.\!\?。！？])\s+", t) if s.strip()]
                for s in sents:
                    # Skip instruction/meta-language sentences.
                    if _is_meta(s):
                        continue
                    sl = s.lower()
                    # If keywords provided, prefer matching ones; otherwise accept any sentence.
                    if kws and not any(k in sl for k in kws):
                        continue
                    # Collect candidates (we will dedupe + language-filter later).
                    candidates_by_lang[_sent_lang(s)].append(s)
                # keep scanning; we dedupe across blocks too

            def _select_from_lang(lang: str, max_n: int) -> list[str]:
                out: list[str] = []
                for s in candidates_by_lang.get(lang, []):
                    if _is_meta(s):
                        continue
                    # Deduplicate semantically similar sentences (approx; no embeddings).
                    thr = float(getattr(self.cfg, "evidence_near_dup_threshold", 0.95))
                    if any(_near_dup(s, prev, thr=thr) for prev in out):
                        continue
                    out.append(s)
                    if len(out) >= max_n:
                        break
                return out

            # Prefer evidence in the same language as the query if available.
            max_n = int(getattr(self.cfg, "evidence_max_sentences", 3))
            picked = _select_from_lang(preferred_lang, max_n)
            if not picked:
                # If a target language was forced but we filtered to empty, warn and fall back to target_lang=None behavior.
                if forced_lang:
                    print("[warn] evidence_language_filtered_to_empty", flush=True)
                    # Retry with query language first, then the other language (still avoid mixing).
                    qlang = _sent_lang(q or "")
                    picked = _select_from_lang(qlang, max_n)
                    if not picked:
                        other = "zh" if qlang == "en" else "en"
                        picked = _select_from_lang(other, max_n)
                else:
                    # Fallback: pick a single language (whichever has more candidates) to avoid CN/EN mixing.
                    alt = "zh" if len(candidates_by_lang["zh"]) >= len(candidates_by_lang["en"]) else "en"
                    picked = _select_from_lang(alt, max_n)

            if not picked:
                # fallback: first 1–2 sentences (after filtering meta).
                t0 = texts[0]
                sents = [s.strip() for s in re.split(r"(?<=[\.\!\?。！？])\s+", t0) if s.strip() and not _is_meta(s)]
                picked = sents[:2] if sents else [t0[:400]]
            picked = [p.replace("\n", " ").strip() for p in picked if p.strip()]
            # Evidence text only (no bullets) to reduce "echo bullets" degeneration.
            return "\n".join(picked[:max_n]).strip()

        def _lang_of_query(q: str) -> str:
            return "zh" if re.search(r"[\u4e00-\u9fff]", q or "") else "en"

        def _has_cjk(s: str) -> bool:
            """Lightweight language detector: any CJK => zh."""
            return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

        def _slot_question(slot: str, *, lang: str) -> str:
            """
            Slot-specific question used for covered-slot generation.
            IMPORTANT: must not mention slot ids/field names; only natural-language questions.
            """
            s = (slot or "").strip().lower()
            if lang == "zh":
                return {
                    "transmission": "该疾病的主要传播途径是什么？",
                    "pathogenesis": "该疾病的发病机制是什么？",
                    "clinical_features": "该疾病的主要临床表现是什么？",
                    "diagnosis": "该疾病如何诊断/检测？",
                    "treatment": "该疾病如何治疗？",
                    "epidemiology": "该疾病的流行病学特征是什么？",
                    "risk_factors": "该疾病的风险因素是什么？",
                    "complications": "该疾病常见并发症有哪些？",
                    "prognosis": "该疾病的预后如何？",
                    "prevention": "该疾病如何预防？",
                    "disease_full_name": "该疾病/病原的全称是什么？",
                    "geographic_distribution": "该疾病/病原的地区分布如何？",
                    "overview": "请简要概述该疾病。",
                    "mechanism": "该疾病的机制是什么？",
                }.get(s, "请回答用户问题中相关方面。")
            return {
                "transmission": "What is the primary mode of transmission?",
                "pathogenesis": "What is the pathogenesis?",
                "clinical_features": "What are the main clinical features?",
                "diagnosis": "How is it diagnosed/tested?",
                "treatment": "How is it treated?",
                "epidemiology": "What are key epidemiological characteristics?",
                "risk_factors": "What are the risk factors?",
                "complications": "What complications are reported?",
                "prognosis": "What is the prognosis?",
                "prevention": "How can it be prevented?",
                "disease_full_name": "What is the full name (expansion) of the disease/pathogen abbreviation?",
                "geographic_distribution": "What is the geographic distribution?",
                "overview": "Provide a brief overview.",
                "mechanism": "What is the mechanism?",
            }.get(s, "Answer the relevant aspect of the user question.")

        def _slot_title(slot: str, *, lang: str) -> str:
            s = (slot or "").strip().lower()
            if lang == "zh":
                return {
                    "transmission": "传播途径",
                    "pathogenesis": "发病机制",
                    "clinical_features": "临床表现",
                    "diagnosis": "诊断",
                    "treatment": "治疗",
                    "epidemiology": "流行病学",
                    "risk_factors": "风险因素",
                    "complications": "并发症",
                    "prognosis": "预后",
                    "prevention": "预防",
                    "disease_full_name": "疾病全称",
                    "geographic_distribution": "地区分布",
                    "overview": "概述",
                    "mechanism": "机制",
                }.get(s, s)
            return s

        def _layer_headings(*, lang: str) -> tuple[str, str, str]:
            # Strict output format (see docs/74_three_knowledge_layers.md and user contract):
            # Must always output L0/L1/L2 sections and never merge/skip.
            if lang == "zh":
                return (
                    "### L0｜证据支持的结论",
                    "### L1｜领域共识（LLM 内部知识）",
                    "### L2｜推测性或解释性补充",
                )
            return (
                "### L0 | Evidence-backed conclusions",
                "### L1 | Domain prior (model knowledge)",
                "### L2 | Speculative / reasoned addendum",
            )

        def _slot_uncertainty(slot: str, *, lang: str) -> str:
            s = (slot or "").strip().lower()
            if lang == "zh":
                # Fixed uncertainty templates (no mechanistic steps).
                return {
                    "pathogenesis": "暂无证据支持（具体发病机制尚不明确）。",
                    "mechanism": "暂无证据支持（具体机制尚不明确）。",
                }.get(s, "暂无证据支持，无法确定。")
            return {
                "pathogenesis": "The specific pathogenesis is not well established based on the provided evidence.",
                "mechanism": "The specific mechanism is not well established based on the provided evidence.",
            }.get(s, "Evidence is insufficient to answer this aspect.")

        prompt_for_final = prompt
        # Grounding/citation: schema-first does NOT bypass evidence lookup.
        # Evidence/raw are retrieval-only; NEVER injected into attention.
        if self.grounding_retriever is None:
            raise ValueError("schema-only injection requires grounding_retriever (schema->evidence->raw routing).")
        if True:
            # IMPORTANT: grounding retrieval must be query-driven.
            # Never fall back to a constant zero vector; that will repeatedly return the same evidence
            # regardless of the user's question.
            if last_query_vec is None:
                if query_embed_fn is not None:
                    try:
                        last_query_vec = query_embed_fn(str(query_text or prompt))
                    except Exception:
                        last_query_vec = None
                if last_query_vec is None:
                    # As a last resort, reuse the current step query_vec if available; otherwise abort grounding.
                    last_query_vec = query_vec if "query_vec" in locals() else np.zeros((1,), dtype=np.float32)
            try:
                gr = self.grounding_retriever.search(
                    last_query_vec, top_k=int(self.cfg.top_k_blocks), filters=None, query_text=query_text
                )
                evidence_texts: List[str] = []
                raw_texts: List[str] = []
                if self.block_text_lookup is not None:
                    for it in (gr.items or []):
                        bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
                        if not bid:
                            continue
                        t = self.block_text_lookup(str(bid))
                        if isinstance(t, str) and t.strip():
                            src = (it.meta or {}).get("retrieval_source")
                            if src == "evidence":
                                evidence_texts.append(t.strip())
                            else:
                                raw_texts.append(t.strip())
            except Exception:
                evidence_texts = []
                raw_texts = []

            evidence = _extract_evidence(evidence_texts, q=str(query_text or ""), keywords=None) if evidence_texts else ""
            # IMPORTANT: generation prompt must contain ONLY user question + selected evidence text.
            # raw_ctx is kept for offline logging only; it MUST NOT be appended to the prompt.
            raw_ctx = _extract_evidence(raw_texts, q=str(query_text or ""), keywords=None) if raw_texts else ""

            # Evidence is mandatory to retrieve; if evidence quotes are empty, we still provide fallback context.
            if not evidence and not raw_ctx:
                print("[grounding] no evidence/raw texts found; proceeding without appended grounding.", flush=True)
            if bool(self.cfg.enable_evidence_slot_gating):
                # Scheme A: evidence-aware slot gating (pre-generation guard).
                # - required_slots from user question
                # - evidence_covered_slots from selected evidence text (may be empty)
                qtxt = str(query_text or prompt)
                required_slots = set(infer_slots_from_query(qtxt))
                evidence_covered_slots = set(infer_slots_from_query(str(evidence or "")))
                # Fail-closed: if we cannot map the question to adjudicable slots,
                # do NOT proceed with free-form GROUNDED generation.
                if not required_slots:
                    lang = _lang_of_query(qtxt)
                    h0, h1, h2 = _layer_headings(lang=lang)
                    fts = set(classify_fact_types(qtxt))
                    if fts:
                        ft_s = "、".join(sorted(fts)) if lang == "zh" else ", ".join(sorted(fts))
                        has_any_evidence = bool((evidence or "").strip()) or bool(evidence_texts)
                        if has_any_evidence:
                            # We retrieved evidence, but schema slots do not cover this fact type (cannot adjudicate via slots).
                            layer0 = (
                                f"- schema 未覆盖该事实类型：{ft_s}；但已检索到相关证据（见下方证据句），当前不做结构化裁决。"
                                if lang == "zh"
                                else f"- Schema has no coverage for fact types: {ft_s}; however, relevant evidence was retrieved (see evidence sentences below). No slot-based adjudication is performed."
                            )
                        else:
                            layer0 = (
                                f"- 暂无证据支持（schema 未覆盖该事实类型：{ft_s}；且未检索到可用证据，已降级为受控领域共识回答）。"
                                if lang == "zh"
                                else f"- No evidence support (schema has no coverage for fact types: {ft_s}; no usable evidence retrieved; downgraded to controlled domain-prior answer)."
                            )
                    else:
                        layer0 = (
                            "- 暂无证据支持（该问题未命中可裁决的 slot，已降级为受控领域共识回答）。"
                            if lang == "zh"
                            else "- No evidence support (question did not map to adjudicable slots; downgraded to controlled domain-prior answer)."
                        )

                    # L1: Domain Prior answer for the whole question (base LLM, no injection)
                    allow_dp = bool(domain_prior_allowed_for_fact_types(fts))
                    if (not allow_dp) and lang == "zh":
                        has_any_evidence = bool((evidence or "").strip()) or bool(evidence_texts)
                        if has_any_evidence:
                            layer1 = "- 该问题涉及需要证据支持的事实类型（例如地区分布等）。已检索到证据，但 schema 未覆盖对应可裁决字段；因此仅在 L0 展示证据句，不在 L1 给出具体结论。"
                        else:
                            layer1 = "- 该问题涉及需要证据支持的事实类型（例如地区分布等），当前 schema 未覆盖且未检索到可用证据，无法给出可靠结论。"
                    elif (not allow_dp) and lang != "zh":
                        has_any_evidence = bool((evidence or "").strip()) or bool(evidence_texts)
                        if has_any_evidence:
                            layer1 = "- This question involves fact types that require evidence (e.g., geographic distribution). Evidence was retrieved, but schema has no adjudicable fields for it; L1 will not produce a specific conclusion."
                        else:
                            layer1 = "- This question involves fact types that require evidence (e.g., geographic distribution). Schema has no coverage and no usable evidence was retrieved; cannot answer reliably."
                    elif lang == "zh":
                        p1 = (
                            "你是医学教科书风格的助手。请基于常识性、长期共识的医学/生物学知识回答用户问题。"
                            "要求：只输出中文；不编造具体论文/实验数据/精确数值；不输出最新研究/假说；避免离题。"
                            "\n\n用户问题：\n"
                            + qtxt
                            + "\n\n请用要点列表输出（每条以“- ”开头），不要输出标题："
                        )
                    else:
                        p1 = (
                            "Answer the user question using only long-standing textbook-level consensus knowledge. "
                            "No paper-specific claims, no precise numbers, no speculative hypotheses, stay on-topic.\n\n"
                            "User question:\n"
                            + qtxt
                            + "\n\nOutput as bullet points (each starts with '- '), no headings:"
                        )
                    if allow_dp:
                        layer1_raw = self._greedy_generate_with_past_prefix(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=p1,
                            device=device,
                            past_key_values=None,
                            max_new_tokens=min(int(getattr(self.cfg, "layer1_max_new_tokens", 256)), 192),
                            no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                            repetition_penalty=max(1.05, float(self.cfg.repetition_penalty)),
                        ).strip()
                        layer1 = layer1_raw.strip()
                        # Deterministic cleanup: drop prompt-echo lines and keep bullets.
                        lines = [ln.strip() for ln in layer1.splitlines() if ln.strip()]
                        cleaned = []
                        for ln in lines:
                            if re.search(r"(用户问题|请用要点|output as bullet|user question)", ln, re.I):
                                continue
                            cleaned.append(ln)
                        layer1 = "\n".join(cleaned).strip()
                        if lang == "zh" and layer1:
                            # drop non-CJK sentences
                            sents = [x.strip() for x in re.split(r"(?<=[。！？!?\.])\s*", layer1) if x.strip()]
                            kept = [x for x in sents if _has_cjk(x)]
                            layer1 = "\n".join(kept).strip() if kept else layer1
                        if layer1 and not re.search(r"^\s*-\s+", layer1):
                            layer1 = "- " + layer1.lstrip("•").lstrip("-").strip()
                        # Hard cap: keep first 3 bullet lines only.
                        bl = [ln for ln in layer1.splitlines() if ln.strip()]
                        layer1 = "\n".join(bl[:3]).strip()

                    # L2: disabled under coverage failure (fail-closed)
                    layer2 = "- （本题已降级，推测层禁用）" if lang == "zh" else "- (Downgraded; speculative layer disabled.)"
                    # If we have evidence sentences, expose them in L0 as evidence-bound text (no expansion).
                    if bool((evidence or "").strip()):
                        ev_lines = []
                        for ln in str(evidence).splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            ev_lines.append(f"- 证据句：{ln}" if lang == "zh" else f"- Evidence: {ln}")
                            if len(ev_lines) >= 2:
                                break
                        layer0 = (layer0 + "\n" + "\n".join(ev_lines)).strip()

                    out = "\n".join([h0, layer0, "", h1, layer1, "", h2, layer2]).strip()
                    return out, step_debugs

                if required_slots:
                    lang = _lang_of_query(qtxt)
                    covered = [s for s in sorted(required_slots) if s in evidence_covered_slots]
                    uncovered = [s for s in sorted(required_slots) if s not in evidence_covered_slots]

                    parts: List[str] = []
                    # Generate ONLY for covered slots (each as a separate generation call).
                    if covered and str(evidence or "").strip():
                        # Cap per-slot length to avoid long looping paraphrases.
                        per_slot_tokens = max(32, int(max_new_tokens // max(1, len(covered))))
                        per_slot_tokens = min(int(per_slot_tokens), 96)

                        def _has_cjk(s: str) -> bool:
                            return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

                        def _shorten_answer(s: str, *, lang: str) -> str:
                            # Keep 1–2 non-duplicate sentences.
                            s0 = (s or "").strip()
                            if not s0:
                                return ""
                            # Strip leading instruction-echo fragments like "(不重复)(不超过20字)".
                            s0 = re.sub(r"^\s*[\(\（][^\)\）]{0,24}(不重复|不超过|简要|一句话|20字|1-2|1–2)[^\)\）]{0,24}[\)\）]\s*", "", s0)
                            s0 = re.sub(r"^\s*[\(\（][^\)\）]{0,24}(do\s*not|concise|1-2)[^\)\）]{0,24}[\)\）]\s*", "", s0, flags=re.IGNORECASE)
                            # Strip leading "rewrite suggestion" like "(简化为：...)" / "(改写为：...)".
                            s0 = re.sub(r"^\s*[\(\（][^\)\）]{0,80}(简化为|改写为|总结为|可简化为)\s*[:：][^\)\）]{0,160}[\)\）]\s*", "", s0)
                            # Strip leading "N sentences" meta like "(3句，但只需前两句即可)" plus trailing punctuation spam.
                            s0 = re.sub(r"^\s*[\(\（][^\)\）]{0,120}(句|sentenc|前两句|前2句|只需|即可)[^\)\）]{0,120}[\)\）]\s*", "", s0, flags=re.IGNORECASE)
                            s0 = re.sub(r"^\s*[:：]{2,}\s*", "", s0)
                            # Normalize repeated colons in-line.
                            s0 = re.sub(r"[:：]{2,}", "：" if lang == "zh" else ":", s0)
                            # Strip language directive echoes (ZH + translated EN).
                            s0 = re.sub(r"^\s*(请用中文回答|用中文回答|请使用中文回答)\s*[:：]?\s*", "", s0)
                            s0 = re.sub(r"^\s*(please\s+use\s+chinese\s+to\s+answer)\s*[:：]?\s*", "", s0, flags=re.IGNORECASE)
                            # Strip leading markdown bullets.
                            s0 = re.sub(r"^\s*[-\*\u2022]\s+", "", s0)
                            # In ZH mode, avoid guessing Chinese common names for Latin species:
                            # drop CJK parenthetical after a Latin binomial (e.g., "Haemaphysalis longicornis (璃眼蜱)").
                            if lang == "zh":
                                s0 = re.sub(
                                    r"(\b[A-Z][a-z]+(?:\s+[a-z][a-z0-9\-]{2,})\b)\s*[\(\（][^\)\）]*[\u4e00-\u9fff][^\)\）]*[\)\）]",
                                    r"\1",
                                    s0,
                                )
                            sents_all = [x.strip() for x in re.split(r"(?<=[。！？!?\.])\s*", s0) if x.strip()]
                            # For Chinese outputs:
                            # - If we already have any CJK sentences, drop non-CJK to prevent EN evidence leakage.
                            # - If we have no CJK at all, keep sentences (rewrite fallback will handle it).
                            if lang == "zh":
                                if any(_has_cjk(x) for x in sents_all):
                                    sents = [x for x in sents_all if _has_cjk(x)]
                                else:
                                    sents = sents_all
                            else:
                                sents = sents_all
                            out_sents: List[str] = []
                            seen_norm: set[str] = set()
                            for x in sents:
                                # Normalize aggressively for de-dupe (handles hidden spaces/punct variations).
                                nx = x
                                nx = re.sub(r"\s+", "", nx)
                                nx = re.sub(r"[，,。\.；;：:！!？?\(\)\（\）\[\]\{\}<>\"'“”‘’\-—_]", "", nx)
                                nx = nx.lower()
                                if nx in seen_norm:
                                    continue
                                seen_norm.add(nx)
                                out_sents.append(x)
                                if len(out_sents) >= 2:
                                    break
                            return " ".join(out_sents).strip()

                        # Deterministic evidence rendering for slots where free-form generation tends to add
                        # bibliographic noise ("(2021)") or partial "结论：" artifacts.
                        def _strip_trailing_citation(t: str) -> str:
                            x = (t or "").strip()
                            if not x:
                                return ""
                            # [1], [12]
                            x = re.sub(r"\s*\[[0-9]{1,3}\]\s*$", "", x).strip()
                            # (Casel et al. 2021) / (2021) / (Smith 2020)
                            x = re.sub(
                                r"\s*[\(\（][^\)\）]{0,80}\b(19|20)\d{2}\b[^\)\）]{0,80}[\)\）]\s*\.?\s*$",
                                "",
                                x,
                            ).strip()
                            return x

                        def _evidence_sentence_for_slot(slot: str, *, evidence_text: str, lang: str) -> str:
                            s0 = (slot or "").strip().lower()
                            lines = [ln.strip() for ln in (evidence_text or "").splitlines() if ln.strip()]
                            if not lines:
                                return ""
                            if s0 == "geographic_distribution":
                                kw = re.compile(
                                    r"(east\s+asia|reported\s+in|province|provinces|country|countries|china|japan|korea|vietnam|taiwan|pakistan|"
                                    r"地区分布|分布|报告|省|中国|日本|韩国|越南|台湾|巴基斯坦)",
                                    re.IGNORECASE,
                                )
                                picked = [ln for ln in lines if kw.search(ln)]
                                picked = picked[:2] if picked else lines[:1]
                                picked = [_strip_trailing_citation(x) for x in picked]
                                picked = [x for x in picked if x]
                                if not picked:
                                    return ""
                                return ("证据句：" if lang == "zh" else "Evidence: ") + (
                                    "；".join(picked) if lang == "zh" else " ".join(picked)
                                )
                            if s0 == "disease_full_name":
                                # Only accept explicit expansion-like evidence sentences.
                                kw = re.compile(
                                    r"(stands\s+for|full\s+name|severe\s+fever\s+with\s+thrombocytopenia|"
                                    r"严重发热伴血小板减少综合征|全称|缩写)",
                                    re.IGNORECASE,
                                )
                                picked = [ln for ln in lines if kw.search(ln)]
                                picked = picked[:1]
                                picked = [_strip_trailing_citation(x) for x in picked]
                                picked = [x for x in picked if x]
                                if not picked:
                                    return ""
                                return ("证据句：" if lang == "zh" else "Evidence: ") + picked[0]
                            return ""

                        for s in covered:
                            # Special deterministic L0 path (no free-form summarization).
                            s_l = str(s or "").strip().lower()
                            if s_l in {"geographic_distribution", "disease_full_name"}:
                                title = _slot_title(s, lang=lang)
                                ev_ans = _evidence_sentence_for_slot(s, evidence_text=str(evidence or ""), lang=lang)
                                if ev_ans:
                                    parts.append(f"- {title}：{ev_ans}" if lang == "zh" else f"- {title}: {ev_ans}")
                                else:
                                    tmpl = _slot_uncertainty(s, lang=lang)
                                    parts.append(f"- {title}：{tmpl}" if lang == "zh" else f"- {title}: {tmpl}")
                                continue
                            q_plain = _slot_question(s, lang=lang).strip()
                            q_slot = q_plain
                            # Covered-slot generation: keep prompt minimal (avoid instruction-echo degradation).
                            # Cross-lingual policy: we do NOT translate evidence text; we may summarize its meaning in target language.
                            p_slot = q_slot + "\n\n" + evidence
                            ans_slot = self._greedy_generate_with_past_prefix(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=p_slot,
                                device=device,
                                past_key_values=last_past_key_values,
                                max_new_tokens=int(per_slot_tokens),
                                no_repeat_ngram_size=int(no_repeat_ngram_size),
                                repetition_penalty=float(self.cfg.repetition_penalty),
                            ).strip()
                            # Remove question echo if present.
                            if q_plain:
                                ans_slot = ans_slot.replace(q_plain, " ").strip()
                            # One deterministic retry if output language mismatches (evidence may be EN-heavy).
                            if lang == "zh" and ans_slot and (not _has_cjk(ans_slot)):
                                # Instead of forcing during first-pass generation, do a rewrite-style summary into Chinese.
                                # Evidence text is included verbatim (unchanged); output must be Chinese only.
                                p_retry = (
                                    "根据以下证据，用中文给出一句结论回答问题。"
                                    "不要翻译/改写证据原文，不要输出英文，不要复述问题。"
                                    "不要给拉丁学名/英文名编造中文俗名；如需提及，请保留原学名/英文名。"
                                    "\n\n问题："
                                    + q_plain
                                    + "\n\n证据：\n"
                                    + evidence
                                )
                                ans_slot = self._greedy_generate_with_past_prefix(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=p_retry,
                                    device=device,
                                    past_key_values=last_past_key_values,
                                    max_new_tokens=min(64, int(per_slot_tokens)),
                                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                                    repetition_penalty=max(1.10, float(self.cfg.repetition_penalty)),
                                ).strip()
                                if q_plain:
                                    ans_slot = ans_slot.replace(q_plain, " ").strip()
                            # Final very-strong retry (still deterministic) if we are not in Chinese yet.
                            if lang == "zh" and ans_slot and (not _has_cjk(ans_slot)):
                                p_retry2 = (
                                    "只输出中文答案（一句话）。不要输出英文。不要复述问题。"
                                    "不要给拉丁学名/英文名编造中文俗名；如需提及，请保留原学名/英文名。"
                                    "\n\n问题："
                                    + q_plain
                                    + "\n\n证据：\n"
                                    + evidence
                                )
                                ans_slot = self._greedy_generate_with_past_prefix(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=p_retry2,
                                    device=device,
                                    past_key_values=last_past_key_values,
                                    max_new_tokens=48,
                                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                                    repetition_penalty=max(1.12, float(self.cfg.repetition_penalty)),
                                ).strip()
                                if q_plain:
                                    ans_slot = ans_slot.replace(q_plain, " ").strip()
                            ans_slot = _shorten_answer(ans_slot, lang=lang)
                            # Evidence-bound safety: if the "answer" degenerates into bibliography/citation, drop it.
                            if ans_slot and re.search(r"(参考文献|references?|doi\s*[:：]|\bet\s+al\.\b|\[[0-9]{1,3}\])", ans_slot, re.IGNORECASE):
                                ans_slot = ""
                            # If still no Chinese, fall back to a minimal template (keeps policy: no hallucinated detail).
                            if lang == "zh" and (not ans_slot or not _has_cjk(ans_slot)):
                                ev_l = (evidence or "").lower()
                                if s.strip().lower() == "transmission" and ("tick" in ev_l or "蜱" in evidence):
                                    ans_slot = "主要通过蜱叮咬传播。"
                                else:
                                    ans_slot = "该方面证据不足，无法确定。"
                            if ans_slot:
                                title = _slot_title(s, lang=lang)
                                # Layer 0 statement only (evidence-bound conclusion).
                                parts.append(f"- {title}：{ans_slot}" if lang == "zh" else f"- {title}: {ans_slot}")

                    # For uncovered required slots, emit fixed uncertainty templates only.
                    for s in uncovered:
                        title = _slot_title(s, lang=lang)
                        tmpl = _slot_uncertainty(s, lang=lang)
                        parts.append(f"- {title}：{tmpl}" if lang == "zh" else f"- {title}: {tmpl}")

                    if parts:
                        # ---- Three knowledge layers (L0 slot-gated; L1/L2 do NOT use slot gating) ----
                        if bool(self.cfg.enable_three_knowledge_layers):
                            h0, h1, h2 = _layer_headings(lang=lang)
                            layer0 = "\n".join(parts).strip()

                            # -------------------------
                            # slot_epistemic_state (NOT in prompt)
                            # -------------------------
                            slot_epistemic_state: Dict[str, Dict[str, Any]] = {}
                            for s in sorted(required_slots):
                                ev = bool(s in set(covered))
                                slot_epistemic_state[str(s)] = {
                                    "evidence": ev,
                                    "allowed_certainty": ("high" if ev else "low"),
                                }

                            def _approx_token_len(s: str, *, lang: str) -> int:
                                # deterministic, cheap proxy
                                if not s:
                                    return 0
                                if lang == "zh":
                                    return len(re.findall(r"[\u4e00-\u9fff]", s)) + max(0, len(s) // 12)
                                return len(re.findall(r"\w+", s))

                            def _compress_text(s: str, *, lang: str, max_sentences: int, max_tokens: int) -> str:
                                t = (s or "").strip()
                                if not t:
                                    return ""
                                # split into sentences
                                sents = [x.strip() for x in re.split(r"(?<=[。！？!?\.])\s*", t) if x.strip()]
                                out: List[str] = []
                                tok = 0
                                for sent in sents:
                                    if len(out) >= int(max_sentences):
                                        break
                                    add = _approx_token_len(sent, lang=lang)
                                    if out and (tok + add) > int(max_tokens):
                                        break
                                    out.append(sent)
                                    tok += add
                                return " ".join(out).strip()

                            def _dedupe_bullets(s: str, *, max_items: int = 12) -> str:
                                lines = [ln.rstrip() for ln in (s or "").splitlines() if ln.strip()]
                                out: List[str] = []
                                seen: set[str] = set()
                                for ln in lines:
                                    m = re.match(r"^\s*-\s*(.+)$", ln)
                                    body = (m.group(1) if m else ln).strip()
                                    key = re.sub(r"\s+", "", body).lower()
                                    key = re.sub(r"[，,。\.；;：:！!？?\(\)\（\）\[\]\{\}<>\"'“”‘’\-—_]", "", key)
                                    if not key or key in seen:
                                        continue
                                    seen.add(key)
                                    out.append(f"- {body}")
                                    if len(out) >= int(max_items):
                                        break
                                return "\n".join(out).strip()

                            # High-risk mechanism entities we forbid when allowed_certainty is low (L1+L2).
                            _MECH_ENTITY_RE = re.compile(
                                r"("
                                r"receptor|受体|pathway|signaling|signal\s+pathway|通路|信号通路|"
                                r"NF-?κB|JAK\s*/?\s*STAT|MAPK|TLR\d*|RIG-?I|MAVS|"
                                r"interferon|IFN|ISG|IL-\d+|TNF|chemokine|cytokine|细胞因子|趋化因子|干扰素|"
                                r"MHC|HLA|endocyt|内吞|"
                                r")",
                                re.IGNORECASE,
                            )
                            _LATIN_BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[a-z][a-z0-9\-]{2,})\b")
                            _UNCERTAINTY_RE_ZH = re.compile(r"(推测|可能|或许|尚未|尚不|不确定|仍不清楚|一般认为|通常认为|通常|一般|多见|往往)")

                            # -------------------------
                            # L1 generation (per-slot, governed by slot_epistemic_state)
                            # -------------------------
                            l1_lines: List[str] = []
                            for s in sorted(required_slots):
                                st = slot_epistemic_state.get(str(s), {})
                                allowed = str(st.get("allowed_certainty") or "low")
                                title = _slot_title(str(s), lang=lang)
                                if lang == "zh":
                                    if allowed == "low":
                                        p1s = (
                                            "用中文输出该方面的【领域共识】要点（1-2句），只描述常见现象与总体认识。"
                                            "不要写具体分子机制/受体/信号通路/细节枚举；不要写具体数值或论文结论；不要推测。"
                                            f"\n\n方面：{title}\n用户问题：{str(query_text or prompt)}\n\n只输出内容本身："
                                        )
                                    else:
                                        p1s = (
                                            "用中文输出该方面的【领域共识】要点（2-3句），教科书级总结即可。"
                                            "不要写具体数值或论文结论；不要推测。"
                                            f"\n\n方面：{title}\n用户问题：{str(query_text or prompt)}\n\n只输出内容本身："
                                        )
                                else:
                                    if allowed == "low":
                                        p1s = (
                                            "Write 1-2 sentences of domain-prior consensus for this aspect. "
                                            "No receptors/pathways/mechanistic enumerations. No numbers. No speculation.\n\n"
                                            f"Aspect: {title}\nUser question: {str(query_text or prompt)}\n\nOutput text only:"
                                        )
                                    else:
                                        p1s = (
                                            "Write 2-3 sentences of textbook-level domain-prior consensus for this aspect. "
                                            "No numbers. No speculation.\n\n"
                                            f"Aspect: {title}\nUser question: {str(query_text or prompt)}\n\nOutput text only:"
                                        )

                                raw = self._greedy_generate_with_past_prefix(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=p1s,
                                    device=device,
                                    past_key_values=None,
                                    max_new_tokens=min(int(getattr(self.cfg, "layer1_max_new_tokens", 256)), 128),
                                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                                    repetition_penalty=max(1.05, float(self.cfg.repetition_penalty)),
                                ).strip()
                                txt = raw.strip()
                                # Language unification: for Chinese queries, drop any non-CJK sentences (no translation).
                                if lang == "zh" and txt:
                                    sents = [x.strip() for x in re.split(r"(?<=[。！？!?\.])\s*", txt) if x.strip()]
                                    kept = [x for x in sents if _has_cjk(x)]
                                    txt = " ".join(kept).strip() if kept else txt.strip()
                                # deterministic filtering for low-certainty: strip risky entities and enforce non-deterministic tone
                                if allowed == "low":
                                    # remove mechanistic-entity sentences
                                    sents = [x.strip() for x in re.split(r"(?<=[。！？!?\.])\s*", txt) if x.strip()]
                                    kept: List[str] = []
                                    for sent in sents:
                                        if _MECH_ENTITY_RE.search(sent):
                                            continue
                                        if _LATIN_BINOMIAL_RE.search(sent):
                                            continue
                                        kept.append(sent)
                                    txt = " ".join(kept).strip()
                                    if lang == "zh" and txt and not _UNCERTAINTY_RE_ZH.search(txt):
                                        txt = "通常认为，" + txt
                                # compress hard
                                txt = _compress_text(
                                    txt,
                                    lang=lang,
                                    max_sentences=int(getattr(self.cfg, "l1_max_sentences", 3)),
                                    max_tokens=int(getattr(self.cfg, "l1_max_tokens", 80)),
                                )
                                if txt:
                                    l1_lines.append(f"- {title}：{txt}" if lang == "zh" else f"- {title}: {txt}")
                                else:
                                    # safe minimal fallback
                                    if lang == "zh":
                                        s_l = str(s or "").strip().lower()
                                        if s_l == "disease_full_name":
                                            l1_lines.append(
                                                f"- {title}：通常指该疾病/病原缩写对应的全称（英文展开或中文全称）。但该表述为领域共识，未由当前证据集裁决。"
                                            )
                                        elif s_l == "geographic_distribution":
                                            l1_lines.append(
                                                f"- {title}：地区分布通常需要具体监测/研究数据支撑；若缺少本库证据或输出被规则过滤，本层不给出具体范围结论。"
                                            )
                                        else:
                                            l1_lines.append(f"- {title}：一般可从临床现象与常识性医学认识进行理解，但缺少本库证据支撑具体细节。")
                                    else:
                                        s_l = str(s or "").strip().lower()
                                        if s_l == "disease_full_name":
                                            l1_lines.append(
                                                f"- {title}: Typically refers to the expansion of the disease/pathogen abbreviation (full name). This is domain-prior knowledge and is not adjudicated by the current evidence set."
                                            )
                                        elif s_l == "geographic_distribution":
                                            l1_lines.append(
                                                f"- {title}: Geographic distribution typically requires specific surveillance/study data; if unsupported by the current evidence set (or filtered), we avoid asserting a concrete range here."
                                            )
                                        else:
                                            l1_lines.append(
                                                f"- {title}: A conservative, textbook-level description is possible, but specific details are not supported by the current evidence set."
                                            )

                            layer1 = _dedupe_bullets("\n".join(l1_lines).strip(), max_items=12)
                            # absolute compression: ≤3 sentences total (across bullets) + ≤80 tokens
                            layer1 = _compress_text(
                                layer1,
                                lang=lang,
                                max_sentences=int(getattr(self.cfg, "l1_max_sentences", 3)),
                                max_tokens=int(getattr(self.cfg, "l1_max_tokens", 80)),
                            )
                            if layer1 and not re.search(r"^\s*-\s+", layer1):
                                layer1 = "- " + layer1.strip()

                            out_sections = [h0, layer0, "", h1, layer1, "", h2]

                            # Layer 2: Speculative/Open (optional)
                            if bool(getattr(self.cfg, "enable_speculative_layer", False)):
                                # L2 is allowed by default only when any slot has low certainty.
                                any_low = any((slot_epistemic_state[s]["allowed_certainty"] == "low") for s in slot_epistemic_state)
                                if any_low:
                                    l2_lines: List[str] = []
                                    for s in sorted(required_slots):
                                        st = slot_epistemic_state.get(str(s), {})
                                        if str(st.get("allowed_certainty") or "low") != "low":
                                            continue
                                        title = _slot_title(str(s), lang=lang)
                                        if lang == "zh":
                                            p2s = (
                                                "针对该方面给出1-2句解释性推测补充。"
                                                "必须包含不确定性措辞（如“推测/可能/尚不清楚/仍待研究”）。"
                                                "禁止具体分子机制、受体、信号通路、具体实验结论或数值。"
                                                "\n\n方面："
                                                + str(title)
                                                + "\n用户问题："
                                                + str(query_text or prompt)
                                                + "\n\n只输出内容本身："
                                            )
                                        else:
                                            p2s = (
                                                "Provide 1-2 sentences of speculative/reasoned addendum for this aspect. "
                                                "Must include uncertainty markers (may/might/unclear/under investigation). "
                                                "No receptors/pathways/molecular entities, no numbers.\n\n"
                                                f"Aspect: {title}\nUser question: {str(query_text or prompt)}\n\nOutput text only:"
                                            )
                                        raw = self._greedy_generate_with_past_prefix(
                                            model=model,
                                            tokenizer=tokenizer,
                                            prompt=p2s,
                                            device=device,
                                            past_key_values=None,
                                            max_new_tokens=min(int(getattr(self.cfg, "layer2_max_new_tokens", 192)), 96),
                                            no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                                            repetition_penalty=max(1.05, float(self.cfg.repetition_penalty)),
                                        ).strip()
                                        txt = raw.strip()
                                        # Hard rules: must contain uncertainty markers; must not contain mech entities / latin binomial
                                        if lang == "zh":
                                            ok_unc = bool(_UNCERTAINTY_RE_ZH.search(txt))
                                        else:
                                            ok_unc = bool(re.search(r"(may|might|unclear|under\s+investigation|not\s+established)", txt, re.I))
                                        if (not ok_unc) or _MECH_ENTITY_RE.search(txt) or _LATIN_BINOMIAL_RE.search(txt):
                                            continue
                                        txt = _compress_text(
                                            txt,
                                            lang=lang,
                                            max_sentences=int(getattr(self.cfg, "l2_max_sentences", 3)),
                                            max_tokens=int(getattr(self.cfg, "l2_max_tokens", 100)),
                                        )
                                        if txt:
                                            l2_lines.append(f"- {title}：{txt}" if lang == "zh" else f"- {title}: {txt}")
                                    layer2 = _dedupe_bullets("\n".join(l2_lines).strip(), max_items=8)
                                    layer2 = _compress_text(
                                        layer2,
                                        lang=lang,
                                        max_sentences=int(getattr(self.cfg, "l2_max_sentences", 3)),
                                        max_tokens=int(getattr(self.cfg, "l2_max_tokens", 100)),
                                    )
                                    out_sections.append(layer2 if layer2.strip() else ("- （推测层内容被规则过滤/暂无合规补充）" if lang == "zh" else "- (Speculative content filtered / none compliant.)"))
                                else:
                                    out_sections.append("- （本次无需推测性补充）" if lang == "zh" else "- (No speculative addendum needed.)")
                            else:
                                # L2 exists but is intentionally empty unless enabled.
                                out_sections.append("- （本次未提供推测性补充）" if lang == "zh" else "- (No speculative addendum in this run.)")

                            return "\n".join([x for x in out_sections if x is not None]).strip(), step_debugs

                        # Legacy: return just Layer 0 statements.
                        return "\n".join(parts).strip(), step_debugs

                # Default: Prompt = user question + evidence text only.
                prompt_for_final = prompt + "\n\n" + evidence

        # final generation using last injected state: simplest approach is do generate without further injection
        # NOTE: do NOT use `model.generate(past_key_values=...)` here; HF generation will treat
        # `past_length` as already-consumed prompt tokens and may slice input_ids to empty when
        # past_length (external prefix length) >> prompt_length.
        txt = self._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_for_final,
            device=device,
            past_key_values=last_past_key_values,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            repetition_penalty=float(self.cfg.repetition_penalty),
        )
        return txt, step_debugs

    @staticmethod
    def _greedy_generate_with_past_prefix(
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
        device: torch.device,
        past_key_values: Any,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.08,
    ) -> str:
        """
        Greedy decode that supports an *external prefix* cache.

        We cannot reliably use `model.generate(past_key_values=...)` here because HF generation
        assumes `past_length == prompt_length` and will slice away the prompt when past_length>0.
        For prefix-injection, past_length is external KV length, so we do manual decoding.
        """

        def _cache_seq_len(pkv: Any) -> int:
            # HF Cache object
            if pkv is None:
                return 0
            if hasattr(pkv, "get_seq_length"):
                try:
                    return int(pkv.get_seq_length())  # type: ignore[attr-defined]
                except Exception:
                    pass
            # legacy tuple: tuple((K,V), ...)
            try:
                k0 = pkv[0][0]
                return int(k0.shape[-2])
            except Exception:
                return 0

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        prefix_len = _cache_seq_len(past_key_values)
        if prefix_len < 0:
            prefix_len = 0
        # IMPORTANT: attention_mask length must account for prefix cache length, otherwise HF models
        # may compute wrong position_ids/cache_position and produce degenerate outputs.
        prefix_mask = torch.ones((prompt_mask.shape[0], prefix_len), dtype=prompt_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

        eos_id = getattr(tokenizer, "eos_token_id", None)

        def _ban_repeated_ngrams(logits: torch.Tensor, seq: torch.Tensor, n: int) -> torch.Tensor:
            """
            Apply a no-repeat-ngram constraint by setting banned token logits to -inf.
            seq: [1, T] token ids (prompt + generated so far)
            logits: [1, V] next-token logits
            """
            if n <= 0:
                return logits
            n = int(n)
            if n < 2:
                return logits
            if seq.numel() < n - 1:
                return logits

            # Build a mapping: (n-1)-gram prefix -> set(next_token)
            tokens = seq[0].tolist()
            prefix_to_next: dict[tuple[int, ...], set[int]] = {}
            for i in range(len(tokens) - n + 1):
                pref = tuple(tokens[i : i + n - 1])
                nxt = int(tokens[i + n - 1])
                s = prefix_to_next.get(pref)
                if s is None:
                    s = set()
                    prefix_to_next[pref] = s
                s.add(nxt)

            cur_pref = tuple(tokens[-(n - 1) :])
            banned = prefix_to_next.get(cur_pref)
            if not banned:
                return logits
            # set to -inf in-place on a copy
            out = logits.clone()
            out[0, list(banned)] = float("-inf")
            return out

        def _apply_repetition_penalty(logits: torch.Tensor, seq: torch.Tensor, penalty: float) -> torch.Tensor:
            """
            Light repetition penalty to reduce stubborn loops that are not caught by n-gram banning.
            penalty > 1.0 discourages already-seen tokens.
            """
            p = float(penalty)
            if p <= 1.0:
                return logits
            if seq.numel() <= 0:
                return logits
            toks = seq[0].tolist()
            if not toks:
                return logits
            out = logits.clone()
            seen = set(int(t) for t in toks[-2048:])  # cap for perf
            idx = torch.tensor(list(seen), device=out.device, dtype=torch.long)
            # Standard repetition penalty: if logit > 0 divide; else multiply.
            vals = out[0, idx]
            out[0, idx] = torch.where(vals > 0, vals / p, vals * p)
            return out

        # First step: consume full prompt with the injected prefix cache.
        # Provide explicit positions starting at prefix_len for RoPE/cache bookkeeping.
        pos0 = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
        cache_pos0 = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=device, dtype=torch.long)
        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=pos0,
                    cache_position=cache_pos0,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            except TypeError:
                # Older HF versions/models may not accept cache_position.
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=pos0,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        past = out.past_key_values
        seq0 = input_ids
        logits0 = out.logits[:, -1, :]
        # Mild penalty to prevent repeating the prompt/evidence tokens.
        logits0 = _apply_repetition_penalty(logits0, seq0, penalty=float(repetition_penalty))
        logits0 = _ban_repeated_ngrams(logits0, seq0, int(no_repeat_ngram_size))
        next_token = torch.argmax(logits0, dim=-1, keepdim=True)  # [1,1]
        generated = [next_token]
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        # Subsequent steps: feed one token at a time with cache.
        for gen_i in range(max(0, int(max_new_tokens) - 1)):
            if isinstance(eos_id, int) and int(next_token.item()) == int(eos_id):
                break
            # Next token position continues after (prefix + prompt + generated_so_far)
            cur_pos = prefix_len + input_ids.shape[1] + gen_i
            pos = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
            with torch.no_grad():
                try:
                    out = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=pos,
                        cache_position=cache_pos,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                except TypeError:
                    out = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=pos,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
            past = out.past_key_values
            seq = torch.cat([input_ids] + generated, dim=1)
            logits = out.logits[:, -1, :]
            logits = _apply_repetition_penalty(logits, seq, penalty=float(repetition_penalty))
            logits = _ban_repeated_ngrams(logits, seq, int(no_repeat_ngram_size))
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        full = torch.cat([input_ids] + generated, dim=1)
        # Return only the newly generated tokens (exclude the prompt) to avoid echoing the prompt
        # in the printed "Answer" output.
        gen_only = torch.cat(generated, dim=1) if generated else full[:, 0:0]
        return tokenizer.decode(gen_only[0], skip_special_tokens=True)


