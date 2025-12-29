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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..retriever import Retriever
from .hf_cache_prefix_injection import ExtKV, build_past_key_values_prefix
from .schema_answerability import SCHEMA_SLOT_ENUM, SchemaAnswerabilityConfig, choose_answerable_schema, infer_slots_from_query


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
    # Schema selection: answerability (NOT similarity ranking).
    schema_max_selected_per_step: int = 1
    # Optional: required slots to answer for this query (derived upstream). If not set, defaults to ALL slots.
    schema_required_slots: Optional[Sequence[str]] = None
    # Stop-rule epsilon for "injection has no effect" sanity check.
    stop_epsilon_logit_delta_vs_zero: float = 0.05


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
        answered_slots: Set[str],
        required_slots: Set[str],
    ) -> Tuple[List[Any], List[str], int, Dict[str, Any]]:
        """
        Select schema candidates (by ID de-dupe + allowlist) and return their TEXT (from block_text_lookup).

        IMPORTANT: Schema injection must NOT use evidence/raw KV. We only use schema texts here.

        Returns: selected_items, selected_texts, redundancy_hits, selector_debug
        """
        if self.block_text_lookup is None:
            raise ValueError("Schema injection requires block_text_lookup to resolve schema text by block_id.")

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
            t = self.block_text_lookup(str(bid))
            if not isinstance(t, str) or not t.strip():
                redundancy_hits += 1
                continue
            candidates.append(it)
            candidate_ids.append(str(bid))
            candidate_texts.append(t.strip())
            if len(candidates) >= int(self.cfg.top_k_blocks):
                break

        # Schema retrieval is NOT ranked similarity search: choose answerable schema(s) by a selector
        # that ignores ANN scores.
        # IMPORTANT: Use answerable_slots for gating (if present); fallback to slots.
        # Only answerable_slots (not generic "slots") can satisfy required_slots.
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

        sel_idx, sel_dbg = choose_answerable_schema(
            query_text=str(query_text or ""),
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            candidate_slots=cand_answerable_slots,  # Use answerable_slots for slot gating
            answered_slots=set(answered_slots),
            required_slots=set(required_slots),
            cfg=SchemaAnswerabilityConfig(max_selected=int(self.cfg.schema_max_selected_per_step)),
        )
        selected: List[Any] = []
        texts: List[str] = []
        for i in sel_idx[: max(1, int(self.cfg.schema_max_selected_per_step))]:
            selected.append(candidates[i])
            texts.append(candidate_texts[i])
            self.used_block_ids.add(candidate_ids[i])

        return selected, texts, redundancy_hits, sel_dbg

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
        """

        step_debugs: List[StepDebug] = []
        last_past_key_values: Any = None
        grounding_texts: List[str] = []
        last_query_vec: Optional[np.ndarray] = None

        if not bool(use_struct_slots):
            # Enforce the new contract: schema KV is the ONLY KV allowed to constrain generation.
            raise ValueError("This runtime enforces schema-only injection. Pass use_struct_slots=True.")

        # Track semantic progress across steps (slot-based).
        answered_slots: Set[str] = set()
        # Dynamic required_slots: if not explicitly provided, infer from user query.
        if self.cfg.schema_required_slots:
            req_slots = set(str(s) for s in self.cfg.schema_required_slots if str(s))
        else:
            req_slots = infer_slots_from_query(str(query_text or prompt))
            if not req_slots:
                # Fallback: use all predefined slots (backward compat).
                req_slots = set(SCHEMA_SLOT_ENUM)

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

        for step in range(self.cfg.max_steps):
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
                answered_slots=answered_slots,
                required_slots=req_slots,
            )

            # Stop-rule (inline, minimal):
            # 1) No selectable schema remains for uncovered slots.
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
                        note=f"stop_reason=no_selectable_schema_for_uncovered_slots selector={sel_dbg}",
                    )
                )
                break

            newly_answered = set(sel_dbg.get("newly_answered_slots") or [])
            # 2) Newly selected schema does NOT introduce new slots.
            if not newly_answered:
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
                        note=f"stop_reason=no_new_slots selector={sel_dbg}",
                    )
                )
                break
            # 3) redundancy_hits > 0
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

            # Commit semantic progress for this step.
            answered_slots |= newly_answered

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
            # Schema injection path:
            # - Build schema from retrieved schema texts
            # - Forward schema text through base LLM to obtain attention cache
            # - Inject that cache at specified layers (optionally with layer-wise decay)
            # Schema text must NOT be concatenated into the user prompt.
            # Schema text must be forwarded through the base LLM to obtain KV cache.
            # We forward the selected schema text(s) directly (no prompt concatenation for schema).
            schema_text = "\n\n".join([t.strip() for t in schema_texts if isinstance(t, str) and t.strip()])
            if not schema_text:
                raise RuntimeError("schema injection: empty schema_text after selection")

            dtype = next(model.parameters()).dtype
            schema_inputs = tokenizer(schema_text, return_tensors="pt").to(device)
            schema_ids = schema_inputs["input_ids"]
            if schema_ids.shape[1] > int(self.cfg.struct_slots_max_prefix_tokens):
                schema_ids = schema_ids[:, : int(self.cfg.struct_slots_max_prefix_tokens)]
            schema_mask = torch.ones_like(schema_ids)

            with torch.no_grad():
                out_schema = model(
                    input_ids=schema_ids,
                    attention_mask=schema_mask,
                    use_cache=True,
                    return_dict=True,
                )
            past_schema = getattr(out_schema, "past_key_values", None)
            if past_schema is None:
                raise RuntimeError("schema injection: model did not return past_key_values")

            def _to_legacy(pkv: Any) -> Any:
                if pkv is None:
                    return None
                if hasattr(pkv, "to_legacy_cache"):
                    try:
                        return pkv.to_legacy_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return pkv

            legacy = _to_legacy(past_schema)

            def _as_bhld(x: torch.Tensor) -> torch.Tensor:
                # Expect either [B, H, L, D] or [B, L, H, D]. Convert to [B, H, L, D].
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if x.ndim != 4:
                    raise ValueError(f"Unexpected KV tensor rank: {x.ndim}")
                d1, d2 = int(x.shape[1]), int(x.shape[2])
                if d1 > 64 and d2 <= 64:
                    return x.permute(0, 2, 1, 3).contiguous()
                return x

            ext_by_layer: Dict[int, ExtKV] = {}
            decay = self.cfg.struct_slots_decay or {}
            for li in self.cfg.inject_layers:
                pair = None
                try:
                    pair = legacy[int(li)]
                except Exception:
                    pair = None
                if pair is None:
                    continue
                k, v = pair
                k = _as_bhld(k).to(device=device, dtype=dtype)
                v = _as_bhld(v).to(device=device, dtype=dtype)
                scale = float(decay.get(int(li), 1.0))
                if scale != 1.0:
                    v = v * scale
                ext_by_layer[int(li)] = ExtKV(K=k, V=v)

            if not ext_by_layer:
                raise RuntimeError("schema injection: empty ext_by_layer (cannot inject)")

            past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
            injected_tokens_effective = int(schema_ids.shape[1])

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

            # stopping policy: (marginal gain下降 + 冗余/安全上限) 至少两个信号
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

            # Instruction/meta-language patterns to strip (case-insensitive).
            meta_patterns = [
                r"^(请|Please)\s*(回答|answer|respond)",
                r"^(依据|Evidence|Proof|证据)[：:]\s*",
                r"(Human|User|Assistant|System)\s*[：:]",
                r"^\[?(注|Note|提示|Hint)\]?[：:]\s*",
                r"^(主要|其他).*(途径|途经|方式)[：:]",  # instruction-like headers
                r"^证据原文[：:]",
                r"^回答要求",
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
                    # Prefer explicit "main route" type sentences.
                    if "main route" in sl or "most likely route" in sl or "primary" in sl or "主要" in s:
                        picked.append(s)
                    else:
                        picked.append(s)
                    if len(picked) >= 3:
                        break
                if len(picked) >= 3:
                    break
            if not picked:
                # fallback: first 1–2 sentences (after filtering meta).
                t0 = texts[0]
                sents = [s.strip() for s in re.split(r"(?<=[\.\!\?。！？])\s+", t0) if s.strip() and not _is_meta(s)]
                picked = sents[:2] if sents else [t0[:400]]
            picked = [p.replace("\n", " ").strip() for p in picked if p.strip()]
            out = "\n".join([f"- {p}" for p in picked[:3]])
            return out

        prompt_for_final = prompt
        # Grounding/citation: schema-first does NOT bypass evidence lookup.
        # Evidence/raw are retrieval-only; NEVER injected into attention.
        if self.grounding_retriever is None:
            raise ValueError("schema-only injection requires grounding_retriever (schema->evidence->raw routing).")
        if True:
            if last_query_vec is None:
                last_query_vec = np.zeros((1,), dtype=np.float32)
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
            raw_ctx = _extract_evidence(raw_texts, q=str(query_text or ""), keywords=None) if raw_texts else ""

            # Evidence is mandatory to retrieve; if evidence quotes are empty, we still provide fallback context.
            if not evidence and not raw_ctx:
                print("[grounding] no evidence/raw texts found; proceeding without appended grounding.", flush=True)
            if evidence or raw_ctx:
                instr = (grounding_instructions or "").strip()
                if not instr:
                    # Generation contract: slot-level rendering exactly once, clean output.
                    # uncovered_slots: slots that were required but NOT answered by any selected schema.
                    uncovered_slots = set(req_slots) - answered_slots
                    uncovered_str = ", ".join(sorted(uncovered_slots)) if uncovered_slots else ""
                    instr = (
                        "请基于【证据句】回答，遵循以下规则：\n"
                        "1) 每个语义槽只输出一次，不重复；\n"
                        "2) 不要复述问题、提示或证据句原文；\n"
                        "3) 不要输出指令性文字（如'请回答'、'依据'等）；\n"
                        "4) 不要编造证据，仅使用提供的证据句；\n"
                    )
                    if uncovered_str:
                        instr += f"5) 对于无法回答的槽位（{uncovered_str}），直接输出：'现有证据不足以回答该问题。'\n"
                    else:
                        instr += "5) 若证据未覆盖某点，输出：'证据未提及'。\n"
                    instr += "6) 回答简洁，不要输出过程性文字。"
                prompt_for_final = prompt
                if evidence:
                    prompt_for_final += "\n\n【证据句（逐字引用，仅用于核对；不要复述）】\n" + evidence
                if raw_ctx and not evidence:
                    prompt_for_final += "\n\n【回退上下文（raw，仅用于补充背景；不要当作已确认事实）】\n" + raw_ctx
                prompt_for_final += "\n\n【回答要求】\n" + instr

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
            logits = _ban_repeated_ngrams(logits, seq, int(no_repeat_ngram_size))
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        full = torch.cat([input_ids] + generated, dim=1)
        # Return only the newly generated tokens (exclude the prompt) to avoid echoing the prompt
        # in the printed "Answer" output.
        gen_only = torch.cat(generated, dim=1) if generated else full[:, 0:0]
        return tokenizer.decode(gen_only[0], skip_special_tokens=True)


