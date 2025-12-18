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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..retriever import Retriever
from .hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer


@dataclass(frozen=True)
class MultiStepConfig:
    inject_layers: Sequence[int] = (0, 1, 2, 3)
    block_tokens: int = 256
    max_step_tokens: int = 1024
    max_total_tokens: int = 2048
    max_steps: int = 8
    top_k_blocks: int = 8  # retrieve candidates

    # stopping thresholds
    min_logit_delta: float = 1e-3
    min_hidden_delta: float = 1e-3
    redundancy_sim_threshold: float = 0.95

    # attention convergence (external KV attention entropy)
    use_attention_entropy: bool = True
    entropy_window: int = 2  # require consecutive decrease over this many steps
    entropy_threshold: float = 0.35  # normalized entropy in [0,1], lower means more concentrated


@dataclass
class StepDebug:
    step: int
    selected_block_ids: List[str]
    injected_tokens: int
    total_injected_tokens: int
    logit_delta: float
    hidden_delta: float
    redundancy_hits: int
    ext_attn_entropy: Optional[float]


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


class MultiStepInjector:
    def __init__(self, *, retriever: Retriever, cfg: MultiStepConfig) -> None:
        self.retriever = retriever
        self.cfg = cfg
        self.used_block_ids: Set[str] = set()
        self.used_keys: List[np.ndarray] = []
        self.total_injected_tokens = 0

    def _select_blocks(self, items: Sequence[Any], query_vec: np.ndarray) -> Tuple[List[Any], int, int]:
        """
        从 retriever 候选中挑选不重复 blocks，满足 max_step_tokens 与 max_total_tokens。
        返回：selected_items, injected_tokens, redundancy_hits
        """

        selected: List[Any] = []
        injected = 0
        redundancy_hits = 0

        for it in items:
            bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
            if bid is None:
                continue
            if bid in self.used_block_ids:
                continue

            # redundancy: compare query_vec to previously used query_vecs (proxy)
            # 生产级可改为 block embedding 相似度；demo 用 query history 的相似度近似
            redundant = False
            for k in self.used_keys[-16:]:
                if _cosine_sim(query_vec, k) >= self.cfg.redundancy_sim_threshold:
                    redundant = True
                    break
            if redundant:
                redundancy_hits += 1
                continue

            kv_len = int(it.meta.get("kv_len", self.cfg.block_tokens))
            if injected + kv_len > self.cfg.max_step_tokens:
                continue
            if self.total_injected_tokens + injected + kv_len > self.cfg.max_total_tokens:
                continue

            selected.append(it)
            injected += kv_len
            self.used_block_ids.add(str(bid))

            if injected >= self.cfg.max_step_tokens:
                break

        return selected, injected, redundancy_hits

    def run(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 128,
        query_embed_fn: Optional[Any] = None,
    ) -> Tuple[str, List[StepDebug]]:
        """
        执行多步注入推理：
        - 每一步：根据当前状态构造 query embedding → 检索 blocks → 注入 → 走一次 forward 更新 hidden/logits
        - 满足 stopping policy 则停止并进入最终 generate
        """

        step_debugs: List[StepDebug] = []

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

        prev_logits = None
        prev_hidden = None
        entropy_hist: List[float] = []

        for step in range(self.cfg.max_steps):
            # ---- build query embedding from current state ----
            if query_embed_fn is not None:
                query_vec = query_embed_fn(prompt)
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

            # ---- retrieve & select blocks ----
            result = self.retriever.search(query_vec, top_k=self.cfg.top_k_blocks, filters=None, query_text=prompt)
            selected, injected_tokens, redundancy_hits = self._select_blocks(result.items, query_vec)

            if not selected:
                # no new info -> stop
                break

            self.used_keys.append(query_vec)
            self.total_injected_tokens += injected_tokens

            # ---- build per-layer ext kv prefix ----
            ext_by_layer = {
                li: stack_ext_kv_items_by_layer(
                    items=selected,
                    layer_id=li,
                    batch_size=1,
                    device=device,
                    dtype=next(model.parameters()).dtype,
                    kv_len_key="kv_len",
                )
                for li in self.cfg.inject_layers
            }
            past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)

            # ---- one forward step to update logits/hidden ----
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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
                    ext_len = int(sum(int(it.meta.get("kv_len", self.cfg.block_tokens)) for it in selected))
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

            step_debugs.append(
                StepDebug(
                    step=step,
                    selected_block_ids=[str(it.meta.get("block_id") or it.meta.get("chunk_id")) for it in selected],
                    injected_tokens=injected_tokens,
                    total_injected_tokens=self.total_injected_tokens,
                    logit_delta=logit_delta,
                    hidden_delta=hidden_delta,
                    redundancy_hits=redundancy_hits,
                    ext_attn_entropy=ext_entropy,
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

            if signals >= 2:
                break

        # final generation using last injected state: simplest approach is do generate without further injection
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True), step_debugs


