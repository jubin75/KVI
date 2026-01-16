"""
KV relevance checks for prefix-injection (runtime).

Goal (docs/06_KVI_IM.md extra requirement):
If the injected KV batch is judged irrelevant for the query, refresh by selecting a new batch
from the KV bank (repeat up to N rounds).

This module provides a model-based irrelevance signal:
Compare logits under injected prefix vs a same-shape ZERO prefix.
If mean(|logits_inj - logits_zero|) is near 0, the prefix likely has no effect for this query.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _cache_seq_len(pkv: Any) -> int:
    if pkv is None:
        return 0
    if hasattr(pkv, "get_seq_length"):
        try:
            return int(pkv.get_seq_length())  # type: ignore[attr-defined]
        except Exception:
            return 0
    try:
        k0 = pkv[0][0]
        return int(k0.shape[-2])
    except Exception:
        return 0


def _zero_like_past_key_values(pkv: Any) -> Any:
    if pkv is None:
        return None
    # HF Cache object: allocating an equivalent zero Cache is not stable across versions.
    if hasattr(pkv, "get_seq_length"):
        return None
    try:
        z = []
        for (k, v) in pkv:
            z.append((torch.zeros_like(k), torch.zeros_like(v)))
        return tuple(z)
    except Exception:
        return None


def _forward_logits_last_token(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    past_key_values: Any,
) -> Optional[torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    prefix_len = _cache_seq_len(past_key_values)
    if prefix_len < 0:
        prefix_len = 0
    if prefix_len > 0:
        prefix_mask = torch.ones((prompt_mask.shape[0], prefix_len), dtype=prompt_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
    else:
        attention_mask = prompt_mask

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
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos0,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

    return out.logits[:, -1, :].to(dtype=torch.float32)


def logit_delta_vs_zero_prefix(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    past_key_values: Any,
) -> Optional[float]:
    """
    One-step forward comparison: injected prefix vs zero prefix.
    If zero-prefix cache is unavailable, fall back to no-prefix baseline.
    Returns mean absolute logit difference on the last prompt token.
    """
    if past_key_values is None:
        return None
    pkv_zero = _zero_like_past_key_values(past_key_values)
    li = _forward_logits_last_token(
        model=model, tokenizer=tokenizer, prompt=prompt, device=device, past_key_values=past_key_values
    )
    if li is None:
        return None
    if pkv_zero is not None:
        lz = _forward_logits_last_token(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device, past_key_values=pkv_zero
        )
    else:
        lz = _forward_logits_last_token(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device, past_key_values=None
        )
    if lz is None:
        return None
    return float((li - lz).abs().mean().item())

