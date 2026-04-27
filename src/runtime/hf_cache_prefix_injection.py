"""
HF Transformers: cache-prefix injection (Demo)

Core idea
- Do not modify attention forward.
- Write external `K_ext/V_ext` into `past_key_values` as a "static prefix KV" to participate in attention.

Scope
- Demo-level pipeline verification (small knowledge base, get it running first).

Production notes (important)
- Many LLMs (Llama/Qwen/DeepSeek, etc.) apply RoPE (rotary position embedding) to Q/K and cache "already-rotated K/V".
  If your `K_ext` is not in the same space (no matching RoPE/alignment applied), effectiveness will degrade or even fail.
  Demo stage allows simplification; production should align projector output to the same attention space as the cache (including rotary processing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


@dataclass(frozen=True)
class ExtKV:
    """
    Single-layer external KV (batch-first, organized by heads).

    Convention shape (semantic):
    - K: [batch, heads, ext_len, head_dim]
    - V: [batch, heads, ext_len, head_dim]
    """

    K: torch.Tensor
    V: torch.Tensor


def _as_cache_obj() -> Any:
    """
    Try to get HF's Cache/DynamicCache.
    Large version differences: do minimal compatibility in demo.
    """

    try:
        from transformers.cache_utils import DynamicCache  # type: ignore

        return DynamicCache()
    except Exception:
        return None


def build_past_key_values_prefix(
    *,
    model: torch.nn.Module,
    ext_kv_by_layer: Dict[int, ExtKV],
) -> Any:
    """
    Construct past_key_values that can be fed to HF model.

    Prefer returning transformers' DynamicCache (if available), otherwise fall back to tuple structure:
    past_key_values = tuple((K,V) for layer in range(num_layers))
    Where (K,V) for non-injected layers is None.
    """

    # We first build a "legacy" tuple-of-(K,V) per layer, then (if supported) convert it
    # into HF's Cache object (e.g. DynamicCache). Newer models (e.g. Qwen2) expect a Cache
    # and will call methods like `get_seq_length()`.
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not isinstance(num_layers, int):
        raise ValueError("Cannot infer num_hidden_layers from model.config.num_hidden_layers")

    # From any ext KV infer dtype/device/kv_heads/head_dim
    sample = next(iter(ext_kv_by_layer.values()), None)
    if sample is None:
        raise ValueError("ext_kv_by_layer is empty")
    # Keep cache length consistent across layers for Transformers 5.x/Qwen2 masking:
    # mask creation may use a single layer's kv_length globally. If non-injected layers
    # have shorter cache than injected layers, attention_mask and key length can mismatch.
    # Therefore we fill missing layers with zero-prefix K/V of the same ext_len.
    sample_k = sample.K
    sample_v = sample.V
    zero_k = torch.zeros_like(sample_k)
    zero_v = torch.zeros_like(sample_v)
    pkv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [(zero_k, zero_v) for _ in range(num_layers)]
    for layer_idx, ext in ext_kv_by_layer.items():
        pkv[int(layer_idx)] = (ext.K, ext.V)
    legacy = tuple(pkv)

    # Try to return a Cache object when available.
    cache = _as_cache_obj()
    if cache is None:
        return legacy

    # Best-effort conversion API (Transformers versions differ).
    try:
        # transformers>=4.39 (commonly) provides this helper.
        if hasattr(cache, "from_legacy_cache"):
            return cache.from_legacy_cache(legacy)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fallback: try to populate a DynamicCache-like object by calling `update` ONLY for injected layers.
    # If this fails, return legacy tuple.
    try:
        if hasattr(cache, "update"):
            for li, pair in enumerate(legacy):
                if pair is None:
                    continue
                k, v = pair
                cache.update(k, v, li)  # type: ignore[call-arg, attr-defined]
            return cache
    except Exception:
        return legacy

    return legacy


def stack_ext_kv_items(
    *,
    items: Sequence[Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ExtKV:
    """
    Concatenate top-k items (each containing K_ext/V_ext) into a single ExtKV.

    Convention items[i].K_ext / items[i].V_ext as numpy or torch:
    - [heads, ext_len, head_dim] or [1, heads, ext_len, head_dim]
    """

    if not items:
        raise ValueError("Cannot stack empty items")

    Ks: List[torch.Tensor] = []
    Vs: List[torch.Tensor] = []
    for it in items:
        K = it.K_ext
        V = it.V_ext
        if not isinstance(K, torch.Tensor):
            # When K/V comes from np.memmap (mmap_mode='r'), it's often non-writable.
            # Make a small copy here (only for selected top-k blocks) to avoid PyTorch warnings/UB.
            try:
                K = K.copy()  # type: ignore[attr-defined]
            except Exception:
                pass
            K = torch.as_tensor(K)
        if not isinstance(V, torch.Tensor):
            try:
                V = V.copy()  # type: ignore[attr-defined]
            except Exception:
                pass
            V = torch.as_tensor(V)
        if K.ndim == 3:
            K = K.unsqueeze(0)
        if V.ndim == 3:
            V = V.unsqueeze(0)
        if K.shape[0] != 1 or V.shape[0] != 1:
            raise ValueError("Each item must have batch dim 1 or no batch dim")
        Ks.append(K)
        Vs.append(V)

    K_cat = torch.cat(Ks, dim=2)  # concat on ext_len
    V_cat = torch.cat(Vs, dim=2)

    # broadcast to batch
    if batch_size != 1:
        K_cat = K_cat.expand(batch_size, *K_cat.shape[1:]).contiguous()
        V_cat = V_cat.expand(batch_size, *V_cat.shape[1:]).contiguous()

    return ExtKV(K=K_cat.to(device=device, dtype=dtype), V=V_cat.to(device=device, dtype=dtype))


def stack_ext_kv_items_by_layer(
    *,
    items: Sequence[Any],
    layer_id: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    kv_len_key: str = "kv_len",
) -> ExtKV:
    """
    For a given layer_id, extract the corresponding layer from each "multi-layer KV" in items and concatenate into ExtKV.

    Convention:
    - item.get_kv_for_layer(layer_id) -> (K,V) numpy/torch, shape [kv_heads, ext_len, head_dim]
    - item.meta[kv_len_key] (optional): actual effective length; used to slice from padding
    """

    if not items:
        raise ValueError("Cannot stack empty items")

    Ks: List[torch.Tensor] = []
    Vs: List[torch.Tensor] = []
    for it in items:
        if hasattr(it, "get_kv_for_layer"):
            K, V = it.get_kv_for_layer(layer_id)
        else:
            # fallback: assume it.K_ext/it.V_ext is already single-layer
            K, V = it.K_ext, it.V_ext

        if not isinstance(K, torch.Tensor):
            try:
                K = K.copy()  # type: ignore[attr-defined]
            except Exception:
                pass
            K = torch.as_tensor(K)
        if not isinstance(V, torch.Tensor):
            try:
                V = V.copy()  # type: ignore[attr-defined]
            except Exception:
                pass
            V = torch.as_tensor(V)

        # slice to valid length if provided
        kv_len = None
        if hasattr(it, "meta") and isinstance(getattr(it, "meta"), dict):
            kv_len = it.meta.get(kv_len_key)
        if isinstance(kv_len, int) and kv_len >= 0:
            # Expect K/V to be [kv_heads, ext_len, head_dim] here (no batch).
            # Be defensive in case a batch dim sneaks in.
            if K.ndim == 3:
                K = K[:, :kv_len, :]
            elif K.ndim == 4 and K.shape[0] == 1:
                K = K[0, :, :kv_len, :]
            V_shape = getattr(V, "shape", None)
            if V.ndim == 3:
                V = V[:, :kv_len, :]
            elif V.ndim == 4 and V.shape[0] == 1:
                V = V[0, :, :kv_len, :]
            # Ensure both are 3D afterwards.
            if K.ndim != 3 or V.ndim != 3:
                raise ValueError(f"Unexpected K/V shapes after kv_len slice: K={tuple(K.shape)} V={V_shape}")

        # add batch dim
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)
        Ks.append(K)
        Vs.append(V)

    K_cat = torch.cat(Ks, dim=2)  # concat on ext_len
    V_cat = torch.cat(Vs, dim=2)

    if batch_size != 1:
        K_cat = K_cat.expand(batch_size, *K_cat.shape[1:]).contiguous()
        V_cat = V_cat.expand(batch_size, *V_cat.shape[1:]).contiguous()

    return ExtKV(K=K_cat.to(device=device, dtype=dtype), V=V_cat.to(device=device, dtype=dtype))


