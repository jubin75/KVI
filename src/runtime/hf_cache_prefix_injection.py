"""
HF Transformers：cache-prefix 注入（Demo）

核心想法
- 不改写 attention forward。
- 把外部 `K_ext/V_ext` 写入 `past_key_values`，作为“静态前缀 KV”参与注意力。

适用范围
- demo 级链路验证（知识库不大、先跑通）。

生产级注意事项（重要）
- 许多 LLM（Llama/Qwen/DeepSeek 等）对 Q/K 应用 RoPE（rotary position embedding），并缓存的是“已旋转后的 K/V”。
  如果你的 `K_ext` 不是处在同一空间（未应用相同 RoPE/对齐），效果会打折甚至失效。
  demo 阶段允许简化；生产级应将 projector 输出对齐到与缓存一致的 attention 空间（含 rotary 处理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


@dataclass(frozen=True)
class ExtKV:
    """
    单层外部 KV（batch-first，已按 heads 组织）。

    约定 shape（语义）：
    - K: [batch, heads, ext_len, head_dim]
    - V: [batch, heads, ext_len, head_dim]
    """

    K: torch.Tensor
    V: torch.Tensor


def _as_cache_obj() -> Any:
    """
    尝试获取 HF 的 Cache/DynamicCache。
    版本差异较大：demo 里做最小兼容。
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
    构造可喂给 HF 模型的 past_key_values。

    优先返回 transformers 的 DynamicCache（若可用），否则退化为 tuple 结构：
    past_key_values = tuple((K,V) for layer in range(num_layers))
    其中未注入层的 (K,V) 为 None。
    """

    # We first build a "legacy" tuple-of-(K,V) per layer, then (if supported) convert it
    # into HF's Cache object (e.g. DynamicCache). Newer models (e.g. Qwen2) expect a Cache
    # and will call methods like `get_seq_length()`.
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not isinstance(num_layers, int):
        raise ValueError("Cannot infer num_hidden_layers from model.config.num_hidden_layers")

    # 从任意一个 ext KV 推导 dtype/device/kv_heads/head_dim
    sample = next(iter(ext_kv_by_layer.values()), None)
    if sample is None:
        raise ValueError("ext_kv_by_layer is empty")
    device = sample.K.device
    dtype = sample.K.dtype
    batch = sample.K.shape[0]
    kv_heads = sample.K.shape[1]
    head_dim = sample.K.shape[-1]

    empty_k = torch.empty((batch, kv_heads, 0, head_dim), device=device, dtype=dtype)
    empty_v = torch.empty((batch, kv_heads, 0, head_dim), device=device, dtype=dtype)

    pkv: List[Tuple[torch.Tensor, torch.Tensor]] = [(empty_k, empty_v) for _ in range(num_layers)]
    for layer_idx, ext in ext_kv_by_layer.items():
        pkv[layer_idx] = (ext.K, ext.V)
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

    # Fallback: try to populate a DynamicCache-like object by calling `update`.
    # If this fails, return legacy tuple.
    try:
        if hasattr(cache, "update"):
            for li, (k, v) in enumerate(legacy):
                # Some implementations expect [batch, heads, seq, dim]
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
    把 top-k items（每条含 K_ext/V_ext）拼成单个 ExtKV。

    约定 items[i].K_ext / items[i].V_ext 为 numpy 或 torch：
    - [heads, ext_len, head_dim] 或 [1, heads, ext_len, head_dim]
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
    针对指定 layer_id，把 items 中的每条“多层 KV”切出对应层并拼接成 ExtKV。

    约定：
    - item.get_kv_for_layer(layer_id) -> (K,V) numpy/torch，shape [kv_heads, ext_len, head_dim]
    - item.meta[kv_len_key]（可选）：实际有效长度；用于从 padding 中切片
    """

    if not items:
        raise ValueError("Cannot stack empty items")

    Ks: List[torch.Tensor] = []
    Vs: List[torch.Tensor] = []
    for it in items:
        if hasattr(it, "get_kv_for_layer"):
            K, V = it.get_kv_for_layer(layer_id)
        else:
            # fallback：认为 it.K_ext/it.V_ext 已经是单层
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


