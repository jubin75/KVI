"""
kv_injector: KV injector that does NOT modify attention forward (runnable implementation)

Implementation approach
- Injects external KV via past_key_values prefix (see runtime/hf_cache_prefix_injection.py)
- Supports two strategies:
  - concat: directly use external KV as prefix KV
  - gate (engineering approximation): control contribution strength by scaling external V (V_ext *= gamma)
    (strict gate mixing requires rewriting attention softmax, which is not done here)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer


@dataclass(frozen=True)
class InjectConfig:
    layers: Sequence[int] = (0, 1, 2, 3)
    strategy: str = "concat"  # concat|gate
    clamp_max: float = 0.10


class KVInjector:
    def __init__(self, cfg: InjectConfig) -> None:
        self.cfg = cfg

    def build_past_key_values(
        self,
        *,
        model: torch.nn.Module,
        items: Sequence[Any],
        gamma: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if not items:
            # return empty prefix (all layers have 0-length kv)
            return build_past_key_values_prefix(model=model, ext_kv_by_layer={})

        ext_by_layer: Dict[int, Any] = {}
        for li in self.cfg.layers:
            ext = stack_ext_kv_items_by_layer(
                items=items,
                layer_id=li,
                batch_size=1,
                device=device,
                dtype=dtype,
                kv_len_key="kv_len",
            )
            if self.cfg.strategy == "gate" and gamma is not None:
                # clamp + scale external V (engineering approximation)
                g = gamma.clamp(min=0.0, max=float(self.cfg.clamp_max)).to(dtype=dtype, device=device)
                ext = type(ext)(K=ext.K, V=ext.V * g)
            ext_by_layer[li] = ext

        return build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)



