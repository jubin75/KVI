"""
model_adapters.deepseek: DeepSeek Adapter (minimal runnable)

Description:
- Same as Qwen, uses past_key_values prefix injection for now; no need to delve into
  MoE routing details.
"""

from __future__ import annotations

import torch

from .base import ModelAdapter


class DeepSeekAdapter(ModelAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)


