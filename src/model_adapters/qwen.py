"""
model_adapters.qwen: Qwen Adapter (minimal runnable)

Current implementation focuses on:
- Using past_key_values prefix injection (no need to rewrite Qwen attention)
- Providing an interface consistent with base.ModelAdapter
"""

from __future__ import annotations

import torch

from .base import ModelAdapter


class QwenAdapter(ModelAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)


