"""
model_adapters.deepseek: DeepSeek 适配器（最小可运行）

说明
- 与 Qwen 一样，当前用 past_key_values 前缀注入，不需要深入 MoE 路由细节。
"""

from __future__ import annotations

import torch

from .base import ModelAdapter


class DeepSeekAdapter(ModelAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)


