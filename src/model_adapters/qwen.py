"""
model_adapters.qwen: Qwen 适配器（最小可运行）

当前实现聚焦于：
- 使用 past_key_values 前缀注入（无需改写 Qwen attention）
- 提供与 base.ModelAdapter 一致的接口
"""

from __future__ import annotations

import torch

from .base import ModelAdapter


class QwenAdapter(ModelAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)


