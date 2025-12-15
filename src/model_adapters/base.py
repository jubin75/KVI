"""
model_adapters.base: 最小可运行的适配器抽象

说明
- 当前工程的主要注入实现是“past_key_values 前缀注入”，无需深度改写 attention。
- 适配器在这里的作用是：统一读取模型元信息、提供一个通用的 generate_with_past_prefix 接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class ModelMetadata:
    num_layers: int
    hidden_size: int


class ModelAdapter:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.meta = self.get_model_metadata()

    def get_model_metadata(self) -> ModelMetadata:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            raise ValueError("model has no .config")
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        hidden_size = int(getattr(cfg, "hidden_size"))
        return ModelMetadata(num_layers=num_layers, hidden_size=hidden_size)

    def supports_kv_cache(self) -> bool:
        return True

    def generate_with_past_prefix(self, *, tokenizer: Any, prompt: str, past_key_values: Any, max_new_tokens: int = 128) -> str:
        device = next(self.model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(**inputs, past_key_values=past_key_values, use_cache=True, do_sample=False, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True)


