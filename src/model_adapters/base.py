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
        # Do greedy decoding manually because `model.generate(past_key_values=...)` assumes
        # the past length corresponds to already-consumed prompt tokens. For prefix-injection
        # the past is *external KV*, so generation may incorrectly slice prompt tokens away.
        device = next(self.model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        eos_id = getattr(tokenizer, "eos_token_id", None)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # [1,1]
        generated = [next_token]
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        for _ in range(max(0, int(max_new_tokens) - 1)):
            if isinstance(eos_id, int) and int(next_token.item()) == int(eos_id):
                break
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        full = torch.cat([input_ids] + generated, dim=1)
        return tokenizer.decode(full[0], skip_special_tokens=True)


