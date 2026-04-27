"""
model_adapters.base: Minimal runnable adapter abstraction

Description:
- The current project's primary injection implementation is "past_key_values prefix injection",
  which does not require deep attention rewriting.
- The adapter's role here is: uniformly read model metadata and provide a common
  generate_with_past_prefix interface.
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
        prompt_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        def _cache_seq_len(pkv: Any) -> int:
            if pkv is None:
                return 0
            if hasattr(pkv, "get_seq_length"):
                try:
                    return int(pkv.get_seq_length())  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                k0 = pkv[0][0]
                return int(k0.shape[-2])
            except Exception:
                return 0

        prefix_len = _cache_seq_len(past_key_values)
        if prefix_len < 0:
            prefix_len = 0
        prefix_mask = torch.ones((prompt_mask.shape[0], prefix_len), dtype=prompt_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

        eos_id = getattr(tokenizer, "eos_token_id", None)

        pos0 = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
        cache_pos0 = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=device, dtype=torch.long)
        with torch.no_grad():
            try:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=pos0,
                    cache_position=cache_pos0,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            except TypeError:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=pos0,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # [1,1]
        generated = [next_token]
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        for gen_i in range(max(0, int(max_new_tokens) - 1)):
            if isinstance(eos_id, int) and int(next_token.item()) == int(eos_id):
                break
            cur_pos = prefix_len + input_ids.shape[1] + gen_i
            pos = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
            with torch.no_grad():
                try:
                    out = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=pos,
                        cache_position=cache_pos,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                except TypeError:
                    out = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=pos,
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


