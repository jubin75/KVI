"""
HF Sentence Encoder（DomainEncoder）

用途
- 用 HF encoder（或任意 AutoModel）把文本编码成向量 embedding，用于：
  - KVBank 的 retrieval_keys（chunk embedding）
  - 在线检索的 query embedding
  - Gate 的 query embedding（建议与检索空间一致）

实现
- mean pooling over last_hidden_state（按 attention_mask）
- 可选 L2 normalize（推荐：与 FAISS inner product 搭配即 cosine）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B,T,H], attention_mask: [B,T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


@dataclass(frozen=True)
class HFSentenceEncoderConfig:
    model_name_or_path: str
    max_length: int = 256
    normalize: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None  # float16|bfloat16|float32
    trust_remote_code: bool = True


class HFSentenceEncoder:
    def __init__(self, cfg: HFSentenceEncoderConfig) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.cfg = cfg
        dev = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = None
        if cfg.dtype:
            torch_dtype = getattr(torch, cfg.dtype)
        elif dev.type == "cuda":
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, use_fast=True, trust_remote_code=bool(cfg.trust_remote_code)
        )
        self.model = AutoModel.from_pretrained(
            cfg.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=bool(cfg.trust_remote_code)
        )
        self.model.to(dev)
        self.model.eval()
        self.device = dev

    @property
    def dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

    def encode(self, texts: Union[str, Sequence[str]], batch_size: int = 16) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        outs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inp = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=True,
                )
                inp = {k: v.to(self.device) for k, v in inp.items()}
                out = self.model(**inp, return_dict=True)
                last_hidden = out.last_hidden_state  # [B,T,H]
                pooled = _mean_pool(last_hidden, inp.get("attention_mask", torch.ones(last_hidden.shape[:2], device=self.device)))
                if self.cfg.normalize:
                    pooled = _l2_normalize(pooled)
                outs.append(pooled.to(torch.float32).cpu().numpy())

        return np.concatenate(outs, axis=0)


