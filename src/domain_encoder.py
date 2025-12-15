"""
domain_encoder: DomainEncoder 的运行期接口（可运行实现）

说明
- 训练 DomainEncoder 的完整流程较长，这里提供工程上可用的“加载/编码”接口：
  - 基于 `encoders.hf_sentence_encoder.HFSentenceEncoder`
  - 可用于：构建 KVBank 的 retrieval_keys、在线检索 query embedding、Gate 的 query embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig


@dataclass(frozen=True)
class DomainEncoderConfig:
    model_name_or_path: str
    max_length: int = 256
    normalize: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None


class DomainEncoder:
    def __init__(self, cfg: DomainEncoderConfig) -> None:
        self.cfg = cfg
        self._enc = HFSentenceEncoder(
            HFSentenceEncoderConfig(
                model_name_or_path=cfg.model_name_or_path,
                max_length=cfg.max_length,
                normalize=cfg.normalize,
                device=cfg.device,
                dtype=cfg.dtype,
            )
        )

    @property
    def dim(self) -> int:
        return self._enc.dim

    def encode(self, texts: Union[str, Sequence[str]], batch_size: int = 16) -> np.ndarray:
        return self._enc.encode(texts, batch_size=batch_size)



