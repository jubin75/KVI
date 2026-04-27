"""
domain_encoder: runtime interface of DomainEncoder (runnable implementation)

Notes
- Training a DomainEncoder end-to-end is a long process; here we provide an engineering-ready "load/encode" interface:
  - Based on `encoders.hf_sentence_encoder.HFSentenceEncoder`
  - Can be used for: building KVBank retrieval_keys, online retrieval query embeddings, Gate query embeddings
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



