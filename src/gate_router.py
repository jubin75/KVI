"""
gate_router: 推理期 γ 生成（可运行实现）

当前实现对齐你已经选定的路线：
- γ 由 query embedding 驱动（DomainEncoder(query) embedding）
- 使用 training/gate_query.py 的 QueryEmbeddingGate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .training.gate_query import GateConfig, QueryEmbeddingGate


@dataclass(frozen=True)
class GateRouterConfig:
    mode: str = "constant"  # constant | learned
    constant_gamma: float = 0.08
    clamp_max: float = 0.10
    gate_ckpt: Optional[str] = None


class GateRouter:
    def __init__(self, cfg: GateRouterConfig, *, input_dim: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.gate: Optional[QueryEmbeddingGate] = None

        if cfg.mode == "learned":
            if cfg.gate_ckpt:
                self.gate = QueryEmbeddingGate.load(cfg.gate_ckpt, map_location="cpu").to(self.device)
            else:
                if input_dim is None:
                    raise ValueError("input_dim required when initializing learned gate without checkpoint")
                self.gate = QueryEmbeddingGate(GateConfig(input_dim=input_dim, clamp_max=cfg.clamp_max)).to(self.device)
            self.gate.eval()

    def gamma(self, q_emb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        q_emb: [B,D] or None
        returns gamma: [B,1]
        """

        if self.cfg.mode == "constant" or self.gate is None:
            if q_emb is None:
                return torch.tensor([[float(self.cfg.constant_gamma)]], device=self.device)
            return torch.full((q_emb.shape[0], 1), float(self.cfg.constant_gamma), device=q_emb.device, dtype=q_emb.dtype)

        if q_emb is None:
            raise ValueError("learned gate requires q_emb")
        with torch.no_grad():
            g = self.gate(q_emb.to(self.device, dtype=torch.float32)).to(dtype=q_emb.dtype)
        return g



