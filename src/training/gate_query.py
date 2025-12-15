"""
Query-Embedding Gate（γ 预测）

目标
- 输入：query embedding（例如 pooled last_hidden / domain encoder embedding）
- 输出：gamma（0~clamp_max），用于控制外部 KV 注入强度

说明
- 本 gate 不训练 LLM layers，可独立训练/部署。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GateConfig:
    input_dim: int
    hidden_dim: int = 256
    clamp_max: float = 0.10


class QueryEmbeddingGate(nn.Module):
    """
    简单 MLP：gamma = clamp_max * sigmoid(MLP(q))
    """

    def __init__(self, cfg: GateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [B, D]
        returns gamma: [B, 1] in [0, clamp_max]
        """

        x = self.net(q)
        g = torch.sigmoid(x) * float(self.cfg.clamp_max)
        return g

    def save(self, path: str) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "cfg": self.cfg.__dict__},
            path,
        )

    @staticmethod
    def load(path: str, map_location: Optional[str] = "cpu") -> "QueryEmbeddingGate":
        ckpt = torch.load(path, map_location=map_location)
        cfg = GateConfig(**ckpt["cfg"])
        m = QueryEmbeddingGate(cfg)
        m.load_state_dict(ckpt["state_dict"])
        return m


