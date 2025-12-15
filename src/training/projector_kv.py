"""
Projector：从 frozen LLM hidden_states 预测 past_key_values 空间的 K/V

设计目标（匹配你的约束）
- 不训练 LLM layers（冻结 base model）
- 训练一个轻量 Projector（外部模块），把 token-level hidden states 映射到每个注入层的 (K,V)

实现说明（demo→可扩展）
- 输入：last_hidden: [B, T, H]
- 输出（每层）：K/V: [B, kv_heads, T, head_dim]
- 采用 per-layer 的线性映射（K 和 V 各一个 Linear），形状：
  - Linear(H -> kv_heads*head_dim)，reshape 后转置到 [B, kv_heads, T, head_dim]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ProjectorConfig:
    hidden_size: int
    kv_heads: int
    head_dim: int
    layer_ids: Sequence[int]


class KVProjector(nn.Module):
    def __init__(self, cfg: ProjectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_ids = list(cfg.layer_ids)

        # per-layer K/V heads
        self.k_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        out_dim = cfg.kv_heads * cfg.head_dim
        for li in self.layer_ids:
            self.k_linears[str(li)] = nn.Linear(cfg.hidden_size, out_dim, bias=False)
            self.v_linears[str(li)] = nn.Linear(cfg.hidden_size, out_dim, bias=False)

    def forward(self, last_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
        - pred_k: [B, L, kv_heads, T, head_dim]
        - pred_v: [B, L, kv_heads, T, head_dim]
        """

        B, T, H = last_hidden.shape
        kv_heads = self.cfg.kv_heads
        head_dim = self.cfg.head_dim

        pred_ks: List[torch.Tensor] = []
        pred_vs: List[torch.Tensor] = []
        for li in self.layer_ids:
            k = self.k_linears[str(li)](last_hidden)  # [B, T, kv_heads*head_dim]
            v = self.v_linears[str(li)](last_hidden)
            k = k.view(B, T, kv_heads, head_dim).transpose(1, 2).contiguous()  # [B, kv_heads, T, head_dim]
            v = v.view(B, T, kv_heads, head_dim).transpose(1, 2).contiguous()
            pred_ks.append(k)
            pred_vs.append(v)

        pred_k = torch.stack(pred_ks, dim=1)
        pred_v = torch.stack(pred_vs, dim=1)
        return pred_k, pred_v


def masked_mse_kv(
    *,
    pred_k: torch.Tensor,
    pred_v: torch.Tensor,
    teacher_k: torch.Tensor,
    teacher_v: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    """
    pred_*:    [B, L, kv_heads, T, head_dim]
    teacher_*: [B, L, kv_heads, T, head_dim]
    kv_len:    [B]  (<=T)
    """

    B, L, Hh, T, D = pred_k.shape
    device = pred_k.device

    # mask: [B, 1, 1, T, 1]
    ar = torch.arange(T, device=device).view(1, 1, 1, T, 1)
    mask = (ar < kv_len.view(B, 1, 1, 1, 1)).to(pred_k.dtype)

    diff_k = (pred_k - teacher_k) * mask
    diff_v = (pred_v - teacher_v) * mask
    denom = mask.sum().clamp_min(1.0)
    loss = (diff_k.square().sum() + diff_v.square().sum()) / denom
    return loss


