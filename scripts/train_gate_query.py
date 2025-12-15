"""
CLI：训练 QueryEmbeddingGate（demo：用伪标签/自监督先跑通）

默认训练目标（无人工标注时）
- 使用“检索 margin”作为 γ 的伪标签：
  - margin = score_top1 - score_top2（或 topk 的统计）
  - gamma_target = clamp_max * sigmoid(a * margin + b)

你可以后续替换为更严格的监督：
- 哪些 query 需要外部知识（0/1 或连续强度）
- 或用 QA 质量/引用正确率作为间接信号

重要一致性要求
- Gate 的输入 embedding 空间必须与 KVBank 的 retrieval_keys 空间一致。
  - 如果 KVBank 用 DomainEncoder 建 keys，则 Gate 也应输入同一 DomainEncoder 的 embedding。
  - 如果 KVBank 用 base pooled hidden 建 keys，则 Gate 输入也应使用同语义（不推荐长期使用）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from external_kv_injection.src.kv_bank import FaissKVBank
from external_kv_injection.src.retriever import Retriever
from external_kv_injection.src.training.gate_query import GateConfig, QueryEmbeddingGate


class GateDataset(Dataset):
    def __init__(self, qs: np.ndarray, targets: np.ndarray) -> None:
        self.qs = qs.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return self.qs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"q": self.qs[idx], "t": self.targets[idx]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kv_dir", required=True)
    p.add_argument("--out", required=True, help="Output gate checkpoint path")
    p.add_argument("--num_samples", type=int, default=2000, help="How many synthetic queries to sample")
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--clamp_max", type=float, default=0.10)
    p.add_argument("--a", type=float, default=3.0, help="Sigmoid scale for margin->gamma_target")
    p.add_argument("--b", type=float, default=0.0, help="Sigmoid bias for margin->gamma_target")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    bank = FaissKVBank.load(Path(args.kv_dir))
    retriever = Retriever(bank)

    # 构造 synthetic queries：从 bank 的 retrieval_keys 加噪声（自监督/伪标签）
    rng = np.random.default_rng(0)
    base_keys = bank.retrieval_keys
    idx = rng.integers(0, base_keys.shape[0], size=args.num_samples)
    qs = base_keys[idx] + 0.05 * rng.standard_normal((args.num_samples, bank.dim)).astype(np.float32)

    # 伪标签：用检索 margin 推 gamma_target
    targets = np.zeros((args.num_samples, 1), dtype=np.float32)
    for i in range(args.num_samples):
        items, debug = bank.search(qs[i], top_k=max(args.top_k, 2))
        if len(items) < 2:
            margin = 0.0
        else:
            margin = float(items[0].score - items[1].score)
        t = float(args.clamp_max) * (1.0 / (1.0 + np.exp(-(args.a * margin + args.b))))
        targets[i, 0] = t

    ds = GateDataset(qs, targets)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = QueryEmbeddingGate(GateConfig(input_dim=bank.dim, hidden_dim=256, clamp_max=float(args.clamp_max))).to(dev)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            q = torch.as_tensor(batch["q"], device=dev)
            t = torch.as_tensor(batch["t"], device=dev)
            pred = gate(q)
            loss = torch.nn.functional.mse_loss(pred, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

    gate.save(args.out)
    print(f"Saved gate to: {args.out}")


if __name__ == "__main__":
    main()


