"""
CLI：训练 KVProjector（对齐到 past_key_values 空间）

训练原则
- 冻结 base model（不训练 LLM layers）
- 训练 KVProjector，使其从 last_hidden 预测每个注入层的 (K,V) cache
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.training.projector_kv import KVProjector, ProjectorConfig, masked_mse_kv  # type: ignore
except ModuleNotFoundError:
    from src.training.projector_kv import KVProjector, ProjectorConfig, masked_mse_kv  # type: ignore


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch items contain: input_ids [T], attention_mask [T], kv_len int, teacher_k [L, kv_heads, T, head_dim]
    input_ids = torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    kv_len = torch.tensor([int(b["kv_len"]) for b in batch], dtype=torch.long)

    # teacher_k/v are already padded to max_kv_tokens, but batch may have varying T if built with different max_kv_tokens.
    teacher_k = torch.stack([b["teacher_k"] for b in batch], dim=0)
    teacher_v = torch.stack([b["teacher_v"] for b in batch], dim=0)
    layer_ids = batch[0]["layer_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kv_len": kv_len,
        "teacher_k": teacher_k,
        "teacher_v": teacher_v,
        "layer_ids": layer_ids,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to teacher_kv_dataset.pt")
    p.add_argument("--model", required=True, help="HF model name/path used to build teacher dataset")
    p.add_argument("--out_dir", required=True, help="Directory to save projector checkpoint")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, help="float16|bfloat16|float32 (default: auto)")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM  # type: ignore

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = None
    if args.dtype:
        torch_dtype = getattr(torch, args.dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16

    # load dataset (list of dict)
    data = torch.load(args.dataset, map_location="cpu")
    dl = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    # load base model frozen
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    base.to(dev)
    base.eval()
    for p_ in base.parameters():
        p_.requires_grad_(False)

    # infer shapes from first batch
    b0 = next(iter(dl))
    teacher_k0 = b0["teacher_k"]  # [B, L, kv_heads, T, head_dim]
    _, L, kv_heads, T, head_dim = teacher_k0.shape
    layer_ids = list(b0["layer_ids"])
    hidden_size = int(base.config.hidden_size)

    proj = KVProjector(ProjectorConfig(hidden_size=hidden_size, kv_heads=kv_heads, head_dim=head_dim, layer_ids=layer_ids))
    proj.to(dev)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            kv_len = batch["kv_len"].to(dev)
            teacher_k = batch["teacher_k"].to(dev)
            teacher_v = batch["teacher_v"].to(dev)

            with torch.no_grad():
                out = base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = out.hidden_states[-1]  # [B, T, H]

            pred_k, pred_v = proj(last_hidden)
            # pred_k/v: [B, L, kv_heads, T, head_dim]
            loss = masked_mse_kv(pred_k=pred_k, pred_v=pred_v, teacher_k=teacher_k, teacher_v=teacher_v, kv_len=kv_len)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f"step={step} loss={loss.item():.6f}")
            step += 1

    ckpt = {
        "projector_state_dict": proj.state_dict(),
        "projector_cfg": {
            "hidden_size": hidden_size,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "layer_ids": layer_ids,
        },
        "base_model": args.model,
    }
    torch.save(ckpt, out_dir / "projector_kv.pt")
    print(f"Saved projector checkpoint to: {out_dir / 'projector_kv.pt'}")


if __name__ == "__main__":
    main()


