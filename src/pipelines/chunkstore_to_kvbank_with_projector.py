"""
Pipeline：ChunkStore(JSONL) → FAISS KVBank（使用训练好的 KVProjector 生成 K/V）

目标
- 不再为每条 chunk 跑 teacher past_key_values（构建更快）
- 用冻结 base model 产出 last_hidden，然后用 KVProjector 生成各注入层的 K/V（对齐到 past_key_values 空间）

注意
- KVProjector 的效果依赖 teacher dataset 的覆盖与训练质量。
- 生产级需要：更好的检索 key（DomainEncoder）、更严格的回归测试（inject off / gamma=0 / 输出一致性）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..training.projector_kv import KVProjector, ProjectorConfig
from ..vector_store.faiss_kv_bank import FaissKVBank


def _read_chunkstore(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def build_faiss_kvbank_with_projector(
    *,
    chunkstore_jsonl: Path,
    out_dir: Path,
    base_model_name_or_path: str,
    projector_ckpt_path: Path,
    max_kv_tokens: int = 256,
    max_chunks: Optional[int] = None,
    normalize_keys: bool = True,
    retrieval_encoder_model: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    """
    构建 KVBank（多层 K/V）：
    - retrieval_key: pooled last_hidden（demo；生产级建议替换为 DomainEncoder）
    - K/V: KVProjector(last_hidden) -> [B, L, kv_heads, T, head_dim]，padding 到 max_kv_tokens
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    base.to(dev)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(projector_ckpt_path, map_location="cpu")
    cfg_d = ckpt["projector_cfg"]
    proj = KVProjector(
        ProjectorConfig(
            hidden_size=int(cfg_d["hidden_size"]),
            kv_heads=int(cfg_d["kv_heads"]),
            head_dim=int(cfg_d["head_dim"]),
            layer_ids=list(cfg_d["layer_ids"]),
        )
    )
    proj.load_state_dict(ckpt["projector_state_dict"])
    proj.to(dev)
    proj.eval()

    layer_ids = list(cfg_d["layer_ids"])

    encoder = None
    if retrieval_encoder_model:
        from ..encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig

        encoder = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=retrieval_encoder_model, max_length=max_kv_tokens, normalize=normalize_keys))

    retrieval_keys: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    k_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []

    written = 0
    for rec in _read_chunkstore(chunkstore_jsonl):
        if max_chunks is not None and written >= max_chunks:
            break
        text = (rec.get("text") or "").strip()
        if len(text) < 20:
            continue

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_kv_tokens)
        input_ids = inputs["input_ids"].to(dev)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
        kv_len = int(input_ids.shape[1])

        with torch.no_grad():
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]  # [1, T, H]

            if encoder is None:
                pooled = _mean_pool_last_hidden(last_hidden, attention_mask)  # [1, H]
                key = pooled[0].to(torch.float32).cpu().numpy()
            else:
                key = encoder.encode(text)[0]

            pred_k, pred_v = proj(last_hidden)  # [1, L, kv_heads, T, head_dim]

        # pad to max_kv_tokens on token dim (T)
        _, L, kv_heads, T, head_dim = pred_k.shape
        if T > max_kv_tokens:
            pred_k = pred_k[:, :, :, :max_kv_tokens, :]
            pred_v = pred_v[:, :, :, :max_kv_tokens, :]
            kv_len = max_kv_tokens
            T = max_kv_tokens

        k_pad = torch.zeros((L, kv_heads, max_kv_tokens, head_dim), dtype=torch.float32)
        v_pad = torch.zeros((L, kv_heads, max_kv_tokens, head_dim), dtype=torch.float32)
        k_pad[:, :, :T, :] = pred_k[0].to(dtype=torch.float32, device="cpu")
        v_pad[:, :, :T, :] = pred_v[0].to(dtype=torch.float32, device="cpu")

        meta = {
            "chunk_id": rec.get("chunk_id"),
            "citation": f"{rec.get('source_uri')}:{rec.get('page_range')}",
            "source_uri": rec.get("source_uri"),
            "page_range": rec.get("page_range"),
            "section_path": rec.get("section_path"),
            "lang": rec.get("lang"),
            "layer_ids": layer_ids,
            "kv_len": kv_len,
            "max_kv_tokens": max_kv_tokens,
        }

        retrieval_keys.append(key.astype(np.float32))
        k_list.append(k_pad.numpy())
        v_list.append(v_pad.numpy())
        metas.append(meta)
        written += 1

    if written == 0:
        raise RuntimeError("No chunks written; check chunkstore and filters.")

    keys_arr = np.stack(retrieval_keys, axis=0).astype(np.float32)
    k_arr = np.stack(k_list, axis=0)
    v_arr = np.stack(v_list, axis=0)

    bank = FaissKVBank.build(
        retrieval_keys=keys_arr,
        k_ext=k_arr,
        v_ext=v_arr,
        metas=metas,
        normalize=normalize_keys,
        metric="ip",
    )
    bank.save(out_dir)


