"""
Pipeline：memory blocks（256-token）→ KVBank（FAISS）

严格对齐 PRD/多步注入的工程实现.md：
- KVBank 存的是 256-token memory blocks 的 K/V（不是 raw text）
- 建库时提取 layers 0..3 的 past_key_values（teacher cache，或后续用 projector）
- 单条 block 的 kv_len 必须 <= 256，且作为注入 token 粒度

检索向量（retrieval_keys）
- 推荐使用 DomainEncoder（独立 encoder）对 block 文本编码并归一化
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig
from ..vector_store.faiss_kv_bank import FaissKVBank


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class BuildBlocksKVBankStats:
    total_read: int
    total_written: int
    skipped_bad_len: int
    skipped_errors: int


def build_kvbank_from_blocks_jsonl(
    *,
    blocks_jsonl: Path,
    out_dir: Path,
    base_llm_name_or_path: str,
    retrieval_encoder_model: str,
    layers: Sequence[int] = (0, 1, 2, 3),
    block_tokens: int = 256,
    max_blocks: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    trust_remote_code: bool = True,
) -> BuildBlocksKVBankStats:
    """
    对每条 block：
    - 用 base LLM tokenizer 截断到 block_tokens，并 forward 抽取 past_key_values(layers)
    - 用 DomainEncoder 编码 block 文本得到 retrieval_key
    - 写入 FaissKVBank（multi-layer KV）
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16

    print(f"[blocks_to_kvbank] Loading base tokenizer/model: {base_llm_name_or_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_llm_name_or_path, use_fast=True, trust_remote_code=bool(trust_remote_code))
    model = AutoModelForCausalLM.from_pretrained(
        base_llm_name_or_path, torch_dtype=torch_dtype, trust_remote_code=bool(trust_remote_code)
    )
    model.to(dev)
    model.eval()
    print(f"[blocks_to_kvbank] model_loaded device={dev.type} dtype={torch_dtype}", flush=True)

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = list(layers)
    for li in layer_ids:
        if li < 0 or li >= num_layers:
            raise ValueError(f"layer {li} out of range [0,{num_layers-1}]")

    print(f"[blocks_to_kvbank] Loading retrieval encoder: {retrieval_encoder_model}", flush=True)
    encoder = HFSentenceEncoder(
        HFSentenceEncoderConfig(model_name_or_path=retrieval_encoder_model, max_length=block_tokens, normalize=True)
    )

    retrieval_keys: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    k_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []

    total_read = 0
    total_written = 0
    skipped_bad_len = 0
    skipped_errors = 0
    error_types: Dict[str, int] = {}
    first_error: Optional[str] = None

    for idx, rec in enumerate(_read_jsonl(blocks_jsonl), start=1):
        total_read += 1
        if max_blocks is not None and total_written >= max_blocks:
            break

        text = (rec.get("text") or "").strip()
        if not text:
            skipped_bad_len += 1
            continue

        try:
            # enforce 256-token block at tokenizer level
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=block_tokens)
            input_ids = inputs["input_ids"].to(dev)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
            kv_len = int(input_ids.shape[1])
            if kv_len > block_tokens:
                kv_len = block_tokens

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            pkv = out.past_key_values
            if pkv is None:
                raise RuntimeError("past_key_values missing")

            # build per-layer padded K/V to block_tokens
            K0, _ = pkv[layer_ids[0]]
            kv_heads = int(K0.shape[1])
            head_dim = int(K0.shape[-1])

            per_layer_k: List[np.ndarray] = []
            per_layer_v: List[np.ndarray] = []
            for li in layer_ids:
                K, V = pkv[li]  # [1, kv_heads, T, head_dim]
                K = K[0, :, :kv_len, :]
                V = V[0, :, :kv_len, :]
                K_pad = torch.zeros((kv_heads, block_tokens, head_dim), device="cpu", dtype=torch.float32)
                V_pad = torch.zeros((kv_heads, block_tokens, head_dim), device="cpu", dtype=torch.float32)
                K_pad[:, :kv_len, :] = K.detach().to(device="cpu", dtype=torch.float32)
                V_pad[:, :kv_len, :] = V.detach().to(device="cpu", dtype=torch.float32)
                per_layer_k.append(K_pad.numpy())
                per_layer_v.append(V_pad.numpy())

            k_item = np.stack(per_layer_k, axis=0)  # [L, kv_heads, block_tokens, head_dim]
            v_item = np.stack(per_layer_v, axis=0)

            key = encoder.encode(text)[0].astype(np.float32)

            meta = {
                "block_id": rec.get("block_id"),
                "parent_chunk_id": rec.get("parent_chunk_id"),
                "source_id": rec.get("source_id"),
                "doc_id": rec.get("doc_id"),
                "layer_ids": layer_ids,
                "kv_len": kv_len,
                "max_kv_tokens": block_tokens,
                "citation": rec.get("block_id"),
                # carry structured metadata (tables/disease/date/paragraph_type, etc.)
                "metadata": rec.get("metadata") or {},
            }

            retrieval_keys.append(key)
            k_list.append(k_item)
            v_list.append(v_item)
            metas.append(meta)
            total_written += 1
            if idx % 50 == 0:
                print(
                    f"[blocks_to_kvbank] processed_blocks={idx} written={total_written} skipped_empty_text={skipped_bad_len} skipped_errors={skipped_errors}",
                    flush=True,
                )

        except Exception as e:
            skipped_errors += 1
            name = type(e).__name__
            error_types[name] = int(error_types.get(name, 0)) + 1
            if first_error is None:
                first_error = f"{name}: {e}"
            continue

    if total_written == 0:
        hint = (
            f"No blocks written from {blocks_jsonl}. "
            f"read={total_read}, skipped_empty_text={skipped_bad_len}, skipped_errors={skipped_errors}. "
        )
        if skipped_errors > 0:
            top = sorted(error_types.items(), key=lambda kv: kv[1], reverse=True)[:5]
            hint += f"error_types={dict(top)}. first_error={first_error}. "
            hint += "Common fixes: try a smaller base model, set device=cpu, ensure enough GPU RAM, and confirm model supports past_key_values."
        else:
            hint += "Common fixes: ensure blocks.jsonl has non-empty 'text' fields; rerun blocks step with --keep_last_incomplete_block."
        raise RuntimeError(hint)

    print(f"[blocks_to_kvbank] Building FAISS KVBank: written={total_written} out_dir={out_dir}", flush=True)
    bank = FaissKVBank.build(
        retrieval_keys=np.stack(retrieval_keys, axis=0),
        k_ext=np.stack(k_list, axis=0),
        v_ext=np.stack(v_list, axis=0),
        metas=metas,
        normalize=True,
        metric="ip",
    )
    bank.save(out_dir)
    print("[blocks_to_kvbank] KVBank saved.", flush=True)

    return BuildBlocksKVBankStats(
        total_read=total_read,
        total_written=total_written,
        skipped_bad_len=skipped_bad_len,
        skipped_errors=skipped_errors,
    )


