"""
Pipeline: ChunkStore(JSONL) → FAISS KVBank (runnable implementation)

Key design (to make the "complete project runnable")
- To solve RoPE/cache space alignment issues across different models, this pipeline directly runs a single forward pass
  on each chunk using the "target base model" and retrieves its `past_key_values` (i.e. real K/V cache). Thus:
  - K/V head_dim, kv_heads, rotary processing are all consistent with the target model
  - During online injection, these K/V values can be directly used as a past_key_values prefix

Cost
- Building KVBank requires running the target model (fully acceptable for small KB demos; production can use batching, caching, sharding/incremental).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..vector_store.faiss_kv_bank import FaissKVBank


@dataclass(frozen=True)
class BuildStats:
    total_read: int
    total_written: int
    skipped_too_short: int
    skipped_errors: int


def _read_chunkstore(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def build_faiss_kvbank_from_chunkstore(
    *,
    chunkstore_jsonl: Path,
    out_dir: Path,
    model_name_or_path: str,
    inject_layers: Sequence[int],
    max_kv_tokens: int = 128,
    max_chunks: Optional[int] = None,
    normalize_keys: bool = True,
    retrieval_encoder_model: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Tuple[FaissKVBank, BuildStats]:
    """
    Build KVBank:
    - retrieval_key: mean pooling over the target base model's last hidden layer (demo uses H-dim vector directly)
    - K_ext/V_ext: directly take the target base model's past_key_values (select layers per inject_layers)

    K/V storage format (multi-layer):
    - k_ext: [N, L, kv_heads, max_kv_tokens, head_dim] (padded to max_kv_tokens)
    - v_ext: same as above
    - meta records: layer_ids, kv_len (effective token length)
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(dev)
    model.eval()

    layer_ids = list(inject_layers)
    num_layers = int(model.config.num_hidden_layers)
    for li in layer_ids:
        if li < 0 or li >= num_layers:
            raise ValueError(f"inject layer {li} out of range [0, {num_layers-1}]")

    # retrieval keys: default uses base model pooled hidden; if retrieval_encoder_model is specified, use DomainEncoder space
    encoder = None
    if retrieval_encoder_model:
        from ..encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig

        encoder = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=retrieval_encoder_model, max_length=max_kv_tokens, normalize=normalize_keys))

    retrieval_keys: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    k_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []

    total_read = 0
    total_written = 0
    skipped_too_short = 0
    skipped_errors = 0

    for rec in _read_chunkstore(chunkstore_jsonl):
        total_read += 1
        if max_chunks is not None and total_written >= max_chunks:
            break

        text = (rec.get("text") or "").strip()
        if len(text) < 20:
            skipped_too_short += 1
            continue

        try:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_kv_tokens)
            input_ids = inputs["input_ids"].to(dev)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if encoder is None:
                last_hidden = out.hidden_states[-1]  # [1, T, H]
                pooled = _mean_pool_last_hidden(last_hidden, attention_mask)  # [1, H]
                key = pooled[0].to(torch.float32).cpu().numpy()
            else:
                # Use DomainEncoder space (recommended for production retrieval)
                key = encoder.encode(text)[0]

            pkv = out.past_key_values
            if pkv is None:
                raise RuntimeError("model did not return past_key_values; cannot build KVBank")

            # pkv is tuple(layer) of (K,V): [B, kv_heads, T, head_dim]
            kv_len = int(input_ids.shape[1])
            # build per-layer arrays and pad to max_kv_tokens
            per_layer_k: List[np.ndarray] = []
            per_layer_v: List[np.ndarray] = []
            for li in layer_ids:
                K, V = pkv[li]
                # slice to kv_len (already) and pad to max_kv_tokens
                K = K[:, :, :kv_len, :]
                V = V[:, :, :kv_len, :]
                kv_heads = K.shape[1]
                head_dim = K.shape[-1]

                K_pad = torch.zeros((1, kv_heads, max_kv_tokens, head_dim), device=K.device, dtype=K.dtype)
                V_pad = torch.zeros((1, kv_heads, max_kv_tokens, head_dim), device=V.device, dtype=V.dtype)
                K_pad[:, :, :kv_len, :] = K
                V_pad[:, :, :kv_len, :] = V
                per_layer_k.append(K_pad[0].detach().cpu().numpy())
                per_layer_v.append(V_pad[0].detach().cpu().numpy())

            # stack layer dim: [L, kv_heads, max_kv_tokens, head_dim]
            k_item = np.stack(per_layer_k, axis=0)
            v_item = np.stack(per_layer_v, axis=0)

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
            k_list.append(k_item)
            v_list.append(v_item)
            metas.append(meta)
            total_written += 1

        except Exception:
            skipped_errors += 1
            continue

    if total_written == 0:
        raise RuntimeError("No chunks were written into KVBank; check your chunkstore and settings.")

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

    stats = BuildStats(
        total_read=total_read,
        total_written=total_written,
        skipped_too_short=skipped_too_short,
        skipped_errors=skipped_errors,
    )
    return bank, stats


