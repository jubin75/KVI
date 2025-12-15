"""
Teacher KV Dataset（对齐到 past_key_values 空间）

目标
- 对每条 chunk，用冻结的目标 LLM forward，抽取指定注入层的 past_key_values 作为 teacher (K_t, V_t)。
- 训练 Projector 时，用 teacher KV 做显式监督，从而直接对齐到“模型缓存 past_key_values 空间”。

保存格式（.pt）
每条样本是一个 dict：
- input_ids: LongTensor [T]
- attention_mask: LongTensor [T]
- kv_len: int（有效 token 长度，<= max_kv_tokens）
- layer_ids: List[int]
- teacher_k: FloatTensor [L, kv_heads, max_kv_tokens, head_dim]（padding）
- teacher_v: FloatTensor [L, kv_heads, max_kv_tokens, head_dim]（padding）
- meta: dict（chunk_id/citation/lang 等）

注意
- 这是 demo→P1 的“可跑通”实现；生产级可做分片、增量、streaming、错误重试与更丰富的质量过滤。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def read_chunkstore(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class BuildTeacherKVDatasetConfig:
    model_name_or_path: str
    layers: Sequence[int]
    max_kv_tokens: int = 256
    max_samples: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None  # float16|bfloat16|float32


@dataclass(frozen=True)
class BuildTeacherKVDatasetStats:
    total_read: int
    total_written: int
    skipped_too_short: int
    skipped_errors: int


def build_teacher_kv_dataset(
    *,
    chunkstore_jsonl: Path,
    out_path: Path,
    cfg: BuildTeacherKVDatasetConfig,
) -> BuildTeacherKVDatasetStats:
    """
    读取 ChunkStore，构建 teacher KV dataset（保存为 torch .pt list）。
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = None
    if cfg.dtype:
        torch_dtype = getattr(torch, cfg.dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16  # L40 上通常 OK

    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch_dtype)
    model.to(dev)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = list(cfg.layers)
    for li in layer_ids:
        if li < 0 or li >= num_layers:
            raise ValueError(f"layer {li} out of range [0,{num_layers-1}]")

    samples: List[Dict[str, Any]] = []

    total_read = 0
    total_written = 0
    skipped_too_short = 0
    skipped_errors = 0

    for rec in read_chunkstore(chunkstore_jsonl):
        total_read += 1
        if cfg.max_samples is not None and total_written >= cfg.max_samples:
            break

        text = (rec.get("text") or "").strip()
        if len(text) < 20:
            skipped_too_short += 1
            continue

        try:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=cfg.max_kv_tokens)
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

            pkv = out.past_key_values
            if pkv is None:
                raise RuntimeError("past_key_values missing")

            T = int(input_ids.shape[1])
            kv_len = T

            # infer kv_heads/head_dim
            K0, _ = pkv[layer_ids[0]]
            kv_heads = int(K0.shape[1])
            head_dim = int(K0.shape[-1])

            teacher_k = torch.zeros((len(layer_ids), kv_heads, cfg.max_kv_tokens, head_dim), device="cpu", dtype=torch.float32)
            teacher_v = torch.zeros((len(layer_ids), kv_heads, cfg.max_kv_tokens, head_dim), device="cpu", dtype=torch.float32)

            for j, li in enumerate(layer_ids):
                K, V = pkv[li]  # [1, kv_heads, T, head_dim]
                K = K[0, :, :kv_len, :].detach().to(dtype=torch.float32, device="cpu")
                V = V[0, :, :kv_len, :].detach().to(dtype=torch.float32, device="cpu")
                teacher_k[j, :, :kv_len, :] = K
                teacher_v[j, :, :kv_len, :] = V

            sample = {
                "input_ids": input_ids[0].detach().cpu(),
                "attention_mask": attention_mask[0].detach().cpu(),
                "kv_len": kv_len,
                "layer_ids": layer_ids,
                "teacher_k": teacher_k,
                "teacher_v": teacher_v,
                "meta": {
                    "chunk_id": rec.get("chunk_id"),
                    "source_uri": rec.get("source_uri"),
                    "page_range": rec.get("page_range"),
                    "section_path": rec.get("section_path"),
                    "lang": rec.get("lang"),
                },
            }
            samples.append(sample)
            total_written += 1

        except Exception:
            skipped_errors += 1
            continue

    torch.save(samples, out_path)
    return BuildTeacherKVDatasetStats(
        total_read=total_read,
        total_written=total_written,
        skipped_too_short=skipped_too_short,
        skipped_errors=skipped_errors,
    )


