"""
Qwen External-KV Injection Demo（HF Transformers）

你现在的目标：先跑通 demo（知识库不大），后续再生产化替换部件。

本 demo 做什么
- 加载 Qwen（或任意支持 past_key_values 的 decoder-only 模型）
- 加载本地 FAISS KVBank（直接存 K_ext/V_ext）
- 用目标模型对 prompt 做一次 forward，生成 query_vec（mean pooled last_hidden）
- 检索 top-k 条目，拼成 ExtKV
- 把 ExtKV（按层）写入 past_key_values（作为静态前缀 KV）
- 调用 model.generate 输出，并打印 debug

注意
- 需要依赖：torch, transformers, faiss（faiss-cpu 或 faiss-gpu）
- 本 demo 的 K_ext/V_ext 来自“目标基模的 past_key_values”，因此 rotary/head_dim/kv_heads 对齐天然成立（适合 demo 小 KB）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import Retriever  # type: ignore
    from external_kv_injection.src.runtime.hf_cache_prefix_injection import (  # type: ignore
        build_past_key_values_prefix,
        stack_ext_kv_items_by_layer,
    )
    from external_kv_injection.src.training.gate_query import GateConfig, QueryEmbeddingGate  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import Retriever  # type: ignore
    from src.runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer  # type: ignore
    from src.training.gate_query import GateConfig, QueryEmbeddingGate  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--kv_dir", required=True, help="Directory containing FAISS KVBank (manifest.json etc.)")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--layers", type=str, default="0,1,2,3", help="Comma-separated layer ids to inject")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--domain_encoder_model",
        default=None,
        help="HF encoder model for query embedding (DomainEncoder). If set, both retriever query_vec and gate input use this embedding.",
    )
    parser.add_argument("--domain_encoder_max_length", type=int, default=256)
    parser.add_argument("--gate_ckpt", default=None, help="Optional QueryEmbeddingGate checkpoint (.pt)")
    parser.add_argument("--gate_clamp_max", type=float, default=0.10)
    parser.add_argument(
        "--gate_mode",
        default="scale_v",
        choices=["scale_v", "onoff"],
        help="How gamma controls injection: scale_v scales ext V; onoff disables injection if gamma<1e-3.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else None)
    model.to(device)
    model.eval()

    bank = FaissKVBank.load(Path(args.kv_dir))
    retriever = Retriever(bank)

    inputs = tok(args.prompt, return_tensors="pt").to(device)

    # query embedding：默认用 DomainEncoder(query)（与检索空间一致）
    if args.domain_encoder_model:
        enc = HFSentenceEncoder(
            HFSentenceEncoderConfig(
                model_name_or_path=args.domain_encoder_model,
                max_length=args.domain_encoder_max_length,
                normalize=True,
            )
        )
        query_vec = enc.encode(args.prompt)[0]
        q_tensor = torch.as_tensor(query_vec[None, :], device=device, dtype=torch.float32)
    else:
        # fallback：用目标模型 pooled hidden（需要 KVBank 用同语义建 key 才能匹配）
        with torch.no_grad():
            out_q = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        pooled = _mean_pool_last_hidden(out_q.hidden_states[-1], inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])))
        query_vec = pooled[0].to(torch.float32).detach().cpu().numpy()
        q_tensor = pooled.to(dtype=torch.float32)

    result = retriever.search(query_vec, top_k=args.top_k, filters=None)
    print("retriever.debug:", result.debug)
    for it in result.items:
        print("hit:", {"score": it.score, "chunk_id": it.meta.get("chunk_id"), "citation": it.meta.get("citation")})

    # ---- Gate：query embedding -> gamma ----
    if args.gate_ckpt:
        gate = QueryEmbeddingGate.load(args.gate_ckpt, map_location="cpu")
        gate.to(device)
    else:
        gate = QueryEmbeddingGate(GateConfig(input_dim=q_tensor.shape[-1], clamp_max=float(args.gate_clamp_max)))
        gate.to(device)
    gate.eval()
    with torch.no_grad():
        gamma = gate(q_tensor).to(dtype=next(model.parameters()).dtype)  # [1,1]
    gamma_val = float(gamma.item())
    print("gamma:", gamma_val)

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
    # 按层拼 ext kv（每层使用其真实 past_kv）
    if args.gate_mode == "onoff" and gamma_val < 1e-3:
        ext_by_layer = {}
    else:
        ext_by_layer = {
            layer_id: stack_ext_kv_items_by_layer(
                items=result.items,
                layer_id=layer_id,
                batch_size=1,
                device=device,
                dtype=next(model.parameters()).dtype,
            )
            for layer_id in layer_ids
        }
        if args.gate_mode == "scale_v":
            # 近似 gate mixing：只缩放外部 V 的贡献（不改写 attention 的前提下的工程近似）
            for li, ext in list(ext_by_layer.items()):
                ext_by_layer[li] = type(ext)(K=ext.K, V=ext.V * gamma)  # ExtKV is a dataclass
    past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=past_key_values,
        )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


