"""
Minimal KVI2Runtime.run_ab test harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


def _load_block_text_lookup(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            bid = rec.get("block_id") or (rec.get("metadata") or {}).get("block_id")
            if bid:
                out[str(bid)] = str(rec.get("text") or "")
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    p = argparse.ArgumentParser(description="Run KVI2Runtime.run_ab test")
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompt", required=True, help="User prompt")
    p.add_argument("--kv_dir", required=True, help="KVBank directory")
    p.add_argument("--blocks_jsonl", required=True, help="blocks.enriched.jsonl path")
    p.add_argument("--pattern_index_dir", required=True, help="pattern sidecar directory")
    p.add_argument("--sidecar_dir", required=True, help="sidecar directory")
    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--kv_refresh_rounds", type=int, default=2)
    p.add_argument("--kv_irrelevant_logit_delta_threshold", type=float, default=0.05)
    p.add_argument("--use_chat_template", action="store_true")
    args = p.parse_args()

    try:
        from external_kv_injection.src.runtime.kvi2_runtime import KVI2Runtime, KVI2Config  # type: ignore
    except ModuleNotFoundError:
        from src.runtime.kvi2_runtime import KVI2Runtime, KVI2Config  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(device).eval()

    block_text_lookup = _load_block_text_lookup(args.blocks_jsonl)

    cfg = KVI2Config(
        top_k=int(args.top_k),
        kv_refresh_rounds=int(args.kv_refresh_rounds),
        kv_irrelevant_logit_delta_threshold=float(args.kv_irrelevant_logit_delta_threshold),
        pattern_index_dir=str(args.pattern_index_dir),
    )
    runtime = KVI2Runtime(cfg=cfg, domain_encoder_model=str(args.domain_encoder_model))

    out = runtime.run_ab(
        model=model,
        tokenizer=tok,
        prompt=str(args.prompt),
        kv_dir=str(args.kv_dir),
        device=device,
        use_chat_template=bool(args.use_chat_template),
        block_text_lookup=block_text_lookup,
        sidecar_dir=str(args.sidecar_dir),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
