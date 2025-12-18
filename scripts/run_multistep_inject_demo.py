"""
CLI：Multi-step Injection Demo（Qwen + blocks KVBank）

严格遵循 PRD/多步注入的工程实现.md 的核心约束：
- 每步注入 ≤1024 tokens（由 256-token blocks 组成）
- 注入层默认 0..3
- retrieval 是推理的一部分：每步基于当前状态重新检索
- stopping policy（边际收益 + 冗余 + 安全上限）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import Retriever, RoutedRetriever, RoutedRetrieverConfig  # type: ignore
    from external_kv_injection.src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import Retriever, RoutedRetriever, RoutedRetrieverConfig  # type: ignore
    from src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--kv_dir", required=True)
    p.add_argument("--kv_dir_tables", default=None, help="Optional tables KVBank dir (built by --split_tables).")
    p.add_argument("--enable_table_routing", action="store_true", help="If set, route table-like queries to kv_dir_tables.")
    p.add_argument("--table_top_k", type=int, default=4, help="When routing hits, retrieve up to N table blocks.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--domain_encoder_model", required=True, help="DomainEncoder for query embedding (must match KVBank keys)")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--max_step_tokens", type=int, default=1024)
    p.add_argument("--max_total_tokens", type=int, default=2048)
    p.add_argument("--top_k_blocks", type=int, default=8)
    p.add_argument("--max_blocks_per_step", type=int, default=8, help="Cap selected blocks per step. For RoPE models, try 1 first.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--print_baseline", action="store_true", help="Print baseline answer without KV injection for A/B compare.")
    p.add_argument("--use_attention_entropy", action="store_true", help="Enable external KV attention entropy stopping signal")
    p.add_argument("--entropy_threshold", type=float, default=0.35, help="Normalized entropy threshold in [0,1]")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else None, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    if bool(args.print_baseline):
        inputs0 = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out0 = model.generate(**inputs0, max_new_tokens=int(args.max_new_tokens), do_sample=False, use_cache=True)
        print("=== Baseline (no injection) ===")
        print(tok.decode(out0[0], skip_special_tokens=True))

    bank = FaissKVBank.load(Path(args.kv_dir))
    if bool(args.enable_table_routing) and args.kv_dir_tables:
        table_bank = FaissKVBank.load(Path(args.kv_dir_tables))
        retriever = RoutedRetriever(
            kv_bank=bank,
            table_kv_bank=table_bank,
            cfg=RoutedRetrieverConfig(enable_table_routing=True, table_top_k=int(args.table_top_k)),
        )
    else:
        retriever = Retriever(bank)

    enc = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=args.domain_encoder_model, max_length=256, normalize=True))

    def query_embed_fn(text: str):
        return enc.encode(text)[0]

    cfg = MultiStepConfig(
        inject_layers=[int(x.strip()) for x in args.layers.split(",") if x.strip() != ""],
        block_tokens=256,
        max_step_tokens=args.max_step_tokens,
        max_total_tokens=args.max_total_tokens,
        max_steps=args.max_steps,
        top_k_blocks=args.top_k_blocks,
        max_blocks_per_step=int(args.max_blocks_per_step),
        use_attention_entropy=bool(args.use_attention_entropy),
        entropy_threshold=float(args.entropy_threshold),
    )
    injector = MultiStepInjector(retriever=retriever, cfg=cfg)
    answer, dbg = injector.run(model=model, tokenizer=tok, prompt=args.prompt, device=device, max_new_tokens=args.max_new_tokens, query_embed_fn=query_embed_fn)

    print("=== Step Debug ===")
    for d in dbg:
        print(d)
    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()


