"""
A/B evaluation for the JSON output protocol with optional KV injection.

Why this script exists
- Our product goal is: baseline vs injection should be comparable and automatically scorable.
- We therefore ask the model to output JSON-only (Output Protocol).
- We compute simple, deterministic metrics:
  - valid_json_rate
  - evidence_coverage (key_points covered by evidence_quotes.claim_supported == key_point)
  - overclaim_count (key_points not covered)

NOTE
- To produce quotes, the model must see the selected evidence text; therefore this script enables
  NOTE (iron law): do NOT use any prompt appendix. KV injection only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import EvidenceFirstRetriever, EvidenceFirstRetrieverConfig, Retriever  # type: ignore
    from external_kv_injection.src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import EvidenceFirstRetriever, EvidenceFirstRetrieverConfig, Retriever  # type: ignore
    from src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _score_protocol(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not obj:
        return {"valid_json": False, "key_points": 0, "evidence_quotes": 0, "covered": 0, "overclaim": 0}
    kps = obj.get("key_points", [])
    evs = obj.get("evidence_quotes", [])
    if not isinstance(kps, list):
        kps = []
    if not isinstance(evs, list):
        evs = []
    kp_strs = [str(x).strip() for x in kps if str(x).strip()]
    claims = set()
    for e in evs:
        if isinstance(e, dict):
            cs = str(e.get("claim_supported") or "").strip()
            if cs:
                claims.add(cs)
    covered = sum(1 for kp in kp_strs if kp in claims)
    overclaim = max(0, len(kp_strs) - covered)
    return {
        "valid_json": True,
        "key_points": len(kp_strs),
        "evidence_quotes": len(evs),
        "covered": covered,
        "overclaim": overclaim,
    }


PROTOCOL_INSTRUCTIONS = """你必须输出严格 JSON（不要输出任何解释/Markdown/代码块/多版本尝试）。
JSON 必须包含字段：
final_answer (string),
key_points (string[]),
evidence_quotes (array of objects with fields: quote, claim_supported, source{doc_id,block_id}).
硬约束：evidence_quotes[].claim_supported 必须逐字等于 key_points 中的某一条（用于自动评测）。
引用约束：evidence_quotes[].quote 必须逐字来自你看到的证据句（不要改写/不要翻译后当引用）。
如果证据不足以支持某条 key_point，请不要输出该 key_point。
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--domain_encoder_model", required=True)
    p.add_argument("--kv_dir", required=True, help="Raw KVBank dir (kvbank_blocks)")
    p.add_argument("--kv_dir_evidence", default=None, help="Evidence KVBank dir (kvbank_evidence). If omitted, uses raw only.")
    p.add_argument("--blocks_jsonl", required=True, help="Raw blocks.jsonl (for block_text lookup)")
    p.add_argument("--blocks_jsonl_evidence", default=None, help="Evidence blocks jsonl (recommended if kv_dir_evidence is set)")
    p.add_argument("--prompts_jsonl", required=True, help="Each line: {id?, prompt}")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--max_steps", type=int, default=1)
    p.add_argument("--max_blocks_per_step", type=int, default=1)
    p.add_argument("--top_k_blocks", type=int, default=16)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--no_repeat_ngram_size", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else None, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    bank_raw = FaissKVBank.load(Path(str(args.kv_dir)))
    bank_ev = FaissKVBank.load(Path(str(args.kv_dir_evidence))) if args.kv_dir_evidence else None
    if bank_ev is not None:
        retriever = EvidenceFirstRetriever(
            evidence_kv_bank=bank_ev, raw_kv_bank=bank_raw, cfg=EvidenceFirstRetrieverConfig()
        )
    else:
        retriever = Retriever(bank_raw)

    enc = HFSentenceEncoder(
        HFSentenceEncoderConfig(model_name_or_path=args.domain_encoder_model, max_length=256, normalize=True)
    )

    def query_embed_fn(text: str):
        return enc.encode(text)[0]

    # Build a text lookup for grounding.
    block_text_by_id: Dict[str, str] = {}
    for pth in [str(args.blocks_jsonl), str(args.blocks_jsonl_evidence) if args.blocks_jsonl_evidence else None]:
        if not pth:
            continue
        path = Path(pth)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                bid = rec.get("block_id")
                if not bid:
                    continue
                if str(bid) not in block_text_by_id:
                    block_text_by_id[str(bid)] = str(rec.get("text") or "")

    cfg = MultiStepConfig(
        inject_layers=[int(x.strip()) for x in str(args.layers).split(",") if x.strip() != ""],
        block_tokens=256,
        max_step_tokens=1024,
        max_total_tokens=2048,
        max_steps=int(args.max_steps),
        top_k_blocks=int(args.top_k_blocks),
        max_blocks_per_step=int(args.max_blocks_per_step),
        use_attention_entropy=False,
        debug_print_candidates_top_n=0,
    )
    injector = MultiStepInjector(
        retriever=retriever,
        cfg=cfg,
        allowed_block_ids=None,
        block_text_lookup=lambda bid: block_text_by_id.get(str(bid)),
    )

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg = {"n": 0, "baseline_valid": 0, "inj_valid": 0, "baseline_covered": 0, "inj_covered": 0, "baseline_over": 0, "inj_over": 0}

    with Path(str(args.prompts_jsonl)).open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            if int(args.max_examples) > 0 and agg["n"] >= int(args.max_examples):
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            prompt = str(rec.get("prompt") or rec.get("query") or "")
            if not prompt.strip():
                continue
            ex_id = rec.get("id", i)
            agg["n"] += 1

            eval_prompt = prompt.strip() + "\n\n" + PROTOCOL_INSTRUCTIONS

            # baseline
            inputs0 = tok(eval_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out0 = model.generate(**inputs0, max_new_tokens=int(args.max_new_tokens), do_sample=False, use_cache=True)
            in_len = int(inputs0["input_ids"].shape[1])
            baseline_text = tok.decode(out0[0][in_len:], skip_special_tokens=True)
            baseline_obj = _extract_first_json_obj(baseline_text)
            baseline_score = _score_protocol(baseline_obj)

            # injection (enable grounding so quotes are possible)
            inj_text, dbg = injector.run(
                model=model,
                tokenizer=tok,
                prompt=eval_prompt,
                query_text=prompt,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                query_embed_fn=query_embed_fn,
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                ground_with_selected_text=False,
                grounding_instructions=PROTOCOL_INSTRUCTIONS,
            )
            inj_obj = _extract_first_json_obj(inj_text)
            inj_score = _score_protocol(inj_obj)

            if baseline_score["valid_json"]:
                agg["baseline_valid"] += 1
                agg["baseline_covered"] += int(baseline_score["covered"])
                agg["baseline_over"] += int(baseline_score["overclaim"])
            if inj_score["valid_json"]:
                agg["inj_valid"] += 1
                agg["inj_covered"] += int(inj_score["covered"])
                agg["inj_over"] += int(inj_score["overclaim"])

            fout.write(
                json.dumps(
                    {
                        "id": ex_id,
                        "prompt": prompt,
                        "baseline_text": baseline_text,
                        "baseline_json": baseline_obj,
                        "baseline_score": baseline_score,
                        "injected_text": inj_text,
                        "injected_json": inj_obj,
                        "injected_score": inj_score,
                        "step_debug": [asdict(d) for d in (dbg or [])],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if agg["n"] == 1 or agg["n"] % 10 == 0:
                print(f"[ab_eval] n={agg['n']} baseline_valid={agg['baseline_valid']} inj_valid={agg['inj_valid']}", flush=True)

    print(f"[ab_eval] done summary={agg} out={out_path}", flush=True)


if __name__ == "__main__":
    main()


