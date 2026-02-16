#!/usr/bin/env python3
"""
Scheme C — CLI: Extract knowledge triples from sentences.

Usage::

    python scripts/extract_triples.py \\
        --sentences_jsonl /path/to/sentences.jsonl \\
        --out_triples /path/to/triples.jsonl \\
        --model /path/to/base_llm \\
        --batch_size 3

Input:  sentences.jsonl  (one JSON object per line, must have "id" and "text")
Output: triples.jsonl    (one Triple JSON per line)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_CITATION_RE = re.compile(r"\b[A-Z][A-Za-z\-']+\s+et al\.?,?\s*\(?\d{4}\)?")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.IGNORECASE)
_YEAR_PAREN_RE = re.compile(r"\(\d{4}\)")


def _is_sentence_noise(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    if _DOI_RE.search(t):
        return True
    if _CITATION_RE.search(t):
        return True
    if len(_YEAR_PAREN_RE.findall(t)) >= 3:
        return True
    return False


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="Extract knowledge triples from sentences")
    p.add_argument("--sentences_jsonl", required=True, help="Input sentences JSONL file")
    p.add_argument("--out_triples", required=True, help="Output triples JSONL file")
    p.add_argument("--model", default="", help="HF model name or local path (base LLM, for local mode)")
    p.add_argument("--batch_size", type=int, default=3, help="Sentences per extraction batch")
    p.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens for extraction output")
    p.add_argument("--relation_types_json", default="", help="Custom relation types JSON file")
    p.add_argument("--entity_types_json", default="", help="Custom entity types JSON file")
    p.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    p.add_argument("--aliases_jsonl", default="", help="Entity aliases JSONL (canonical + aliases)")
    # DeepSeek API mode
    p.add_argument("--use_deepseek", action="store_true", help="Use DeepSeek API instead of local LLM")
    p.add_argument("--deepseek_base_url", default="https://api.deepseek.com", help="DeepSeek API base URL")
    p.add_argument("--deepseek_model", default="deepseek-chat", help="DeepSeek model name")
    p.add_argument("--deepseek_api_key_env", default="DEEPSEEK_API_KEY", help="Env var for DeepSeek API key")
    args = p.parse_args()

    # Load sentences
    sentences_path = Path(args.sentences_jsonl)
    if not sentences_path.exists():
        print(f"ERROR: sentences file not found: {sentences_path}", file=sys.stderr)
        sys.exit(1)

    sentences = []
    with sentences_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                if isinstance(rec, dict) and rec.get("text"):
                    text = str(rec.get("text") or "").strip()
                    if _is_sentence_noise(text):
                        continue
                    sentences.append(rec)
            except Exception:
                continue

    print(f"Loaded {len(sentences)} sentences from {sentences_path}", file=sys.stderr)
    if not sentences:
        print("ERROR: no sentences to process", file=sys.stderr)
        sys.exit(1)

    # Load optional configs
    from src.graph.schema import load_relation_types, load_entity_types
    rel_path = Path(args.relation_types_json) if args.relation_types_json else None
    ent_path = Path(args.entity_types_json) if args.entity_types_json else None
    rel_types = load_relation_types(rel_path)
    ent_types = load_entity_types(ent_path)

    from src.graph.triple_extractor import TripleExtractor

    if args.use_deepseek:
        # ---- DeepSeek API mode (no GPU needed) ----
        from src.llm_filter.deepseek_client import DeepSeekClient, DeepSeekClientConfig
        print(f"Using DeepSeek API: model={args.deepseek_model} base_url={args.deepseek_base_url}", file=sys.stderr)
        client = DeepSeekClient(DeepSeekClientConfig(
            base_url=args.deepseek_base_url,
            model=args.deepseek_model,
            api_key_env=args.deepseek_api_key_env,
            timeout_s=120,
        ))
        extractor = TripleExtractor(
            deepseek_client=client,
            relation_types=rel_types,
            entity_types=ent_types,
            batch_size=args.batch_size,
        )
    else:
        # ---- Local LLM mode ----
        if not args.model:
            print("ERROR: --model is required when not using --use_deepseek", file=sys.stderr)
            sys.exit(1)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        print(f"Loading model: {args.model} → {device}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.to(device).eval()

        extractor = TripleExtractor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            relation_types=rel_types,
            entity_types=ent_types,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    print(f"Extracting triples (batch_size={args.batch_size})...", file=sys.stderr)
    triples = extractor.extract_from_sentences(sentences)
    print(f"Extracted {len(triples)} triples", file=sys.stderr)

    # Write output
    out_path = Path(args.out_triples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    print(f"Written {len(triples)} triples to {out_path}", file=sys.stderr)

    # Print summary
    relations_count: dict = {}
    for t in triples:
        relations_count[t.predicate] = relations_count.get(t.predicate, 0) + 1
    print("\n=== Extraction Summary ===", file=sys.stderr)
    print(f"  Sentences: {len(sentences)}", file=sys.stderr)
    print(f"  Triples:   {len(triples)}", file=sys.stderr)
    print(f"  Relations:", file=sys.stderr)
    for rel, cnt in sorted(relations_count.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {cnt}", file=sys.stderr)


if __name__ == "__main__":
    main()
