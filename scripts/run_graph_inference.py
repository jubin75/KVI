#!/usr/bin/env python3
"""
Scheme C — Graph Inference CLI.

Lightweight runtime: graph-based retrieval + LLM generation.
Replaces Mode A's tag routing with entity-anchored graph traversal.

Usage::

    python scripts/run_graph_inference.py \\
        --model /path/to/base_llm \\
        --prompt "SFTSV的临床症状有哪些？" \\
        --graph_index /path/to/graph_index.json \\
        [--use_chat_template] [--local_files_only]

Output (JSON to stdout):
    {
        "diagnosis_result": "...",
        "base_llm_result": "...",
        "graph_debug": {...},
        "grounding_report": {...}
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Keyword-based intent classification (lightweight, no LLM needed)
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "symptom":       ["症状", "表现", "临床", "symptom", "sign", "表征", "不适", "起病"],
    "drug":          ["药物", "治疗", "drug", "treatment", "therapy", "用药", "给药", "法匹拉韦"],
    "mechanism":     ["机制", "mechanism", "pathway", "通路", "过程", "原理", "发病机制"],
    "transmission":  ["传播", "transmission", "传染", "途径", "感染途径", "蜱虫"],
    "definition":    ["是什么", "定义", "什么是", "identity", "属于", "缩写", "全称"],
    "epidemiology":  ["流行", "分布", "epidemiology", "prevalence", "发病率", "流行病学"],
    "prevention":    ["预防", "prevention", "防控", "防护"],
    "location":      ["部位", "位置", "location", "anatomy", "器官"],
    "etiology":      ["病因", "病原", "etiology", "原因", "致病"],
}


def _classify_intent(query: str) -> str:
    """Classify query intent via keyword matching."""
    q = query.lower()
    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in q)
        if score > 0:
            scores[intent] = score
    if not scores:
        return ""
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Simple grounding filter (token-overlap based)
# ---------------------------------------------------------------------------

def _tokenize_chinese(text: str) -> set[str]:
    """Bigram + word tokenization for Chinese + English."""
    tokens: set[str] = set()
    # Chinese bigrams
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])
    # English words
    for w in re.findall(r"[a-zA-Z]{2,}", text.lower()):
        tokens.add(w)
    return tokens


def _simple_grounding(output_text: str, evidence_texts: list[str]) -> dict:
    """Simple grounding check based on token overlap with evidence."""
    evidence_tokens: set[str] = set()
    for et in evidence_texts:
        evidence_tokens |= _tokenize_chinese(et)

    sents = re.split(r"[。！？\n]+", output_text)
    sents = [s.strip() for s in sents if s.strip()]

    details = []
    grounded = 0
    for s in sents:
        s_tokens = _tokenize_chinese(s)
        if not s_tokens:
            details.append({"sentence": s, "overlap": 0, "grounded": False})
            continue
        overlap = len(s_tokens & evidence_tokens) / len(s_tokens) if s_tokens else 0
        is_grounded = overlap >= 0.15
        if is_grounded:
            grounded += 1
        details.append({"sentence": s, "overlap": round(overlap, 4), "grounded": is_grounded})

    return {
        "total_sentences": len(sents),
        "grounded": grounded,
        "dropped_or_tagged": len(sents) - grounded,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="Graph-based inference (Scheme C)")
    p.add_argument("--model", required=True, help="Base LLM model path")
    p.add_argument("--prompt", required=True, help="User query")
    p.add_argument("--graph_index", required=True, help="graph_index.json path")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_evidence", type=int, default=10)
    p.add_argument("--max_hops", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    # ---- 1. Load graph ----
    from src.graph.schema import KnowledgeGraphIndex
    from src.graph.graph_retriever import GraphRetriever

    graph_path = Path(args.graph_index)
    if not graph_path.exists():
        print(json.dumps({"error": f"graph_index not found: {graph_path}"}, ensure_ascii=False))
        sys.exit(1)

    graph = KnowledgeGraphIndex.load(graph_path)
    print(
        f"[graphC] Loaded graph: nodes={len(graph.nodes)} "
        f"triples={len(graph.triples)} index={len(graph.entity_index)}",
        file=sys.stderr,
    )

    # ---- 2. Classify intent ----
    intent = _classify_intent(args.prompt)
    print(f"[graphC] intent={intent}", file=sys.stderr)

    # ---- 3. Graph retrieval ----
    retriever = GraphRetriever(
        graph=graph,
        max_hops=args.max_hops,
        max_evidence=args.max_evidence,
    )
    gr = retriever.retrieve(args.prompt, intent=intent)

    evidence_texts = [e["text"] for e in gr.evidence_sentences]
    entity_context = gr.entity_context
    print(
        f"[graphC] entities matched={len(gr.matched_entities)} "
        f"evidence={len(evidence_texts)}",
        file=sys.stderr,
    )

    if not evidence_texts:
        print(json.dumps({
            "diagnosis_result": "现有知识图谱中未找到相关证据。",
            "base_llm_result": "",
            "graph_debug": gr.debug,
            "evidence_texts": [],
            "entity_context": entity_context,
            "intent": intent,
            "grounding_report": {
                "total_sentences": 0, "grounded": 0,
                "dropped_or_tagged": 0, "details": [],
            },
        }, ensure_ascii=False))
        sys.exit(0)

    # ---- 4. Load LLM ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)

    print(f"[graphC] Loading model: {args.model} → {device}", file=sys.stderr)
    tok_kwargs: dict = {"use_fast": True, "trust_remote_code": True}
    model_kwargs: dict = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.local_files_only:
        tok_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device).eval()
    print("[graphC] Model loaded", file=sys.stderr)

    # ---- 5. Build prompt ----
    evidence_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(evidence_texts))

    prompt_parts: list[str] = []
    if entity_context:
        prompt_parts.append(entity_context)
    prompt_parts.append(f"### 参考证据\n{evidence_block}")
    prompt_parts.append(f"### 问题\n{args.prompt}")
    prompt_parts.append(
        "### 要求\n"
        "请仅根据上述参考证据回答问题。使用中文简洁作答，不引入外部知识。"
        "如果证据不足以完整回答，请如实说明。"
    )
    full_prompt = "\n\n".join(prompt_parts)

    # ---- 6. Generate (graph + evidence) ----
    def _build_input(messages: list[dict]) -> str:
        if args.use_chat_template:
            apply_fn = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_fn):
                try:
                    return apply_fn(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
        return messages[-1]["content"]

    main_messages = [
        {"role": "system", "content": "你是一个医学专业助手，根据提供的参考证据回答问题。"},
        {"role": "user", "content": full_prompt},
    ]
    text_input = _build_input(main_messages)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=6,
            repetition_penalty=1.15,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    print(f"[graphC] Generated answer ({len(answer)} chars)", file=sys.stderr)

    # ---- 7. Generate baseline (no evidence) ----
    base_messages = [
        {"role": "system", "content": "你是一个医学专业助手。"},
        {"role": "user", "content": args.prompt},
    ]
    base_input = _build_input(base_messages)
    base_inputs = tokenizer(base_input, return_tensors="pt").to(device)

    with torch.no_grad():
        base_ids = model.generate(
            **base_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=6,
            repetition_penalty=1.15,
        )
    base_new = base_ids[0][base_inputs["input_ids"].shape[1] :]
    base_answer = tokenizer.decode(base_new, skip_special_tokens=True).strip()

    # ---- 8. Grounding filter ----
    grounding = _simple_grounding(answer, evidence_texts)

    # Assemble grounded output (only include grounded sentences)
    grounded_sents = [d["sentence"] for d in grounding["details"] if d["grounded"]]
    grounded_text = "。".join(grounded_sents) + "。" if grounded_sents else answer

    # ---- 9. Output ----
    result_json = {
        "diagnosis_result": grounded_text,
        "diagnosis_result_raw": answer,
        "base_llm_result": base_answer,
        "graph_debug": gr.debug,
        "evidence_texts": evidence_texts,
        "entity_context": entity_context,
        "intent": intent,
        "grounding_report": grounding,
    }
    print(json.dumps(result_json, ensure_ascii=False))


if __name__ == "__main__":
    main()
