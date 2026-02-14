#!/usr/bin/env python3
"""
Scheme C — Graph Inference CLI (v4: DRM + Relation Gating + 三元 KVI).

Runtime: graph-based retrieval + **DRM scoring** + **relation gating**
+ triple KV injection + LLM generation.

v4 key change (fixing the "injecting irrelevant triples" problem):

    Graph Walk → DRM Score → Relation Gating → KV Budget → KV Assembly

1. **DRM (Document Relevance Model)**: Score every walk triple's provenance
   sentence against the query using lightweight bigram overlap.
2. **Relation Gating**: Group scored triples by relation type, compute
   aggregate scores, keep only top-k relation groups.
3. **KV Budget**: From selected groups, take at most ``--max_kv_triples``
   triples (sorted by DRM score) for KV injection.
4. **Evidence Ranking**: Evidence sentences are also ranked by DRM score;
   highest-scoring evidence appears first in the prompt.

Design follows 15_架构调整.md + ToG-2's DRM → Relation Prune pattern:
  "DRM 过滤必须在 KVI 之前"

Usage::

    python scripts/run_graph_inference.py \\
        --model /path/to/base_llm \\
        --prompt "SFTSV的临床症状有哪些？" \\
        --graph_index /path/to/graph_index.json \\
        [--triple_kvbank_dir /path/to/triple_kvbank] \\
        [--max_kv_triples 3] [--drm_threshold 0.05] \\
        [--use_chat_template] [--local_files_only]

Output (JSON to stdout):
    {
        "diagnosis_result": "...",
        "base_llm_result": "...",
        "graph_debug": {...},
        "grounding_report": {...},
        "kv_injection_debug": {...}
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


# ---------------------------------------------------------------------------
# DRM (Document Relevance Model) — lightweight scoring
# ---------------------------------------------------------------------------

def _score_triple_relevance(query: str, evidence_text: str) -> float:
    """
    Lightweight DRM scoring: bigram overlap between *query* and *evidence_text*.

    Adapted from ToG-2's DRM concept and 15_架构调整.md §2:
      "DRM 过滤必须在 KVI 之前"

    Returns a score in [0, 1] representing the fraction of query tokens
    that appear in the evidence text.  Higher = more relevant.
    """
    q_tokens = _tokenize_chinese(query)
    e_tokens = _tokenize_chinese(evidence_text)
    if not q_tokens or not e_tokens:
        return 0.0
    overlap = len(q_tokens & e_tokens)
    return overlap / len(q_tokens)


def _relation_gating(
    scored_triples: list[dict],
    top_k_relations: int = 2,
) -> list[dict]:
    """
    Relation Gating Engine — 15_架构调整.md §3 / ToG-2 Relation Prune.

    1. Group scored triples by relation type.
    2. Compute aggregate score per group (sum of member DRM scores).
    3. Return only triples from the top-k highest-scoring groups.

    This prevents relation types with low aggregate relevance from
    contributing KV items (e.g., ``located_in`` triples when query is
    about symptoms).
    """
    if not scored_triples:
        return []

    # Group by relation type
    groups: dict[str, list[dict]] = {}
    for t in scored_triples:
        rel = t.get("relation", "unknown")
        groups.setdefault(rel, []).append(t)

    # Score each group
    group_ranking: list[tuple[str, float, list[dict]]] = []
    for rel, members in groups.items():
        total_score = sum(m["drm_score"] for m in members)
        group_ranking.append((rel, total_score, members))

    # Sort by aggregate score, select top-k
    group_ranking.sort(key=lambda x: x[1], reverse=True)
    selected = group_ranking[:top_k_relations]

    # Flatten selected groups
    result: list[dict] = []
    for _rel, _score, members in selected:
        result.extend(members)
    return result


# ---------------------------------------------------------------------------
# Simple grounding filter (token-overlap based)
# ---------------------------------------------------------------------------

def _simple_grounding(
    output_text: str,
    evidence_texts: list[str],
    entity_context: str = "",
    kv_texts: list[str] | None = None,
) -> dict:
    """Simple grounding check based on token overlap with evidence + entity context + KV texts."""
    evidence_tokens: set[str] = set()
    for et in evidence_texts:
        evidence_tokens |= _tokenize_chinese(et)
    # Entity context is also a valid grounding source
    if entity_context:
        evidence_tokens |= _tokenize_chinese(entity_context)
    # Triple KV texts are also valid grounding sources
    if kv_texts:
        for kt in kv_texts:
            evidence_tokens |= _tokenize_chinese(kt)

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

    p = argparse.ArgumentParser(description="Graph-based inference (Scheme C + 三元 KVI)")
    p.add_argument("--model", required=True, help="Base LLM model path")
    p.add_argument("--prompt", required=True, help="User query")
    p.add_argument("--graph_index", required=True, help="graph_index.json path")
    p.add_argument("--triple_kvbank_dir", default="", help="Triple KV bank directory (compiled by triple_kv_compiler)")
    # DRM + Relation Gating parameters
    p.add_argument("--max_kv_triples", type=int, default=3,
                   help="Max triple KVs to inject after DRM scoring (budget control)")
    p.add_argument("--drm_threshold", type=float, default=0.05,
                   help="Min DRM score for a triple to be considered for KV injection")
    p.add_argument("--top_k_relations", type=int, default=2,
                   help="Max relation groups to keep after Relation Gating")
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
            "kv_injection_debug": {"enabled": False, "reason": "no_evidence"},
        }, ensure_ascii=False))
        sys.exit(0)

    # ---- 4. Load Triple KV Bank (if available) ----
    kv_injection_debug: dict = {"enabled": False}
    triple_kv_manifest = None
    triple_kv_cache_dict = None
    triple_kvbank_dir = Path(args.triple_kvbank_dir) if args.triple_kvbank_dir else None

    if triple_kvbank_dir and triple_kvbank_dir.exists():
        try:
            from src.graph.triple_kv_compiler import (
                load_triple_kvbank,
                assemble_kv_for_entities,
            )
            triple_kv_manifest, triple_kv_cache_dict = load_triple_kvbank(triple_kvbank_dir)
            kv_injection_debug["enabled"] = True
            kv_injection_debug["kvbank_dir"] = str(triple_kvbank_dir)
            kv_injection_debug["num_items"] = len(triple_kv_manifest.items)
            kv_injection_debug["num_entities"] = len(triple_kv_manifest.entity_items)
            print(
                f"[graphC] Triple KV bank loaded: {len(triple_kv_manifest.items)} items "
                f"for {len(triple_kv_manifest.entity_items)} entities",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[graphC] WARNING: Failed to load triple KV bank: {e}", file=sys.stderr)
            kv_injection_debug["error"] = str(e)
    else:
        kv_injection_debug["reason"] = "no_kvbank_dir" if not triple_kvbank_dir else "dir_not_found"
        print(f"[graphC] Triple KV injection disabled ({kv_injection_debug['reason']})", file=sys.stderr)

    # ---- 5. Load LLM ----
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

    # ---- 6. DRM Scoring → Relation Gating → KV Budget → Assembly ----
    #
    # Core fix for "injecting irrelevant triples" (15_架构调整.md §2,8):
    #   DRM → Relation Aggregation → Layer Gating → KVI Injection
    #
    assembled_kv = None
    if triple_kv_manifest and triple_kv_cache_dict:
        matched_names = [m["entity_name"] for m in gr.matched_entities]

        # 6a. DRM: Score each walk triple against the query
        scored_walk_triples: list[dict] = []
        for tid in gr.walk_triple_ids:
            triple = graph.triples.get(tid)
            if not triple:
                continue
            prov_text = str(triple.provenance.get("sentence_text") or "")
            drm_score = _score_triple_relevance(args.prompt, prov_text)
            scored_walk_triples.append({
                "triple_id": tid,
                "relation": triple.predicate,
                "subject": triple.subject,
                "object": triple.object,
                "drm_score": round(drm_score, 4),
                "provenance": prov_text[:100],
            })

        # Sort by DRM score (highest first) for debug display
        scored_walk_triples.sort(key=lambda x: x["drm_score"], reverse=True)
        kv_injection_debug["drm_scores"] = scored_walk_triples
        kv_injection_debug["drm_threshold"] = args.drm_threshold

        # 6b. Filter by DRM threshold
        above_threshold = [
            t for t in scored_walk_triples
            if t["drm_score"] >= args.drm_threshold
        ]
        kv_injection_debug["drm_passed"] = len(above_threshold)

        # 6c. Relation Gating: select top-k relation groups
        gated = _relation_gating(above_threshold, top_k_relations=args.top_k_relations)
        kv_injection_debug["gated_count"] = len(gated)

        # 6d. Budget: take top-N triples by DRM score
        gated.sort(key=lambda x: x["drm_score"], reverse=True)
        budget_selected = gated[:args.max_kv_triples]
        kv_triple_ids = [t["triple_id"] for t in budget_selected]
        kv_injection_debug["budget_selected"] = len(kv_triple_ids)
        kv_injection_debug["selected_triples"] = [
            {"triple_id": t["triple_id"], "relation": t["relation"],
             "subject": t["subject"], "object": t["object"],
             "drm_score": t["drm_score"]}
            for t in budget_selected
        ]

        print(
            f"[graphC] DRM: {len(scored_walk_triples)} walk triples → "
            f"{len(above_threshold)} above threshold ({args.drm_threshold}) → "
            f"{len(gated)} after gating (top-{args.top_k_relations} rels) → "
            f"{len(kv_triple_ids)} selected for KV (budget={args.max_kv_triples})",
            file=sys.stderr,
        )

        # 6e. Assemble KV — always pass a list (never None)
        try:
            assembled_kv, selected_item_ids = assemble_kv_for_entities(
                matched_entity_names=matched_names,
                walk_triple_ids=kv_triple_ids,  # DRM-filtered list, ALWAYS a list
                manifest=triple_kv_manifest,
                kv_cache_dict=triple_kv_cache_dict,
                device=device,
                dtype=dtype,
            )
            if assembled_kv is not None:
                # Build debug from ACTUALLY selected items only
                active_items = []
                for iid in selected_item_ids:
                    meta = triple_kv_manifest.items.get(iid)
                    if meta:
                        active_items.append({
                            "item_id": iid,
                            "type": meta.item_type,
                            "text": meta.text,
                            "relation": meta.relation,
                            "object": meta.object_name,
                            "layers": f"{meta.layer_start}-{meta.layer_end}",
                            "tokens": meta.token_count,
                        })
                kv_injection_debug["active_items"] = active_items
                kv_injection_debug["total_kv_tokens"] = sum(
                    it.get("tokens", 0) for it in active_items
                )
                kv_seq_len = assembled_kv[0][0].shape[2] if assembled_kv[0] is not None else 0
                kv_injection_debug["assembled_seq_len"] = int(kv_seq_len)
                print(
                    f"[graphC] KV assembled: {len(active_items)} items, "
                    f"seq_len={kv_seq_len}",
                    file=sys.stderr,
                )
            else:
                kv_injection_debug["assembled"] = False
                kv_injection_debug["reason"] = "no_matching_items_after_drm"
        except Exception as e:
            print(f"[graphC] WARNING: KV assembly failed: {e}", file=sys.stderr)
            kv_injection_debug["assembly_error"] = str(e)
            assembled_kv = None

    # ---- 7. Build prompt (rank evidence by DRM score) ----
    # Rank evidence sentences by DRM relevance to the query (highest first)
    evidence_with_scores = []
    for et in evidence_texts:
        score = _score_triple_relevance(args.prompt, et)
        evidence_with_scores.append((et, score))
    evidence_with_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_evidence_texts = [et for et, _s in evidence_with_scores]
    evidence_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ranked_evidence_texts))

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

    # ---- 8. Generate (graph + evidence + KV injection) ----
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

    generate_kwargs: dict = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "no_repeat_ngram_size": 6,
        "repetition_penalty": 1.15,
    }

    # Inject triple KV as past_key_values if available
    if assembled_kv is not None:
        generate_kwargs["past_key_values"] = assembled_kv
        print("[graphC] Generating with triple KV injection", file=sys.stderr)
    else:
        print("[graphC] Generating without KV injection", file=sys.stderr)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    print(f"[graphC] Generated answer ({len(answer)} chars)", file=sys.stderr)

    # ---- 9. Generate baseline (no evidence, no KV) ----
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

    # ---- 10. Grounding filter ----
    # Include triple KV texts as additional grounding sources
    kv_grounding_texts: list[str] = []
    if kv_injection_debug.get("active_items"):
        for it in kv_injection_debug["active_items"]:
            if it.get("text"):
                kv_grounding_texts.append(it["text"])

    grounding = _simple_grounding(
        answer,
        evidence_texts,
        entity_context=entity_context,
        kv_texts=kv_grounding_texts,
    )

    # Assemble grounded output (only include grounded sentences)
    grounded_sents = [d["sentence"] for d in grounding["details"] if d["grounded"]]
    grounded_text = "。".join(grounded_sents) + "。" if grounded_sents else answer

    # ---- 11. Output ----
    result_json = {
        "diagnosis_result": grounded_text,
        "diagnosis_result_raw": answer,
        "base_llm_result": base_answer,
        "graph_debug": gr.debug,
        "evidence_texts": evidence_texts,
        "entity_context": entity_context,
        "intent": intent,
        "grounding_report": grounding,
        "kv_injection_debug": kv_injection_debug,
    }
    print(json.dumps(result_json, ensure_ascii=False))


if __name__ == "__main__":
    main()
