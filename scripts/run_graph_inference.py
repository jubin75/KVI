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
from typing import List


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


_DB_ID_RE = re.compile(r"DB\d+", re.IGNORECASE)


def _drugbank_id_grounding_key(pred: str, support: str) -> tuple:
    """Lexicographic sort: higher overlap of DB ids with support wins; then shorter pred."""
    sup = set(_DB_ID_RE.findall(support or ""))
    if not sup:
        return (0, 0)
    ids = _DB_ID_RE.findall(pred or "")
    if not ids:
        return (-1, -min(len(pred or ""), 1_000_000))
    sup_u = {s.upper() for s in sup}
    overlap = sum(1 for i in ids if i.upper() in sup_u)
    return (overlap, -min(len(pred), 1_000_000))


# ---------------------------------------------------------------------------
# Simple grounding filter (token-overlap based)
# ---------------------------------------------------------------------------

def _tokenize_chinese(text: str) -> set[str]:
    """
    Bigram + word tokenization for Chinese + English.

    NOTE: For ID-style QA (e.g. MedHop DrugBank IDs like DB04844),
    we must treat alphanumeric IDs as tokens; otherwise DRM/text_search/grounding
    will see near-zero overlap and systematically drop correct answers.
    """
    tokens: set[str] = set()
    # Chinese bigrams
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])
    # English words
    for w in re.findall(r"[a-zA-Z]{2,}", text.lower()):
        tokens.add(w)
    # Alphanumeric IDs with digits (keep short-ish to avoid flooding)
    # Examples: db04844, p13569, q9h3n8
    for w in re.findall(r"\b[a-zA-Z]{1,6}\d{2,8}\b", text.lower()):
        tokens.add(w)
    # DrugBank IDs (explicit)
    for w in re.findall(r"\bdb\d+\b", text.lower()):
        tokens.add(w)
    # Pure numbers (years, counts) — helps numeric-only answers in Hotpot/NQ.
    for w in re.findall(r"\b\d{2,8}\b", text):
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
# Text search fallback (hybrid retrieval)
# ---------------------------------------------------------------------------

def _text_search(
    query: str,
    sentences_jsonl: Path,
    max_results: int = 10,
    min_score: float = 0.05,
) -> list[dict]:
    """
    Keyword-based text search over raw evidence sentences (sentences.jsonl).

    Complements graph retrieval: catches evidence that was never extracted
    as triples (URLs, references, opinions, etc.).

    Returns a list of dicts:
        [{"text": "...", "block_id": "...", "score": 0.25}, ...]
    sorted by score descending.
    """
    if not sentences_jsonl.exists():
        return []
    q_tokens = _tokenize_chinese(query)
    if not q_tokens:
        return []
    results: list[dict] = []
    try:
        for line in sentences_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = str(rec.get("text") or "").strip()
            if not text:
                continue
            t_tokens = _tokenize_chinese(text)
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens) / len(q_tokens)
            if overlap >= min_score:
                results.append({
                    "text": text,
                    "block_id": str(rec.get("block_id") or ""),
                    "score": round(overlap, 4),
                })
    except Exception:
        return []
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# URL / verbatim detection
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r'https?://\S+', re.IGNORECASE)


def _has_url(text: str) -> bool:
    """Return True if *text* contains an HTTP(S) URL."""
    return bool(_URL_PATTERN.search(text))


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
        is_grounded = overlap >= 0.10
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
    # KVI is OFF by default — pure RAG mode.  Use --enable_kvi to turn it on.
    p.add_argument("--enable_kvi", action="store_true",
                   help="Enable triple KV injection (default: OFF, pure RAG mode)")
    # DRM + Relation Gating parameters (only used when --enable_kvi is set)
    p.add_argument("--max_kv_triples", type=int, default=3,
                   help="Max triple KVs to inject after DRM scoring (budget control)")
    p.add_argument("--drm_threshold", type=float, default=0.05,
                   help="Min DRM score for a triple to be considered for KV injection")
    p.add_argument("--top_k_relations", type=int, default=2,
                   help="Max relation groups to keep after Relation Gating")
    # Hybrid retrieval: text search fallback over raw evidence sentences
    p.add_argument("--sentences_jsonl", default="",
                   help="Path to sentences.tagged.jsonl (or sentences.jsonl) for text search fallback")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_evidence", type=int, default=10)
    p.add_argument("--max_hops", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    # Open-domain QA (Hotpot/NQ): avoid medical-domain Chinese prompts that hurt English QA + KVI.
    p.add_argument(
        "--openqa_mode",
        action="store_true",
        help="Use English open-QA system/instructions (Hotpot/NQ). Default is legacy medical Chinese prompts.",
    )
    # When KVI injects KV, optionally drop long evidence from the *prompt* to test dual-channel (KV structure vs prompt text).
    p.add_argument(
        "--kvi_minimal_prompt",
        action="store_true",
        help="If --enable_kvi and KV assembled: user prompt = question + short instruction only (no evidence block). "
        "Tests whether prompt evidence + KV duplicates or KV deflects attention.",
    )
    p.add_argument(
        "--kvi_reconcile_no_kv_decode",
        action="store_true",
        help="If --enable_kvi and KV assembled: run a second decode on the same prompt without past_key_values, "
        "then pick KV vs no-KV output by which cites more DrugBank IDs present in entity+full evidence block.",
    )
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

    graph_evidence_texts = [e["text"] for e in gr.evidence_sentences]
    entity_context = gr.entity_context
    print(
        f"[graphC] entities matched={len(gr.matched_entities)} "
        f"graph_evidence={len(graph_evidence_texts)}",
        file=sys.stderr,
    )

    # ---- 3b. Text search fallback (hybrid retrieval) ----
    # Always run text search alongside graph retrieval, then merge.
    # Graph evidence is primary; text search fills coverage gaps
    # (e.g., URLs, references, sentences not extracted as triples).
    text_search_results: list[dict] = []
    sentences_jsonl_path = Path(args.sentences_jsonl) if args.sentences_jsonl else None
    if sentences_jsonl_path and sentences_jsonl_path.exists():
        text_search_results = _text_search(
            args.prompt, sentences_jsonl_path,
            max_results=args.max_evidence,
        )
        print(
            f"[graphC] text_search: {len(text_search_results)} hits from {sentences_jsonl_path.name}",
            file=sys.stderr,
        )
    else:
        print("[graphC] text_search: skipped (no --sentences_jsonl)", file=sys.stderr)

    # Merge: graph evidence first, text search supplements (dedup by text)
    seen_texts: set[str] = set()
    evidence_texts: list[str] = []
    evidence_source: list[str] = []  # track provenance for debug
    for et in graph_evidence_texts:
        norm = et.strip()
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            evidence_texts.append(norm)
            evidence_source.append("graph")
    for ts in text_search_results:
        norm = ts["text"].strip()
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            evidence_texts.append(norm)
            evidence_source.append(f"text_search(score={ts['score']})")

    gr.debug["graph_evidence_count"] = len(graph_evidence_texts)
    gr.debug["text_search_count"] = len(text_search_results)
    gr.debug["text_search_added"] = sum(1 for s in evidence_source if s.startswith("text_search"))
    gr.debug["merged_evidence_count"] = len(evidence_texts)

    # ---- 3c. Separate verbatim evidence (URLs, etc.) from regular evidence ----
    # Evidence containing URLs is passed verbatim to the output — never through
    # the LLM — to avoid hallucinated/mangled URLs.
    verbatim_evidence: list[str] = []
    regular_evidence_texts: list[str] = []
    regular_evidence_source: list[str] = []
    for i, et in enumerate(evidence_texts):
        if _has_url(et):
            verbatim_evidence.append(et)
        else:
            regular_evidence_texts.append(et)
            regular_evidence_source.append(evidence_source[i])
    gr.debug["verbatim_evidence_count"] = len(verbatim_evidence)
    gr.debug["regular_evidence_count"] = len(regular_evidence_texts)

    print(
        f"[graphC] merged evidence: {len(evidence_texts)} "
        f"(graph={len(graph_evidence_texts)}, "
        f"text_search_added={gr.debug['text_search_added']}, "
        f"verbatim={len(verbatim_evidence)})",
        file=sys.stderr,
    )

    if not evidence_texts:
        print(json.dumps({
            "diagnosis_result": "现有知识图谱和文本检索均未找到相关证据。",
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

    # ---- 4. Load Triple KV Bank (only if --enable_kvi) ----
    kv_injection_debug: dict = {"enabled": False}
    triple_kv_manifest = None
    triple_kv_cache_dict = None
    triple_kvbank_dir = Path(args.triple_kvbank_dir) if args.triple_kvbank_dir else None

    if not args.enable_kvi:
        kv_injection_debug["reason"] = "kvi_disabled_by_default"
        print("[graphC] KVI disabled (pure RAG mode). Use --enable_kvi to turn on.", file=sys.stderr)
    elif triple_kvbank_dir and triple_kvbank_dir.exists():
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
    model_kwargs.setdefault("attn_implementation", "eager")
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

    # ---- 7. Build prompt (rank evidence by DRM score, relaxed RAG) ----
    # Only *regular* evidence (no URLs) goes into the LLM prompt.
    # URL-containing evidence is appended verbatim after grounding (step 10).
    evidence_with_scores = []
    for et in regular_evidence_texts:
        score = _score_triple_relevance(args.prompt, et)
        evidence_with_scores.append((et, score))
    evidence_with_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_evidence_texts = [et for et, _s in evidence_with_scores]
    evidence_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ranked_evidence_texts))

    use_openqa = bool(getattr(args, "openqa_mode", False))
    use_kvi_minimal = (
        bool(getattr(args, "kvi_minimal_prompt", False))
        and bool(args.enable_kvi)
        and assembled_kv is not None
    )

    if use_openqa:
        prompt_parts: list[str] = []
        # Keep entity context whenever present (including KVI minimal). Stripping
        # entity+evidence while KV is only in past_key_values caused degenerate
        # outputs on MedHop official (kvituned sweep).
        if entity_context:
            prompt_parts.append(f"Entity context:\n{entity_context}")
        if not use_kvi_minimal:
            prompt_parts.append(f"Evidence:\n{evidence_block}")
        elif ranked_evidence_texts:
            k_tail = min(3, len(ranked_evidence_texts))
            thin = "\n".join(
                f"{i+1}. {t}" for i, t in enumerate(ranked_evidence_texts[:k_tail])
            )
            prompt_parts.append(f"Evidence (brief):\n{thin}")
        prompt_parts.append(f"Question: {args.prompt}")
        prompt_parts.append(
            "Answer concisely with the final answer only when possible (entity, yes/no, or short phrase)."
        )
        full_prompt = "\n\n".join(prompt_parts)
        if use_kvi_minimal:
            system_msg = (
                "You are a careful assistant for open-domain question answering. "
                "Ground answers in the brief evidence when helpful; answer concisely."
            )
        else:
            system_msg = (
                "You are a careful assistant for open-domain question answering. "
                "Ground answers in the provided evidence when it is present."
            )
    elif use_kvi_minimal:
        prompt_parts = []
        if entity_context:
            prompt_parts.append(entity_context)
        prompt_parts.append(f"### 问题\n{args.prompt}")
        prompt_parts.append(
            "### 要求\n"
            "已注入结构化知识（KV）。请直接作答，避免冗长推理。"
            "使用中文简洁作答。"
        )
        full_prompt = "\n\n".join(prompt_parts)
        system_msg = "你是一个医学专业助手。根据提供的参考资料回答问题，必要时可结合医学常识。"
    else:
        prompt_parts = []
        if entity_context:
            prompt_parts.append(entity_context)
        prompt_parts.append(f"### 参考证据\n{evidence_block}")
        prompt_parts.append(f"### 问题\n{args.prompt}")
        prompt_parts.append(
            "### 要求\n"
            "请根据上述实体背景和参考证据回答问题。"
            "以参考证据为主，必要时可结合医学常识补充。"
            "使用中文简洁作答。"
        )
        full_prompt = "\n\n".join(prompt_parts)
        system_msg = "你是一个医学专业助手。根据提供的参考资料回答问题，必要时可结合医学常识。"

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
        {"role": "system", "content": system_msg},
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

    # Inject triple KV as past_key_values if available.
    # Transformers 5.x requires Cache object (legacy tuple is rejected).
    if assembled_kv is not None:
        pkv_for_generate = assembled_kv
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore

            if not isinstance(pkv_for_generate, DynamicCache):
                # Manual legacy(tuple) -> DynamicCache conversion.
                if isinstance(pkv_for_generate, (tuple, list)):
                    legacy_layers = list(pkv_for_generate)
                    sample_pair = None
                    for lp in legacy_layers:
                        if isinstance(lp, (tuple, list)) and len(lp) >= 2 and lp[0] is not None and lp[1] is not None:
                            sample_pair = (lp[0], lp[1])
                            break
                    if sample_pair is not None:
                        sk, sv = sample_pair
                        dc = DynamicCache(config=model.config)
                        for li, lp in enumerate(legacy_layers):
                            if isinstance(lp, (tuple, list)) and len(lp) >= 2 and lp[0] is not None and lp[1] is not None:
                                lk, lv = lp[0], lp[1]
                            else:
                                lk = torch.zeros_like(sk)
                                lv = torch.zeros_like(sv)
                            dc.update(lk, lv, li)
                        pkv_for_generate = dc
        except Exception:
            pass
        generate_kwargs["past_key_values"] = pkv_for_generate
        print("[graphC] Generating with triple KV injection", file=sys.stderr)
    else:
        print("[graphC] Generating without KV injection", file=sys.stderr)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    print(f"[graphC] Generated answer ({len(answer)} chars)", file=sys.stderr)

    kvi_reconcile_debug = None
    if (
        bool(args.enable_kvi)
        and assembled_kv is not None
        and bool(getattr(args, "kvi_reconcile_no_kv_decode", False))
    ):
        support_for_score = "\n".join(
            str(x) for x in (entity_context or "", evidence_block or "") if str(x).strip()
        )
        gen_plain = {k: v for k, v in generate_kwargs.items() if k != "past_key_values"}
        with torch.no_grad():
            out_plain = model.generate(**inputs, **gen_plain)
        plain_new = out_plain[0][inputs["input_ids"].shape[1] :]
        answer_plain = tokenizer.decode(plain_new, skip_special_tokens=True).strip()
        k_kv = _drugbank_id_grounding_key(answer, support_for_score)
        k_pl = _drugbank_id_grounding_key(answer_plain, support_for_score)
        # Prefer the no-KV decode when it is at least as evidence-grounded (ties → GraphRAG-like path).
        picked = "kv"
        if k_pl >= k_kv:
            answer = answer_plain
            picked = "no_kv"
        kvi_reconcile_debug = {
            "picked": picked,
            "key_kv": list(k_kv),
            "key_no_kv": list(k_pl),
            "preview_plain": answer_plain[:240],
        }
        print(
            f"[graphC] KVI reconcile: picked={picked} key_kv={k_kv} key_plain={k_pl}",
            file=sys.stderr,
        )

    # ---- 9. Generate baseline (no evidence, no KV) ----
    base_system = (
        "You are a helpful assistant."
        if use_openqa
        else "你是一个医学专业助手。"
    )
    base_messages = [
        {"role": "system", "content": base_system},
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

    # ---- 10. Grounding check (filter: keep only grounded sentences) ----
    # Grounding catches hallucinated content (e.g., "气促或呼吸困难" that
    # doesn't appear in any evidence or entity context).  With the improved
    # topic-scoped retrieval, the model should receive enough evidence to
    # generate well-grounded answers.
    kv_grounding_texts: list[str] = []
    if kv_injection_debug.get("active_items"):
        for it in kv_injection_debug["active_items"]:
            if it.get("text"):
                kv_grounding_texts.append(it["text"])

    grounding = _simple_grounding(
        answer,
        evidence_texts,   # grounding checks against ALL evidence (incl. URL-based)
        entity_context=entity_context,
        kv_texts=kv_grounding_texts,
    )

    # Build grounded text: keep only sentences that overlap with evidence
    grounded_sentences = [
        d["sentence"] for d in grounding["details"] if d["grounded"]
    ]
    grounded_text = "\n".join(grounded_sentences) if grounded_sentences else answer

    # ---- 10b. Append verbatim evidence (URLs, references) ----
    # Only when the user is actually asking about references / literature.
    _REF_KEYWORDS = {"参考文献", "文献", "综述", "链接", "reference", "literature", "推荐文献", "论文"}
    query_wants_refs = any(kw in args.prompt for kw in _REF_KEYWORDS)
    if verbatim_evidence and query_wants_refs:
        ref_lines = "\n".join(f"- {v}" for v in verbatim_evidence)
        grounded_text += f"\n\n### 参考文献\n{ref_lines}"

    # ---- 11. Output ----
    result_json = {
        "diagnosis_result": grounded_text,
        "diagnosis_result_raw": answer,
        "base_llm_result": base_answer,
        "graph_debug": gr.debug,
        "evidence_texts": evidence_texts,
        "evidence_source": evidence_source,
        "entity_context": entity_context,
        "intent": intent,
        "grounding_report": grounding,
        "kv_injection_debug": kv_injection_debug,
        "verbatim_evidence": verbatim_evidence,
        "openqa_mode": bool(getattr(args, "openqa_mode", False)),
        "kvi_minimal_prompt": bool(
            getattr(args, "kvi_minimal_prompt", False)
            and bool(args.enable_kvi)
            and assembled_kv is not None
        ),
        "kvi_reconcile_debug": kvi_reconcile_debug,
    }
    print(json.dumps(result_json, ensure_ascii=False))


if __name__ == "__main__":
    main()
