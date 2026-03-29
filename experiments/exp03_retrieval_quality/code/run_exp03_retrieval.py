#!/usr/bin/env python3
"""
Experiment 3 — Retrieval quality (HotpotQA).
See ../README.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_rgi_path = _REPO / "scripts" / "run_graph_inference.py"
_spec = importlib.util.spec_from_file_location("graph_inf", _rgi_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
_text_search = _mod._text_search
_classify_intent = _mod._classify_intent


def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def _hit_rank(gold_sents: Sequence[str], ranked_texts: Sequence[str]) -> int:
    if not gold_sents or not ranked_texts:
        return 0
    gold_n = [_norm(g) for g in gold_sents if str(g).strip()]
    if not gold_n:
        return 0
    for i, rt in enumerate(ranked_texts, start=1):
        rn = _norm(rt)
        if not rn:
            continue
        for g in gold_n:
            if len(g) >= 8 and g in rn:
                return i
            if len(g) < 8 and g in rn:
                return i
    return 0


def _rr(rank: int) -> float:
    return 1.0 / float(rank) if rank > 0 else 0.0


def _recall_at(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


def _load_graph_evidence_order(
    *,
    question: str,
    graph_index: Path,
    sentences_jsonl: Path,
    max_evidence: int,
    max_hops: int,
) -> List[str]:
    sys.path.insert(0, str(_REPO))
    from src.graph.schema import KnowledgeGraphIndex
    from src.graph.graph_retriever import GraphRetriever

    graph = KnowledgeGraphIndex.load(graph_index)
    intent = _classify_intent(question)
    retriever = GraphRetriever(graph=graph, max_hops=max_hops, max_evidence=max_evidence)
    gr = retriever.retrieve(question, intent=intent)
    graph_evidence_texts = [e["text"] for e in gr.evidence_sentences]
    text_search_results = _text_search(question, sentences_jsonl, max_results=max_evidence)
    seen: set[str] = set()
    merged: List[str] = []
    for et in graph_evidence_texts:
        n = et.strip()
        if n and n not in seen:
            seen.add(n)
            merged.append(n)
    for ts in text_search_results:
        n = ts["text"].strip()
        if n and n not in seen:
            seen.add(n)
            merged.append(n)
    return merged


def _ann_ranked_dense(
    question: str,
    texts: List[str],
    encoder_model: str,
    device: str,
    top_n: int,
) -> List[str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_model, device=device)
    qv = model.encode([question], convert_to_numpy=True, show_progress_bar=False)
    mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = (qv @ mat.T)[0]
    order = np.argsort(-sims)[:top_n]
    return [texts[int(i)] for i in order]


def _ann_ranked_lexical(question: str, texts: List[str], top_n: int) -> List[str]:
    """Fallback when sentence-transformers is not installed: token Jaccard-like score."""
    q = set(_norm(question).split())
    if not q:
        return texts[:top_n]
    scored: List[tuple[float, int]] = []
    for i, t in enumerate(texts):
        ts = set(_norm(t).split())
        if not ts:
            continue
        inter = len(q & ts)
        union = len(q | ts) or 1
        scored.append((inter / union, i))
    scored.sort(key=lambda x: -x[0])
    return [texts[i] for _, i in scored[:top_n]]


def _ann_ranked(
    question: str,
    sentences_jsonl: Path,
    encoder_model: str,
    device: str,
    top_n: int,
) -> List[str]:
    texts: List[str] = []
    with sentences_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = str(rec.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        return []
    try:
        return _ann_ranked_dense(question, texts, encoder_model, device, top_n)
    except Exception:
        return _ann_ranked_lexical(question, texts, top_n)


def main() -> None:
    p = argparse.ArgumentParser(description="Exp3: ANN vs Graph retrieval metrics on Hotpot")
    p.add_argument("--dataset_jsonl", required=True)
    p.add_argument("--graph_index", required=True)
    p.add_argument("--sentences_jsonl", required=True)
    p.add_argument("--domain_encoder_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_evidence", type=int, default=10)
    p.add_argument("--max_hops", type=int, default=2)
    p.add_argument("--out_dir", default="", help="Write metrics.json + metrics.md")
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    with Path(args.dataset_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    ann_rr: List[float] = []
    graph_rr: List[float] = []
    ann_r5: List[float] = []
    ann_r10: List[float] = []
    graph_r5: List[float] = []
    graph_r10: List[float] = []
    skipped = 0

    for rec in rows:
        gold = rec.get("gold_supporting_sentences") or []
        if not gold or not isinstance(gold, list):
            skipped += 1
            continue
        q = str(rec.get("question") or "").strip()
        if not q:
            skipped += 1
            continue

        try:
            ann_list = _ann_ranked(
                q,
                Path(args.sentences_jsonl),
                str(args.domain_encoder_model),
                str(args.device),
                top_n=20,
            )
            gr_list = _load_graph_evidence_order(
                question=q,
                graph_index=Path(args.graph_index),
                sentences_jsonl=Path(args.sentences_jsonl),
                max_evidence=int(args.max_evidence),
                max_hops=int(args.max_hops),
            )
        except Exception as e:
            print(f"[warn] skip id={rec.get('id')}: {e}", file=sys.stderr)
            skipped += 1
            continue

        ra = _hit_rank(gold, ann_list)
        rg = _hit_rank(gold, gr_list)
        ann_rr.append(_rr(ra))
        graph_rr.append(_rr(rg))
        ann_r5.append(_recall_at(ra, 5))
        ann_r10.append(_recall_at(ra, 10))
        graph_r5.append(_recall_at(rg, 5))
        graph_r10.append(_recall_at(rg, 10))

    n = len(ann_rr)
    out: Dict[str, Any] = {
        "experiment": "exp03_retrieval_quality",
        "dataset_jsonl": str(args.dataset_jsonl),
        "n_evaluated": n,
        "n_skipped_no_gold": skipped,
        "ANN": {
            "Recall@5": round(100.0 * float(np.mean(ann_r5)) if n else 0.0, 2),
            "Recall@10": round(100.0 * float(np.mean(ann_r10)) if n else 0.0, 2),
            "MRR": round(float(np.mean(ann_rr)) if n else 0.0, 4),
        },
        "Graph": {
            "Recall@5": round(100.0 * float(np.mean(graph_r5)) if n else 0.0, 2),
            "Recall@10": round(100.0 * float(np.mean(graph_r10)) if n else 0.0, 2),
            "MRR": round(float(np.mean(graph_rr)) if n else 0.0, 4),
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out_dir).strip():
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        md = [
            "## Experiment 3 — Retrieval Quality (HotpotQA)\n\n",
            f"- Evaluated: **{n}** (skipped no gold: {skipped})\n\n",
            "| Retrieval | Recall@5 | Recall@10 | MRR |\n",
            "|---|---:|---:|---:|\n",
            f"| ANN | {out['ANN']['Recall@5']} | {out['ANN']['Recall@10']} | {out['ANN']['MRR']} |\n",
            f"| Graph | {out['Graph']['Recall@5']} | {out['Graph']['Recall@10']} | {out['Graph']['MRR']} |\n",
        ]
        (od / "metrics.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
