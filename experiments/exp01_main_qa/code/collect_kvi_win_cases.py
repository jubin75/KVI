#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _index_supporting_sentences(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    for r in _read_jsonl(path):
        rid = str(r.get("id") or "").strip()
        if not rid:
            continue
        vals = r.get("gold_supporting_sentences")
        if isinstance(vals, list):
            out[rid] = [str(x).strip() for x in vals if str(x).strip()]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Collect case studies where KVI succeeds but GraphRAG/RAG fail.")
    p.add_argument("--predictions_jsonl", required=True, help="run_exp01 predictions.jsonl")
    p.add_argument("--dataset_jsonl", default="", help="optional dataset with gold_supporting_sentences")
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--max_cases", type=int, default=30)
    p.add_argument("--require_rag_fail", action="store_true", help="if set, require rag em == 0 as well")
    args = p.parse_args()

    preds = _read_jsonl(Path(args.predictions_jsonl))
    support_map = _index_supporting_sentences(Path(args.dataset_jsonl)) if str(args.dataset_jsonl).strip() else {}

    cases: List[Dict[str, Any]] = []
    for r in preds:
        em = r.get("em") or {}
        if not isinstance(em, dict):
            continue
        kvi_ok = int(em.get("kvi") or 0) == 1
        graph_fail = int(em.get("graphrag") or 0) == 0
        rag_fail = int(em.get("rag") or 0) == 0
        if not kvi_ok or not graph_fail:
            continue
        if args.require_rag_fail and not rag_fail:
            continue

        rid = str(r.get("id") or "")
        pred = r.get("predictions") or {}
        if not isinstance(pred, dict):
            pred = {}
        q = str(r.get("question") or "").strip()
        answers = r.get("gold_answers") if isinstance(r.get("gold_answers"), list) else []
        gold = [str(x).strip() for x in answers if str(x).strip()]

        def _trim(x: str, n: int = 500) -> str:
            s = str(x or "").strip()
            return s if len(s) <= n else s[:n] + " ..."

        cases.append(
            {
                "id": rid,
                "question": q,
                "gold_answers": gold,
                "pred_kvi": _trim(str(pred.get("kvi") or "")),
                "pred_graphrag": _trim(str(pred.get("graphrag") or "")),
                "pred_rag": _trim(str(pred.get("rag") or "")),
                "em": {
                    "kvi": int(em.get("kvi") or 0),
                    "graphrag": int(em.get("graphrag") or 0),
                    "rag": int(em.get("rag") or 0),
                },
                "gold_supporting_sentences": support_map.get(rid, []),
            }
        )
        if len(cases) >= int(args.max_cases):
            break

    out_json = {
        "n_cases": len(cases),
        "source_predictions": str(args.predictions_jsonl),
        "require_rag_fail": bool(args.require_rag_fail),
        "cases": cases,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# KVI win cases (KVI succeeds; GraphRAG fails)\n\n")
    md.append(f"- Source predictions: `{args.predictions_jsonl}`\n")
    md.append(f"- Cases collected: **{len(cases)}**\n")
    md.append(f"- Require RAG fail: **{bool(args.require_rag_fail)}**\n\n")
    for i, c in enumerate(cases, start=1):
        md.append(f"## Case {i} — {c['id']}\n\n")
        md.append(f"**Question**: {c['question']}\n\n")
        md.append(f"**Gold**: {', '.join(c['gold_answers']) if c['gold_answers'] else '(empty)'}\n\n")
        md.append(
            f"**EM**: KVI={c['em']['kvi']}, GraphRAG={c['em']['graphrag']}, RAG={c['em']['rag']}\n\n"
        )
        md.append(f"**KVI prediction**: {c['pred_kvi']}\n\n")
        md.append(f"**GraphRAG prediction**: {c['pred_graphrag']}\n\n")
        md.append(f"**RAG prediction**: {c['pred_rag']}\n\n")
        if c["gold_supporting_sentences"]:
            md.append("**Gold supporting sentences**:\n")
            for s in c["gold_supporting_sentences"][:4]:
                md.append(f"- {s}\n")
            md.append("\n")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print(json.dumps({"ok": True, "n_cases": len(cases), "out_md": args.out_md, "out_json": args.out_json}, ensure_ascii=False))


if __name__ == "__main__":
    main()

