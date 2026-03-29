#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_hotpot_answers(ex: Dict[str, Any]) -> List[str]:
    ans = str(ex.get("answer") or "").strip()
    if not ans:
        return []
    return [ans]


def _extract_hotpot_supporting_sentences(ex: Dict[str, Any]) -> List[str]:
    """Gold evidence sentences (for Exp3 retrieval Recall@k / MRR)."""
    sf = ex.get("supporting_facts") or {}
    titles = sf.get("title") or []
    sids = sf.get("sent_id") or []
    ctx = ex.get("context") or {}
    ctitles = ctx.get("title") or []
    sents = ctx.get("sentences") or []
    if not titles or not isinstance(ctitles, list) or not isinstance(sents, list):
        return []
    out: List[str] = []
    for tit, sid in zip(titles, sids):
        try:
            pi = ctitles.index(tit)
            para = sents[pi]
            if isinstance(sid, int) and 0 <= sid < len(para):
                t = str(para[sid] or "").strip()
                if t:
                    out.append(t)
        except (ValueError, IndexError, TypeError):
            continue
    return _dedup_keep_order(out)


def _extract_nq_answers(ex: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    # Some mirrors expose "answer" list directly.
    ans = ex.get("answer")
    if isinstance(ans, list):
        vals.extend([str(x) for x in ans if str(x).strip()])
    elif isinstance(ans, str) and ans.strip():
        vals.append(ans.strip())

    # Common `natural_questions` format: annotations[*].short_answers[*].text
    anns = ex.get("annotations")
    if isinstance(anns, list):
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            sa = ann.get("short_answers")
            if isinstance(sa, list):
                for x in sa:
                    if isinstance(x, dict):
                        t = str(x.get("text") or "").strip()
                        if t:
                            vals.append(t)
                    elif isinstance(x, str) and x.strip():
                        vals.append(x.strip())

            la = ann.get("long_answer")
            if isinstance(la, dict):
                t = str(la.get("text") or "").strip()
                if t:
                    vals.append(t)
    elif isinstance(anns, dict):
        # natural_questions streaming format often stores annotations as dict-of-lists
        sa_list = anns.get("short_answers")
        if isinstance(sa_list, list):
            for sa in sa_list:
                if not isinstance(sa, dict):
                    continue
                texts = sa.get("text")
                if isinstance(texts, list):
                    for t in texts:
                        s = str(t or "").strip()
                        if s:
                            vals.append(s)
        la_list = anns.get("long_answer")
        if isinstance(la_list, list):
            for la in la_list:
                if not isinstance(la, dict):
                    continue
                t = str(la.get("text") or "").strip()
                if t:
                    vals.append(t)

    return _dedup_keep_order(vals)


def _get_question(ex: Dict[str, Any]) -> str:
    q = ex.get("question")
    if isinstance(q, dict):
        t = str(q.get("text") or "").strip()
        return t
    return str(q or "").strip()


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Download + convert HotpotQA/NQ into Exp01 JSONL")
    p.add_argument("--out_dir", required=True, help="Output folder for converted JSONL")
    p.add_argument("--hotpot_config", default="distractor", help="hotpot_qa config: distractor/fullwiki")
    p.add_argument("--hotpot_split", default="validation", help="hotpot_qa split")
    p.add_argument("--nq_split", default="validation", help="natural_questions split")
    p.add_argument("--hotpot_max", type=int, default=0, help="Max converted Hotpot examples (0=all, streaming not recommended)")
    p.add_argument("--nq_max", type=int, default=0, help="Max converted NQ examples (0=all, streaming not recommended)")
    p.add_argument("--streaming", action="store_true", help="Use HF streaming mode to avoid full dataset materialization")
    p.add_argument(
        "--include_hotpot_supporting_sentences",
        action="store_true",
        help="Add gold_supporting_sentences to each Hotpot row (for Exp3 retrieval metrics).",
    )
    p.add_argument(
        "--hotpot_only",
        action="store_true",
        help="Only download/convert HotpotQA (skip NQ; writes empty nq_eval.jsonl).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] loading hotpot_qa config={args.hotpot_config} split={args.hotpot_split}")
    ds_hotpot = load_dataset("hotpot_qa", args.hotpot_config, split=args.hotpot_split, streaming=bool(args.streaming))
    hotpot_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds_hotpot):
        q = _get_question(ex)
        answers = _extract_hotpot_answers(ex)
        if not q or not answers:
            continue
        row: Dict[str, Any] = {
            "id": str(ex.get("id") or f"hotpot_{i}"),
            "question": q,
            "answer": answers[0],
            "answers": answers,
            "dataset": "HotpotQA",
        }
        if args.include_hotpot_supporting_sentences:
            row["gold_supporting_sentences"] = _extract_hotpot_supporting_sentences(ex)
        hotpot_rows.append(row)
        if args.hotpot_max and len(hotpot_rows) >= int(args.hotpot_max):
            break

    nq_rows: List[Dict[str, Any]] = []
    if not args.hotpot_only:
        print(f"[prepare] loading natural_questions split={args.nq_split}")
        ds_nq = load_dataset("natural_questions", split=args.nq_split, streaming=bool(args.streaming))
        for i, ex in enumerate(ds_nq):
            q = _get_question(ex)
            answers = _extract_nq_answers(ex)
            if not q or not answers:
                continue
            nq_rows.append(
                {
                    "id": str(ex.get("id") or ex.get("example_id") or f"nq_{i}"),
                    "question": q,
                    "answer": answers[0],
                    "answers": answers,
                    "dataset": "NQ",
                }
            )
            if args.nq_max and len(nq_rows) >= int(args.nq_max):
                break
    else:
        print("[prepare] skipping NQ (--hotpot_only)")

    hotpot_path = out_dir / "hotpot_eval.jsonl"
    nq_path = out_dir / "nq_eval.jsonl"
    _write_jsonl(hotpot_path, hotpot_rows)
    _write_jsonl(nq_path, nq_rows)

    manifest = {
        "hotpot_source": {"name": "hotpot_qa", "config": args.hotpot_config, "split": args.hotpot_split},
        "nq_source": {"name": "natural_questions", "split": args.nq_split},
        "counts": {"hotpot": len(hotpot_rows), "nq": len(nq_rows)},
        "outputs": {"hotpot_jsonl": str(hotpot_path), "nq_jsonl": str(nq_path)},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

