#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", _norm(s))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _score(sent: str, gold_answers: List[str]) -> float:
    st = set(_tok(sent))
    if not st:
        return 0.0
    best = 0.0
    for g in gold_answers:
        gt = set(_tok(g))
        if not gt:
            continue
        ov = len(st & gt) / max(1, len(gt))
        if ov > best:
            best = ov
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Build weak oracle evidence from answer overlap")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--sentences_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--min_score", type=float, default=0.18)
    args = ap.parse_args()

    ds = _read_jsonl(Path(args.dataset_jsonl))
    sents = _read_jsonl(Path(args.sentences_jsonl))
    sent_texts = [str(x.get("text") or "").strip() for x in sents if str(x.get("text") or "").strip()]

    if args.limit and args.limit > 0:
        ds = ds[: int(args.limit)]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []

    for i, ex in enumerate(ds, start=1):
        qid = str(ex.get("id") or ex.get("qid") or f"idx_{i}")
        answers = ex.get("answers")
        if not isinstance(answers, list):
            a = ex.get("answer")
            answers = a if isinstance(a, list) else [a]
        gold = [str(a).strip() for a in (answers or []) if str(a).strip()]
        scored: List[Tuple[float, str]] = []
        for st in sent_texts:
            sc = _score(st, gold)
            if sc >= float(args.min_score):
                scored.append((sc, st))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [t for _s, t in scored[: max(1, int(args.top_k))]]
        rows.append(
            json.dumps(
                {
                    "id": qid,
                    "question": str(ex.get("question") or ""),
                    "gold_evidence_texts": picked,
                    "oracle_type": "weak_answer_overlap",
                },
                ensure_ascii=False,
            )
        )

    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
