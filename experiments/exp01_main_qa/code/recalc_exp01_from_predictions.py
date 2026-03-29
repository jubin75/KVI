#!/usr/bin/env python3
"""Recompute results.md / summary.json / results.csv from predictions.jsonl (no model calls)."""
from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import argparse
import csv
import json
import random
from typing import Any, Dict, List

from metrics import best_exact_match, best_f1, best_relaxed_em

METHODS_ORDER = ["llm", "rag", "graphrag", "kv_prefix", "kvi"]
METHODS_META = {
    "llm": {"label": "LLM", "retrieval": "none", "injection": "none"},
    "rag": {"label": "RAG", "retrieval": "ANN", "injection": "prompt"},
    "graphrag": {"label": "GraphRAG", "retrieval": "graph", "injection": "prompt"},
    "kv_prefix": {"label": "KV Prefix", "retrieval": "ANN", "injection": "KV"},
    "kvi": {"label": "KVI", "retrieval": "graph", "injection": "KV + prompt"},
}


def _bootstrap_ci_percent(samples01: List[int], *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    n = len(samples01)
    if n <= 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(max(10, int(n_boot))):
        s = sum(samples01[rng.randrange(0, n)] for _ in range(n))
        vals.append(100.0 * float(s) / float(n))
    vals.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(vals)) - 1)
    hi_idx = min(len(vals) - 1, int((1.0 - alpha / 2.0) * len(vals)) - 1)
    mean = 100.0 * float(sum(samples01)) / float(n)
    return {"mean": float(mean), "lo": float(vals[lo_idx]), "hi": float(vals[hi_idx])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="predictions.jsonl from run_exp01")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_name", default="")
    p.add_argument("--em_mode", choices=["strict", "relaxed"], default="relaxed")
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--random_seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    em_fn = best_exact_match if args.em_mode == "strict" else best_relaxed_em

    methods = list(METHODS_ORDER)
    method_correct = {m: 0 for m in methods}
    method_em_series: Dict[str, List[int]] = {m: [] for m in methods}
    method_f1_sum = {m: 0.0 for m in methods}
    n = 0

    with Path(args.predictions).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            n += 1
            gold = rec.get("gold_answers") or []
            preds = rec.get("predictions") or {}
            for m in methods:
                pred = str(preds.get(m) or "")
                em = int(em_fn(pred, gold))
                method_correct[m] += em
                method_em_series[m].append(em)
                method_f1_sum[m] += float(best_f1(pred, gold))

    em_percent = {m: (100.0 * method_correct[m] / n if n else 0.0) for m in methods}
    rows = []
    for m in methods:
        meta = METHODS_META[m]
        ci = _bootstrap_ci_percent(method_em_series[m], n_boot=int(args.bootstrap_samples), seed=int(args.random_seed))
        f1_mean = (method_f1_sum[m] / n) if n else 0.0
        rows.append(
            {
                "method_key": m,
                "method": meta["label"],
                "retrieval": meta["retrieval"],
                "injection": meta["injection"],
                "em": round(em_percent[m], 4),
                "em_ci95_lo": round(ci["lo"], 4),
                "em_ci95_hi": round(ci["hi"], 4),
                "f1_mean": round(f1_mean, 4),
                "correct": int(method_correct[m]),
                "total": int(n),
            }
        )

    summary: Dict[str, Any] = {
        "dataset": str(args.dataset_name or Path(args.predictions).stem),
        "n": n,
        "methods": rows,
        "statistics": {"em_mode": str(args.em_mode)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["Method", "Retrieval", "Injection", "EM", "CI95 Low", "CI95 High", "F1 Mean"])
        for r in rows:
            w.writerow(
                [
                    r["method"],
                    r["retrieval"],
                    r["injection"],
                    f"{r['em']:.1f}",
                    f"{r['em_ci95_lo']:.1f}",
                    f"{r['em_ci95_hi']:.1f}",
                    f"{r['f1_mean']:.4f}",
                ]
            )

    md: List[str] = []
    md.append("## Experiment 1 — Main QA Performance (single dataset)\n\n")
    md.append(f"- Dataset: {summary['dataset']}\n")
    md.append(f"- N: {n}\n")
    md.append(f"- EM mode: **{args.em_mode}**\n\n")
    md.append("| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |\n")
    md.append("|---|---|---|---:|---:|---:|\n")
    for m in METHODS_ORDER:
        r = next(x for x in rows if x["method_key"] == m)
        md.append(
            f"| {r['method']} | {r['retrieval']} | {r['injection']} | {r['em']:.1f} | "
            f"[{r['em_ci95_lo']:.1f}, {r['em_ci95_hi']:.1f}] | {r['f1_mean']:.3f} |\n"
        )
    (out_dir / "results.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"n": n, "em_mode": args.em_mode, "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
