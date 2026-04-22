#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    return rows


def _text_flags(text: str) -> Dict[str, bool]:
    t = str(text or "")
    return {
        "has_url": bool(re.search(r"https?://", t, flags=re.IGNORECASE)),
        "has_image_md": bool(re.search(r"!\[[^\]]*\]\([^)]*\)", t)),
        "has_many_exclaim": t.count("!") >= 3,
        "is_too_short": len(t.strip()) < 16,
        "has_hedging": bool(
            re.search(
                r"\b(?:generally|typically|usually|it is recommended|likely|possibly|in general)\b",
                t,
                flags=re.IGNORECASE,
            )
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bucket TruthfulQA KVI underperforming examples by text-shape features.")
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.predictions_jsonl))
    if not rows:
        raise SystemExit("No rows in predictions jsonl.")

    losers: List[Dict[str, Any]] = []
    for r in rows:
        mc2 = r.get("truthfulqa_mc2_proxy") or {}
        try:
            d = float(mc2.get("kvi", 0.0)) - float(mc2.get("graphrag", 0.0))
        except Exception:
            d = 0.0
        if d < 0:
            pred_kvi = str((r.get("predictions") or {}).get("kvi") or "")
            pred_gr = str((r.get("predictions") or {}).get("graphrag") or "")
            losers.append(
                {
                    "id": r.get("id"),
                    "question": r.get("question"),
                    "delta_kvi_minus_graphrag": d,
                    "kvi_mc2": float(mc2.get("kvi", 0.0)),
                    "graphrag_mc2": float(mc2.get("graphrag", 0.0)),
                    "kvi_text_flags": _text_flags(pred_kvi),
                    "kvi_pred_preview": pred_kvi[:240],
                    "graphrag_pred_preview": pred_gr[:240],
                }
            )

    feat_counter: Counter[str] = Counter()
    for x in losers:
        for k, v in (x.get("kvi_text_flags") or {}).items():
            if v:
                feat_counter[k] += 1

    n = len(rows)
    n_loser = len(losers)
    out_obj = {
        "n_total": n,
        "n_kvi_under_graphrag": n_loser,
        "underperform_ratio": (float(n_loser) / float(n)) if n > 0 else 0.0,
        "kvi_text_feature_hits_in_underperformers": dict(feat_counter),
        "worst_cases": sorted(losers, key=lambda x: x["delta_kvi_minus_graphrag"])[:10],
    }
    Path(args.out_json).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# TruthfulQA KVI Error Buckets")
    lines.append("")
    lines.append(f"- Total examples: {n}")
    lines.append(f"- KVI under GraphRAG (MC2): {n_loser} ({out_obj['underperform_ratio']:.3f})")
    lines.append("")
    lines.append("## Feature Hits In Underperformers")
    lines.append("")
    for k, v in sorted(feat_counter.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {k}: {v}/{n_loser}")
    lines.append("")
    lines.append("## Worst 10 (KVI - GraphRAG MC2)")
    lines.append("")
    lines.append("| id | delta | kvi_mc2 | graphrag_mc2 | flags |")
    lines.append("|---|---:|---:|---:|---|")
    for x in out_obj["worst_cases"]:
        flags = [k for k, v in (x.get("kvi_text_flags") or {}).items() if v]
        lines.append(
            f"| {x.get('id')} | {x.get('delta_kvi_minus_graphrag', 0.0):.4f} | "
            f"{x.get('kvi_mc2', 0.0):.4f} | {x.get('graphrag_mc2', 0.0):.4f} | {', '.join(flags)} |"
        )
    Path(args.out_md).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
