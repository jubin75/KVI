#!/usr/bin/env python3
"""Per-example MC2 attribution: KVI vs GraphRAG on TruthfulQA predictions.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _mc2(r: Dict[str, Any], m: str) -> float:
    mc2 = r.get("truthfulqa_mc2_proxy") or {}
    try:
        return float(mc2.get(m, 0.0))
    except Exception:
        return 0.0


def _pred(r: Dict[str, Any], m: str) -> str:
    p = r.get("predictions") or {}
    return str(p.get(m) or "")


def _flags(text: str) -> Dict[str, Any]:
    t = str(text or "")
    ex = t.count("!")
    return {
        "len": len(t),
        "exclaim_count": ex,
        "high_exclaim": ex >= 5,
        "has_url": bool(re.search(r"https?://", t, flags=re.I)),
        "has_image_md": bool(re.search(r"!\[[^\]]*\]\([^)]*\)", t)),
        "has_many_exclaim": ex >= 3,
        "is_too_short": len(t.strip()) < 16,
        "has_hedging": bool(
            re.search(
                r"\b(?:generally|typically|usually|it is recommended|likely|possibly|in general)\b",
                t,
                flags=re.I,
            )
        ),
        "broken_spacing": bool(re.search(r"\b\w(?:\s\w){4,}\b", t)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TruthfulQA KVI vs GraphRAG MC2 attribution from predictions.jsonl")
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    path = Path(args.predictions_jsonl)
    rows = _read_jsonl(path)
    if not rows:
        raise SystemExit(f"empty or unreadable: {path}")

    deltas: List[float] = []
    win = tie = lose = 0
    loser_rows: List[Dict[str, Any]] = []
    winner_rows: List[Dict[str, Any]] = []

    for r in rows:
        k = _mc2(r, "kvi")
        g = _mc2(r, "graphrag")
        d = k - g
        deltas.append(d)
        if d > 1e-9:
            win += 1
            winner_rows.append(r)
        elif d < -1e-9:
            lose += 1
            loser_rows.append(r)
        else:
            tie += 1

    n = len(rows)

    def _agg_feat(rs: List[Dict[str, Any]], method: str) -> Counter:
        c: Counter = Counter()
        for r in rs:
            fl = _flags(_pred(r, method))
            for k2, v in fl.items():
                if v is True:
                    c[k2] += 1
        return c

    loser_kvi_feat = _agg_feat(loser_rows, "kvi")
    loser_gr_feat = _agg_feat(loser_rows, "graphrag")
    winner_kvi_feat = _agg_feat(winner_rows, "kvi")

    # On KVI-lose examples: how often GraphRAG looks "cleaner" (shorter, fewer !) than KVI?
    gr_shorter_when_kvi_loses = 0
    gr_fewer_exclaim_when_kvi_loses = 0
    both_high_exclaim = 0
    len_k_losers: List[int] = []
    len_g_losers: List[int] = []
    for r in loser_rows:
        pk = _flags(_pred(r, "kvi"))
        pg = _flags(_pred(r, "graphrag"))
        len_k_losers.append(int(pk["len"]))
        len_g_losers.append(int(pg["len"]))
        if pg["len"] < pk["len"]:
            gr_shorter_when_kvi_loses += 1
        if pg["exclaim_count"] < pk["exclaim_count"]:
            gr_fewer_exclaim_when_kvi_loses += 1
        if pk["high_exclaim"] and pg["high_exclaim"]:
            both_high_exclaim += 1

    def _mean(xs: List[int]) -> float:
        return float(sum(xs)) / float(len(xs))) if xs else 0.0

    mean_delta = sum(deltas) / float(n) if n else 0.0
    sorted_by_delta = sorted(
        (
            {
                "id": r.get("id"),
                "delta": _mc2(r, "kvi") - _mc2(r, "graphrag"),
                "kvi_mc2": _mc2(r, "kvi"),
                "gr_mc2": _mc2(r, "graphrag"),
                "kvi_preview": _pred(r, "kvi")[:200],
                "gr_preview": _pred(r, "graphrag")[:200],
            }
            for r in rows
        ),
        key=lambda x: x["delta"],
    )

    out: Dict[str, Any] = {
        "predictions_jsonl": str(path),
        "n": n,
        "kvi_win_mc2": win,
        "kvi_tie_mc2": tie,
        "kvi_lose_mc2": lose,
        "mean_delta_kvi_minus_graphrag_mc2": round(mean_delta, 6),
        "loser_count": lose,
        "when_kvi_loses_mc2": {
            "graphrag_shorter_than_kvi": gr_shorter_when_kvi_loses,
            "graphrag_fewer_exclaims_than_kvi": gr_fewer_exclaim_when_kvi_loses,
            "both_high_exclaim_ge5": both_high_exclaim,
            "mean_chars_kvi_on_losers": round(_mean(len_k_losers), 2),
            "mean_chars_graphrag_on_same_losers": round(_mean(len_g_losers), 2),
        },
        "kvi_text_feature_hits_in_losers": dict(loser_kvi_feat),
        "graphrag_text_feature_hits_in_same_losers": dict(loser_gr_feat),
        "kvi_text_feature_hits_in_winners": dict(winner_kvi_feat),
        "worst_15_by_delta": sorted_by_delta[:15],
        "best_15_by_delta": list(reversed(sorted_by_delta[-15:])),
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# TruthfulQA D full — KVI vs GraphRAG MC2 attribution")
    md.append("")
    md.append(f"- Source: `{path}`")
    md.append(f"- N: **{n}**")
    md.append(
        f"- Per-example MC2: KVI better / tie / worse than GraphRAG: **{win} / {tie} / {lose}** "
        f"({100.0*win/n:.1f}% / {100.0*tie/n:.1f}% / {100.0*lose/n:.1f}%)"
    )
    md.append(f"- Mean (KVI−GraphRAG) MC2: **{mean_delta:+.4f}**")
    md.append("")
    md.append("## When KVI loses MC2 (structural checks)")
    md.append("")
    wh = out["when_kvi_loses_mc2"]
    md.append(f"- On those **{lose}** items: GraphRAG answer shorter than KVI: **{wh['graphrag_shorter_than_kvi']}**")
    md.append(f"- GraphRAG fewer `!` than KVI: **{wh['graphrag_fewer_exclaims_than_kvi']}**")
    md.append(f"- Both KVI and GraphRAG have `!`≥5: **{wh['both_high_exclaim_ge5']}**")
    md.append(
        f"- Mean answer length (chars) on losers — KVI: **{wh['mean_chars_kvi_on_losers']}**, "
        f"GraphRAG: **{wh['mean_chars_graphrag_on_same_losers']}**"
    )
    md.append("")
    md.append("## Text-shape hits (KVI predictions, losers only)")
    md.append("")
    for k, v in sorted(loser_kvi_feat.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- `{k}`: **{v}** / {lose}")
    md.append("")
    md.append("## Worst 15 (lowest KVI−GraphRAG MC2)")
    md.append("")
    md.append("| id | Δ | kvi_mc2 | gr_mc2 |")
    md.append("|---|---:|---:|---:|")
    for x in out["worst_15_by_delta"]:
        md.append(f"| {x.get('id')} | {x['delta']:.4f} | {x['kvi_mc2']:.4f} | {x['gr_mc2']:.4f} |")
    md.append("")
    Path(args.out_md).write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out_json": args.out_json, "out_md": args.out_md}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
