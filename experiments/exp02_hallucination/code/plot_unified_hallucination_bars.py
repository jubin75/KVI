#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

METHOD_ORDER = ["llm", "rag", "graphrag", "kv_prefix", "kvi"]
METHOD_LABEL = {
    "llm": "LLM",
    "rag": "RAG",
    "graphrag": "GraphRAG",
    "kv_prefix": "KV Prefix",
    "kvi": "KVI",
}


def _load(path: Path) -> List[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid unified summary: {path}")
    return rows


def _panel(title: str, rates: Dict[str, float], x0: float, y0: float, w: float, h: float) -> str:
    ml, mb, mt = 64.0, 44.0, 34.0
    pw, ph = w - ml - 14.0, h - mt - mb
    maxv = max(100.0, max(rates.values()) if rates else 100.0)
    s: List[str] = []
    s.append(f'<text x="{x0 + w/2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="13" font-weight="600">{title}</text>')
    s.append(f'<text transform="translate({x0+16:.1f},{y0+mt+ph/2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">Hallucination Rate (%)</text>')
    for t in (0, 25, 50, 75, 100):
        ty = y0 + mt + ph * (1.0 - t / maxv)
        s.append(f'<line x1="{x0+ml:.1f}" y1="{ty:.1f}" x2="{x0+ml+pw:.1f}" y2="{ty:.1f}" stroke="#dddddd" stroke-width="1"/>')
        s.append(f'<text x="{x0+ml-6:.1f}" y="{ty+4:.1f}" text-anchor="end" font-size="9">{t}</text>')
    n = len(METHOD_ORDER)
    bw = pw / n * 0.62
    gap = pw / n * 0.38
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, mk in enumerate(METHOD_ORDER):
        v = float(rates.get(mk, 0.0))
        bh = ph * (v / maxv)
        bx = x0 + ml + i * (bw + gap) + gap * 0.15
        by = y0 + mt + ph - bh
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{colors[i]}" rx="2"/>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{by-3:.1f}" text-anchor="middle" font-size="9">{v:.1f}</text>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y0+mt+ph+13:.1f}" text-anchor="middle" font-size="9">{METHOD_LABEL[mk]}</text>')
    return "\n  ".join(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot unified Exp02 hallucination summary")
    p.add_argument("--summary_json", default="/home/zd/dev/KVI/experiments/exp02_hallucination/results/summary.json")
    p.add_argument("--out_dir", default="/home/zd/dev/KVI/experiments/exp02_hallucination/results")
    args = p.parse_args()

    rows = _load(Path(args.summary_json))
    fever = {str(r.get("method_key")): float(r.get("hallucination_rate") or 0.0) for r in rows if str(r.get("dataset")) == "fever" and str(r.get("metric_source")) == "fever_label_accuracy"}
    tqa_mc1 = {str(r.get("method_key")): float(r.get("hallucination_rate") or 0.0) for r in rows if str(r.get("dataset")) == "truthfulqa" and str(r.get("metric_source")) == "mc1_proxy"}
    tqa_mc2 = {str(r.get("method_key")): float(r.get("hallucination_rate") or 0.0) for r in rows if str(r.get("dataset")) == "truthfulqa" and str(r.get("metric_source")) == "mc2_proxy"}

    W, H = 1320, 390
    pw = W / 3 - 18
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="14" font-weight="700">Unified hallucination rate (proxy, %)</text>
  {_panel("TruthfulQA (MC1 proxy -> Hallucination Rate)", tqa_mc1, 6, 24, pw, H-30)}
  {_panel("TruthfulQA (MC2 proxy -> Hallucination Rate)", tqa_mc2, 12+pw, 24, pw, H-30)}
  {_panel("FEVER (Label Accuracy -> Hallucination Rate)", fever, 18+2*pw, 24, pw, H-30)}
</svg>'''
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unified_hallucination_bars.svg").write_text(svg, encoding="utf-8")
    print(json.dumps({"ok": True, "svg": str(out_dir / "unified_hallucination_bars.svg")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
