#!/usr/bin/env python3
"""
Build bar charts for Exp02 hallucination proxy: Hallucination Rate (%) = 100 - relaxed EM.

Reads experiments/exp02_hallucination/results/hallucination_proxy_summary.json
(produced by run_exp02_hallucination.py after TruthfulQA + FEVER fullmethods runs).

Outputs SVG + HTML (no matplotlib required).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


METHOD_ORDER: List[Tuple[str, str]] = [
    ("llm", "LLM"),
    ("rag", "RAG"),
    ("graphrag", "GraphRAG"),
    ("kv_prefix", "KV Prefix"),
    ("kvi", "KVI"),
]

DATASET_ORDER = ["truthfulqa", "fever"]
DATASET_TITLE = {"truthfulqa": "TruthfulQA", "fever": "FEVER"}


def _load_proxy_json(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid summary JSON (no rows): {path}")
    return rows


def _rates_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {d: {} for d in DATASET_ORDER}
    for r in rows:
        ds = str(r.get("dataset") or "").lower()
        mk = str(r.get("method_key") or "")
        if ds in out and mk:
            out[ds][mk] = float(r.get("hallucination_rate") or 0.0)
    return out


def _build_from_summaries(truthfulqa_summary: Path, fever_summary: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds_key, path, name in [
        ("truthfulqa", truthfulqa_summary, "TruthfulQA"),
        ("fever", fever_summary, "FEVER"),
    ]:
        summ = json.loads(path.read_text(encoding="utf-8"))
        for m in summ.get("methods", []):
            mk = str(m.get("method_key") or "")
            em = float(m.get("em") or 0.0)
            rows.append(
                {
                    "dataset": ds_key,
                    "method_key": mk,
                    "method": str(m.get("method") or mk),
                    "em": em,
                    "hallucination_rate": round(100.0 - em, 4),
                }
            )
    return rows


def _svg_panel(
    *,
    title: str,
    rates: Dict[str, float],
    x0: float,
    y0: float,
    panel_w: float,
    panel_h: float,
    max_rate: float,
) -> str:
    margin_l, margin_b, margin_t = 72.0, 48.0, 36.0
    plot_w = panel_w - margin_l - 16
    plot_h = panel_h - margin_t - margin_b
    inner = []
    inner.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="15" font-weight="600" fill="#1a1a1a">{title}</text>')
    inner.append(
        f'<text x="{x0 + margin_l + plot_w / 2:.1f}" y="{y0 + panel_h - 8:.1f}" text-anchor="middle" font-size="12" fill="#444">Method</text>'
    )
    inner.append(
        f'<text transform="translate({x0 + 18:.1f},{y0 + margin_t + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" fill="#444">Hallucination Rate (%)</text>'
    )
    mx = max(max_rate, 1.0)
    # y-axis ticks 0, 25, 50, 75, 100 or scaled
    for tick in (0, 25, 50, 75, 100):
        if tick > mx + 5:
            continue
        ty = y0 + margin_t + plot_h * (1.0 - tick / mx)
        inner.append(
            f'<line x1="{x0 + margin_l:.1f}" y1="{ty:.1f}" x2="{x0 + margin_l + plot_w:.1f}" y2="{ty:.1f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        inner.append(
            f'<text x="{x0 + margin_l - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end" font-size="10" fill="#666">{tick:g}</text>'
        )
    n = len(METHOD_ORDER)
    bw = plot_w / max(n, 1) * 0.62
    gap = plot_w / max(n, 1) * 0.38
    colors = ["#c94c4c", "#d97b2a", "#c9a227", "#2a8f9d", "#3d6ab5"]
    for i, ((mk, label), c) in enumerate(zip(METHOD_ORDER, colors)):
        rate = float(rates.get(mk, 0.0))
        h = plot_h * (rate / mx) if mx else 0.0
        bx = x0 + margin_l + i * (bw + gap) + gap * 0.15
        by = y0 + margin_t + plot_h - h
        inner.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{c}" opacity="0.92"/>')
        inner.append(
            f'<text x="{bx + bw / 2:.1f}" y="{y0 + margin_t + plot_h + 14:.1f}" text-anchor="middle" font-size="9" fill="#333">{label}</text>'
        )
        inner.append(
            f'<text x="{bx + bw / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9" fill="#333">{rate:.1f}%</text>'
        )
    return "\n    ".join(inner)


def build_svg(rows: List[Dict[str, Any]]) -> str:
    by_ds = _rates_by_dataset(rows)
    all_rates = [by_ds[d][mk] for d in DATASET_ORDER for mk, _ in METHOD_ORDER if mk in by_ds[d]]
    max_rate = max(all_rates) if all_rates else 100.0
    max_rate = min(100.0, max(max_rate * 1.08, 10.0))

    W, H = 920, 380
    pw = W / 2 - 24
    ph = H - 40
    left = _svg_panel(
        title=DATASET_TITLE["truthfulqa"],
        rates=by_ds["truthfulqa"],
        x0=8,
        y0=24,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
    )
    right = _svg_panel(
        title=DATASET_TITLE["fever"],
        rates=by_ds["fever"],
        x0=8 + pw + 16,
        y0=24,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="{W / 2:.0f}" y="18" text-anchor="middle" font-size="14" font-weight="700" fill="#111">Exp02 — Hallucination rate proxy (100 − relaxed EM)</text>
  {left}
  {right}
</svg>'''


def build_html(svg_body: str) -> str:
    # Strip XML declaration for inline SVG
    inline = svg_body.split("\n", 1)[1] if svg_body.startswith("<?xml") else svg_body
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Exp02 Hallucination Rate</title>
  <style> body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f5f5f5; }} </style>
</head>
<body>
{inline}
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Exp02 hallucination proxy bar charts")
    p.add_argument(
        "--summary_json",
        default="/home/zd/dev/KVI/experiments/exp02_hallucination/results/hallucination_proxy_summary.json",
        help="From run_exp02_hallucination.py",
    )
    p.add_argument("--truthfulqa_summary", default="", help="Optional: rebuild from run_exp01 summary.json")
    p.add_argument("--fever_summary", default="", help="Optional: rebuild from run_exp01 summary.json")
    p.add_argument(
        "--out_dir",
        default="/home/zd/dev/KVI/experiments/exp02_hallucination/results",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.truthfulqa_summary).strip() and str(args.fever_summary).strip():
        rows = _build_from_summaries(Path(args.truthfulqa_summary), Path(args.fever_summary))
    else:
        path = Path(args.summary_json)
        if not path.exists():
            raise SystemExit(f"Missing {path}; pass --truthfulqa_summary and --fever_summary or run Exp02 first.")
        rows = _load_proxy_json(path)

    svg = build_svg(rows)
    (out_dir / "hallucination_proxy_bars.svg").write_text(svg, encoding="utf-8")
    (out_dir / "hallucination_proxy_bars.html").write_text(build_html(svg), encoding="utf-8")
    print(json.dumps({"ok": True, "svg": str(out_dir / "hallucination_proxy_bars.svg"), "html": str(out_dir / "hallucination_proxy_bars.html")}, indent=2))


if __name__ == "__main__":
    main()
