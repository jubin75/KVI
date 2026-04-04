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
    paper: bool,
    y_axis_label: str = "Hallucination Rate (%)",
) -> str:
    margin_l, margin_b, margin_t = 72.0, 48.0, 36.0
    plot_w = panel_w - margin_l - 16
    plot_h = panel_h - margin_t - margin_b
    inner = []
    title_c = "#111111" if paper else "#1a1a1a"
    sub_c = "#222222" if paper else "#444444"
    tick_c = "#333333" if paper else "#666666"
    grid_c = "#dddddd" if paper else "#e8e8e8"
    ff = "Helvetica, Arial, sans-serif" if paper else "system-ui, sans-serif"
    inner.append(
        f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="{"14" if paper else "15"}" '
        f'font-weight="600" font-family="{ff}" fill="{title_c}">{title}</text>'
    )
    inner.append(
        f'<text x="{x0 + margin_l + plot_w / 2:.1f}" y="{y0 + panel_h - 8:.1f}" text-anchor="middle" '
        f'font-size="{"11" if paper else "12"}" font-family="{ff}" fill="{sub_c}">Method</text>'
    )
    inner.append(
        f'<text transform="translate({x0 + 18:.1f},{y0 + margin_t + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" '
        f'font-size="{"11" if paper else "12"}" font-family="{ff}" fill="{sub_c}">{y_axis_label}</text>'
    )
    mx = max(max_rate, 1.0)
    # y-axis ticks 0, 25, 50, 75, 100 or scaled
    for tick in (0, 25, 50, 75, 100):
        if tick > mx + 5:
            continue
        ty = y0 + margin_t + plot_h * (1.0 - tick / mx)
        inner.append(
            f'<line x1="{x0 + margin_l:.1f}" y1="{ty:.1f}" x2="{x0 + margin_l + plot_w:.1f}" y2="{ty:.1f}" '
            f'stroke="{grid_c}" stroke-width="{"1" if paper else "1"}"/>'
        )
        inner.append(
            f'<text x="{x0 + margin_l - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end" font-size="{"10" if paper else "10"}" '
            f'font-family="{ff}" fill="{tick_c}">{tick:g}</text>'
        )
    n = len(METHOD_ORDER)
    bw = plot_w / max(n, 1) * 0.62
    gap = plot_w / max(n, 1) * 0.38
    colors = (
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        if paper
        else ["#c94c4c", "#d97b2a", "#c9a227", "#2a8f9d", "#3d6ab5"]
    )
    label_fill = "#111111" if paper else "#333333"
    for i, ((mk, label), c) in enumerate(zip(METHOD_ORDER, colors)):
        rate = float(rates.get(mk, 0.0))
        h = plot_h * (rate / mx) if mx else 0.0
        bx = x0 + margin_l + i * (bw + gap) + gap * 0.15
        by = y0 + margin_t + plot_h - h
        op = "1" if paper else "0.92"
        inner.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{c}" opacity="{op}"/>')
        inner.append(
            f'<text x="{bx + bw / 2:.1f}" y="{y0 + margin_t + plot_h + 14:.1f}" text-anchor="middle" font-size="{"9" if paper else "9"}" '
            f'font-family="{ff}" fill="{label_fill}">{label}</text>'
        )
        inner.append(
            f'<text x="{bx + bw / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9" font-family="{ff}" '
            f'fill="{label_fill}">{rate:.1f}%</text>'
        )
    return "\n    ".join(inner)


def build_svg(rows: List[Dict[str, Any]], *, paper: bool = False) -> str:
    by_ds = _rates_by_dataset(rows)
    all_rates = [by_ds[d][mk] for d in DATASET_ORDER for mk, _ in METHOD_ORDER if mk in by_ds[d]]
    max_rate = max(all_rates) if all_rates else 100.0
    max_rate = min(100.0, max(max_rate * 1.08, 10.0))

    W, H = 920, 400 if paper else 380
    pw = W / 2 - 24
    ph = H - (52 if paper else 40)
    bg = "#ffffff" if paper else "#fafafa"
    ff = "Helvetica, Arial, sans-serif" if paper else "system-ui, sans-serif"
    cap = (
        "Metric: hallucination proxy = 100 − relaxed EM (substring match; not TruthfulQA official)."
        if paper
        else ""
    )
    left = _svg_panel(
        title=DATASET_TITLE["truthfulqa"],
        rates=by_ds["truthfulqa"],
        x0=8,
        y0=24,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
    )
    right = _svg_panel(
        title=DATASET_TITLE["fever"],
        rates=by_ds["fever"],
        x0=8 + pw + 16,
        y0=24,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
    )
    cap_el = ""
    if paper and cap:
        cap_el = f'\n  <text x="{W / 2:.0f}" y="{H - 6:.0f}" text-anchor="middle" font-size="9" font-family="{ff}" fill="#555555">{cap}</text>'
    title_main = "Exp02 — Hallucination proxy (100 − relaxed EM)"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="{bg}"/>
  <text x="{W / 2:.0f}" y="17" text-anchor="middle" font-size="{"13" if paper else "14"}" font-weight="700" font-family="{ff}" fill="#111111">{title_main}</text>
  {left}
  {right}{cap_el}
</svg>'''


def build_svg_fever_label_accuracy(fever_summary: Path, *, paper: bool = False) -> str:
    summ = json.loads(fever_summary.read_text(encoding="utf-8"))
    acc: Dict[str, float] = {mk: 0.0 for mk, _ in METHOD_ORDER}
    for m in summ.get("methods", []):
        mk = str(m.get("method_key") or "")
        if mk in acc and "fever_label_accuracy" in m:
            acc[mk] = float(m["fever_label_accuracy"])
    if not any(isinstance(m, dict) and "fever_label_accuracy" in m for m in summ.get("methods", [])):
        raise ValueError(
            f"No fever_label_accuracy field in {fever_summary}; rerun run_exp01.py on FEVER after updating metrics.py."
        )
    max_rate = min(100.0, max(acc.values()) * 1.08 + 5 if acc else 100.0)
    max_rate = max(max_rate, 20.0)
    W, H = 640, 360 if not paper else 640, 380
    pw, ph = W - 48, H - (52 if paper else 44)
    bg = "#ffffff" if paper else "#fafafa"
    ff = "Helvetica, Arial, sans-serif" if paper else "system-ui, sans-serif"
    panel = _svg_panel(
        title="FEVER — Veracity label accuracy",
        rates=acc,
        x0=24,
        y0=28,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
        y_axis_label="Accuracy (%)",
    )
    cap = (
        "Parsed first occurrence of SUPPORTS / REFUTES / NOT ENOUGH INFO in model output (vs gold label)."
        if paper
        else ""
    )
    cap_el = ""
    if paper and cap:
        cap_el = f'\n  <text x="{W / 2:.0f}" y="{H - 6:.0f}" text-anchor="middle" font-size="9" font-family="{ff}" fill="#555555">{cap}</text>'
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="{bg}"/>
  <text x="{W / 2:.0f}" y="18" text-anchor="middle" font-size="{"13" if paper else "14"}" font-weight="700" font-family="{ff}" fill="#111111">Exp02 — FEVER label accuracy</text>
  {panel}{cap_el}
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
    p.add_argument(
        "--paper",
        action="store_true",
        help="Publication style: white background, Helvetica, caption line, separate *_paper.svg files",
    )
    p.add_argument(
        "--fever_label_figure",
        action="store_true",
        help="Also write fever_label_accuracy_bars.svg from fever summary (requires fever_label_accuracy in JSON)",
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

    out: Dict[str, Any] = {"ok": True}
    svg_screen = build_svg(rows, paper=False)
    (out_dir / "hallucination_proxy_bars.svg").write_text(svg_screen, encoding="utf-8")
    (out_dir / "hallucination_proxy_bars.html").write_text(build_html(svg_screen), encoding="utf-8")
    out["hallucination_proxy_bars_svg"] = str(out_dir / "hallucination_proxy_bars.svg")

    if args.paper:
        svg_p = build_svg(rows, paper=True)
        (out_dir / "hallucination_proxy_bars_paper.svg").write_text(svg_p, encoding="utf-8")
        (out_dir / "hallucination_proxy_bars_paper.html").write_text(build_html(svg_p), encoding="utf-8")
        out["hallucination_proxy_bars_paper_svg"] = str(out_dir / "hallucination_proxy_bars_paper.svg")

    fever_path = Path(str(args.fever_summary).strip()) if str(args.fever_summary).strip() else out_dir / "fever_fullmethods_qwen25_7b" / "summary.json"
    if args.fever_label_figure:
        try:
            fl = build_svg_fever_label_accuracy(fever_path, paper=False)
            (out_dir / "fever_label_accuracy_bars.svg").write_text(fl, encoding="utf-8")
            out["fever_label_accuracy_bars_svg"] = str(out_dir / "fever_label_accuracy_bars.svg")
            if args.paper:
                flp = build_svg_fever_label_accuracy(fever_path, paper=True)
                (out_dir / "fever_label_accuracy_bars_paper.svg").write_text(flp, encoding="utf-8")
                out["fever_label_accuracy_bars_paper_svg"] = str(out_dir / "fever_label_accuracy_bars_paper.svg")
        except Exception as e:
            out["fever_label_error"] = str(e)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
