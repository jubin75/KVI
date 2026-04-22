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
    for ds_key, path, _name in [
        ("truthfulqa", truthfulqa_summary, "TruthfulQA"),
        ("fever", fever_summary, "FEVER"),
    ]:
        summ = json.loads(path.read_text(encoding="utf-8"))
        for m in summ.get("methods", []):
            mk = str(m.get("method_key") or "")
            em = float(m.get("em") or 0.0)
            if ds_key == "fever" and m.get("fever_label_accuracy") is not None:
                try:
                    fla = float(m["fever_label_accuracy"])
                    hall = round(100.0 - fla, 4)
                except (TypeError, ValueError):
                    hall = round(100.0 - em, 4)
            else:
                hall = round(100.0 - em, 4)
            rows.append(
                {
                    "dataset": ds_key,
                    "method_key": mk,
                    "method": str(m.get("method") or mk),
                    "em": em,
                    "hallucination_rate": hall,
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
        "TruthfulQA: 100 − relaxed EM (substring match; not official MC). "
        "FEVER: 100 − veracity label accuracy (SUPPORTS / REFUTES / NOT ENOUGH INFO)."
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
    title_main = "Hallucination rate by method (proxy metrics)"
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
    W, H = (640, 380) if paper else (640, 360)
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
  <text x="{W / 2:.0f}" y="18" text-anchor="middle" font-size="{"13" if paper else "14"}" font-weight="700" font-family="{ff}" fill="#111111">FEVER veracity label accuracy</text>
  {panel}{cap_el}
</svg>'''


def build_svg_truthfulqa_mc_proxy(truthfulqa_summary: Path, *, paper: bool = False) -> str:
    summ = json.loads(truthfulqa_summary.read_text(encoding="utf-8"))
    mc1: Dict[str, float] = {mk: 0.0 for mk, _ in METHOD_ORDER}
    mc2: Dict[str, float] = {mk: 0.0 for mk, _ in METHOD_ORDER}
    found = False
    for m in summ.get("methods", []):
        mk = str(m.get("method_key") or "")
        if mk in mc1 and "truthfulqa_mc1_proxy" in m and "truthfulqa_mc2_proxy" in m:
            mc1[mk] = 100.0 * float(m["truthfulqa_mc1_proxy"])
            mc2[mk] = 100.0 * float(m["truthfulqa_mc2_proxy"])
            found = True
    if not found:
        raise ValueError(
            f"No truthfulqa_mc1_proxy/truthfulqa_mc2_proxy fields in {truthfulqa_summary}; rerun run_exp01.py on TRUTHFULQA."
        )
    max_rate = max(20.0, min(100.0, max(max(mc1.values()), max(mc2.values())) * 1.12 + 5))
    W, H = (960, 390) if paper else (960, 360)
    bg = "#ffffff" if paper else "#fafafa"
    ff = "Helvetica, Arial, sans-serif" if paper else "system-ui, sans-serif"
    left = _svg_panel(
        title="TruthfulQA — MC1 (proxy, %)",
        rates=mc1,
        x0=8,
        y0=28,
        panel_w=W / 2 - 16,
        panel_h=H - (52 if paper else 40),
        max_rate=max_rate,
        paper=paper,
        y_axis_label="MC1 Proxy (%)",
    )
    right = _svg_panel(
        title="TruthfulQA — MC2 (proxy, %)",
        rates=mc2,
        x0=W / 2 + 4,
        y0=28,
        panel_w=W / 2 - 16,
        panel_h=H - (52 if paper else 40),
        max_rate=max_rate,
        paper=paper,
        y_axis_label="MC2 Proxy (%)",
    )
    cap = (
        "Proxy metrics: option likelihood conditioned on question + method answer; not official TruthfulQA script."
        if paper
        else ""
    )
    cap_el = ""
    if paper and cap:
        cap_el = f'\n  <text x="{W / 2:.0f}" y="{H - 6:.0f}" text-anchor="middle" font-size="9" font-family="{ff}" fill="#555555">{cap}</text>'
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="{bg}"/>
  <text x="{W / 2:.0f}" y="18" text-anchor="middle" font-size="{"13" if paper else "14"}" font-weight="700" font-family="{ff}" fill="#111111">TruthfulQA multiple-choice likelihood proxy</text>
  {left}
  {right}{cap_el}
</svg>'''


def _rates_three_panel_from_unified_summary(path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load results/summary.json produced for the unified 3-panel figure (MC1 / MC2 / FEVER label)."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid unified summary (no rows): {path}")
    fever: Dict[str, float] = {}
    tqa_mc1: Dict[str, float] = {}
    tqa_mc2: Dict[str, float] = {}
    for r in rows:
        ds = str(r.get("dataset") or "")
        mk = str(r.get("method_key") or "")
        src = str(r.get("metric_source") or "")
        if not mk:
            continue
        try:
            h = float(r.get("hallucination_rate") or 0.0)
        except (TypeError, ValueError):
            continue
        if ds == "fever" and src == "fever_label_accuracy":
            fever[mk] = h
        elif ds == "truthfulqa" and src == "mc1_proxy":
            tqa_mc1[mk] = h
        elif ds == "truthfulqa" and src == "mc2_proxy":
            tqa_mc2[mk] = h
    if not fever or not tqa_mc1 or not tqa_mc2:
        raise ValueError(
            f"{path} missing fever_label_accuracy / mc1_proxy / mc2_proxy rows; "
            "regenerate summary.json or run plot_unified_hallucination_bars data pipeline."
        )
    return tqa_mc1, tqa_mc2, fever


def build_svg_three_panel_unified(unified_summary: Path, *, paper: bool = False) -> str:
    """One figure: TruthfulQA MC1 & MC2 proxy hallucination + FEVER label-based hallucination (same metrics as summary.json)."""
    mc1, mc2, fever = _rates_three_panel_from_unified_summary(unified_summary)
    all_rates = [mc1[mk] for mk, _ in METHOD_ORDER if mk in mc1]
    all_rates += [mc2[mk] for mk, _ in METHOD_ORDER if mk in mc2]
    all_rates += [fever[mk] for mk, _ in METHOD_ORDER if mk in fever]
    max_rate = max(all_rates) if all_rates else 100.0
    max_rate = min(100.0, max(max_rate * 1.08, 10.0))

    W, H = (1320, 410) if paper else (1320, 390)
    gap = 16.0
    side_margin = 8.0
    pw = (W - side_margin * 2 - 2 * gap) / 3.0
    ph = H - (56 if paper else 44)
    bg = "#ffffff" if paper else "#fafafa"
    ff = "Helvetica, Arial, sans-serif" if paper else "system-ui, sans-serif"
    cap = (
        "TruthfulQA: 100 − MC1/MC2 likelihood proxy (not official MC). "
        "FEVER: 100 − veracity label accuracy."
        if paper
        else ""
    )
    x1 = side_margin
    x2 = side_margin + pw + gap
    x3 = side_margin + 2 * (pw + gap)
    y0 = 26.0
    p1 = _svg_panel(
        title="TruthfulQA — MC1 proxy",
        rates=mc1,
        x0=x1,
        y0=y0,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
        y_axis_label="Hallucination Rate (%)",
    )
    p2 = _svg_panel(
        title="TruthfulQA — MC2 proxy",
        rates=mc2,
        x0=x2,
        y0=y0,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
        y_axis_label="Hallucination Rate (%)",
    )
    p3 = _svg_panel(
        title="FEVER — Label accuracy",
        rates=fever,
        x0=x3,
        y0=y0,
        panel_w=pw,
        panel_h=ph,
        max_rate=max_rate,
        paper=paper,
        y_axis_label="Hallucination Rate (%)",
    )
    cap_el = ""
    if paper and cap:
        cap_el = f'\n  <text x="{W / 2:.0f}" y="{H - 6:.0f}" text-anchor="middle" font-size="9" font-family="{ff}" fill="#555555">{cap}</text>'
    title_main = "Hallucination rate by method (unified proxy metrics, 3 panels)"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="{bg}"/>
  <text x="{W / 2:.0f}" y="17" text-anchor="middle" font-size="{"13" if paper else "14"}" font-weight="700" font-family="{ff}" fill="#111111">{title_main}</text>
  {p1}
  {p2}
  {p3}{cap_el}
</svg>'''


def build_html(svg_body: str) -> str:
    # Strip XML declaration for inline SVG
    inline = svg_body.split("\n", 1)[1] if svg_body.startswith("<?xml") else svg_body
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Hallucination rate (proxy benchmarks)</title>
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
    p.add_argument(
        "--truthfulqa_mc_figure",
        action="store_true",
        help="Also write truthfulqa_mc_proxy_bars.svg from truthfulqa summary (requires truthfulqa_mc1_proxy/mc2 fields).",
    )
    p.add_argument(
        "--three_panel_unified",
        action="store_true",
        help="Also write 3-panel figure (TruthfulQA MC1 + MC2 + FEVER label hallucination) from unified summary.json.",
    )
    p.add_argument(
        "--unified_summary_json",
        default="",
        help="Path to results/summary.json for --three_panel_unified (default: <out_dir>/summary.json).",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fever_path = Path(str(args.fever_summary).strip()) if str(args.fever_summary).strip() else out_dir / "fever_fullmethods_qwen25_7b" / "summary.json"
    tqa_path = Path(str(args.truthfulqa_summary).strip()) if str(args.truthfulqa_summary).strip() else out_dir / "truthfulqa_fullmethods_qwen25_7b" / "summary.json"

    if str(args.truthfulqa_summary).strip() and str(args.fever_summary).strip():
        rows = _build_from_summaries(tqa_path, fever_path)
    else:
        path = Path(args.summary_json)
        if not path.exists():
            raise SystemExit(f"Missing {path}; pass --truthfulqa_summary and --fever_summary or run Exp02 first.")
        rows = _load_proxy_json(path)
        # FEVER panel: prefer label-accuracy-based hallucination when per-method summaries exist
        if fever_path.exists():
            rows = _build_from_summaries(tqa_path, fever_path)

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
    if args.truthfulqa_mc_figure:
        try:
            tqa = build_svg_truthfulqa_mc_proxy(tqa_path, paper=False)
            (out_dir / "truthfulqa_mc_proxy_bars.svg").write_text(tqa, encoding="utf-8")
            out["truthfulqa_mc_proxy_bars_svg"] = str(out_dir / "truthfulqa_mc_proxy_bars.svg")
            if args.paper:
                tqap = build_svg_truthfulqa_mc_proxy(tqa_path, paper=True)
                (out_dir / "truthfulqa_mc_proxy_bars_paper.svg").write_text(tqap, encoding="utf-8")
                out["truthfulqa_mc_proxy_bars_paper_svg"] = str(out_dir / "truthfulqa_mc_proxy_bars_paper.svg")
        except Exception as e:
            out["truthfulqa_mc_error"] = str(e)

    if args.three_panel_unified:
        uni_path = Path(str(args.unified_summary_json).strip()) if str(args.unified_summary_json).strip() else out_dir / "summary.json"
        try:
            svg3 = build_svg_three_panel_unified(uni_path, paper=False)
            (out_dir / "hallucination_proxy_three_panel.svg").write_text(svg3, encoding="utf-8")
            out["hallucination_proxy_three_panel_svg"] = str(out_dir / "hallucination_proxy_three_panel.svg")
            if args.paper:
                svg3p = build_svg_three_panel_unified(uni_path, paper=True)
                (out_dir / "hallucination_proxy_three_panel_paper.svg").write_text(svg3p, encoding="utf-8")
                out["hallucination_proxy_three_panel_paper_svg"] = str(out_dir / "hallucination_proxy_three_panel_paper.svg")
        except Exception as e:
            out["three_panel_unified_error"] = str(e)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
