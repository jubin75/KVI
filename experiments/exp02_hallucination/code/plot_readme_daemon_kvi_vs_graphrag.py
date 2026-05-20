#!/usr/bin/env python3
"""
Aggregate README progressive-daemon Exp02 runs (truthfulqa_readme_daemon_*/fever_readme_daemon_*)
and plot KVI-family methods vs GraphRAG.

Outputs under --out-dir:
  - readme_daemon_kvi_vs_graphrag.png
  - readme_daemon_kvi_vs_graphrag_by_schema_group.png
  - readme_daemon_kvi_vs_graphrag.json  (machine-readable stats)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _save_fig_png_pdf(fig: Any, png_path: Path) -> Tuple[Path, Path]:
    """Write raster for screens and vector PDF for LaTeX (same basename)."""
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return png_path, pdf_path
import numpy as np


def _parse_tag(tag: str) -> Dict[str, Any]:
    """Parse suffix like '<hex>_source_id_mx12_mn0p04'."""
    out: Dict[str, Any] = {"raw": tag}
    m = re.search(r"(doc_id|source_id)_mx(\d+)_mn0p(\d+)$", tag)
    if m:
        out["schema_group_by"] = m.group(1)
        out["graph_max_schema_evidence"] = int(m.group(2))
        out["graph_schema_min_score"] = int(m.group(3)) / 100.0
    return out


def _load_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tpath in sorted(results_dir.glob("truthfulqa_readme_daemon*/summary.json")):
        tag = tpath.parent.name.replace("truthfulqa_readme_daemon_", "")
        fpath = results_dir / f"fever_readme_daemon_{tag}" / "summary.json"
        if not fpath.is_file():
            continue
        tqa = json.loads(tpath.read_text(encoding="utf-8"))
        fev = json.loads(fpath.read_text(encoding="utf-8"))
        meta = _parse_tag(tag)
        methods_tqa = {str(m["method_key"]): m for m in tqa.get("methods") or []}
        methods_fev = {str(m["method_key"]): m for m in fev.get("methods") or []}
        method_keys = sorted(set(methods_tqa.keys()) | set(methods_fev.keys()))
        rows.append(
            {
                "tag": tag,
                "meta": meta,
                "method_keys": method_keys,
                "tqa": methods_tqa,
                "fev": methods_fev,
            }
        )
    return rows


def _extract(rows: List[Dict[str, Any]], dataset: str, metric: str) -> Dict[str, List[float]]:
    """dataset: 'tqa'|'fev', metric: mc2|fever_label."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        src = r["tqa"] if dataset == "tqa" else r["fev"]
        for mk, row in src.items():
            if metric == "mc2":
                v = row.get("truthfulqa_mc2_proxy")
            else:
                v = row.get("fever_label_accuracy")
            if v is None:
                continue
            acc[mk].append(float(v))
    return dict(acc)


def _method_order(keys: List[str]) -> List[str]:
    pref = [
        "graphrag",
        "kvi_triple_legacy",
        "kvi_schema_writer",
        "kvi_schema_verifier",
        "kvi_noinject_planner",
    ]
    out = [k for k in pref if k in keys]
    for k in sorted(keys):
        if k not in out:
            out.append(k)
    return out


def _plot_main(
    results_dir: Path,
    out_dir: Path,
    by_mc2: Dict[str, List[float]],
    by_fev: Dict[str, List[float]],
    stats: Dict[str, Any],
) -> Path:
    order = _method_order(list(set(by_mc2.keys()) | set(by_fev.keys())))
    means_mc2 = [float(np.mean(by_mc2[k])) if by_mc2.get(k) else float("nan") for k in order]
    std_mc2 = [float(np.std(by_mc2[k])) if by_mc2.get(k) and len(by_mc2[k]) > 1 else 0.0 for k in order]
    means_fev = [float(np.mean(by_fev[k])) if by_fev.get(k) else float("nan") for k in order]
    std_fev = [float(np.std(by_fev[k])) if by_fev.get(k) and len(by_fev[k]) > 1 else 0.0 for k in order]

    labels = {
        "graphrag": "GraphRAG",
        "kvi_triple_legacy": "KVI triple\n(legacy)",
        "kvi_schema_writer": "KVI schema\nwriter",
        "kvi_schema_verifier": "KVI schema\nverifier",
        "kvi_noinject_planner": "KVI no-inject\nplanner",
    }
    xlabs = [labels.get(k, k) for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.suptitle(
        f"README daemon: KVI variants vs GraphRAG (Exp02 pre100, n={stats['n_runs']} configs)",
        fontsize=12,
    )

    colors = []
    g_mc2 = means_mc2[order.index("graphrag")] if "graphrag" in order else None
    g_fev = means_fev[order.index("graphrag")] if "graphrag" in order else None
    for i, k in enumerate(order):
        if k == "graphrag":
            colors.append("#6c757d")
        elif k.startswith("kvi_"):
            if g_mc2 is not None and i < len(means_mc2) and means_mc2[i] > g_mc2:
                colors.append("#2e7d32")
            else:
                colors.append("#1976d2")
        else:
            colors.append("#888888")

    x = np.arange(len(order))
    w = 0.55
    axes[0].bar(x, means_mc2, w, yerr=std_mc2, capsize=3, color=colors, edgecolor="#333", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabs, fontsize=9)
    axes[0].set_ylabel("TruthfulQA MC2 proxy (↑ better)")
    axes[0].set_title("TruthfulQA (likelihood MC2 proxy)")
    axes[0].axhline(g_mc2, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7, label="GraphRAG mean")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylim(0, max(1.0, max(means_mc2) + 0.08))

    colors2 = []
    for i, k in enumerate(order):
        if k == "graphrag":
            colors2.append("#6c757d")
        elif k.startswith("kvi_"):
            if g_fev is not None and means_fev[i] > g_fev:
                colors2.append("#2e7d32")
            else:
                colors2.append("#1976d2")
        else:
            colors2.append("#888888")

    axes[1].bar(x, means_fev, w, yerr=std_fev, capsize=3, color=colors2, edgecolor="#333", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabs, fontsize=9)
    axes[1].set_ylabel("FEVER label accuracy (%)")
    axes[1].set_title("FEVER (label accuracy)")
    axes[1].axhline(g_fev, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7, label="GraphRAG mean")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_ylim(0, 100)

    fig.tight_layout()
    outp = out_dir / "readme_daemon_kvi_vs_graphrag.png"
    _save_fig_png_pdf(fig, outp)
    plt.close(fig)
    return outp


def _plot_by_schema_group(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    stats: Dict[str, Any],
) -> Optional[Path]:
    groups = {"source_id": [], "doc_id": []}
    for r in rows:
        gb = r["meta"].get("schema_group_by")
        if gb in groups:
            groups[gb].append(r)
    if not any(groups.values()):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    fig.suptitle("Mean by schema block grouping (source_id vs doc_id)", fontsize=12)

    for gi, (gname, sub) in enumerate(groups.items()):
        if not sub:
            continue
        by_mc2 = _extract(sub, "tqa", "mc2")
        by_fev = _extract(sub, "fev", "fever_label")
        order = _method_order(list(set(by_mc2.keys()) | set(by_fev.keys())))
        means_mc2 = [float(np.mean(by_mc2[k])) if by_mc2.get(k) else float("nan") for k in order]
        means_fev = [float(np.mean(by_fev[k])) if by_fev.get(k) else float("nan") for k in order]
        x = np.arange(len(order))
        labels_short = [k.replace("kvi_", "").replace("_", "\n") for k in order]

        axes[gi, 0].bar(x, means_mc2, color=["#6c757d" if k == "graphrag" else "#1565c0" for k in order])
        axes[gi, 0].set_xticks(x)
        axes[gi, 0].set_xticklabels(labels_short, fontsize=8, rotation=15, ha="right")
        axes[gi, 0].set_title(f"TruthfulQA MC2 — {gname} (n={len(sub)})")
        axes[gi, 0].set_ylabel("MC2 proxy")

        axes[gi, 1].bar(x, means_fev, color=["#6c757d" if k == "graphrag" else "#1565c0" for k in order])
        axes[gi, 1].set_xticks(x)
        axes[gi, 1].set_xticklabels(labels_short, fontsize=8, rotation=15, ha="right")
        axes[gi, 1].set_title(f"FEVER label acc — {gname}")
        axes[gi, 1].set_ylabel("accuracy %")
        axes[gi, 1].set_ylim(0, 100)

    fig.tight_layout()
    outp = out_dir / "readme_daemon_kvi_vs_graphrag_by_schema_group.png"
    _save_fig_png_pdf(fig, outp)
    plt.close(fig)
    return outp


def _win_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-run: which KVI methods beat GraphRAG on each metric."""
    tqa_wins: List[Dict[str, Any]] = []
    fev_wins: List[Dict[str, Any]] = []
    for r in rows:
        tag = r["tag"]
        meta = r["meta"]
        g_mc2 = r["tqa"].get("graphrag", {}).get("truthfulqa_mc2_proxy")
        g_fev = r["fev"].get("graphrag", {}).get("fever_label_accuracy")
        if g_mc2 is None or g_fev is None:
            continue
        g_mc2, g_fev = float(g_mc2), float(g_fev)
        for mk, row in r["tqa"].items():
            if mk == "graphrag" or not mk.startswith("kvi_"):
                continue
            v = row.get("truthfulqa_mc2_proxy")
            if v is not None and float(v) > g_mc2:
                tqa_wins.append(
                    {
                        "tag": tag,
                        "schema_group_by": meta.get("schema_group_by"),
                        "method": mk,
                        "mc2": float(v),
                        "graphrag_mc2": g_mc2,
                        "delta": float(v) - g_mc2,
                    }
                )
        for mk, row in r["fev"].items():
            if mk == "graphrag" or not mk.startswith("kvi_"):
                continue
            v = row.get("fever_label_accuracy")
            if v is not None and float(v) > g_fev:
                fev_wins.append(
                    {
                        "tag": tag,
                        "schema_group_by": meta.get("schema_group_by"),
                        "method": mk,
                        "fever_label_accuracy": float(v),
                        "graphrag_fever": g_fev,
                        "delta": float(v) - g_fev,
                    }
                )
    return {
        "truthfulqa_mc2_kvi_beats_graphrag": tqa_wins,
        "fever_label_kvi_beats_graphrag": fev_wins,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/zd/dev/KVI/experiments/exp02_hallucination/results"),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_summaries(results_dir)
    if not rows:
        raise SystemExit(f"No paired readme_daemon summaries under {results_dir}")

    by_mc2 = _extract(rows, "tqa", "mc2")
    by_fev = _extract(rows, "fev", "fever_label")
    wins = _win_analysis(rows)

    stats: Dict[str, Any] = {
        "n_runs": len(rows),
        "results_dir": str(results_dir),
        "mean_truthfulqa_mc2_proxy": {k: float(np.mean(v)) for k, v in by_mc2.items()},
        "std_truthfulqa_mc2_proxy": {k: float(np.std(v)) if len(v) > 1 else 0.0 for k, v in by_mc2.items()},
        "mean_fever_label_accuracy": {k: float(np.mean(v)) for k, v in by_fev.items()},
        "std_fever_label_accuracy": {k: float(np.std(v)) if len(v) > 1 else 0.0 for k, v in by_fev.items()},
        "per_run_wins": wins,
    }

    png1 = _plot_main(results_dir, out_dir, by_mc2, by_fev, stats)
    pdf1 = png1.with_suffix(".pdf")
    png2 = _plot_by_schema_group(rows, out_dir, stats)
    pdf2 = png2.with_suffix(".pdf") if png2 else None

    jpath = out_dir / "readme_daemon_kvi_vs_graphrag.json"
    jpath.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "png": str(png1),
                "pdf": str(pdf1),
                "png_by_group": str(png2) if png2 else None,
                "pdf_by_group": str(pdf2) if pdf2 else None,
                "json": str(jpath),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
