#!/usr/bin/env python3
"""
Rebuild results/summary.json, hallucination_proxy_summary.json/.md, summary.md
from existing truthfulqa_fullmethods_qwen25_7b/summary.json and fever_fullmethods_qwen25_7b/summary.json.
Use after TruthfulQA (or FEVER) eval so unified Hallucination plots stay in sync without re-running inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/zd/dev/KVI")
    p.add_argument("--model", default="/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct")
    args = p.parse_args()
    root = Path(args.root)
    res = root / "experiments/exp02_hallucination/results"
    run_summaries: Dict[str, Dict[str, Any]] = {}
    for ds_key in ("truthfulqa", "fever"):
        summ_path = res / f"{ds_key}_fullmethods_qwen25_7b/summary.json"
        if not summ_path.exists():
            raise SystemExit(f"Missing {summ_path}")
        run_summaries[ds_key] = _load(summ_path)

    out_rows: List[Dict[str, Any]] = []
    for ds_name in ("truthfulqa", "fever"):
        summ = run_summaries.get(ds_name) or {}
        for m in summ.get("methods", []) or []:
            if not isinstance(m, dict):
                continue
            em = float(m.get("em") or 0.0)
            if ds_name == "fever" and m.get("fever_label_accuracy") is not None:
                try:
                    fla = float(m["fever_label_accuracy"])
                    hall = round(100.0 - fla, 4)
                except (TypeError, ValueError):
                    hall = round(100.0 - em, 4)
            else:
                hall = round(100.0 - em, 4)
            out_rows.append(
                {
                    "dataset": ds_name,
                    "method_key": m.get("method_key"),
                    "method": m.get("method"),
                    "em": em,
                    "hallucination_rate": hall,
                }
            )

    out = {
        "note": (
            "Hallucination proxy: TruthfulQA = 100 - relaxed EM; "
            "FEVER = 100 - fever_label_accuracy (parsed label vs gold)."
        ),
        "rows": out_rows,
    }
    (res / "hallucination_proxy_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "## Experiment 2 — Hallucination Reduction (Proxy)\n\n",
        "TruthfulQA: Hallucination Rate (%) = `100 - relaxed EM`. "
        "FEVER: Hallucination Rate (%) = `100 - fever_label_accuracy`.\n\n",
        "| Dataset | Method | Relaxed EM (%) | Hallucination Rate (%) |\n",
        "|---|---|---:|---:|\n",
    ]
    for r in out_rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['em']:.1f} | {r['hallucination_rate']:.1f} |\n")

    tqa = run_summaries.get("truthfulqa") or {}
    tqa_methods = tqa.get("methods") if isinstance(tqa, dict) else None
    if isinstance(tqa_methods, list):
        rows_mc = []
        for m in tqa_methods:
            if not isinstance(m, dict):
                continue
            if "truthfulqa_mc1_proxy" in m and "truthfulqa_mc2_proxy" in m:
                rows_mc.append(
                    (
                        str(m.get("method") or m.get("method_key") or ""),
                        float(m.get("truthfulqa_mc1_proxy") or 0.0),
                        float(m.get("truthfulqa_mc2_proxy") or 0.0),
                        int(m.get("truthfulqa_mc_valid_n") or 0),
                    )
                )
        if rows_mc:
            md += [
                "\n### TruthfulQA MC1/MC2 (proxy)\n\n",
                "| Method | MC1 Proxy (%) | MC2 Proxy (%) | Valid N |\n",
                "|---|---:|---:|---:|\n",
            ]
            for method, mc1, mc2, vn in rows_mc:
                md.append(f"| {method} | {100.0 * mc1:.1f} | {100.0 * mc2:.1f} | {vn} |\n")
    (res / "hallucination_proxy_summary.md").write_text("".join(md), encoding="utf-8")

    unified_rows: List[Dict[str, Any]] = []
    if isinstance(tqa, dict):
        for m in tqa.get("methods") or []:
            if not isinstance(m, dict):
                continue
            method = str(m.get("method") or m.get("method_key") or "")
            mk = str(m.get("method_key") or "")
            if "truthfulqa_mc1_proxy" in m:
                v = float(m.get("truthfulqa_mc1_proxy") or 0.0)
                unified_rows.append(
                    {
                        "dataset": "truthfulqa",
                        "method_key": mk,
                        "method": method,
                        "metric_source": "mc1_proxy",
                        "score_percent": round(100.0 * v, 4),
                        "hallucination_rate": round(100.0 * (1.0 - v), 4),
                    }
                )
            if "truthfulqa_mc2_proxy" in m:
                v = float(m.get("truthfulqa_mc2_proxy") or 0.0)
                unified_rows.append(
                    {
                        "dataset": "truthfulqa",
                        "method_key": mk,
                        "method": method,
                        "metric_source": "mc2_proxy",
                        "score_percent": round(100.0 * v, 4),
                        "hallucination_rate": round(100.0 * (1.0 - v), 4),
                    }
                )

    fever = run_summaries.get("fever") or {}
    if isinstance(fever, dict):
        for m in fever.get("methods") or []:
            if not isinstance(m, dict) or "fever_label_accuracy" not in m:
                continue
            v = float(m.get("fever_label_accuracy") or 0.0)
            unified_rows.append(
                {
                    "dataset": "fever",
                    "method_key": str(m.get("method_key") or ""),
                    "method": str(m.get("method") or m.get("method_key") or ""),
                    "metric_source": "fever_label_accuracy",
                    "score_percent": round(v, 4),
                    "hallucination_rate": round(100.0 - v, 4),
                }
            )

    unified = {
        "note": (
            "Unified Hallucination Rate (%): TruthfulQA uses 100 - MC proxy (MC1/MC2); "
            "FEVER uses 100 - fever_label_accuracy."
        ),
        "rows": unified_rows,
    }
    (res / "summary.json").write_text(json.dumps(unified, ensure_ascii=False, indent=2), encoding="utf-8")

    order = {"llm": 0, "rag": 1, "graphrag": 2, "kv_prefix": 3, "kvi": 4}
    source_order = {"mc1_proxy": 0, "mc2_proxy": 1, "fever_label_accuracy": 2}
    unified_rows_sorted = sorted(
        unified_rows,
        key=lambda r: (
            str(r.get("dataset")),
            int(order.get(str(r.get("method_key")), 99)),
            int(source_order.get(str(r.get("metric_source")), 99)),
        ),
    )
    tqa_n = int((run_summaries.get("truthfulqa") or {}).get("n") or 0)
    fever_n = int((run_summaries.get("fever") or {}).get("n") or 0)
    smd: List[str] = [
        "## Experiment 2 — Unified Hallucination Rate Summary\n\n",
        "| Dataset | Method | Metric Source | Score (%) | Hallucination Rate (%) |\n",
        "|---|---|---|---:|---:|\n",
    ]
    for r in unified_rows_sorted:
        smd.append(
            f"| {r['dataset']} | {r['method']} | {r['metric_source']} | "
            f"{float(r['score_percent']):.1f} | {float(r['hallucination_rate']):.1f} |\n"
        )
    smd.append(f"\n_TruthfulQA n={tqa_n}, FEVER n={fever_n}, model `{args.model}`._\n")
    (res / "summary.md").write_text("".join(smd), encoding="utf-8")

    print(json.dumps({"ok": True, "unified_rows": len(unified_rows)}, indent=2))


if __name__ == "__main__":
    main()
