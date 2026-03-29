#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_summary(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid summary JSON: {path}")
    return obj


def _method_em_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in summary.get("methods", []) or []:
        if not isinstance(r, dict):
            continue
        key = str(r.get("method_key") or "").strip()
        if key:
            out[key] = {
                "em": float(r.get("em") or 0.0),
                "lo": float(r.get("em_ci95_lo") or 0.0),
                "hi": float(r.get("em_ci95_hi") or 0.0),
            }
    return out


def _dataset_metrics(method_map: Dict[str, Dict[str, float]], key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if key not in method_map:
        return None, None, None
    m = method_map[key]
    return float(m.get("em", 0.0)), float(m.get("lo", 0.0)), float(m.get("hi", 0.0))


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.1f}"


def _fmt_ci(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "N/A"
    return f"[{float(lo):.1f}, {float(hi):.1f}]"


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Exp01 Hotpot+MedHop+NQ summaries into one main table")
    p.add_argument("--hotpot_summary", required=True)
    p.add_argument("--medhop_summary", default="")
    p.add_argument("--nq_summary", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    hotpot = _load_summary(Path(args.hotpot_summary))
    medhop = _load_summary(Path(args.medhop_summary)) if str(args.medhop_summary).strip() else {}
    nq = _load_summary(Path(args.nq_summary))

    h = _method_em_map(hotpot)
    m = _method_em_map(medhop) if medhop else {}
    n = _method_em_map(nq)

    methods = [
        ("llm", "LLM", "none", "none"),
        ("rag", "RAG", "ANN", "prompt"),
        ("graphrag", "GraphRAG", "graph", "prompt"),
        ("kv_prefix", "KV Prefix", "ANN", "KV"),
        ("kvi", "KVI", "graph", "KV + prompt"),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, retrieval, injection in methods:
        he, hlo, hhi = _dataset_metrics(h, key)
        me, mlo, mhi = _dataset_metrics(m, key)
        ne, nlo, nhi = _dataset_metrics(n, key)
        rows.append(
            {
                "method_key": key,
                "method": label,
                "retrieval": retrieval,
                "injection": injection,
                "hotpot_em": he,
                "hotpot_ci95_lo": hlo,
                "hotpot_ci95_hi": hhi,
                "medhop_em": me,
                "medhop_ci95_lo": mlo,
                "medhop_ci95_hi": mhi,
                "nq_em": ne,
                "nq_ci95_lo": nlo,
                "nq_ci95_hi": nhi,
            }
        )

    summary = {
        "datasets": {
            "hotpot_summary": str(Path(args.hotpot_summary)),
            "medhop_summary": (str(Path(args.medhop_summary)) if str(args.medhop_summary).strip() else ""),
            "nq_summary": str(Path(args.nq_summary)),
        },
        "table": rows,
    }
    (out_dir / "main_table_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "main_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Retrieval", "Injection", "HotpotQA EM", "HotpotQA CI95", "MedHopQA EM", "MedHopQA CI95", "NQ EM", "NQ CI95"])
        for r in rows:
            w.writerow(
                [
                    r["method"],
                    r["retrieval"],
                    r["injection"],
                    _fmt_pct(r["hotpot_em"]),
                    _fmt_ci(r["hotpot_ci95_lo"], r["hotpot_ci95_hi"]),
                    _fmt_pct(r["medhop_em"]),
                    _fmt_ci(r["medhop_ci95_lo"], r["medhop_ci95_hi"]),
                    _fmt_pct(r["nq_em"]),
                    _fmt_ci(r["nq_ci95_lo"], r["nq_ci95_hi"]),
                ]
            )

    md = []
    md.append("## Experiment 1 — Main QA Performance\n\n")
    md.append("| Method | Retrieval | Injection | HotpotQA EM | HotpotQA CI95 | MedHopQA EM | MedHopQA CI95 | NQ EM | NQ CI95 |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            f"| {r['method']} | {r['retrieval']} | {r['injection']} | "
            f"{_fmt_pct(r['hotpot_em'])} | {_fmt_ci(r['hotpot_ci95_lo'], r['hotpot_ci95_hi'])} | "
            f"{_fmt_pct(r['medhop_em'])} | {_fmt_ci(r['medhop_ci95_lo'], r['medhop_ci95_hi'])} | "
            f"{_fmt_pct(r['nq_em'])} | {_fmt_ci(r['nq_ci95_lo'], r['nq_ci95_hi'])} |\n"
        )
    (out_dir / "main_table.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

