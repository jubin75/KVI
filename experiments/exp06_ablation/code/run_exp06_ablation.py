#!/usr/bin/env python3
"""Experiment 6 — Ablation table template + optional fill from Exp01 KVI row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _kvi_em(summary_path: Optional[Path]) -> Optional[float]:
    if not summary_path or not summary_path.exists():
        return None
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    for r in obj.get("methods") or []:
        if isinstance(r, dict) and r.get("method_key") == "kvi":
            return float(r.get("em") or 0.0)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Exp6: ablation table")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exp01_summary", default="", help="Optional Exp01 summary.json to fill Full KVI EM")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kvi_em = _kvi_em(Path(args.exp01_summary) if str(args.exp01_summary).strip() else None)

    rows: List[Dict[str, Any]] = [
        {
            "variant": "Full KVI",
            "accuracy_pct": kvi_em,
            "hallucination_pct": None,
            "notes": "graph + KV + prompt (dual-channel), openqa_mode",
        },
        {"variant": "− graph retrieval (ANN-only baseline)", "accuracy_pct": None, "hallucination_pct": None, "notes": "run modeA_rag"},
        {"variant": "− KV injection", "accuracy_pct": None, "hallucination_pct": None, "notes": "max_kv_triples 0 or GraphRAG"},
        {"variant": "− DRM (threshold ablation)", "accuracy_pct": None, "hallucination_pct": None, "notes": "drm_threshold sweep"},
        {"variant": "− dual channel (minimal prompt)", "accuracy_pct": None, "hallucination_pct": None, "notes": "kvi_minimal_prompt"},
    ]

    payload = {"experiment": "exp06_ablation", "rows": rows}
    (out / "ablation_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = [
        "## Experiment 6 — Ablation Study\n\n",
        "| Variant | Accuracy (%) | Hallucination (%) | Notes |\n",
        "|---|---:|---:|---|\n",
    ]
    for r in rows:
        acc = "" if r["accuracy_pct"] is None else f"{r['accuracy_pct']:.1f}"
        hal = "" if r["hallucination_pct"] is None else f"{r['hallucination_pct']:.1f}"
        md.append(f"| {r['variant']} | {acc} | {hal} | {r['notes']} |\n")
    (out / "ablation_table.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
