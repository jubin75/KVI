#!/usr/bin/env python3
"""
Run KVI ablations vs GraphRAG on one dataset (Hotpot by default).

Each variant invokes run_exp01.py with --methods graphrag,kvi only (no ANN deps).

Outputs:
  <out_dir>/variant_<name>/summary.json
  <out_dir>/ablation_summary.json
  <out_dir>/ablation_table.md
  <out_dir>/CONCLUSION.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _em_for_keys(summary: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in summary.get("methods") or []:
        if not isinstance(r, dict):
            continue
        k = str(r.get("method_key") or "").strip()
        if k in keys:
            out[k] = float(r.get("em") or 0.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--dataset_jsonl", default="experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl")
    p.add_argument("--dataset_name", default="HotpotQA")
    p.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--artifacts_dir", default="experiments/exp01_main_qa/artifacts/hotpot")
    p.add_argument("--exp01_script", default="experiments/exp01_main_qa/code/run_exp01.py")
    p.add_argument("--out_dir", default="experiments/exp06_ablation/results/kvi_ablation_hotpot")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--bootstrap_samples", type=int, default=500)
    p.add_argument("--permutation_samples", type=int, default=1000)
    p.add_argument("--timeout_s", type=int, default=300)
    p.add_argument(
        "--variants",
        default="",
        help="Comma list of variant names to run (default: all). E.g. full_kvi,kvi_kv0,kvi_minimal_prompt",
    )
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    art = Path(args.artifacts_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    exp01 = repo / args.exp01_script

    all_variants: List[Dict[str, Any]] = [
        {
            "name": "full_kvi",
            "label": "Full KVI (default DRM / budget)",
            "extra": [],
        },
        {
            "name": "kvi_minimal_prompt",
            "label": "KVI + minimal prompt (no evidence block in user prompt)",
            "extra": ["--kvi_minimal_prompt"],
        },
        {
            "name": "kvi_kv0",
            "label": "KVI with max_kv_triples=0 (no KV injection)",
            "extra": ["--kvi_max_kv_triples", "0"],
        },
        {
            "name": "kvi_drm_loose",
            "label": "KVI drm_threshold=0.0 (looser triple filter)",
            "extra": ["--kvi_drm_threshold", "0.0"],
        },
        {
            "name": "kvi_drm_tight",
            "label": "KVI drm_threshold=0.25 (stricter triple filter)",
            "extra": ["--kvi_drm_threshold", "0.25"],
        },
        {
            "name": "kvi_max1",
            "label": "KVI with max_kv_triples=1 (smaller KV budget)",
            "extra": ["--kvi_max_kv_triples", "1"],
        },
    ]
    if str(args.variants).strip():
        want = {x.strip() for x in str(args.variants).split(",") if x.strip()}
        variants = [v for v in all_variants if v["name"] in want]
        missing = want - {v["name"] for v in variants}
        if missing:
            raise SystemExit(f"Unknown variant names: {sorted(missing)}")
    else:
        variants = all_variants

    rows: List[Dict[str, Any]] = []

    for v in variants:
        name = str(v["name"])
        vdir = out_root / f"variant_{name}"
        vdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(exp01),
            "--dataset",
            str(repo / args.dataset_jsonl),
            "--dataset_name",
            str(args.dataset_name),
            "--model",
            str(repo / args.model),
            "--graph_index",
            str((art / "graph_index.json").resolve()),
            "--triple_kvbank_dir",
            str((art / "triple_kvbank").resolve()),
            "--graph_sentences_jsonl",
            str((art / "sentences.jsonl").resolve()),
            "--methods",
            "graphrag,kvi",
            "--out_dir",
            str(vdir),
            "--limit",
            str(int(args.limit)),
            "--bootstrap_samples",
            str(int(args.bootstrap_samples)),
            "--permutation_samples",
            str(int(args.permutation_samples)),
            "--timeout_s",
            str(int(args.timeout_s)),
            "--em_mode",
            "relaxed",
            "--openqa_mode",
        ]
        cmd.extend(v["extra"])
        print("[ablation] running:", name, flush=True)
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Variant {name} failed rc={proc.returncode}\n"
                f"stdout_tail={proc.stdout[-4000:]}\nstderr_tail={proc.stderr[-4000:]}"
            )
        summ_path = (vdir / "summary.json").resolve()
        summary = json.loads(summ_path.read_text(encoding="utf-8"))
        ems = _em_for_keys(summary, ("graphrag", "kvi"))
        delta = ems.get("kvi", 0.0) - ems.get("graphrag", 0.0)
        try:
            rel_summ = str(summ_path.relative_to(repo))
        except ValueError:
            rel_summ = str(summ_path)
        rows.append(
            {
                "variant": name,
                "label": v["label"],
                "graphrag_em": ems.get("graphrag"),
                "kvi_em": ems.get("kvi"),
                "kvi_minus_graphrag_em": round(delta, 4),
                "summary_path": rel_summ,
            }
        )

    (out_root / "ablation_summary.json").write_text(
        json.dumps({"variants": rows, "n": int(args.limit), "dataset": args.dataset_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = ["## KVI ablation vs GraphRAG (relaxed EM)\n\n"]
    md.append(f"- Dataset: **{args.dataset_name}**\n")
    md.append(f"- N (limit): **{int(args.limit)}**\n")
    md.append("- Methods per run: `graphrag`, `kvi` only\n\n")
    md.append("| Variant | GraphRAG EM | KVI EM | KVI − GraphRAG |\n")
    md.append("|---|---:|---:|---:|\n")
    for r in rows:
        gr = r["graphrag_em"]
        kv = r["kvi_em"]
        d = r["kvi_minus_graphrag_em"]
        md.append(
            f"| {r['label']} | {gr:.1f} | {kv:.1f} | {d:+.1f} |\n"
        )
    (out_root / "ablation_table.md").write_text("".join(md), encoding="utf-8")

    # Short conclusion heuristics
    baseline = next((x for x in rows if x["variant"] == "full_kvi"), rows[0])
    kv0 = next((x for x in rows if x["variant"] == "kvi_kv0"), None)
    minimal = next((x for x in rows if x["variant"] == "kvi_minimal_prompt"), None)

    lines = ["## Conclusion (auto)\n\n"]
    lines.append(
        f"- **Baseline Full KVI** vs GraphRAG: KVI EM **{baseline['kvi_em']:.1f}**, GraphRAG **{baseline['graphrag_em']:.1f}** "
        f"(Δ **{baseline['kvi_minus_graphrag_em']:+.1f}** pts).\n"
    )
    if kv0:
        lines.append(
            f"- **KV off (`max_kv_triples=0`)**: KVI EM **{kv0['kvi_em']:.1f}** vs GraphRAG **{kv0['graphrag_em']:.1f}**. "
            "If this tracks GraphRAG closely, injected KV (vs none) is the main differentiator for this run.\n"
        )
    if minimal:
        lines.append(
            f"- **Minimal prompt**: KVI EM **{minimal['kvi_em']:.1f}** (vs full KVI **{baseline['kvi_em']:.1f}**). "
            "Large swing suggests dual-channel prompt+KV interaction matters.\n"
        )
    lines.append(
        "\n**Interpretation:** Positive Δ means KVI beat GraphRAG on this slice; negative means KV path underperformed. "
        "Synthetic `build_assets_from_dataset` graphs can distort Hotpot-style multi-hop behavior — treat this as pipeline-level evidence, not official benchmark claims.\n"
    )
    (out_root / "CONCLUSION.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_root), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
