#!/usr/bin/env python3
"""
Grid search KVI hyperparameters on MedHopQA_official until KVI EM > GraphRAG EM
(or all configs exhausted). Uses --methods graphrag,kvi only.

Requires: artifacts/medhop_official, resident /infer/graph, venv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _em_pair(summary_path: Path) -> Tuple[float, float, int]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    n = int(data.get("n") or 0)
    g: float | None = None
    k: float | None = None
    for row in data.get("methods") or []:
        mk = row.get("method_key")
        em = float(row.get("em") or 0.0)
        if mk == "graphrag":
            g = em
        elif mk == "kvi":
            k = em
    if g is None or k is None:
        raise SystemExit(f"Missing graphrag/kvi in {summary_path}")
    return g, k, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/zd/dev/KVI"))
    p.add_argument("--port", type=int, default=18888)
    p.add_argument("--stop-when-ahead", action="store_true", default=True)
    p.add_argument("--no-reconcile", action="store_true", help="Disable dual-decode KVI reconcile")
    args = p.parse_args()

    root: Path = args.root
    code = root / "experiments" / "exp01_main_qa" / "code"
    data = root / "experiments" / "exp01_main_qa" / "data" / "medhop_official"
    art = root / "experiments" / "exp01_main_qa" / "artifacts" / "medhop_official"
    results = root / "experiments" / "exp01_main_qa" / "results"
    model = root / "models" / "Qwen2.5-7B-Instruct"
    py = root / "KVI" / "bin" / "python"

    if not py.exists():
        py = Path(sys.executable)

    # (name_suffix, extra_args)
    grid: List[Tuple[str, List[str]]] = [
        # Anchor-only triple budget (walk triples filtered to 0 → subject_anchor KV only)
        ("kvi_t0_d005_rel2", ["--kvi_max_kv_triples", "0", "--kvi_drm_threshold", "0.05", "--kvi_top_k_relations", "2"]),
        ("kvi_t1_d25_rel1", ["--kvi_max_kv_triples", "1", "--kvi_drm_threshold", "0.25", "--kvi_top_k_relations", "1"]),
        ("kvi_t1_d20_rel1", ["--kvi_max_kv_triples", "1", "--kvi_drm_threshold", "0.20", "--kvi_top_k_relations", "1"]),
        ("kvi_t1_d15_rel1", ["--kvi_max_kv_triples", "1", "--kvi_drm_threshold", "0.15", "--kvi_top_k_relations", "1"]),
        ("kvi_t2_d10_rel2", ["--kvi_max_kv_triples", "2", "--kvi_drm_threshold", "0.10", "--kvi_top_k_relations", "2"]),
        ("kvi_t2_d12_rel2", ["--kvi_max_kv_triples", "2", "--kvi_drm_threshold", "0.12", "--kvi_top_k_relations", "2"]),
        ("kvi_t3_d05_rel2", ["--kvi_max_kv_triples", "3", "--kvi_drm_threshold", "0.05", "--kvi_top_k_relations", "2"]),
        # Minimal + thin evidence (post-fix): sometimes reduces prompt–KV conflict
        ("kvi_t2_d08_rel2_min", ["--kvi_max_kv_triples", "2", "--kvi_drm_threshold", "0.08", "--kvi_top_k_relations", "2", "--kvi_minimal_prompt"]),
        ("kvi_t1_d12_rel1_min", ["--kvi_max_kv_triples", "1", "--kvi_drm_threshold", "0.12", "--kvi_top_k_relations", "1", "--kvi_minimal_prompt"]),
    ]

    base_cmd = [
        str(py),
        "-u",
        str(code / "run_exp01.py"),
        "--dataset",
        str(data / "medhop_eval.jsonl"),
        "--dataset_name",
        "MedHopQA_official",
        "--model",
        str(model),
        "--graph_index",
        str(art / "graph_index.json"),
        "--triple_kvbank_dir",
        str(art / "triple_kvbank"),
        "--graph_sentences_jsonl",
        str(art / "sentences.tagged.jsonl"),
        "--methods",
        "graphrag,kvi",
        "--timeout_s",
        "600",
        "--bootstrap_samples",
        "200",
        "--permutation_samples",
        "500",
        "--ann_force_cpu",
        "--inference_service_url",
        f"http://127.0.0.1:{args.port}",
    ]
    if not bool(args.no_reconcile):
        base_cmd.append("--kvi_reconcile_no_kv_decode")

    best: Dict[str, Any] = {"kvi": -1.0, "graphrag": -1.0, "name": "", "out": ""}

    for name, extra in grid:
        out_dir = results / f"medhop_official_kvi_sweep_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summ = out_dir / "summary.json"
        cmd = base_cmd + ["--out_dir", str(out_dir)] + extra
        print(json.dumps({"run": name, "cmd_tail": extra}, ensure_ascii=False), flush=True)
        import os as _os

        env = {**dict(_os.environ), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        subprocess.run(cmd, cwd=str(root), env=env, check=True)
        g_em, k_em, n = _em_pair(summ)
        row = {
            "name": name,
            "out_dir": str(out_dir),
            "n": n,
            "graphrag_em": round(g_em, 4),
            "kvi_em": round(k_em, 4),
            "kvi_minus_graphrag": round(k_em - g_em, 4),
        }
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        if k_em > best["kvi"]:
            best = {**row, "kvi": k_em, "graphrag": g_em}
        if bool(args.stop_when_ahead) and k_em > g_em:
            print(
                json.dumps(
                    {"ok": True, "message": "KVI ahead of GraphRAG", "winner": row},
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            (results / "medhop_official_kvi_sweep_BEST.json").write_text(
                json.dumps({**row, "best_overall_tracked": best}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        (results / "medhop_official_kvi_sweep_BEST.json").write_text(
            json.dumps({"last": row, "best_kvi_so_far": best}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "ok": False,
                "message": "No config in grid beat GraphRAG; best attempt",
                "best_kvi_em": best.get("kvi_em"),
                "best_name": best.get("name"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
