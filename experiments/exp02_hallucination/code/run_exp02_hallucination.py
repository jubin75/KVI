#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set


def _run(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout[-2000:]}\nstderr:\n{p.stderr[-2000:]}")


def _load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_matches(data: Path, *, t_max: int, f_max: int) -> bool:
    mpath = data / "dataset_manifest.json"
    if not mpath.exists():
        return False
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        c = m.get("counts") or {}
        return int(c.get("truthfulqa", 0)) == int(t_max) and int(c.get("fever", 0)) == int(f_max)
    except Exception:
        return False


def _can_reuse_artifacts(ds_art: Path) -> bool:
    """Graph + KV bank + ANN artifacts already built for this dataset."""
    if not (ds_art / "graph_index.json").exists():
        return False
    if not (ds_art / "sentences.tagged.jsonl").exists():
        return False
    kvdir = ds_art / "kvbank_sentences"
    if not kvdir.is_dir():
        return False
    if not (kvdir / "pattern_sidecar").is_dir():
        return False
    tdir = ds_art / "triple_kvbank"
    if not tdir.is_dir():
        return False
    for p in tdir.iterdir():
        if p.suffix == ".pt" or p.name == "manifest.json":
            return True
    return False


def _run_exp01_with_graph_fallback(cmd: list[str], root: Path) -> None:
    try:
        _run(cmd, root)
    except RuntimeError as e:
        msg = str(e)
        if "endpoint=/infer/graph" not in msg and "HTTP Error 500" not in msg:
            raise
        fallback_cmd: list[str] = []
        skip_next = False
        for tok in cmd:
            if skip_next:
                skip_next = False
                continue
            if tok == "--methods":
                fallback_cmd.extend(["--methods", "llm,rag"])
                skip_next = True
                continue
            fallback_cmd.append(tok)
        _run(fallback_cmd, root)


def main() -> None:
    p = argparse.ArgumentParser(description="Run Exp02 hallucination reduction with Exp01 pipeline")
    p.add_argument("--root", default="/home/zd/dev/KVI")
    p.add_argument("--model", default="/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct")
    p.add_argument("--resident_url", default="")
    p.add_argument(
        "--ann_via_resident",
        action="store_true",
        help="Route ANN (/infer/kvi) through the same resident URL; default is graph-only resident + local ANN on CPU to reduce GPU OOM kills.",
    )
    p.add_argument("--build_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--truthfulqa_max", type=int, default=500)
    p.add_argument("--fever_max", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--only_datasets",
        default="truthfulqa,fever",
        help="Comma list: truthfulqa, fever (subset to run build+eval for)",
    )
    p.add_argument(
        "--skip_mirror_and_prepare",
        action="store_true",
        help="Skip download_mirror_datasets + prepare_exp02 when dataset_manifest.json counts match maxima",
    )
    p.add_argument(
        "--reuse_artifacts",
        action="store_true",
        help="Skip per-dataset asset/KV/graph/triple_kv compile when artifacts dir already complete",
    )
    p.add_argument(
        "--resume_eval",
        action="store_true",
        help="Pass --resume to run_exp01: continue from existing predictions.jsonl in each out_dir (validates prefix).",
    )
    args = p.parse_args()

    print(
        json.dumps(
            {
                "exp02_start": True,
                "only_datasets": str(args.only_datasets),
                "skip_mirror_and_prepare": bool(args.skip_mirror_and_prepare),
                "reuse_artifacts": bool(args.reuse_artifacts),
                "ann_via_resident": bool(args.ann_via_resident),
                "resume_eval": bool(args.resume_eval),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    root = Path(args.root)
    code = root / "experiments/exp01_main_qa/code"
    exp2 = root / "experiments/exp02_hallucination"
    data = exp2 / "data"
    art = exp2 / "artifacts"
    res = exp2 / "results"
    for d in [data, art, res]:
        d.mkdir(parents=True, exist_ok=True)

    wanted: Set[str] = {x.strip().lower() for x in str(args.only_datasets).split(",") if x.strip()}
    for w in wanted:
        if w not in {"truthfulqa", "fever"}:
            raise SystemExit(f"Unknown dataset in --only_datasets: {w}")

    skip_prep = bool(args.skip_mirror_and_prepare) and _manifest_matches(
        data, t_max=int(args.truthfulqa_max), f_max=int(args.fever_max)
    )
    if bool(args.skip_mirror_and_prepare) and not skip_prep:
        raise SystemExit(
            f"--skip_mirror_and_prepare set but {data / 'dataset_manifest.json'} missing or counts "
            f"do not match truthfulqa_max={args.truthfulqa_max} fever_max={args.fever_max}"
        )

    if not skip_prep:
        # HF Hub may be blocked; ensure mirror-resolved files are present first.
        _run(
            [
                "python3",
                str(root / "experiments" / "code" / "download_mirror_datasets.py"),
                "--out_dir",
                str(root / "experiments" / "_mirror_data_resolved"),
            ],
            root,
        )

        _run(
            [
                "python3",
                str(exp2 / "code" / "prepare_exp02_datasets.py"),
                "--out_dir",
                str(data),
                "--mirror_root",
                str(root / "experiments" / "_mirror_data_resolved"),
                "--mirror_data_root",
                str(root / "experiments" / "_mirror_data"),
                "--truthfulqa_max",
                str(args.truthfulqa_max),
                "--fever_max",
                str(args.fever_max),
                "--streaming",
            ],
            root,
        )

    datasets: List[tuple[str, Path]] = [
        ("truthfulqa", data / "truthfulqa_eval.jsonl"),
        ("fever", data / "fever_eval.jsonl"),
    ]
    datasets = [pair for pair in datasets if pair[0] in wanted]

    run_summaries: Dict[str, Dict] = {}
    for name, ds in datasets:
        ds_art = art / name
        reuse = bool(args.reuse_artifacts) and _can_reuse_artifacts(ds_art)
        if not reuse:
            _run(
                [
                    "python3",
                    str(code / "build_assets_from_dataset.py"),
                    "--dataset_jsonl",
                    str(ds),
                    "--out_dir",
                    str(ds_art),
                    "--max_examples",
                    str(args.limit if args.limit > 0 else 0),
                ],
                root,
            )
            _run(
                [
                    "python3",
                    str(root / "scripts" / "annotate_sentences_semantic_tags.py"),
                    "--in_jsonl",
                    str(ds_art / "sentences.jsonl"),
                    "--out_jsonl",
                    str(ds_art / "sentences.tagged.jsonl"),
                    "--domain_encoder_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "--semantic_type_specs",
                    str(root / "experiments/exp01_main_qa/artifacts/medhop_n40/kvbank_sentences/pattern_sidecar/semantic_type_specs.json"),
                    "--device",
                    str(args.build_device),
                ],
                root,
            )
            _run(
                [
                    "python3",
                    str(root / "scripts" / "build_kvbank_from_blocks_jsonl.py"),
                    "--blocks_jsonl",
                    str(ds_art / "sentences.tagged.jsonl"),
                    "--disable_enriched",
                    "--out_dir",
                    str(ds_art / "kvbank_sentences"),
                    "--base_llm",
                    str(args.model),
                    "--domain_encoder_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "--layers",
                    "0,1,2,3",
                    "--block_tokens",
                    "128",
                    "--shard_size",
                    "1024",
                    "--device",
                    str(args.build_device),
                    "--dtype",
                    ("bfloat16" if args.build_device == "cuda" else "float32"),
                ],
                root,
            )
            _run(
                [
                    "python3",
                    str(root / "scripts" / "build_knowledge_graph.py"),
                    "--triples_jsonl",
                    str(ds_art / "triples.jsonl"),
                    "--out_graph",
                    str(ds_art / "graph_index.json"),
                ],
                root,
            )
            _run(
                [
                    "python3",
                    str(root / "src/graph/triple_kv_compiler.py"),
                    "--graph_index",
                    str(ds_art / "graph_index.json"),
                    "--model",
                    str(args.model),
                    "--out_dir",
                    str(ds_art / "triple_kvbank"),
                    "--device",
                    str(args.build_device),
                    "--dtype",
                    ("bfloat16" if args.build_device == "cuda" else "float32"),
                ],
                root,
            )

        out_dir = res / f"{name}_fullmethods_qwen25_7b"
        cmd = [
            str(root / "KVI" / "bin" / "python"),
            "-u",
            str(code / "run_exp01.py"),
            "--dataset",
            str(ds),
            "--dataset_name",
            name.upper(),
            "--model",
            str(args.model),
            "--graph_index",
            str(ds_art / "graph_index.json"),
            "--triple_kvbank_dir",
            str(ds_art / "triple_kvbank"),
            "--graph_sentences_jsonl",
            str(ds_art / "sentences.tagged.jsonl"),
            "--ann_kv_dir",
            str(ds_art / "kvbank_sentences"),
            "--ann_sentences_jsonl",
            str(ds_art / "sentences.tagged.jsonl"),
            "--ann_semantic_type_specs",
            str(ds_art / "kvbank_sentences/pattern_sidecar/semantic_type_specs.json"),
            "--ann_pattern_index_dir",
            str(ds_art / "kvbank_sentences/pattern_sidecar"),
            "--ann_sidecar_dir",
            str(ds_art / "kvbank_sentences/pattern_sidecar"),
            "--methods",
            "llm,rag,graphrag,kv_prefix,kvi",
            "--out_dir",
            str(out_dir),
            "--timeout_s",
            "600",
            "--bootstrap_samples",
            "1000",
            "--permutation_samples",
            "2000",
        ]
        if args.resident_url:
            cmd += ["--inference_service_url", str(args.resident_url)]
            t_ann = str(args.resident_url) if args.ann_via_resident else ""
            cmd += ["--ann_inference_service_url", t_ann]
            if not args.ann_via_resident:
                cmd += ["--ann_force_cpu"]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if bool(args.resume_eval):
            cmd += ["--resume"]
        _run_exp01_with_graph_fallback(cmd, root)
        run_summaries[name] = _load_summary(out_dir / "summary.json")

    # Allow partial runs: merge summaries from disk for datasets not in this invocation.
    for ds_key in ("truthfulqa", "fever"):
        if ds_key not in run_summaries:
            summ_path = res / f"{ds_key}_fullmethods_qwen25_7b/summary.json"
            if summ_path.exists():
                run_summaries[ds_key] = _load_summary(summ_path)

    # Hallucination rate proxy = 100 - relaxed EM (stable dataset order)
    out_rows = []
    for ds_name in ("truthfulqa", "fever"):
        summ = run_summaries.get(ds_name)
        if not summ:
            continue
        for m in summ.get("methods", []):
            em = float(m.get("em") or 0.0)
            out_rows.append(
                {
                    "dataset": ds_name,
                    "method_key": m.get("method_key"),
                    "method": m.get("method"),
                    "em": em,
                    "hallucination_rate": round(100.0 - em, 4),
                }
            )
    out = {"note": "Hallucination Rate here is proxy: 100 - relaxed EM.", "rows": out_rows}
    (res / "hallucination_proxy_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["## Experiment 2 — Hallucination Reduction (Proxy)\n\n", "Hallucination Rate (%) = `100 - relaxed EM`.\n\n"]
    md += ["| Dataset | Method | EM (%) | Hallucination Rate (%) |\n", "|---|---|---:|---:|\n"]
    for r in out_rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['em']:.1f} | {r['hallucination_rate']:.1f} |\n")
    (res / "hallucination_proxy_summary.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

