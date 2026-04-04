#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict


def _run(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout[-2000:]}\nstderr:\n{p.stderr[-2000:]}")


def _load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    p.add_argument("--build_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--truthfulqa_max", type=int, default=500)
    p.add_argument("--fever_max", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    code = root / "experiments/exp01_main_qa/code"
    exp2 = root / "experiments/exp02_hallucination"
    data = exp2 / "data"
    art = exp2 / "artifacts"
    res = exp2 / "results"
    for d in [data, art, res]:
        d.mkdir(parents=True, exist_ok=True)

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

    datasets = [
        ("truthfulqa", data / "truthfulqa_eval.jsonl"),
        ("fever", data / "fever_eval.jsonl"),
    ]

    run_summaries: Dict[str, Dict] = {}
    for name, ds in datasets:
        ds_art = art / name
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
            cmd += ["--inference_service_url", args.resident_url, "--ann_inference_service_url", args.resident_url]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        _run_exp01_with_graph_fallback(cmd, root)
        run_summaries[name] = _load_summary(out_dir / "summary.json")

    # Hallucination rate proxy = 100 - relaxed EM
    out_rows = []
    for ds_name, summ in run_summaries.items():
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

