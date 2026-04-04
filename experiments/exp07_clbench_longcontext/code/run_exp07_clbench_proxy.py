#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def _run(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout[-2000:]}\nstderr:\n{p.stderr[-2000:]}")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prepare(root: Path, out_jsonl: Path, max_n: int) -> Dict[str, Any]:
    src_parquet_candidates = [
        root / "experiments/_mirror_data_resolved/cl_bench_train_0000.parquet",
        root / "experiments/_mirror_data/CL-bench-parquet/default/train/0000.parquet",
    ]
    src_jsonl_candidates = [
        root / "experiments/_mirror_data_resolved/cl_bench.jsonl",
        root / "experiments/_mirror_data/CL-bench/CL-bench.jsonl",
    ]
    src_parquet = next((p for p in src_parquet_candidates if p.exists()), None)
    src_jsonl = next((p for p in src_jsonl_candidates if p.exists()), None)
    if src_parquet is None and src_jsonl is None:
        raise RuntimeError(
            "Missing local CL-bench mirror files. "
            "Expected one of: "
            f"{src_parquet_candidates[0]}, {src_parquet_candidates[1]}, "
            f"{src_jsonl_candidates[0]}, {src_jsonl_candidates[1]}"
        )
    rows: List[Dict[str, Any]] = []
    lens: List[int] = []
    if src_parquet is not None:
        iterable = load_dataset("parquet", data_files=str(src_parquet), split="train")
    else:
        iterable = load_dataset("json", data_files=str(src_jsonl), split="train")
    for i, ex in enumerate(iterable):
        msgs = ex.get("messages") or []
        rub = ex.get("rubrics") or []
        user_content = ""
        for m in msgs:
            if isinstance(m, dict) and str(m.get("role")) == "user":
                user_content = str(m.get("content") or "")
                break
        if not user_content or not isinstance(rub, list) or not rub:
            continue
        # Proxy target: first rubric string (CL-bench official metric uses rubric scoring;
        # here we keep a local comparable proxy for method deltas).
        ans = str(rub[0]).strip()
        if not ans:
            continue
        q = user_content.strip()
        rows.append(
            {
                "id": f"clbench_{i}",
                "question": q,
                "answer": ans,
                "answers": [ans],
                "dataset": "CL_bench_proxy",
                "context_len_chars": len(q),
            }
        )
        lens.append(len(q))
        if max_n and len(rows) >= max_n:
            break
    _write_jsonl(out_jsonl, rows)
    lens_sorted = sorted(lens)
    q1 = lens_sorted[len(lens_sorted) // 3] if lens_sorted else 0
    q2 = lens_sorted[(2 * len(lens_sorted)) // 3] if lens_sorted else 0
    return {"n": len(rows), "q1": q1, "q2": q2}


def main() -> None:
    p = argparse.ArgumentParser(description="Run CL-bench long-context proxy with Exp01 methods")
    p.add_argument("--root", default="/home/zd/dev/KVI")
    p.add_argument("--model", default="/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct")
    p.add_argument("--resident_url", default="")
    p.add_argument("--build_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max_examples", type=int, default=300)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    exp = root / "experiments/exp07_clbench_longcontext"
    data = exp / "data"
    art = exp / "artifacts"
    res = exp / "results"
    code = root / "experiments/exp01_main_qa/code"
    for d in [data, art, res]:
        d.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python3",
            str(root / "experiments" / "code" / "download_mirror_datasets.py"),
            "--out_dir",
            str(root / "experiments" / "_mirror_data_resolved"),
        ],
        root,
    )

    ds_jsonl = data / "clbench_proxy_eval.jsonl"
    bins = _prepare(root, ds_jsonl, int(args.max_examples))

    _run(["python3", str(code / "build_assets_from_dataset.py"), "--dataset_jsonl", str(ds_jsonl), "--out_dir", str(art)], root)
    _run(
        [
            "python3",
            str(root / "scripts/annotate_sentences_semantic_tags.py"),
            "--in_jsonl",
            str(art / "sentences.jsonl"),
            "--out_jsonl",
            str(art / "sentences.tagged.jsonl"),
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
            str(root / "scripts/build_kvbank_from_blocks_jsonl.py"),
            "--blocks_jsonl",
            str(art / "sentences.tagged.jsonl"),
            "--disable_enriched",
            "--out_dir",
            str(art / "kvbank_sentences"),
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
    _run(["python3", str(root / "scripts/build_knowledge_graph.py"), "--triples_jsonl", str(art / "triples.jsonl"), "--out_graph", str(art / "graph_index.json")], root)
    _run(
        [
            "python3",
            str(root / "src/graph/triple_kv_compiler.py"),
            "--graph_index",
            str(art / "graph_index.json"),
            "--model",
            str(args.model),
            "--out_dir",
            str(art / "triple_kvbank"),
            "--device",
            str(args.build_device),
            "--dtype",
            ("bfloat16" if args.build_device == "cuda" else "float32"),
        ],
        root,
    )

    out_dir = res / "clbench_proxy_fullmethods_qwen25_7b"
    cmd = [
        str(root / "KVI/bin/python"),
        "-u",
        str(code / "run_exp01.py"),
        "--dataset",
        str(ds_jsonl),
        "--dataset_name",
        "CL_bench_proxy",
        "--model",
        str(args.model),
        "--graph_index",
        str(art / "graph_index.json"),
        "--triple_kvbank_dir",
        str(art / "triple_kvbank"),
        "--graph_sentences_jsonl",
        str(art / "sentences.tagged.jsonl"),
        "--ann_kv_dir",
        str(art / "kvbank_sentences"),
        "--ann_sentences_jsonl",
        str(art / "sentences.tagged.jsonl"),
        "--ann_semantic_type_specs",
        str(art / "kvbank_sentences/pattern_sidecar/semantic_type_specs.json"),
        "--ann_pattern_index_dir",
        str(art / "kvbank_sentences/pattern_sidecar"),
        "--ann_sidecar_dir",
        str(art / "kvbank_sentences/pattern_sidecar"),
        "--methods",
        "llm,rag,graphrag,kv_prefix,kvi",
        "--out_dir",
        str(out_dir),
        "--timeout_s",
        "600",
        "--bootstrap_samples",
        "500",
        "--permutation_samples",
        "1000",
    ]
    if args.resident_url:
        cmd += ["--inference_service_url", args.resident_url, "--ann_inference_service_url", args.resident_url]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    try:
        _run(cmd, root)
    except RuntimeError as e:
        msg = str(e)
        if "endpoint=/infer/graph" not in msg and "HTTP Error 500" not in msg:
            raise
        # Keep experiment alive when graph endpoint is unstable.
        fallback_cmd = []
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

    # Length-bin proxy analysis.
    preds = out_dir / "predictions.jsonl"
    rows = [json.loads(x) for x in preds.read_text(encoding="utf-8").splitlines() if x.strip()]
    q_len = {}
    for r in rows:
        q = str(r.get("question") or "")
        q_len[str(r.get("id"))] = len(q)
    q1, q2 = int(bins["q1"]), int(bins["q2"])
    bucket = lambda l: "short" if l <= q1 else ("medium" if l <= q2 else "long")
    agg: Dict[str, Dict[str, List[int]]] = {}
    for r in rows:
        b = bucket(q_len.get(str(r.get("id")), 0))
        em = r.get("em") or {}
        agg.setdefault(b, {})
        for m, v in em.items():
            agg[b].setdefault(m, []).append(int(v))
    out = {"bins": {"q1": q1, "q2": q2}, "bucket_em": {}}
    for b, mm in agg.items():
        out["bucket_em"][b] = {m: (100.0 * sum(vs) / len(vs) if vs else 0.0) for m, vs in mm.items()}
    (res / "clbench_proxy_length_bucket_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

