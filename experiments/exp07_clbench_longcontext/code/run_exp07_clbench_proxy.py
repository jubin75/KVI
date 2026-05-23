#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset


DEFAULT_REGIMES = "short:4000,medium:16000,long:32000,extreme:64000"
DEFAULT_METHODS = "llm,rag,graphrag,kv_prefix,kvi_triple_legacy,kvi_schema_writer,kvi_noinject_planner"


@dataclass
class SourceExample:
    source_id: str
    prompt: str
    answer: str
    answers: List[str]
    prompt_tokens: int
    prompt_chars: int
    hop_count_proxy: int
    evidence_sparsity_proxy: str
    metadata: Dict[str, Any]


def _run(cmd: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}"
        )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _parse_regimes(spec: str) -> List[Tuple[str, int]]:
    regimes: List[Tuple[str, int]] = []
    for part in str(spec or "").split(","):
        item = part.strip()
        if not item:
            continue
        name, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"Invalid regime spec: {item}")
        regimes.append((name.strip(), int(value.strip())))
    if not regimes:
        raise ValueError("No context-length regimes configured")
    return regimes


def _make_token_counter(model_name_or_path: str) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        def _count(text: str) -> int:
            return int(len(tok.encode(text, add_special_tokens=False)))

        return _count
    except Exception:
        pattern = re.compile(r"\S+")

        def _fallback(text: str) -> int:
            return len(pattern.findall(text))

        return _fallback


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in (
            "answer",
            "reference_answer",
            "reference",
            "rubric",
            "content",
            "text",
            "ideal",
            "target",
        ):
            text = _first_text(value.get(key))
            if text:
                return text
        for item in value.values():
            text = _first_text(item)
            if text:
                return text
    return ""


def _extract_user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "user":
            return str(msg.get("content") or "").strip()
    return ""


def _estimate_hop_count(prompt: str) -> int:
    text = str(prompt or "").lower()
    score = 2
    score += min(2, len(re.findall(r"\b(?:then|after|before|next|finally)\b", text)))
    score += min(2, len(re.findall(r"\b(?:compare|combine|synthesize|cross-reference)\b", text)))
    score += min(2, len(re.findall(r"\n\s*(?:[-*]|\d+\.)\s+", prompt)))
    return max(2, min(6, score))


def _estimate_sparsity(prompt: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", str(prompt or "")) if p.strip()]
    if len(paragraphs) >= 12:
        return "sparse"
    if len(paragraphs) >= 6:
        return "medium"
    return "dense"


def _split_distractor_blocks(text: str, *, max_chars: int = 1200) -> List[str]:
    raw_parts = [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]
    blocks: List[str] = []
    for part in raw_parts:
        if len(part) <= max_chars:
            blocks.append(part)
            continue
        start = 0
        while start < len(part):
            end = min(len(part), start + max_chars)
            blocks.append(part[start:end].strip())
            start = end
    return [b for b in blocks if len(b) >= 80]


def _inflate_prompt(
    *,
    base_prompt: str,
    target_tokens: int,
    base_tokens: int,
    distractor_blocks: Sequence[str],
    token_count: Callable[[str], int],
) -> Tuple[str, int, int]:
    if base_tokens >= target_tokens or not distractor_blocks:
        return base_prompt, 0, base_tokens
    docs: List[str] = []
    for idx, block in enumerate(distractor_blocks, start=1):
        docs.append(f"[Distractor Doc {idx}]\n{block}")
        candidate = (
            "Additional retrieved documents follow. Ignore them unless they help answer the original task.\n\n"
            + "\n\n".join(docs)
            + "\n\n[Original Task]\n"
            + base_prompt
        )
        if token_count(candidate) >= target_tokens:
            return candidate, idx, token_count(candidate)
    final_text = (
        "Additional retrieved documents follow. Ignore them unless they help answer the original task.\n\n"
        + "\n\n".join(docs)
        + "\n\n[Original Task]\n"
        + base_prompt
    )
    return final_text, len(docs), token_count(final_text)


def _load_source_examples(
    *,
    root: Path,
    max_examples: int,
    token_count: Callable[[str], int],
) -> List[SourceExample]:
    src_parquet_candidates = [
        root / "experiments/_mirror_data_resolved/cl_bench_train_0000.parquet",
        root / "experiments/_mirror_data/CL-bench-parquet/default/train/0000.parquet",
    ]
    src_jsonl_candidates = [
        root / "experiments/_mirror_data_resolved/cl_bench.jsonl",
        root / "experiments/_mirror_data/CL-bench/CL-bench.jsonl",
        root / "experiments/_mirror_data/cl_bench.jsonl",
    ]
    src_parquet = next((p for p in src_parquet_candidates if p.exists()), None)
    src_jsonl = next((p for p in src_jsonl_candidates if p.exists()), None)
    if src_parquet is None and src_jsonl is None:
        raise RuntimeError(
            "Missing local CL-bench mirror files. Expected one of: "
            f"{src_parquet_candidates[0]}, {src_parquet_candidates[1]}, "
            f"{src_jsonl_candidates[0]}, {src_jsonl_candidates[1]}, {src_jsonl_candidates[2]}"
        )
    if src_parquet is not None:
        iterable = load_dataset("parquet", data_files=str(src_parquet), split="train")
    else:
        iterable = load_dataset("json", data_files=str(src_jsonl), split="train")

    rows: List[SourceExample] = []
    for idx, ex in enumerate(iterable):
        prompt = _extract_user_prompt(ex.get("messages"))
        answer = _first_text(ex.get("rubrics"))
        if not prompt or not answer:
            continue
        prompt_tokens = int(token_count(prompt))
        rows.append(
            SourceExample(
                source_id=f"clbench_src_{idx}",
                prompt=prompt,
                answer=answer,
                answers=[answer],
                prompt_tokens=prompt_tokens,
                prompt_chars=len(prompt),
                hop_count_proxy=_estimate_hop_count(prompt),
                evidence_sparsity_proxy=_estimate_sparsity(prompt),
                metadata={
                    "source_index": idx,
                    "prompt_line_count": len([x for x in prompt.splitlines() if x.strip()]),
                    "rubric_count": len(ex.get("rubrics") or []) if isinstance(ex.get("rubrics"), list) else 0,
                    "task": str(ex.get("task") or ""),
                    "category": str(ex.get("category") or ""),
                    "subtask": str(ex.get("subtask") or ex.get("sub_task") or ""),
                },
            )
        )
        if max_examples > 0 and len(rows) >= max_examples:
            break
    if not rows:
        raise RuntimeError("No usable CL-bench rows found after filtering")
    return rows


def _build_variant_dataset(
    *,
    examples: Sequence[SourceExample],
    regimes: Sequence[Tuple[str, int]],
    token_count: Callable[[str], int],
    seed: int,
    out_jsonl: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    block_pool_by_source = {ex.source_id: _split_distractor_blocks(ex.prompt) for ex in examples}
    all_rows: List[Dict[str, Any]] = []
    regime_counts: Dict[str, int] = {name: 0 for name, _ in regimes}

    for ex_idx, ex in enumerate(examples):
        local_rng = random.Random(seed + ex_idx)
        distractor_candidates: List[str] = []
        other_ids = [x.source_id for x in examples if x.source_id != ex.source_id]
        rng.shuffle(other_ids)
        for other_id in other_ids:
            distractor_candidates.extend(block_pool_by_source.get(other_id, []))
        if distractor_candidates:
            local_rng.shuffle(distractor_candidates)

        eligible_regimes = [(name, budget) for name, budget in regimes if budget >= ex.prompt_tokens]
        if not eligible_regimes:
            eligible_regimes = [max(regimes, key=lambda item: item[1])]

        for regime_name, target_tokens in eligible_regimes:
            prompt_text, distractor_docs, actual_tokens = _inflate_prompt(
                base_prompt=ex.prompt,
                target_tokens=target_tokens,
                base_tokens=ex.prompt_tokens,
                distractor_blocks=distractor_candidates,
                token_count=token_count,
            )
            row = {
                "id": f"{ex.source_id}_{regime_name}",
                "question": prompt_text,
                "answer": ex.answer,
                "answers": ex.answers,
                "dataset": "CL_bench_longcontext_proxy",
                "source_id": ex.source_id,
                "length_regime": regime_name,
                "target_context_tokens": int(target_tokens),
                "context_tokens_est": int(actual_tokens),
                "source_prompt_tokens": int(ex.prompt_tokens),
                "source_prompt_chars": int(ex.prompt_chars),
                "distractor_docs_used": int(distractor_docs),
                "hop_count_proxy": int(ex.hop_count_proxy),
                "evidence_sparsity_proxy": str(ex.evidence_sparsity_proxy),
                "metadata": ex.metadata,
            }
            all_rows.append(row)
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

    _write_jsonl(out_jsonl, all_rows)
    manifest = {
        "dataset": "CL_bench_longcontext_proxy",
        "source_examples": len(examples),
        "variant_examples": len(all_rows),
        "seed": int(seed),
        "regimes": [{"name": name, "target_tokens": int(tokens)} for name, tokens in regimes],
        "regime_counts": regime_counts,
        "notes": [
            "Variants preserve the original CL-bench user prompt and pad distractor documents ahead of it.",
            "This is a long-context proxy benchmark aligned to Exp07, not the final paper-grade schema-first benchmark.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _resolve_python(root: Path) -> str:
    for cand in (root / "KVI/bin/python", root / ".venv/bin/python"):
        if cand.exists():
            return str(cand)
    return sys.executable or "python3"


def _ensure_clbench_mirror(root: Path) -> None:
    needed = root / "experiments/_mirror_data_resolved/cl_bench.jsonl"
    if needed.exists() and needed.stat().st_size > 0:
        return
    _run(
        [
            _resolve_python(root),
            str(root / "experiments/code/download_mirror_datasets.py"),
            "--out_dir",
            str(root / "experiments/_mirror_data_resolved"),
        ],
        root,
    )


def _build_assets(root: Path, dataset_jsonl: Path, artifacts_dir: Path, model: str, build_device: str) -> None:
    py = _resolve_python(root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            py,
            str(root / "experiments/exp01_main_qa/code/build_assets_from_dataset.py"),
            "--dataset_jsonl",
            str(dataset_jsonl),
            "--out_dir",
            str(artifacts_dir),
        ],
        root,
    )
    _run(
        [
            py,
            str(root / "scripts/annotate_sentences_semantic_tags.py"),
            "--in_jsonl",
            str(artifacts_dir / "sentences.jsonl"),
            "--out_jsonl",
            str(artifacts_dir / "sentences.tagged.jsonl"),
            "--domain_encoder_model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--device",
            str(build_device),
        ],
        root,
    )
    _run(
        [
            py,
            str(root / "scripts/build_kvbank_from_blocks_jsonl.py"),
            "--blocks_jsonl",
            str(artifacts_dir / "sentences.tagged.jsonl"),
            "--disable_enriched",
            "--out_dir",
            str(artifacts_dir / "kvbank_sentences"),
            "--base_llm",
            str(model),
            "--domain_encoder_model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--layers",
            "0,1,2,3",
            "--block_tokens",
            "128",
            "--shard_size",
            "1024",
            "--device",
            str(build_device),
            "--dtype",
            ("bfloat16" if build_device == "cuda" else "float32"),
        ],
        root,
    )
    _run(
        [
            py,
            str(root / "scripts/build_knowledge_graph.py"),
            "--triples_jsonl",
            str(artifacts_dir / "triples.jsonl"),
            "--out_graph",
            str(artifacts_dir / "graph_index.json"),
        ],
        root,
    )
    _run(
        [
            py,
            str(root / "src/graph/triple_kv_compiler.py"),
            "--graph_index",
            str(artifacts_dir / "graph_index.json"),
            "--model",
            str(model),
            "--out_dir",
            str(artifacts_dir / "triple_kvbank"),
            "--device",
            str(build_device),
            "--dtype",
            ("bfloat16" if build_device == "cuda" else "float32"),
        ],
        root,
    )


def _run_exp01_eval(
    *,
    root: Path,
    dataset_jsonl: Path,
    artifacts_dir: Path,
    out_dir: Path,
    model: str,
    methods: str,
    resident_url: str,
    limit: int,
    max_new_tokens: int,
    timeout_s: int,
    bootstrap_samples: int,
    permutation_samples: int,
) -> None:
    py = _resolve_python(root)
    cmd = [
        py,
        "-u",
        str(root / "experiments/exp01_main_qa/code/run_exp01.py"),
        "--dataset",
        str(dataset_jsonl),
        "--dataset_name",
        "CL_bench_longcontext_proxy",
        "--model",
        str(model),
        "--graph_index",
        str(artifacts_dir / "graph_index.json"),
        "--triple_kvbank_dir",
        str(artifacts_dir / "triple_kvbank"),
        "--graph_sentences_jsonl",
        str(artifacts_dir / "sentences.tagged.jsonl"),
        "--ann_kv_dir",
        str(artifacts_dir / "kvbank_sentences"),
        "--ann_sentences_jsonl",
        str(artifacts_dir / "sentences.tagged.jsonl"),
        "--ann_semantic_type_specs",
        str(artifacts_dir / "kvbank_sentences/pattern_sidecar/semantic_type_specs.json"),
        "--ann_pattern_index_dir",
        str(artifacts_dir / "kvbank_sentences/pattern_sidecar"),
        "--ann_sidecar_dir",
        str(artifacts_dir / "kvbank_sentences/pattern_sidecar"),
        "--methods",
        methods,
        "--out_dir",
        str(out_dir),
        "--timeout_s",
        str(timeout_s),
        "--max_new_tokens",
        str(max_new_tokens),
        "--bootstrap_samples",
        str(bootstrap_samples),
        "--permutation_samples",
        str(permutation_samples),
        "--resume",
    ]
    if resident_url:
        cmd += ["--inference_service_url", resident_url, "--ann_inference_service_url", resident_url]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    _run(cmd, root)


def _aggregate_by_regime(
    *,
    dataset_rows: Sequence[Dict[str, Any]],
    predictions_rows: Sequence[Dict[str, Any]],
    out_json: Path,
    out_csv: Path,
    out_md: Path,
    methods: Sequence[str],
) -> Dict[str, Any]:
    meta_by_id = {str(row.get("id")): row for row in dataset_rows}
    regimes = [str(row.get("length_regime") or "unknown") for row in dataset_rows]
    ordered_regimes = [x for x in ["short", "medium", "long", "extreme"] if x in regimes]
    if "unknown" in regimes and "unknown" not in ordered_regimes:
        ordered_regimes.append("unknown")

    summary: Dict[str, Any] = {"by_regime": {}, "by_hop_proxy": {}}
    for pred in predictions_rows:
        meta = meta_by_id.get(str(pred.get("id")))
        if not meta:
            continue
        regime = str(meta.get("length_regime") or "unknown")
        hop_key = str(meta.get("hop_count_proxy") or "unknown")
        em_map = pred.get("em") or {}
        f1_map = pred.get("f1") or {}
        summary["by_regime"].setdefault(regime, {"n": 0, "metrics": {}})
        summary["by_regime"][regime]["n"] += 1
        summary["by_hop_proxy"].setdefault(hop_key, {"n": 0, "metrics": {}})
        summary["by_hop_proxy"][hop_key]["n"] += 1
        for method in methods:
            summary["by_regime"][regime]["metrics"].setdefault(method, {"em": [], "f1": []})
            summary["by_hop_proxy"][hop_key]["metrics"].setdefault(method, {"em": [], "f1": []})
            summary["by_regime"][regime]["metrics"][method]["em"].append(float(em_map.get(method, 0.0)))
            summary["by_regime"][regime]["metrics"][method]["f1"].append(float(f1_map.get(method, 0.0)))
            summary["by_hop_proxy"][hop_key]["metrics"][method]["em"].append(float(em_map.get(method, 0.0)))
            summary["by_hop_proxy"][hop_key]["metrics"][method]["f1"].append(float(f1_map.get(method, 0.0)))

    def _finalize(section: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for bucket, payload in section.items():
            out[bucket] = {"n": int(payload["n"]), "metrics": {}}
            for method, values in payload["metrics"].items():
                em_vals = values["em"]
                f1_vals = values["f1"]
                out[bucket]["metrics"][method] = {
                    "em": (100.0 * sum(em_vals) / len(em_vals) if em_vals else 0.0),
                    "f1": (100.0 * sum(f1_vals) / len(f1_vals) if f1_vals else 0.0),
                }
        return out

    finalized = {
        "by_regime": _finalize(summary["by_regime"]),
        "by_hop_proxy": _finalize(summary["by_hop_proxy"]),
    }
    out_json.write_text(json.dumps(finalized, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket_type", "bucket", "n", "method", "em", "f1"])
        for bucket_type in ("by_regime", "by_hop_proxy"):
            section = finalized[bucket_type]
            bucket_order = ordered_regimes if bucket_type == "by_regime" else sorted(section.keys(), key=lambda x: (x == "unknown", x))
            for bucket in bucket_order:
                if bucket not in section:
                    continue
                for method in methods:
                    if method not in section[bucket]["metrics"]:
                        continue
                    metric = section[bucket]["metrics"][method]
                    writer.writerow(
                        [
                            bucket_type,
                            bucket,
                            section[bucket]["n"],
                            method,
                            f"{metric['em']:.2f}",
                            f"{metric['f1']:.2f}",
                        ]
                    )

    lines: List[str] = []
    lines.append("# Exp07 CL-Bench Long-Context Proxy Summary\n\n")
    lines.append("## By Length Regime\n\n")
    lines.append("| Regime | N | Method | EM | F1 |\n")
    lines.append("|---|---:|---|---:|---:|\n")
    for regime in ordered_regimes:
        if regime not in finalized["by_regime"]:
            continue
        payload = finalized["by_regime"][regime]
        for method in methods:
            metric = payload["metrics"].get(method)
            if not metric:
                continue
            lines.append(
                f"| {regime} | {payload['n']} | {method} | {metric['em']:.2f} | {metric['f1']:.2f} |\n"
            )
    lines.append("\n## By Hop Proxy\n\n")
    lines.append("| Hop Proxy | N | Method | EM | F1 |\n")
    lines.append("|---|---:|---|---:|---:|\n")
    for hop_key in sorted(finalized["by_hop_proxy"].keys(), key=lambda x: (x == "unknown", x)):
        payload = finalized["by_hop_proxy"][hop_key]
        for method in methods:
            metric = payload["metrics"].get(method)
            if not metric:
                continue
            lines.append(
                f"| {hop_key} | {payload['n']} | {method} | {metric['em']:.2f} | {metric['f1']:.2f} |\n"
            )
    out_md.write_text("".join(lines), encoding="utf-8")
    return finalized


def main() -> None:
    p = argparse.ArgumentParser(description="Run Exp07 CL-bench long-context proxy benchmark")
    p.add_argument("--root", default="/home/zd/dev/KVI")
    p.add_argument("--model", default="/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct")
    p.add_argument("--resident_url", default="")
    p.add_argument("--build_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max_examples", type=int, default=120)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regimes", default=DEFAULT_REGIMES)
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--timeout_s", type=int, default=900)
    p.add_argument("--bootstrap_samples", type=int, default=500)
    p.add_argument("--permutation_samples", type=int, default=1000)
    p.add_argument("--prepare_only", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    exp = root / "experiments/exp07_clbench_longcontext"
    data_dir = exp / "data"
    artifacts_dir = exp / "artifacts/clbench_proxy_v2"
    results_dir = exp / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _ensure_clbench_mirror(root)
    token_count = _make_token_counter(str(args.model))
    regimes = _parse_regimes(str(args.regimes))
    source_examples = _load_source_examples(
        root=root,
        max_examples=int(args.max_examples),
        token_count=token_count,
    )

    dataset_jsonl = data_dir / "clbench_longcontext_proxy_eval.jsonl"
    manifest_json = data_dir / "clbench_longcontext_proxy_manifest.json"
    manifest = _build_variant_dataset(
        examples=source_examples,
        regimes=regimes,
        token_count=token_count,
        seed=int(args.seed),
        out_jsonl=dataset_jsonl,
        manifest_path=manifest_json,
    )

    _build_assets(
        root=root,
        dataset_jsonl=dataset_jsonl,
        artifacts_dir=artifacts_dir,
        model=str(args.model),
        build_device=str(args.build_device),
    )

    if args.prepare_only:
        print(json.dumps({"manifest": manifest, "dataset_jsonl": str(dataset_jsonl)}, ensure_ascii=False, indent=2))
        return

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    out_dir = results_dir / "clbench_proxy_fullmethods_qwen25_7b"
    _run_exp01_eval(
        root=root,
        dataset_jsonl=dataset_jsonl,
        artifacts_dir=artifacts_dir,
        out_dir=out_dir,
        model=str(args.model),
        methods=",".join(methods),
        resident_url=str(args.resident_url),
        limit=int(args.limit),
        max_new_tokens=int(args.max_new_tokens),
        timeout_s=int(args.timeout_s),
        bootstrap_samples=int(args.bootstrap_samples),
        permutation_samples=int(args.permutation_samples),
    )

    predictions = _read_jsonl(out_dir / "predictions.jsonl")
    dataset_rows = _read_jsonl(dataset_jsonl)
    summary = _aggregate_by_regime(
        dataset_rows=dataset_rows,
        predictions_rows=predictions,
        out_json=results_dir / "clbench_proxy_length_bucket_summary.json",
        out_csv=results_dir / "clbench_proxy_length_bucket_summary.csv",
        out_md=results_dir / "clbench_proxy_length_bucket_summary.md",
        methods=methods,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
