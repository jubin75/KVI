#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics import exact_match


@dataclass
class Example:
    id: str
    question: str
    answer: str


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Example]:
    out: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            ex = Example(
                id=str(obj.get("id") or ""),
                question=str(obj.get("question") or ""),
                answer=str(obj.get("answer") or ""),
            )
            if ex.id and ex.question:
                out.append(ex)
            if limit and len(out) >= limit:
                break
    return out


def _run_graph_inference(
    *,
    repo_root: Path,
    model: str,
    question: str,
    graph_index: Path,
    triple_kvbank_dir: Optional[Path],
    enable_kvi: bool,
    sentences_jsonl: Optional[Path],
    max_new_tokens: int,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "external_kv_injection" / "scripts" / "run_graph_inference.py"),
        "--model",
        model,
        "--prompt",
        question,
        "--graph_index",
        str(graph_index),
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    if sentences_jsonl:
        cmd += ["--sentences_jsonl", str(sentences_jsonl)]
    if enable_kvi:
        cmd += ["--enable_kvi"]
        if triple_kvbank_dir:
            cmd += ["--triple_kvbank_dir", str(triple_kvbank_dir)]
    # stdout is a single JSON object
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    if p.returncode != 0:
        raise RuntimeError(
            "run_graph_inference failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={p.stdout[-2000:]}\n"
            f"stderr={p.stderr[-2000:]}"
        )
    try:
        return json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(
            "Failed to parse JSON from run_graph_inference\n"
            f"error={e}\n"
            f"stdout_tail={p.stdout[-2000:]}\n"
            f"stderr_tail={p.stderr[-2000:]}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 01: Main QA performance (EM)")
    p.add_argument("--dataset", required=True, help="JSONL: {id, question, answer}")
    p.add_argument("--model", required=True, help="Base LLM model path/name")
    p.add_argument("--graph_index", required=True, help="graph_index.json path")
    p.add_argument("--triple_kvbank_dir", default="", help="triple_kvbank directory (optional)")
    p.add_argument("--sentences_jsonl", default="", help="sentences.jsonl for hybrid text fallback (optional)")
    p.add_argument("--out_dir", required=True, help="Output directory for results")
    p.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[3]  # .../external_kv_injection/experiments/exp01_main_qa/code
    dataset_path = Path(args.dataset)
    graph_index = Path(args.graph_index)
    triple_kvbank_dir = Path(args.triple_kvbank_dir) if args.triple_kvbank_dir else None
    sentences_jsonl = Path(args.sentences_jsonl) if args.sentences_jsonl else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _read_jsonl(dataset_path, limit=(args.limit or None))
    if not examples:
        raise SystemExit(f"Empty dataset: {dataset_path}")

    preds_path = out_dir / "predictions.jsonl"
    rows: List[Dict[str, Any]] = []

    # Aggregate counters
    n = 0
    em_llm = 0
    em_graphrag = 0
    em_kvi = 0

    with preds_path.open("w", encoding="utf-8") as wf:
        for ex in examples:
            n += 1

            # GraphRAG run (KVI disabled) — also returns a prompt-only baseline.
            out_gr = _run_graph_inference(
                repo_root=repo_root,
                model=args.model,
                question=ex.question,
                graph_index=graph_index,
                triple_kvbank_dir=triple_kvbank_dir,
                enable_kvi=False,
                sentences_jsonl=sentences_jsonl,
                max_new_tokens=args.max_new_tokens,
            )
            pred_llm = str(out_gr.get("base_llm_result") or "")
            pred_gr = str(out_gr.get("diagnosis_result_raw") or out_gr.get("diagnosis_result") or "")

            # KVI run (KVI enabled, if kvbank is available)
            out_kvi = _run_graph_inference(
                repo_root=repo_root,
                model=args.model,
                question=ex.question,
                graph_index=graph_index,
                triple_kvbank_dir=triple_kvbank_dir,
                enable_kvi=True,
                sentences_jsonl=sentences_jsonl,
                max_new_tokens=args.max_new_tokens,
            )
            pred_kvi = str(out_kvi.get("diagnosis_result_raw") or out_kvi.get("diagnosis_result") or "")

            em_llm_i = exact_match(pred_llm, ex.answer)
            em_gr_i = exact_match(pred_gr, ex.answer)
            em_kvi_i = exact_match(pred_kvi, ex.answer)

            em_llm += em_llm_i
            em_graphrag += em_gr_i
            em_kvi += em_kvi_i

            rec = {
                "id": ex.id,
                "question": ex.question,
                "gold": ex.answer,
                "pred_llm": pred_llm,
                "pred_graphrag": pred_gr,
                "pred_kvi": pred_kvi,
                "em_llm": em_llm_i,
                "em_graphrag": em_gr_i,
                "em_kvi": em_kvi_i,
                "debug": {
                    "graphrag": {
                        "intent": out_gr.get("intent"),
                        "matched_entities": (out_gr.get("graph_debug") or {}).get("matched_entities", []),
                        "kv_enabled": (out_gr.get("kv_injection_debug") or {}).get("enabled", False),
                    },
                    "kvi": {
                        "intent": out_kvi.get("intent"),
                        "matched_entities": (out_kvi.get("graph_debug") or {}).get("matched_entities", []),
                        "kv_enabled": (out_kvi.get("kv_injection_debug") or {}).get("enabled", False),
                        "selected_triples": (out_kvi.get("kv_injection_debug") or {}).get("selected_triples", [])[:5],
                    },
                },
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    def _rate(x: int, denom: int) -> float:
        return 0.0 if denom <= 0 else 100.0 * float(x) / float(denom)

    summary = {
        "n": n,
        "em": {
            "llm": _rate(em_llm, n),
            "graphrag": _rate(em_graphrag, n),
            "kvi": _rate(em_kvi, n),
        },
        "counts": {
            "llm": em_llm,
            "graphrag": em_graphrag,
            "kvi": em_kvi,
        },
        "paths": {
            "predictions": str(preds_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("## Experiment 01 — Main QA Performance (EM)\n")
    md.append(f"- N = {n}\n")
    md.append("\n")
    md.append("| Method | Retrieval | Injection | EM (%) |\n")
    md.append("|---|---|---:|---:|\n")
    md.append(f"| LLM | none | none | {summary['em']['llm']:.1f} |\n")
    md.append(f"| GraphRAG | graph | prompt | {summary['em']['graphrag']:.1f} |\n")
    md.append(f"| KVI | graph | KV + prompt | {summary['em']['kvi']:.1f} |\n")
    md.append("\n")
    (out_dir / "results.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

