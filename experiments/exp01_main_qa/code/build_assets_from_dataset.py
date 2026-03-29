#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build Exp01 graph/ann source artifacts from unified dataset JSONL")
    p.add_argument("--dataset_jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_examples", type=int, default=0)
    args = p.parse_args()

    ds_path = Path(args.dataset_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(ds_path)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: int(args.max_examples)]
    if not rows:
        raise SystemExit(f"Empty dataset rows: {ds_path}")

    sentences_path = out_dir / "sentences.jsonl"
    triples_path = out_dir / "triples.jsonl"
    manifest_path = out_dir / "asset_manifest.json"

    n_sent = 0
    n_tri = 0
    with sentences_path.open("w", encoding="utf-8") as sf, triples_path.open("w", encoding="utf-8") as tf:
        for i, r in enumerate(rows):
            q = str(r.get("question") or "").strip()
            ans_list = r.get("answers") if isinstance(r.get("answers"), list) else []
            ans = str(r.get("answer") or "").strip()
            if not ans and ans_list:
                ans = str(ans_list[0] or "").strip()
            if not q or not ans:
                continue

            rid = str(r.get("id") or f"ex_{i}")
            sid = f"sent_{rid}"
            sentence_text = f"Q: {q}\nA: {ans}"

            sent = {
                "block_id": sid,
                "text": sentence_text,
                "source_id": rid,
                "doc_id": str(r.get("dataset") or "exp01"),
                "metadata": {
                    "kind": "qa_pair",
                    "dataset": str(r.get("dataset") or ""),
                    "question": q,
                    "answer": ans,
                },
            }
            sf.write(json.dumps(sent, ensure_ascii=False) + "\n")
            n_sent += 1

            tri = {
                "triple_id": f"tri_{rid}",
                "subject": q,
                "subject_type": "disease",
                "predicate": "associated_with",
                "object": ans,
                "object_type": "symptom",
                "confidence": 0.9,
                "provenance": {
                    "sentence_id": sid,
                    "sentence_text": sentence_text,
                    "source_block_id": sid,
                    "source_doc_id": str(r.get("dataset") or "exp01"),
                },
            }
            tf.write(json.dumps(tri, ensure_ascii=False) + "\n")
            n_tri += 1

    manifest = {
        "dataset_jsonl": str(ds_path),
        "out_dir": str(out_dir),
        "num_examples_input": len(rows),
        "num_sentences_written": n_sent,
        "num_triples_written": n_tri,
        "outputs": {
            "sentences_jsonl": str(sentences_path),
            "triples_jsonl": str(triples_path),
        },
        "notes": "Synthetic QA-derived artifacts for Exp01 pipeline automation.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

