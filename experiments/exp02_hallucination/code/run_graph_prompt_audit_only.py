#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if limit > 0 and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run retrieval/DRM/prompt audit only (no generation)")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph_index", required=True)
    ap.add_argument("--graph_sentences_jsonl", required=True)
    ap.add_argument("--triple_kvbank_dir", required=True)
    ap.add_argument("--audit_jsonl", required=True)
    ap.add_argument("--audit_oracle_jsonl", default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    graph_cli = repo_root / "scripts" / "run_graph_inference.py"
    rows = _read_jsonl(Path(args.dataset_jsonl), int(args.limit))

    for i, ex in enumerate(rows, start=1):
        q = str(ex.get("question") or "").strip()
        qid = str(ex.get("id") or ex.get("qid") or f"idx_{i}")
        if not q:
            continue
        base_argv = [
            sys.executable,
            str(graph_cli),
            "--model",
            str(args.model),
            "--prompt",
            q,
            "--graph_index",
            str(args.graph_index),
            "--sentences_jsonl",
            str(args.graph_sentences_jsonl),
            "--openqa_mode",
            "--audit_only",
            "--audit_jsonl",
            str(args.audit_jsonl),
            "--audit_query_id",
            qid,
        ]
        if str(args.audit_oracle_jsonl).strip():
            base_argv += ["--audit_oracle_jsonl", str(args.audit_oracle_jsonl)]

        # GraphRAG-like audit (no KVI)
        a1 = base_argv + ["--audit_method_key", "graphrag"]
        subprocess.run(a1, cwd=str(repo_root), check=True, capture_output=True, text=True)

        # KVI-like audit (retrieval/DRM/prompt with KV assembly path enabled)
        a2 = base_argv + [
            "--enable_kvi",
            "--triple_kvbank_dir",
            str(args.triple_kvbank_dir),
            "--audit_method_key",
            "kvi",
        ]
        subprocess.run(a2, cwd=str(repo_root), check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
