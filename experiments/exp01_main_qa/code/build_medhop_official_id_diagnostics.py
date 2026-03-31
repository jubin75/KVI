#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DB_ID_RE = re.compile(r"DB\d+", re.IGNORECASE)
STRICT_DB_RE = re.compile(r"^\s*DB\d+\s*$", re.IGNORECASE)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _extract_first_db(pred: str) -> str:
    m = DB_ID_RE.search(pred or "")
    return m.group(0).upper() if m else ""


def _is_strict_single_db(pred: str) -> bool:
    return bool(STRICT_DB_RE.match(pred or ""))


def _eval_method(predictions_path: Path, method_key: str) -> Tuple[int, float, float]:
    total = 0
    strict_ok = 0
    extracted_em_ok = 0
    for row in _iter_jsonl(predictions_path):
        preds = row.get("predictions") or {}
        if method_key not in preds:
            continue
        pred = str(preds.get(method_key) or "")
        golds = [str(x).upper() for x in (row.get("gold_answers") or [])]
        total += 1
        if _is_strict_single_db(pred):
            strict_ok += 1
        first_db = _extract_first_db(pred)
        if first_db and first_db in golds:
            extracted_em_ok += 1
    if total == 0:
        return (0, 0.0, 0.0)
    valid_rate = 100.0 * strict_ok / total
    extract_em = 100.0 * extracted_em_ok / total
    return (total, valid_rate, extract_em)


def main() -> None:
    root = Path("/home/zd/dev/KVI")
    out_path = root / "experiments/exp01_main_qa/results/main_table/medhop_official_id_diagnostics.md"

    # Align with current main-table sources:
    # - Qwen: llm/rag/graphrag/kv_prefix from fullmethods_qwen, kvi from reconcile run.
    qwen_full = root / "experiments/exp01_main_qa/results/medhop_official_fullmethods_qwen25_7b/predictions.jsonl"
    qwen_kvi = root / "experiments/exp01_main_qa/results/medhop_official_kvi_reconcile_final/predictions.jsonl"
    mistral = root / "experiments/exp01_main_qa/results/medhop_official_fullmethods_mistral7b_v0_3/predictions.jsonl"

    rows: List[Tuple[str, str, int, float, float]] = []
    for m in ["llm", "rag", "graphrag", "kv_prefix"]:
        n, vr, eem = _eval_method(qwen_full, m)
        rows.append(("Qwen2.5-7B-Instruct", m, n, vr, eem))
    n, vr, eem = _eval_method(qwen_kvi, "kvi")
    rows.append(("Qwen2.5-7B-Instruct", "kvi", n, vr, eem))

    for m in ["llm", "rag", "graphrag", "kv_prefix", "kvi"]:
        n, vr, eem = _eval_method(mistral, m)
        rows.append(("Mistral-7B-Instruct-v0.3", m, n, vr, eem))

    method_label = {
        "llm": "LLM",
        "rag": "RAG",
        "graphrag": "GraphRAG",
        "kv_prefix": "KV Prefix",
        "kvi": "KVI",
    }

    lines = [
        "## MedHopQA_official ID Diagnostics (for explaining EM=0)",
        "",
        "This table supplements EM with two diagnostics:",
        "- **Valid-ID Rate (%)**: prediction is exactly one `DB` id (`^DB\\\\d+$`) with no extra text.",
        "- **Extract-then-EM (%)**: extract the first `DB\\\\d+` from prediction, then compare to gold ID.",
        "",
        "| Backbone | Method | N | Valid-ID Rate (%) | Extract-then-EM (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for backbone, method, n, vr, eem in rows:
        lines.append(f"| {backbone} | {method_label[method]} | {n} | {vr:.1f} | {eem:.1f} |")

    lines += [
        "",
        "### Source runs",
        f"- Qwen (`LLM/RAG/GraphRAG/KV Prefix`): `{qwen_full}`",
        f"- Qwen (`KVI`): `{qwen_kvi}`",
        f"- Mistral (all five methods): `{mistral}`",
        "",
        "Note: these diagnostics are for analysis only; the main table metric remains EM.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

