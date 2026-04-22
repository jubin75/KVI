#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Exp02 datasets (TruthfulQA, FEVER) to Exp01 JSONL format")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mirror_root", default="/home/zd/dev/KVI/experiments/_mirror_data_resolved")
    p.add_argument("--mirror_data_root", default="/home/zd/dev/KVI/experiments/_mirror_data")
    p.add_argument("--truthfulqa_max", type=int, default=500)
    p.add_argument("--fever_max", type=int, default=1000)
    p.add_argument("--streaming", action="store_true")
    p.add_argument(
        "--offline_only",
        action="store_true",
        help="Use local mirrored parquet files only (no HF network fallback).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    mirror_root = Path(args.mirror_root)
    mirror_data_root = Path(args.mirror_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TruthfulQA generation split (retry for transient HF limits)
    last_err = None
    ds_t = None
    local_truth = mirror_root / "truthful_qa_generation_val.parquet"
    if local_truth.exists():
        ds_t = load_dataset("parquet", data_files=str(local_truth), split="train")
    elif not args.offline_only:
        for i in range(6):
            try:
                ds_t = load_dataset("truthful_qa", "generation", split="validation", streaming=bool(args.streaming))
                break
            except Exception as e:
                last_err = e
                time.sleep(15 * (i + 1))
    if ds_t is None:
        if args.offline_only:
            raise RuntimeError(
                f"offline_only is set but missing local TruthfulQA generation parquet: {local_truth}"
            )
        raise RuntimeError(f"Failed loading truthful_qa after retries: {last_err}")

    # Optional TruthfulQA multiple-choice targets (for MC proxy/offline alignment).
    ds_t_mc = None
    local_truth_mc_candidates = [
        mirror_root / "truthful_qa_multiple_choice_val.parquet",
        mirror_data_root / "truthful_qa/multiple_choice/validation-00000-of-00001.parquet",
    ]
    local_truth_mc = next((p for p in local_truth_mc_candidates if p.exists()), None)
    if local_truth_mc is not None:
        try:
            ds_t_mc = load_dataset("parquet", data_files=str(local_truth_mc), split="train")
        except Exception:
            ds_t_mc = None
    if ds_t_mc is None and (not args.offline_only):
        # best-effort online fallback; many environments are rate-limited/forbidden.
        for i in range(3):
            try:
                ds_t_mc = load_dataset("truthful_qa", "multiple_choice", split="validation", streaming=bool(args.streaming))
                break
            except Exception:
                time.sleep(10 * (i + 1))

    mc_by_question: Dict[str, Dict[str, Any]] = {}
    if ds_t_mc is not None:
        for ex in ds_t_mc:
            q = str(ex.get("question") or "").strip()
            if not q:
                continue
            mc1 = ex.get("mc1_targets")
            mc2 = ex.get("mc2_targets")
            row: Dict[str, Any] = {}
            if isinstance(mc1, dict):
                row["mc1_targets"] = mc1
            if isinstance(mc2, dict):
                row["mc2_targets"] = mc2
            if row:
                mc_by_question[q] = row
    t_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds_t):
        q = str(ex.get("question") or "").strip()
        best = str(ex.get("best_answer") or "").strip()
        corr = ex.get("correct_answers")
        answers = [best] if best else []
        if isinstance(corr, list):
            for x in corr:
                s = str(x or "").strip()
                if s and s not in answers:
                    answers.append(s)
        if not q or not answers:
            continue
        row: Dict[str, Any] = {
            "id": f"truthfulqa_{i}",
            "question": q,
            "answer": answers[0],
            "answers": answers,
            "dataset": "TruthfulQA",
        }
        # Always build MC-style targets from generation split as fallback.
        incorrect = ex.get("incorrect_answers")
        if isinstance(corr, list) and isinstance(incorrect, list):
            cands: List[str] = []
            labels: List[int] = []
            for s in corr:
                t = str(s or "").strip()
                if t and t not in cands:
                    cands.append(t)
                    labels.append(1)
            for s in incorrect:
                t = str(s or "").strip()
                if t and t not in cands:
                    cands.append(t)
                    labels.append(0)
            if cands and any(labels):
                row["mc1_targets"] = {"choices": cands, "labels": labels}
                row["mc2_targets"] = {"choices": cands, "labels": labels}
        if q in mc_by_question:
            row.update(mc_by_question[q])
        t_rows.append(row)
        if args.truthfulqa_max and len(t_rows) >= int(args.truthfulqa_max):
            break

    # FEVER: old 'fever' script is unsupported in recent datasets.
    # Try modern mirrored variants in order.
    ds_f = None
    local_fever_candidates = [
        mirror_root / "kilt_fever_validation.parquet",
        mirror_data_root / "kilt_tasks/fever/validation-00000-of-00001.parquet",
    ]
    local_fever = next((p for p in local_fever_candidates if p.exists()), None)
    if local_fever is not None:
        ds_f = load_dataset("parquet", data_files=str(local_fever), split="train")
    elif not args.offline_only:
        fever_candidates = [
            ("pietrolesci/fever", None, "validation"),
            ("SetFit/fever", None, "validation"),
            ("feverous/fever", None, "validation"),
        ]
        last_err = None
        for name, cfg, split in fever_candidates:
            for i in range(4):
                try:
                    if cfg:
                        ds_f = load_dataset(name, cfg, split=split, streaming=bool(args.streaming))
                    else:
                        ds_f = load_dataset(name, split=split, streaming=bool(args.streaming))
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(12 * (i + 1))
            if ds_f is not None:
                break
    if ds_f is None:
        if args.offline_only:
            raise RuntimeError(
                "offline_only is set but local FEVER parquet not found under mirror roots. "
                f"Checked: {[str(x) for x in local_fever_candidates]}"
            )
        raise RuntimeError(
            "Failed loading FEVER from local mirror and fallback candidates. "
            "Try: python3 /home/zd/dev/KVI/experiments/code/download_mirror_datasets.py "
            f"--out_dir {mirror_root}. Last error: {last_err}"
        )
    label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
    f_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds_f):
        claim = str(ex.get("claim") or ex.get("input") or "").strip()
        lbl = ex.get("label")
        ans = ""
        if isinstance(lbl, int):
            ans = label_map.get(lbl, "")
        elif isinstance(lbl, str):
            ans = lbl.strip().upper()
        else:
            out = ex.get("output")
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    ans = str(first.get("answer") or "").strip().upper()
        if not claim or not ans:
            continue
        q = (
            f'Claim: "{claim}"\n'
            "Based on evidence, answer with one label only: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
        )
        f_rows.append(
            {
                "id": f"fever_{i}",
                "question": q,
                "answer": ans,
                "answers": [ans],
                "dataset": "FEVER",
            }
        )
        if args.fever_max and len(f_rows) >= int(args.fever_max):
            break

    t_path = out_dir / "truthfulqa_eval.jsonl"
    f_path = out_dir / "fever_eval.jsonl"
    _write_jsonl(t_path, t_rows)
    _write_jsonl(f_path, f_rows)

    manifest = {
        "counts": {"truthfulqa": len(t_rows), "fever": len(f_rows)},
        "truthfulqa_mc_targets_covered": int(sum(1 for r in t_rows if isinstance(r.get("mc1_targets"), dict) and isinstance(r.get("mc2_targets"), dict))),
        "outputs": {"truthfulqa": str(t_path), "fever": str(f_path)},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

