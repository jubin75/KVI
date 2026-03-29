#!/usr/bin/env python3
"""
Build MedHopQA official-style eval JSONL + Exp01 assets from medhop_raw.

Source rows (e.g. medhop_source_validation.parquet.jsonl) use:
  query: "interacts_with DB01171?"
  answer: partner DrugBank ID
  supports: list of PubMed-style text snippets (often "DB08820 : abstract...")

Outputs under --out_dir:
  medhop_official_eval.jsonl — paper-facing schema (question, short_answer, long_answer, supporting_facts, ...)
  medhop_eval.jsonl — runnable Exp01 file {id, question, answers, ...}
  sentences_medhop.jsonl, triples_medhop.jsonl — same construction as prepare_medhopqa_assets.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_code_dir = Path(__file__).resolve().parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

# Reuse helpers from the ID-variant preparer
from prepare_medhopqa_assets import (
    _dedup_keep_order,
    _extract_docs,
    _extract_first_db_code,
    _norm_for_id,
    _read_jsonl,
    _read_json,
    _write_jsonl,
)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def _parse_supporting_fact(text: str, idx: int) -> Tuple[str, str]:
    s = str(text or "").strip()
    if not s:
        return f"support_{idx}", ""
    m = re.match(r"^(\bDB\d+)\s*:\s*(.+)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    head = s[:160].strip()
    if len(head) < len(s):
        head = head.rsplit(" ", 1)[0] + "..."
    return f"passage_{idx}", s


def _extract_answers(ex: Dict[str, Any]) -> List[str]:
    vals = ex.get("answers")
    out: List[str] = []
    if isinstance(vals, list):
        out.extend([str(x).strip() for x in vals if str(x).strip()])
    ans = ex.get("answer")
    if isinstance(ans, list):
        out.extend([str(x).strip() for x in ans if str(x).strip()])
    elif isinstance(ans, str) and ans.strip():
        out.append(ans.strip())
    return _dedup_keep_order(out)


def _nl_question(query_code: str) -> str:
    return (
        f"Which DrugBank-listed drug interacts with {query_code}? "
        "Answer using the partner's canonical DrugBank identifier."
    )


def _build_official_row(
    ex: Dict[str, Any],
    qid: str,
    query_code: str,
    answer_code: str,
    answers: List[str],
    supporting_facts: List[Dict[str, str]],
    long_answer: str,
    append_hint: bool,
) -> Dict[str, Any]:
    q_eval = _nl_question(query_code)
    if append_hint:
        q_eval += (
            "\n\nAnswer with only the partner entity DB id (format: DB followed by digits), and nothing else."
        )
    return {
        "id": qid,
        "question": q_eval,
        "raw_query": str(ex.get("query") or "").strip(),
        "answer": answers[0],
        "answers": answers,
        "short_answer": answer_code,
        "long_answer": long_answer,
        "type": "drug_drug_interaction_completion",
        "supporting_facts": supporting_facts,
        "candidates": ex.get("candidates") if isinstance(ex.get("candidates"), list) else [],
        "dataset": "MedHopQA_official_nl",
        "gold_note": (
            "Gold short_answer is the partner DrugBank ID (DB + digits). "
            "Natural drug names are not provided in this release; ID-level EM matches Exp01 MedHop-ID."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MedHop raw → official-style eval + Exp01 assets.")
    p.add_argument(
        "--medhop_raw",
        required=True,
        help="medhop_source_validation.parquet.jsonl (or compatible jsonl)",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument(
        "--append_id_only_hint",
        action="store_true",
        default=True,
        help="Append ID-only answer hint to question (default: on, align EM with MedHop-ID).",
    )
    p.add_argument(
        "--no_append_id_only_hint",
        dest="append_id_only_hint",
        action="store_false",
        help="Omit ID-only hint (open-form question only).",
    )
    args = p.parse_args()

    input_path = Path(args.medhop_raw)
    rows = _load_records(input_path)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: int(args.max_examples)]

    official_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    sent_rows: List[Dict[str, Any]] = []
    tri_rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(rows):
        qid = str(ex.get("id") or ex.get("_id") or f"medhop_{i}")
        q_raw = str(ex.get("question") or ex.get("query") or "").strip()
        answers = _extract_answers(ex)
        supports = ex.get("supports")
        if not isinstance(supports, Sequence):
            supports = []
        docs = _extract_docs(ex)
        if not q_raw or not answers:
            continue

        query_code = _extract_first_db_code(q_raw)
        answer_code = _extract_first_db_code(answers[0]) or str(answers[0]).strip()
        if not query_code or not answer_code:
            continue

        supporting_facts: List[Dict[str, str]] = []
        for si, sp in enumerate(supports):
            if not isinstance(sp, str):
                continue
            title, sentence = _parse_supporting_fact(sp, si)
            supporting_facts.append({"title": title, "sentence": sentence})

        la_parts: List[str] = []
        for sp in supports[:2]:
            if isinstance(sp, str) and sp.strip():
                la_parts.append(sp.strip())
        long_answer = "\n\n".join(la_parts)[:2500]

        official = _build_official_row(
            ex,
            qid,
            query_code,
            answer_code,
            answers,
            supporting_facts,
            long_answer,
            bool(args.append_id_only_hint),
        )
        official_rows.append(official)

        supporting_titles = [f.get("title") or "" for f in supporting_facts if f.get("title")]
        eval_rows.append(
            {
                "id": qid,
                "question": official["question"],
                "answer": answers[0],
                "answers": answers,
                "dataset": "MedHopQA_official_nl",
                "supporting_titles": supporting_titles,
            }
        )

        both_blocks: List[Dict[str, str]] = []
        answer_blocks: List[Dict[str, str]] = []
        fallback_block: Dict[str, str] | None = None

        for title, sents in docs:
            for sid, txt in enumerate(sents):
                block_id = f"mh_{qid}_{_norm_for_id(title)}_{sid}"
                sent_rows.append(
                    {
                        "block_id": block_id,
                        "text": txt,
                        "source_id": qid,
                        "doc_id": title,
                        "metadata": {
                            "dataset": "MedHopQA_official_nl",
                            "question_id": qid,
                            "title": title,
                            "sentence_id": sid,
                            "is_supporting_title": title in supporting_titles,
                        },
                    }
                )
                if fallback_block is None:
                    fallback_block = {"block_id": block_id, "text": txt, "title": title}
                txt_norm = str(txt)
                contains_query = bool(query_code and query_code in txt_norm)
                contains_answer = bool(answer_code and answer_code in txt_norm)
                if contains_query and contains_answer:
                    both_blocks.append({"block_id": block_id, "text": txt_norm, "title": title})
                elif contains_answer:
                    answer_blocks.append({"block_id": block_id, "text": txt_norm, "title": title})

        max_tri_per_example = 2
        chosen_blocks = (both_blocks[:max_tri_per_example]) or (answer_blocks[:max_tri_per_example])
        if not chosen_blocks and fallback_block:
            chosen_blocks = [fallback_block]
        if not chosen_blocks:
            continue

        for ti, blk in enumerate(chosen_blocks):
            short_stmt = f"{query_code} interacts_with {answer_code}."
            prov_text = short_stmt
            long_txt = str(blk.get("text") or "").strip()
            if long_txt:
                prov_text = prov_text + " " + long_txt[:800]
            tri_rows.append(
                {
                    "triple_id": f"tri_interacts_{qid}_{ti}",
                    "subject": query_code,
                    "subject_type": "entity",
                    "predicate": "associated_with",
                    "object": answers[0],
                    "object_type": "entity",
                    "confidence": 0.9,
                    "provenance": {
                        "sentence_id": blk["block_id"],
                        "sentence_text": prov_text,
                        "source_block_id": blk["block_id"],
                        "source_doc_id": blk.get("title") or qid,
                    },
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    off_path = out_dir / "medhop_official_eval.jsonl"
    eval_path = out_dir / "medhop_eval.jsonl"
    sent_path = out_dir / "sentences_medhop.jsonl"
    tri_path = out_dir / "triples_medhop.jsonl"

    n_off = _write_jsonl(off_path, official_rows)
    n_eval = _write_jsonl(eval_path, eval_rows)
    n_sent = _write_jsonl(sent_path, sent_rows)
    n_tri = _write_jsonl(tri_path, tri_rows)

    manifest = {
        "source_file": str(input_path),
        "append_id_only_hint": bool(args.append_id_only_hint),
        "counts": {
            "official_examples": n_off,
            "exp01_examples": n_eval,
            "sentences": n_sent,
            "triples": n_tri,
        },
        "outputs": {
            "medhop_official_eval_jsonl": str(off_path),
            "medhop_eval_jsonl": str(eval_path),
            "sentences_jsonl": str(sent_path),
            "triples_jsonl": str(tri_path),
        },
        "notes": (
            "medhop_official_eval.jsonl is paper-facing (long_answer, supporting_facts). "
            "medhop_eval.jsonl is the runnable Exp01 dataset (same questions/answers)."
        ),
    }
    (out_dir / "manifest_medhop_official.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
