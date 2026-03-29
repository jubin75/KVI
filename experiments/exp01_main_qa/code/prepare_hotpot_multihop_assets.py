#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _norm_for_id(x: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(x or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "unk"


def _iter_context_sentences(ex: Dict[str, Any]) -> Iterable[Tuple[str, int, str]]:
    ctx = ex.get("context") or {}
    titles = ctx.get("title") or []
    sents = ctx.get("sentences") or []
    if not isinstance(titles, Sequence) or not isinstance(sents, Sequence):
        return
    for t, para in zip(titles, sents):
        title = str(t or "").strip()
        if not title or not isinstance(para, Sequence):
            continue
        for sid, sent in enumerate(para):
            txt = str(sent or "").strip()
            if txt:
                yield title, sid, txt


def _supporting_pairs(ex: Dict[str, Any]) -> List[Tuple[str, int]]:
    sf = ex.get("supporting_facts") or {}
    titles = sf.get("title") or []
    sids = sf.get("sent_id") or []
    out: List[Tuple[str, int]] = []
    for t, sid in zip(titles, sids):
        title = str(t or "").strip()
        if title and isinstance(sid, int):
            out.append((title, sid))
    return out


def _build_gold_supporting_sentences(ex: Dict[str, Any]) -> List[str]:
    ctx_map: Dict[Tuple[str, int], str] = {}
    for title, sid, txt in _iter_context_sentences(ex):
        ctx_map[(title, sid)] = txt
    out: List[str] = []
    for key in _supporting_pairs(ex):
        txt = ctx_map.get(key, "")
        if txt:
            out.append(txt)
    return _dedup_keep_order(out)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare Hotpot multihop-oriented eval + graph/kv source assets from original context paragraphs."
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hotpot_config", default="distractor", help="hotpot_qa config: distractor/fullwiki")
    p.add_argument("--hotpot_split", default="validation")
    p.add_argument("--hotpot_max", type=int, default=0, help="max examples (0=all)")
    p.add_argument("--streaming", action="store_true")
    p.add_argument(
        "--keep_only_supporting_titles",
        action="store_true",
        help="Only keep sentences from titles that appear in supporting_facts.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpot_qa", args.hotpot_config, split=args.hotpot_split, streaming=bool(args.streaming))

    eval_rows: List[Dict[str, Any]] = []
    sent_rows: List[Dict[str, Any]] = []
    tri_rows: List[Dict[str, Any]] = []

    n_ex = 0
    for i, ex in enumerate(ds):
        q = str(ex.get("question") or "").strip()
        ans = str(ex.get("answer") or "").strip()
        qid = str(ex.get("id") or f"hotpot_{i}")
        if not q or not ans:
            continue

        supporting_pairs = _supporting_pairs(ex)
        supporting_titles = _dedup_keep_order([t for t, _sid in supporting_pairs])
        supporting_set = {(t, sid) for t, sid in supporting_pairs}
        supporting_texts = _build_gold_supporting_sentences(ex)

        eval_rows.append(
            {
                "id": qid,
                "question": q,
                "answer": ans,
                "answers": [ans],
                "dataset": "HotpotQA",
                "gold_supporting_sentences": supporting_texts,
                "supporting_titles": supporting_titles,
            }
        )

        # Sentence pool from original context paragraphs (long-text multi-hop friendly).
        local_block_ids: List[str] = []
        for title, sid, txt in _iter_context_sentences(ex):
            if args.keep_only_supporting_titles and title not in supporting_titles:
                continue
            block_id = f"hp_{qid}_{_norm_for_id(title)}_{sid}"
            local_block_ids.append(block_id)
            sent_rows.append(
                {
                    "block_id": block_id,
                    "text": txt,
                    "source_id": qid,
                    "doc_id": title,
                    "metadata": {
                        "dataset": "HotpotQA",
                        "question_id": qid,
                        "title": title,
                        "sentence_id": sid,
                        "is_supporting_fact": (title, sid) in supporting_set,
                    },
                }
            )

        # Lightweight graph triples from supporting title chain (for better multi-hop signal than Q->A template).
        # Bridge edges among supporting titles, and answer-anchor edges from each supporting title to answer.
        for ti in range(len(supporting_titles) - 1):
            t1 = supporting_titles[ti]
            t2 = supporting_titles[ti + 1]
            tri_rows.append(
                {
                    "triple_id": f"tri_bridge_{qid}_{ti}",
                    "subject": t1,
                    "subject_type": "entity",
                    "predicate": "bridge_to",
                    "object": t2,
                    "object_type": "entity",
                    "confidence": 0.95,
                    "provenance": {
                        "sentence_id": "",
                        "sentence_text": " | ".join(supporting_texts[:2]),
                        "source_block_id": "",
                        "source_doc_id": qid,
                    },
                }
            )
        for ti, title in enumerate(supporting_titles):
            tri_rows.append(
                {
                    "triple_id": f"tri_answer_{qid}_{ti}",
                    "subject": title,
                    "subject_type": "entity",
                    "predicate": "answer_anchor",
                    "object": ans,
                    "object_type": "answer",
                    "confidence": 0.9,
                    "provenance": {
                        "sentence_id": "",
                        "sentence_text": " | ".join(supporting_texts[:2]),
                        "source_block_id": "",
                        "source_doc_id": qid,
                    },
                }
            )

        n_ex += 1
        if args.hotpot_max and n_ex >= int(args.hotpot_max):
            break

    eval_path = out_dir / "hotpot_eval_multihop.jsonl"
    sent_path = out_dir / "sentences_multihop.jsonl"
    tri_path = out_dir / "triples_multihop.jsonl"

    n_eval = _write_jsonl(eval_path, eval_rows)
    n_sent = _write_jsonl(sent_path, sent_rows)
    n_tri = _write_jsonl(tri_path, tri_rows)

    manifest = {
        "source": {"name": "hotpot_qa", "config": args.hotpot_config, "split": args.hotpot_split},
        "counts": {"examples": n_eval, "sentences": n_sent, "triples": n_tri},
        "flags": {"keep_only_supporting_titles": bool(args.keep_only_supporting_titles)},
        "outputs": {
            "eval_jsonl": str(eval_path),
            "sentences_jsonl": str(sent_path),
            "triples_jsonl": str(tri_path),
        },
        "notes": "Uses original Hotpot context paragraphs + supporting_facts; designed for multi-hop long-text evaluation and case analysis.",
    }
    (out_dir / "manifest_multihop.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

