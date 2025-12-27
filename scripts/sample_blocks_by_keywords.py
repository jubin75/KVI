"""
Sample blocks from a blocks*.jsonl file by keyword match (streaming).

Use cases:
- Verify whether your evidence library contains "mechanism/pathogenesis/immune" evidence sentences.
- Quickly inspect representative blocks without loading the whole file into memory.

Example:
  python -u scripts/sample_blocks_by_keywords.py \
    --blocks_jsonl /home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl \
    --kw pathogenesis --kw mechanism --kw immune --kw cytokine \
    --sample 20 --seed 0 --max_chars 600
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MatchRec:
    line_no: int
    block_id: str
    doc_id: Optional[str]
    source_uri: Optional[str]
    token_count: Optional[int]
    matched_keywords: List[str]
    text: str


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _get_source_uri(rec: Dict[str, Any]) -> Optional[str]:
    # Prefer top-level source_uri (newer pipeline), fallback to metadata.
    src = rec.get("source_uri")
    if isinstance(src, str) and src.strip():
        return src.strip()
    md = rec.get("metadata") or {}
    if isinstance(md, dict):
        for k in ("source_uri", "source_path", "pdf_path", "uri", "path"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _compile_keywords(keywords: List[str], *, ignore_case: bool) -> List[Tuple[str, re.Pattern[str]]]:
    flags = re.IGNORECASE if ignore_case else 0
    out: List[Tuple[str, re.Pattern[str]]] = []
    for kw in keywords:
        kw2 = kw.strip()
        if not kw2:
            continue
        # Treat keywords as literal substrings (safe default); users can pass regex via --regex.
        out.append((kw2, re.compile(re.escape(kw2), flags=flags)))
    return out


def _compile_regexes(patterns: List[str], *, ignore_case: bool) -> List[Tuple[str, re.Pattern[str]]]:
    flags = re.IGNORECASE if ignore_case else 0
    out: List[Tuple[str, re.Pattern[str]]] = []
    for pat in patterns:
        p2 = pat.strip()
        if not p2:
            continue
        out.append((p2, re.compile(p2, flags=flags)))
    return out


def _reservoir_add(
    reservoir: List[MatchRec], item: MatchRec, *, k: int, seen: int, rng: random.Random
) -> None:
    # Standard reservoir sampling (Algorithm R).
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randrange(seen)
    if j < k:
        reservoir[j] = item


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl", required=True, help="Path to blocks.jsonl / blocks.evidence.jsonl")
    p.add_argument("--kw", action="append", default=[], help="Keyword to match (literal substring). Can repeat.")
    p.add_argument(
        "--keywords",
        default="",
        help="Comma-separated keywords (literal). Alternative to repeating --kw.",
    )
    p.add_argument(
        "--regex",
        action="append",
        default=[],
        help="Regex pattern to match (advanced). Can repeat. Example: '(pathogen|mechan)ism'.",
    )
    p.add_argument("--ignore_case", action="store_true", default=True, help="Case-insensitive match (default true).")
    p.add_argument("--case_sensitive", action="store_false", dest="ignore_case", help="Case-sensitive match.")
    p.add_argument("--sample", type=int, default=20, help="How many matching blocks to sample-print.")
    p.add_argument("--max_chars", type=int, default=800, help="Max chars to print per matched block text (0=no trunc).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--limit_lines", type=int, default=0, help="Only scan first N lines (0=scan all).")
    args = p.parse_args()

    blocks_path = Path(str(args.blocks_jsonl))
    if not blocks_path.exists():
        raise SystemExit(f"blocks_jsonl not found: {blocks_path}")

    keywords: List[str] = []
    for kw in (args.kw or []):
        if str(kw).strip():
            keywords.append(str(kw).strip())
    if str(args.keywords).strip():
        keywords.extend([x.strip() for x in str(args.keywords).split(",") if x.strip()])
    regexes: List[str] = [str(x).strip() for x in (args.regex or []) if str(x).strip()]

    if not keywords and not regexes:
        raise SystemExit("No keywords provided. Use --kw / --keywords / --regex.")

    kw_pats = _compile_keywords(keywords, ignore_case=bool(args.ignore_case))
    rx_pats = _compile_regexes(regexes, ignore_case=bool(args.ignore_case))
    all_pats: List[Tuple[str, re.Pattern[str]]] = kw_pats + rx_pats

    rng = random.Random(int(args.seed))

    total = 0
    matched = 0
    per_pat_counts: Dict[str, int] = {name: 0 for name, _ in all_pats}
    reservoir: List[MatchRec] = []

    with blocks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if int(args.limit_lines) > 0 and line_no > int(args.limit_lines):
                break
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            text = str(rec.get("text") or "")
            if not text:
                continue

            hit_names: List[str] = []
            for name, pat in all_pats:
                if pat.search(text):
                    hit_names.append(name)
                    per_pat_counts[name] = per_pat_counts.get(name, 0) + 1
            if not hit_names:
                continue

            matched += 1
            block_id = rec.get("block_id") or rec.get("chunk_id") or rec.get("id")
            if not isinstance(block_id, str) or not block_id.strip():
                block_id = f"<missing_block_id_line_{line_no}>"
            item = MatchRec(
                line_no=int(line_no),
                block_id=str(block_id),
                doc_id=(str(rec.get("doc_id")) if rec.get("doc_id") is not None else None),
                source_uri=_get_source_uri(rec),
                token_count=(int(rec.get("token_count")) if rec.get("token_count") is not None else None),
                matched_keywords=sorted(set(hit_names)),
                text=text,
            )
            _reservoir_add(reservoir, item, k=int(args.sample), seen=matched, rng=rng)

    # Output summary
    print("=== Keyword Sampling Summary ===", flush=True)
    print(f"blocks_jsonl={blocks_path}", flush=True)
    print(f"scanned_lines={total} matched_blocks={matched}", flush=True)
    print(f"sample_size={min(int(args.sample), len(reservoir))} seed={int(args.seed)}", flush=True)
    print(f"ignore_case={bool(args.ignore_case)}", flush=True)
    if keywords:
        print(f"keywords(literal)={keywords}", flush=True)
    if regexes:
        print(f"regexes={regexes}", flush=True)
    print("per_pattern_counts:", flush=True)
    for name, _ in all_pats:
        print(f"- {name}: {int(per_pat_counts.get(name, 0))}", flush=True)

    # Print sampled matches
    if not reservoir:
        print("\n=== Samples ===\n(no matches)", flush=True)
        return

    # Stable print order: sort by line_no for easier file lookup
    reservoir_sorted = sorted(reservoir, key=lambda r: int(r.line_no))
    print("\n=== Samples ===", flush=True)
    for i, r in enumerate(reservoir_sorted, start=1):
        txt = _norm_ws(r.text)
        if int(args.max_chars) > 0:
            txt = txt[: int(args.max_chars)]
        print(f"\n--- sample {i}/{len(reservoir_sorted)} ---", flush=True)
        print(
            f"line_no={r.line_no} block_id={r.block_id} doc_id={r.doc_id} token_count={r.token_count}",
            flush=True,
        )
        print(f"source_uri={r.source_uri}", flush=True)
        print(f"matched={r.matched_keywords}", flush=True)
        print(f"text_snippet={txt}", flush=True)


if __name__ == "__main__":
    main()



