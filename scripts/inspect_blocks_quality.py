"""
Inspect block text quality for blocks.jsonl.

This is the same utility as repo-root `scripts/inspect_blocks_quality.py`, but placed
under `external_kv_injection/scripts/` so that users who only sync the
`external_kv_injection/` directory can still run the inspection commands in the runbook.

Usage (repo root, where `external_kv_injection/` is a subdir):
  python -u external_kv_injection/scripts/inspect_blocks_quality.py \
    --blocks_jsonl /home/jb/KVI/_exp_prod/blocks.jsonl --sample 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_repo_root_on_syspath() -> None:
    # This file is: <repo_root>/external_kv_injection/scripts/inspect_blocks_quality.py
    repo_root = Path(__file__).resolve().parents[2]
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    # Also add `<repo_root>/src` for the repo-root layout where we may want to import
    # modules directly (e.g. `import cleaning_and_dedupe`) even if `src` isn't treated
    # as a package in some environments.
    src_dir = repo_root / "src"
    if src_dir.exists():
        s2 = str(src_dir)
        if s2 not in sys.path:
            sys.path.insert(0, s2)

    # Extra robustness: locate `cleaning_and_dedupe.py` even if the repo was moved/renamed.
    candidates = [
        repo_root / "src" / "cleaning_and_dedupe.py",
        repo_root / "external_kv_injection" / "src" / "cleaning_and_dedupe.py",
    ]
    found_parent: Path | None = None
    for c in candidates:
        if c.exists():
            found_parent = c.parent
            break
    if found_parent is None:
        try:
            for p in repo_root.rglob("cleaning_and_dedupe.py"):
                found_parent = p.parent
                break
        except Exception:
            found_parent = None
    if found_parent is not None:
        sp = str(found_parent)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_repo_root_on_syspath()

try:
    from external_kv_injection.src.cleaning_and_dedupe import normalize_text, quality_score, simhash64  # type: ignore
except ModuleNotFoundError:
    try:
        # Repo-root KVI layout: `<repo_root>/src/...`
        from src.cleaning_and_dedupe import normalize_text, quality_score, simhash64  # type: ignore
    except ModuleNotFoundError:
        try:
            # Fallback when `<repo_root>/src` (or the discovered parent dir) is on sys.path.
            from cleaning_and_dedupe import normalize_text, quality_score, simhash64  # type: ignore
        except ModuleNotFoundError as e:
            repo_root = Path(__file__).resolve().parents[2]
            src_dir = repo_root / "src"
            raise ModuleNotFoundError(
                "Failed to import `cleaning_and_dedupe`. "
                f"repo_root={repo_root} src_exists={src_dir.exists()}. "
                "Expected one of: "
                "`<repo_root>/src/cleaning_and_dedupe.py` (KVI root layout) or "
                "`<repo_root>/external_kv_injection/src/cleaning_and_dedupe.py` (monorepo layout). "
                "Fix: ensure you pulled the full repo, and that `src/cleaning_and_dedupe.py` exists."
            ) from e


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _quantiles(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"min": 0, "avg": 0, "p50": 0, "p95": 0, "max": 0}
    xs2 = sorted(xs)
    n = len(xs2)
    avg = sum(xs2) / n
    p50 = xs2[int(0.50 * (n - 1))]
    p95 = xs2[int(0.95 * (n - 1))]
    return {"min": float(xs2[0]), "avg": float(avg), "p50": float(p50), "p95": float(p95), "max": float(xs2[-1])}


def _nonprintable_ratio(s: str) -> float:
    if not s:
        return 0.0
    bad = 0
    for ch in s:
        o = ord(ch)
        # keep common whitespace
        if ch in "\n\t\r":
            continue
        # printable basic + CJK
        if 32 <= o <= 126 or (0x4E00 <= o <= 0x9FFF):
            continue
        bad += 1
    return bad / max(1, len(s))


def _extract_table_ids(meta: Dict[str, Any]) -> List[int]:
    tables = meta.get("tables") or {}
    ids = tables.get("table_ids") or []
    out: List[int] = []
    for x in ids:
        try:
            out.append(int(x))
        except Exception:
            continue
    return sorted(set(out))


_TABLE_MARKER_RE = re.compile(
    r"<\s*!\s*-\s*-\s*table\s*:\s*(\d+)\s*-\s*-\s*>",
    flags=re.IGNORECASE,
)


def _has_table(text: str, meta: Dict[str, Any]) -> bool:
    if _extract_table_ids(meta):
        return True
    if _TABLE_MARKER_RE.search(text):
        return True
    if text.count("|") >= 8:
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl", required=True)
    p.add_argument("--limit", type=int, default=None, help="Only read first N blocks (for quick checks).")
    p.add_argument("--sample", type=int, default=8, help="How many blocks to print as samples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tables_only", action="store_true", help="Sample only blocks that contain table markers/metadata.")
    args = p.parse_args()

    path = Path(args.blocks_jsonl)
    recs = _read_jsonl(path, limit=args.limit)
    if not recs:
        raise RuntimeError(f"No records read from {path}")

    total = len(recs)
    empty = 0
    token_counts: List[int] = []
    q_scores: List[float] = []
    nonprint_high = 0
    has_table = 0
    simhashes: List[str] = []
    # table-only subset stats (non-empty blocks that look like tables)
    table_token_counts: List[int] = []
    table_q_scores: List[float] = []
    table_nonprint_high = 0
    table_simhashes: List[str] = []

    for r in recs:
        text = normalize_text(str(r.get("text") or ""))
        if not text:
            empty += 1
            continue
        tc = int(r.get("token_count") or 0)
        token_counts.append(tc)
        q_scores.append(float(quality_score(text)))
        if _nonprintable_ratio(text) >= 0.02:
            nonprint_high += 1
        meta = r.get("metadata") or {}
        is_table = _has_table(text, meta)
        if is_table:
            has_table += 1
            table_token_counts.append(tc)
            table_q_scores.append(float(quality_score(text)))
            if _nonprintable_ratio(text) >= 0.02:
                table_nonprint_high += 1
            table_simhashes.append(simhash64(text))
        simhashes.append(simhash64(text))

    uniq_simhash = len(set(simhashes))
    uniq_table_simhash = len(set(table_simhashes))

    print(f"[inspect] blocks_jsonl={path}", flush=True)
    print(f"[inspect] total={total} empty_text={empty} ({_pct(empty/total)})", flush=True)
    if token_counts:
        q = _quantiles(token_counts)
        print(
            f"[inspect] token_count: min={q['min']:.0f} avg={q['avg']:.1f} p50={q['p50']:.0f} p95={q['p95']:.0f} max={q['max']:.0f}",
            flush=True,
        )
    if q_scores:
        qs = sorted(q_scores)
        n = len(qs)
        p10 = qs[int(0.10 * (n - 1))]
        p50 = qs[int(0.50 * (n - 1))]
        p90 = qs[int(0.90 * (n - 1))]
        print(f"[inspect] quality_score: p10={p10:.2f} p50={p50:.2f} p90={p90:.2f}", flush=True)
    if simhashes:
        print(f"[inspect] simhash_unique={uniq_simhash}/{len(simhashes)} ({_pct(uniq_simhash/max(1,len(simhashes)))})", flush=True)
    if total - empty > 0:
        print(f"[inspect] nonprintable_ratio>=2%: {nonprint_high}/{total-empty} ({_pct(nonprint_high/max(1,total-empty))})", flush=True)
        print(f"[inspect] has_table: {has_table}/{total-empty} ({_pct(has_table/max(1,total-empty))})", flush=True)
        if args.tables_only:
            denom = max(1, total - empty)
            t_total = len(table_token_counts)
            print(
                f"[inspect] tables_subset: {t_total}/{total-empty} ({_pct(t_total/denom)})",
                flush=True,
            )
            if table_token_counts:
                tq = _quantiles(table_token_counts)
                print(
                    f"[inspect] tables_subset token_count: min={tq['min']:.0f} avg={tq['avg']:.1f} p50={tq['p50']:.0f} p95={tq['p95']:.0f} max={tq['max']:.0f}",
                    flush=True,
                )
            if table_q_scores:
                qs = sorted(table_q_scores)
                n = len(qs)
                p10 = qs[int(0.10 * (n - 1))]
                p50 = qs[int(0.50 * (n - 1))]
                p90 = qs[int(0.90 * (n - 1))]
                print(f"[inspect] tables_subset quality_score: p10={p10:.2f} p50={p50:.2f} p90={p90:.2f}", flush=True)
            if table_simhashes:
                print(
                    f"[inspect] tables_subset simhash_unique={uniq_table_simhash}/{len(table_simhashes)} ({_pct(uniq_table_simhash/max(1,len(table_simhashes)))})",
                    flush=True,
                )
            if t_total > 0:
                print(
                    f"[inspect] tables_subset nonprintable_ratio>=2%: {table_nonprint_high}/{t_total} ({_pct(table_nonprint_high/max(1,t_total))})",
                    flush=True,
                )

    # sampling
    random.seed(int(args.seed))
    candidates: List[Dict[str, Any]] = []
    for r in recs:
        text = normalize_text(str(r.get("text") or ""))
        if not text:
            continue
        meta = r.get("metadata") or {}
        if args.tables_only:
            if not _has_table(text, meta):
                continue
        candidates.append(r)

    if not candidates:
        print("[inspect] no candidates for sampling (try without --tables_only)", flush=True)
        return

    k = min(int(args.sample), len(candidates))
    picks = random.sample(candidates, k=k)
    print(f"[inspect] samples={k} tables_only={bool(args.tables_only)}", flush=True)
    for i, r in enumerate(picks, start=1):
        text = normalize_text(str(r.get("text") or ""))
        meta = r.get("metadata") or {}
        tables = _extract_table_ids(meta)
        ex = (meta.get("extraction_stats") or {}) if isinstance(meta, dict) else {}
        ocr_used = ex.get("ocr_used")
        text_chars = ex.get("text_chars")
        table_chars = ex.get("table_chars")
        snippet = text[:700]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        print(f"\n--- sample {i}/{k} ---", flush=True)
        print(f"block_id={r.get('block_id')} doc_id={r.get('doc_id')} token_count={r.get('token_count')}", flush=True)
        print(f"ocr_used={ocr_used} text_chars={text_chars} table_chars={table_chars} table_ids={tables}", flush=True)
        print(f"text_snippet={snippet}", flush=True)


if __name__ == "__main__":
    main()


