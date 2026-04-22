#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _classify_failure(rec: Dict[str, Any]) -> str:
    oh = rec.get("oracle_hits") or {}
    if not oh.get("available"):
        return "NO_ORACLE"
    r = bool(oh.get("retrieval_hit_any"))
    d = bool(oh.get("drm_hit_any"))
    p = bool(oh.get("prompt_hit_any"))
    if not r:
        return "MISS_RETRIEVAL"
    if r and not d:
        return "MISS_DRM"
    if d and not p:
        return "MISS_PROMPT"
    return "HIT_PROMPT"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze graph retrieval/DRM/prompt audit JSONL")
    ap.add_argument("--audit_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.audit_jsonl))
    if not rows:
        raise SystemExit("No audit rows found.")

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("method_key") or "unknown")
        by_method.setdefault(m, []).append(r)

    report: Dict[str, Any] = {"total_rows": len(rows), "methods": {}}
    md_lines: List[str] = []
    md_lines.append("## Graph Prompt Audit")
    md_lines.append("")
    md_lines.append(f"- Total rows: {len(rows)}")
    md_lines.append("")

    for m, recs in sorted(by_method.items()):
        oracle_recs = [r for r in recs if (r.get("oracle_hits") or {}).get("available")]
        n = len(recs)
        n_oracle = len(oracle_recs)
        retrieval_any = sum(1 for r in oracle_recs if (r.get("oracle_hits") or {}).get("retrieval_hit_any"))
        drm_any = sum(1 for r in oracle_recs if (r.get("oracle_hits") or {}).get("drm_hit_any"))
        prompt_any = sum(1 for r in oracle_recs if (r.get("oracle_hits") or {}).get("prompt_hit_any"))
        denom = max(1, n_oracle)
        fail_counts = Counter(_classify_failure(r) for r in recs)
        method_obj = {
            "rows": n,
            "oracle_rows": n_oracle,
            "retrieval_hit_any_rate": retrieval_any / denom,
            "drm_hit_any_rate": drm_any / denom,
            "prompt_hit_any_rate": prompt_any / denom,
            "failure_breakdown": dict(fail_counts),
        }
        report["methods"][m] = method_obj

        md_lines.append(f"### {m}")
        md_lines.append(f"- Rows: {n} (oracle-available: {n_oracle})")
        md_lines.append(f"- R@retrieval(any): {retrieval_any}/{denom} = {retrieval_any/denom:.3f}")
        md_lines.append(f"- R@drm(any): {drm_any}/{denom} = {drm_any/denom:.3f}")
        md_lines.append(f"- R@prompt(any): {prompt_any}/{denom} = {prompt_any/denom:.3f}")
        md_lines.append(f"- Failure breakdown: {dict(fail_counts)}")
        md_lines.append("")

    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
