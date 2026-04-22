#!/usr/bin/env python3
"""Remove per-example truthfulqa_mc*_proxy so run_exp01 --resume recomputes them with current metrics code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions_jsonl", type=Path)
    args = p.parse_args()
    path = args.predictions_jsonl
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj.pop("truthfulqa_mc1_proxy", None)
        obj.pop("truthfulqa_mc2_proxy", None)
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "path": str(path), "n": len(out_lines)}, indent=2))


if __name__ == "__main__":
    main()
