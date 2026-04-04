#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import requests


FILES = {
    "truthful_qa_generation_val.parquet": "https://hf-mirror.com/datasets/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet",
    "kilt_fever_validation.parquet": "https://hf-mirror.com/datasets/kilt_tasks/resolve/main/fever/validation-00000-of-00001.parquet",
    "cl_bench.jsonl": "https://hf-mirror.com/datasets/tencent/CL-bench/resolve/main/CL-bench.jsonl",
}


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Download required datasets via HF mirror resolve URLs")
    p.add_argument("--out_dir", default="/home/zd/dev/KVI/experiments/_mirror_data_resolved")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    for fname, url in FILES.items():
        out = out_dir / fname
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {fname} size={out.stat().st_size}")
            continue
        print(f"download {fname} ...")
        _download(url, out)
        print(f"done {fname} size={out.stat().st_size}")


if __name__ == "__main__":
    main()

