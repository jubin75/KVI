#!/usr/bin/env python3
"""Download Qwen/Qwen2.5-7B-Instruct to a local dir (uses HF_ENDPOINT e.g. hf-mirror.com)."""
from __future__ import annotations

import os
import sys

# Respect HF_ENDPOINT from environment (e.g. https://hf-mirror.com)
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "Qwen2.5-7B-Instruct",
)


def main() -> None:
    local_dir = os.environ.get("QWEN25_LOCAL_DIR", DEFAULT_LOCAL)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}", flush=True)
    print(f"[download] repo={REPO_ID} -> {local_dir}", flush=True)
    snapshot_download(REPO_ID, local_dir=local_dir)
    print(f"[download] done: {local_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
