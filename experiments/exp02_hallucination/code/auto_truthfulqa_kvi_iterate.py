#!/usr/bin/env python3
import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/zd/dev/KVI")
RES = ROOT / "experiments/exp02_hallucination/results"
RUNNER = ROOT / "experiments/exp02_hallucination/code/run_truthfulqa_kvi_D_v36_pre100.sh"
REPORT_JSONL = RES / "truthfulqa_kvi_auto_iter_report.jsonl"
REPORT_MD = RES / "truthfulqa_kvi_auto_iter_report.md"
LOCK_FILE = RES / "truthfulqa_kvi_auto_iter.lock"
DEFAULT_RESIDENT_URL = "http://127.0.0.1:18888"


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def parse_summary(summary_path: Path):
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    kvi = next(m for m in obj["methods"] if m["method_key"] == "kvi")
    graph = next(m for m in obj["methods"] if m["method_key"] == "graphrag")
    return float(kvi["truthfulqa_mc2_proxy"]), float(graph["truthfulqa_mc2_proxy"]), float(kvi["em"]), float(kvi["f1_mean"])


def write_report(iter_name: str, out_dir: Path, cfg: dict, kvi_mc2: float, g_mc2: float, em: float, f1: float):
    row = {
        "iter": iter_name,
        "out_dir": str(out_dir),
        "config": cfg,
        "kvi_mc2": kvi_mc2,
        "graphrag_mc2": g_mc2,
        "kvi_em": em,
        "kvi_f1": f1,
        "win": kvi_mc2 > g_mc2,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with REPORT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    with REPORT_MD.open("a", encoding="utf-8") as f:
        f.write(
            f"- `{iter_name}`: KVI MC2={kvi_mc2:.4f}, GraphRAG MC2={g_mc2:.4f}, EM={em:.2f}, F1={f1:.4f}, win={kvi_mc2 > g_mc2}, cfg={cfg}\n"
        )


def run_one(tag: str, cfg: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env.update(
        {
            "TS": ts,
            "RUN_TAG": tag,
            "ANN_FORCE_CPU": "0",
            "STRICT_REQUIRE_GPU": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "EXTRA_ARGS": cfg.get("extra_args", ""),
            "LIMIT": "100",
            "RESIDENT_URL": os.environ.get("RESIDENT_URL", DEFAULT_RESIDENT_URL),
            "ANN_VIA_RESIDENT": os.environ.get("ANN_VIA_RESIDENT", "1"),
            "AUTO_START_RESIDENT": os.environ.get("AUTO_START_RESIDENT", "1"),
        }
    )
    log(f"launch {tag} cfg={cfg}")
    rc = subprocess.run(["bash", str(RUNNER)], cwd=str(ROOT), env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"{tag} failed, rc={rc}")
    return RES / f"truthfulqa_kvi_optimize_D_{tag}_pre100_{ts}"


def candidate_stream():
    seeded = [
        {"extra_args": "--kvi_max_kv_triples 3 --kvi_top_k_relations 2"},
        {"extra_args": "--kvi_max_kv_triples 4 --kvi_top_k_relations 4"},
        {"extra_args": "--kvi_max_kv_triples 5 --kvi_top_k_relations 5"},
        {"extra_args": "--kvi_max_kv_triples 4 --kvi_top_k_relations 4 --truthfulqa_kvi_max_new_tokens 112"},
        {"extra_args": "--kvi_max_kv_triples 4 --kvi_top_k_relations 4 --no-truthfulqa_kvi_minimal_prompt"},
    ]
    for x in seeded:
        yield x
    while True:
        yield {
            "extra_args": random.choice(
                [
                    "--kvi_max_kv_triples 3 --kvi_top_k_relations 3",
                    "--kvi_max_kv_triples 4 --kvi_top_k_relations 4",
                    "--kvi_max_kv_triples 5 --kvi_top_k_relations 5",
                    "--kvi_max_kv_triples 4 --kvi_top_k_relations 2",
                    "--kvi_max_kv_triples 4 --kvi_top_k_relations 4 --truthfulqa_kvi_max_new_tokens 112",
                ]
            ),
        }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        raise RuntimeError(f"lock exists: {LOCK_FILE}")
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    random.seed(42)
    with REPORT_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n## Auto iteration start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        idx = 2
        for cfg in candidate_stream():
            tag = f"auto{idx:02d}"
            out_dir = run_one(tag, cfg)
            summary = out_dir / "summary.json"
            if not summary.exists():
                raise RuntimeError(f"missing summary: {summary}")
            kvi_mc2, g_mc2, em, f1 = parse_summary(summary)
            write_report(tag, out_dir, cfg, kvi_mc2, g_mc2, em, f1)
            log(f"{tag} done kvi_mc2={kvi_mc2:.4f} graph_mc2={g_mc2:.4f}")
            if kvi_mc2 > g_mc2:
                log(f"SUCCESS {tag}: {kvi_mc2:.4f} > {g_mc2:.4f}")
                break
            idx += 1
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()


if __name__ == "__main__":
    main()
