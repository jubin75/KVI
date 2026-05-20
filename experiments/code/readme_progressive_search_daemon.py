#!/usr/bin/env python3
"""
Background daemon aligned with experiments/README.md (2026-04-28 progressive plan).

Goals (automated, no prompts):
- Step through P0 (retrieval floor), P2 (schema-first artifacts), P4-style Exp02-vNext
  comparisons to reduce dual-channel competition vs legacy triple-KV.
- Search a bounded optimization space (methods + schema retrieval knobs + schema grouping).
- When a method beats GraphRAG on BOTH TruthfulQA MC2 proxy and FEVER label accuracy
  (README Exp02-vNext pass rules), append a durable record to README_DAEMON_WINS.jsonl.

This process is intended to be started under nohup; it is serial (one eval at a time).

Env overrides:
  ROOT, MODEL, RESIDENT_URL, README_DAEMON_SLEEP_S, README_DAEMON_LIMIT,
  README_DAEMON_SCHEMA_DEVICE (cpu|cuda), README_DAEMON_MAX_ROUNDS (0=infinite)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_python(root: Path) -> str:
    v = root / "KVI" / "bin" / "python"
    if v.is_file():
        return str(v)
    v3 = root / "KVI" / "bin" / "python3"
    if v3.is_file():
        return str(v3)
    return sys.executable


def _journal(state_dir: Path, msg: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{_utc_now()}] {msg}\n"
    (state_dir / "README_DAEMON_JOURNAL.log").open("a", encoding="utf-8").write(line)
    print(line, end="", flush=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _resident_ok(url: str, timeout_s: float = 3.0) -> bool:
    base = str(url or "").strip().rstrip("/")
    if not base:
        return False
    try:
        req = urllib.request.Request(f"{base}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _run(
    cmd: List[str],
    *,
    cwd: Path,
    journal: Path,
    timeout_s: Optional[int] = None,
) -> None:
    _journal(journal.parent, "RUN " + " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    tail = (p.stdout or "")[-8000:]
    if p.returncode != 0:
        _journal(journal.parent, f"FAIL rc={p.returncode} tail=\n{tail}")
        raise RuntimeError(f"command_failed rc={p.returncode}")
    _journal(journal.parent, "OK tail=\n" + tail[-2000:])


def _phase_p0(root: Path, state_dir: Path, *, limit: int, device: str) -> Dict[str, Any]:
    """README P0: Exp03 retrieval metrics non-degenerate."""
    hotpot = root / "experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl"
    graph = root / "experiments/exp01_main_qa/artifacts/hotpot_distractor/graph_index.json"
    sents = root / "experiments/exp01_main_qa/artifacts/hotpot_distractor/sentences.jsonl"
    out_dir = state_dir / "exp03_daemon_probe"
    if not hotpot.is_file() or not graph.is_file() or not sents.is_file():
        msg = "P0 skip: missing hotpot_eval.jsonl or Hotpot artifacts"
        _journal(state_dir, msg)
        return {"skipped": True, "reason": msg}
    py = _resolve_python(root)
    cmd = [
        py,
        str(root / "experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py"),
        "--dataset_jsonl",
        str(hotpot),
        "--graph_index",
        str(graph),
        "--sentences_jsonl",
        str(sents),
        "--device",
        device,
        "--limit",
        str(int(limit)),
        "--out_dir",
        str(out_dir),
    ]
    _run(cmd, cwd=root, journal=state_dir / "README_DAEMON_JOURNAL.log", timeout_s=7200)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        return {"skipped": False, "error": "no metrics.json"}
    m = _load_json(metrics_path)
    _journal(state_dir, f"P0 metrics={json.dumps(m, ensure_ascii=False)[:900]}")
    return {"skipped": False, "metrics": m}


def _build_schema_for_dataset(
    root: Path,
    ds: str,
    *,
    group_by: str,
    state_dir: Path,
) -> Dict[str, Any]:
    art = root / "experiments/exp02_hallucination/artifacts" / ds
    src = art / "sentences.tagged.jsonl"
    out = art / "blocks.schema.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src}")
    py = _resolve_python(root)
    cmd = [
        py,
        str(root / "scripts/build_schema_blocks_from_evidence_jsonl.py"),
        "--blocks_jsonl_evidence",
        str(src),
        "--out_jsonl",
        str(out),
        "--group_by",
        str(group_by),
    ]
    _run(cmd, cwd=root, journal=state_dir / "README_DAEMON_JOURNAL.log", timeout_s=600)
    return {"dataset": ds, "blocks_schema": str(out), "group_by": group_by}


def _build_kvbank_schema(
    root: Path,
    ds: str,
    *,
    model: Path,
    schema_device: str,
    state_dir: Path,
) -> Dict[str, Any]:
    art = root / "experiments/exp02_hallucination/artifacts" / ds
    blocks = art / "blocks.schema.jsonl"
    out_dir = art / "kvbank_schema"
    if not blocks.is_file():
        raise FileNotFoundError(f"missing {blocks}")
    py = _resolve_python(root)
    dtype = "bfloat16" if schema_device == "cuda" else "float32"
    cmd = [
        py,
        str(root / "scripts/build_kvbank_from_blocks_jsonl.py"),
        "--blocks_jsonl",
        str(blocks),
        "--disable_enriched",
        "--out_dir",
        str(out_dir),
        "--base_llm",
        str(model),
        "--domain_encoder_model",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--layers",
        "0,1,2,3",
        "--block_tokens",
        "128",
        "--shard_size",
        "1024",
        "--device",
        str(schema_device),
        "--dtype",
        dtype,
    ]
    _run(cmd, cwd=root, journal=state_dir / "README_DAEMON_JOURNAL.log", timeout_s=14400)
    return {"dataset": ds, "kvbank_schema": str(out_dir)}


def _parse_summary(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (mc2_or_none_by_method, fever_label_by_method).
    TruthfulQA summary uses truthfulqa_mc2_proxy; FEVER uses fever_label_accuracy.
    """
    s = _load_json(path)
    by_mc2: Dict[str, float] = {}
    by_fever: Dict[str, float] = {}
    for row in s.get("methods") or []:
        mk = str(row.get("method_key") or "")
        if not mk:
            continue
        if "truthfulqa_mc2_proxy" in row:
            by_mc2[mk] = float(row["truthfulqa_mc2_proxy"])
        if "fever_label_accuracy" in row:
            by_fever[mk] = float(row["fever_label_accuracy"])
    return by_mc2, by_fever


def _winners_vs_graphrag(
    tqa: Path,
    fever: Path,
) -> List[Dict[str, Any]]:
    mc2_map, _ = _parse_summary(tqa)
    _, fev_map = _parse_summary(fever)
    base_mc2 = mc2_map.get("graphrag")
    base_fev = fev_map.get("graphrag")
    out: List[Dict[str, Any]] = []
    if base_mc2 is None or base_fev is None:
        return out
    for mk in mc2_map:
        if mk == "graphrag":
            continue
        m2 = mc2_map.get(mk)
        fv = fev_map.get(mk)
        if m2 is None or fv is None:
            continue
        if m2 > base_mc2 and fv > base_fev:
            out.append(
                {
                    "method_key": mk,
                    "truthfulqa_mc2_proxy": m2,
                    "graphrag_mc2": base_mc2,
                    "delta_mc2": round(m2 - base_mc2, 6),
                    "fever_label_accuracy": fv,
                    "graphrag_fever_label": base_fev,
                    "delta_fever": round(fv - base_fev, 6),
                }
            )
    return out


@dataclass
class Candidate:
    methods: str
    graph_max_schema_evidence: int
    graph_schema_min_score: float
    schema_group_by: str  # doc_id | source_id
    label: str = ""

    def stable_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _candidates() -> List[Candidate]:
    """
    README default methods + schema-writer (Exp01-oriented successor) and
    schema retrieval / grouping knobs that address P2 granularity.
    """
    method_sets = [
        "graphrag,kvi_triple_legacy,kvi_schema_verifier,kvi_noinject_planner",
        "graphrag,kvi_triple_legacy,kvi_schema_writer,kvi_schema_verifier,kvi_noinject_planner",
    ]
    knobs = [(8, 0.05), (12, 0.04), (16, 0.03), (6, 0.06)]
    groups = ["source_id", "doc_id"]
    out: List[Candidate] = []
    for ms in method_sets:
        for mx, mn in knobs:
            for g in groups:
                out.append(
                    Candidate(
                        methods=ms,
                        graph_max_schema_evidence=int(mx),
                        graph_schema_min_score=float(mn),
                        schema_group_by=g,
                        label=f"{g}_mx{mx}_mn{mn}",
                    )
                )
    return out


def _run_exp02(
    root: Path,
    c: Candidate,
    *,
    result_tag: str,
    model: Path,
    resident_url: str,
    limit: int,
    ann_via_resident: bool,
    state_dir: Path,
) -> None:
    py = _resolve_python(root)
    cmd = [
        py,
        "-u",
        str(root / "experiments/exp02_hallucination/code/run_exp02_hallucination.py"),
        "--root",
        str(root),
        "--model",
        str(model),
        "--resident_url",
        str(resident_url),
        "--methods",
        str(c.methods),
        "--result_tag",
        str(result_tag),
        "--only_datasets",
        "truthfulqa,fever",
        "--limit",
        str(int(limit)),
        "--skip_mirror_and_prepare",
        "--reuse_artifacts",
        "--graph_max_schema_evidence",
        str(int(c.graph_max_schema_evidence)),
        "--graph_schema_min_score",
        str(float(c.graph_schema_min_score)),
    ]
    if ann_via_resident:
        cmd.append("--ann_via_resident")
    _run(cmd, cwd=root, journal=state_dir / "README_DAEMON_JOURNAL.log", timeout_s=86400)


def _append_win_record(
    state_dir: Path,
    *,
    candidate: Candidate,
    result_tag: str,
    winners: List[Dict[str, Any]],
    paths: Dict[str, str],
) -> None:
    rec = {
        "ts": _utc_now(),
        "readme_section": "2026-04-28 Progressive Experiment Plan / Exp02-vNext pass rules",
        "implementation_notes": (
            "Daemon ran run_exp02_hallucination.py with --reuse_artifacts after rebuilding "
            "blocks.schema.jsonl via scripts/build_schema_blocks_from_evidence_jsonl.py "
            f"(group_by={candidate.schema_group_by}) and kvbank_schema via "
            "scripts/build_kvbank_from_blocks_jsonl.py. "
            "Dual-channel mitigation per README: prefer kvi_schema_verifier / kvi_noinject_planner "
            "(schema for plan/verify; evidence-only writer) over kvi_triple_legacy."
        ),
        "candidate": asdict(candidate),
        "result_tag": result_tag,
        "summaries": paths,
        "winners_vs_graphrag": winners,
    }
    path = state_dir / "README_DAEMON_WINS.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _journal(state_dir, f"WIN_RECORD written -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="README-aligned progressive search daemon")
    p.add_argument("--root", default=os.environ.get("ROOT", "/home/zd/dev/KVI"))
    p.add_argument("--model", default=os.environ.get("MODEL", ""))
    p.add_argument("--resident-url", default=os.environ.get("RESIDENT_URL", "http://127.0.0.1:18888"))
    p.add_argument("--state-dir", default="", help="Default: <root>/experiments/exp02_hallucination/results")
    p.add_argument("--sleep-s", type=float, default=float(os.environ.get("README_DAEMON_SLEEP_S", "30")))
    p.add_argument("--limit", type=int, default=int(os.environ.get("README_DAEMON_LIMIT", "100")))
    p.add_argument("--schema-device", default=os.environ.get("README_DAEMON_SCHEMA_DEVICE", "cpu"))
    p.add_argument("--p0-limit", type=int, default=80)
    p.add_argument("--p0-device", default="cpu")
    p.add_argument("--skip-p0", action="store_true")
    p.add_argument("--require-resident", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ann-via-resident", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-rounds", type=int, default=int(os.environ.get("README_DAEMON_MAX_ROUNDS", "0")))
    p.add_argument("--once", action="store_true", help="Run a single candidate then exit")
    args = p.parse_args()

    root = Path(str(args.root)).resolve()
    model = Path(str(args.model or (root / "models/Qwen2.5-7B-Instruct"))).resolve()
    state_dir = Path(args.state_dir) if str(args.state_dir).strip() else (root / "experiments/exp02_hallucination/results")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "README_DAEMON_STATE.json"

    resident = str(args.resident_url).strip()
    if bool(args.require_resident) and not _resident_ok(resident):
        _journal(state_dir, f"ERROR: resident not healthy at {resident}/health ; start exp01_resident_infer_service.py first")
        sys.exit(2)

    if not model.is_dir():
        _journal(state_dir, f"ERROR: model path not found: {model}")
        sys.exit(3)

    st: Dict[str, Any] = {}
    if state_path.is_file():
        try:
            st = _load_json(state_path)
        except Exception:
            st = {}

    cand_list = _candidates()
    n_c = len(cand_list)
    if n_c == 0:
        _journal(state_dir, "ERROR: empty candidate list")
        sys.exit(4)

    if not bool(args.skip_p0) and not st.get("p0_done"):
        try:
            st["p0"] = _phase_p0(root, state_dir, limit=int(args.p0_limit), device=str(args.p0_device))
            st["p0_done"] = True
        except Exception as e:
            st["p0"] = {"error": str(e)}
            st["p0_done"] = True
        _save_json(state_path, st)

    rounds = int(st.get("rounds_completed", 0))
    while int(args.max_rounds) <= 0 or rounds < int(args.max_rounds):
        try:
            ci = int(st.get("next_candidate_index", 0)) % n_c
        except Exception:
            ci = 0
        c = cand_list[ci]

        tag = f"readme_daemon_{c.stable_id()}_{c.label}".replace(".", "p")[:120]
        _journal(state_dir, f"=== candidate[{ci}/{n_c}] id={c.stable_id()} tag={tag} {asdict(c)}")

        try:
            for ds in ("truthfulqa", "fever"):
                _build_schema_for_dataset(root, ds, group_by=c.schema_group_by, state_dir=state_dir)
            for ds in ("truthfulqa", "fever"):
                _build_kvbank_schema(
                    root,
                    ds,
                    model=model,
                    schema_device=str(args.schema_device),
                    state_dir=state_dir,
                )
            _run_exp02(
                root,
                c,
                result_tag=tag,
                model=model,
                resident_url=resident,
                limit=int(args.limit),
                ann_via_resident=bool(args.ann_via_resident),
                state_dir=state_dir,
            )
            tqa_path = state_dir / f"truthfulqa_{tag}" / "summary.json"
            fev_path = state_dir / f"fever_{tag}" / "summary.json"
            winners = _winners_vs_graphrag(tqa_path, fev_path)
            st["last_ok"] = {"tag": tag, "winners": winners, "ts": _utc_now()}
            if winners:
                _append_win_record(
                    state_dir,
                    candidate=c,
                    result_tag=tag,
                    winners=winners,
                    paths={"truthfulqa": str(tqa_path), "fever": str(fev_path)},
                )
        except Exception as e:
            st["last_error"] = {"tag": tag, "error": str(e), "ts": _utc_now()}
            _journal(state_dir, f"ERROR candidate {tag}: {e}")

        st["next_candidate_index"] = int(ci) + 1
        rounds += 1
        st["rounds_completed"] = rounds
        _save_json(state_path, st)

        if bool(args.once):
            break
        time.sleep(max(1.0, float(args.sleep_s)))


if __name__ == "__main__":
    main()
