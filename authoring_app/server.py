from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
import re
import shutil
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root, repo_root.parent]
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_repo_root_on_syspath()

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # external_kv_injection/
TOPICS_DIR = PROJECT_ROOT / "config" / "topics"

# ----------------------------
# Sentence-KVBank hard budgets
# ----------------------------
# These budgets are enforced server-side regardless of UI input.
DEFAULT_MAX_SENTENCE_TOKENS = 128
HARD_MAX_INJECTED_SENTENCES_PER_STEP = 4
HARD_MAX_TOTAL_INJECTED_TOKENS = 512

# Build-pipeline runtime state (for UI polling + per-topic mutex)
_PIPELINE_STATE_LOCK = threading.Lock()
_PIPELINE_STATE: Dict[str, Dict[str, Any]] = {}
_PIPELINE_TOPIC_LOCKS: Dict[str, threading.Lock] = {}

_DEFAULT_SEMANTIC_TYPE_SPECS = {
    "symptom": {
        "description": "临床表现、症状体征、实验室异常、常见表现的枚举或陈述句。",
        "threshold": 0.45,
        # Optional runtime guidance (config-driven; no evidence text in prompt).
        "focus_terms": ["症状", "体征", "临床表现", "实验室异常", "出血", "呕吐", "腹泻", "白细胞", "血小板"],
        "deny_terms": ["机制", "发病机制", "致病机制", "感染", "免疫", "内皮", "通透性", "复制", "通路", "汉滩病毒", "汉坦病毒"],
        "keywords": ["症状", "体征", "发热", "乏力", "恶心", "呕吐", "腹泻", "出血", "意识障碍", "皮疹", "白细胞", "血小板"],
        "keyword_boost": 0.06,
    },
    "drug": {
        "description": "治疗、用药、药物、获批/批准、疗效、不良反应等相关陈述句。",
        "threshold": 0.55,
        "focus_terms": ["治疗", "用药", "药物", "疗效", "不良反应", "获批", "批准"],
        "deny_terms": ["地区分布", "流行区域", "机制", "发病机制"],
        "keywords": ["药物", "治疗", "用药", "法维拉韦", "获批", "批准", "不良反应", "抗病毒", "支持治疗"],
        "keyword_boost": 0.06,
    },
    "location": {
        "description": "地区分布、流行区域、病例报告地点、地理范围等相关陈述句。",
        "threshold": 0.50,
        "focus_terms": ["地区", "分布", "流行", "报告", "病例", "省", "市", "国家", "区域"],
        "deny_terms": ["治疗", "用药", "药物", "机制", "发病机制"],
        "keywords": ["地区", "分布", "流行", "报告", "病例", "省", "市", "国家", "区域"],
        "keyword_boost": 0.05,
    },
    "mechanism": {
        "description": "作用机制/发病机制：感染哪些细胞、免疫应答/免疫抑制、炎症反应、病理过程、通透性改变、多器官损伤等。",
        "threshold": 0.50,
        "focus_terms": ["机制", "发病机制", "致病机制", "感染", "免疫", "炎症", "内皮", "通透性", "细胞", "病理过程"],
        "deny_terms": ["临床表现", "症状", "体征", "治疗", "用药", "地区分布", "流行区域"],
        "keywords": ["机制", "发病机制", "致病机制", "感染", "免疫", "炎症", "内皮", "通透性", "病理", "多器官"],
        "keyword_boost": 0.06,
    },
}

_INDEX_FALLBACK_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KVI UI (static missing)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }}
    .box {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; max-width: 980px; }}
    .muted {{ color: #555; }}
  </style>
</head>
<body>
  <div class="box">
    <h2>KVI UI static files not found</h2>
    <p class="muted">
      The HTTP server is running, but <code>static/index.html</code> was not found on disk.
    </p>
    <p>
      Expected directory: <code>{STATIC_DIR.as_posix()}</code>
    </p>
    <p>
      Quick checks (run on server):
      <ul>
        <li><code>ls -la {STATIC_DIR.as_posix()}</code></li>
        <li><code>ls -la { (STATIC_DIR / "index.html").as_posix() }</code></li>
        <li><code>curl -v http://127.0.0.1:8765/api/health</code></li>
      </ul>
    </p>
    <p>
      API is reachable: <a href="/api/health">/api/health</a> · <a href="/api/kvi/topics">/api/kvi/topics</a>
    </p>
  </div>
</body>
</html>
"""


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = _safe_json_loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _safe_parse_last_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last JSON object from a mixed stdout (common for CLIs).
    """
    import re

    s = str(text or "").strip()
    if not s:
        return None
    # Try to locate a JSON object at the end.
    m = re.search(r"\{[\s\S]*\}\s*$", s)
    if m:
        obj = _safe_json_loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    # Fallback: first JSON object
    m2 = re.search(r"\{[\s\S]*\}", s)
    if not m2:
        return None
    obj2 = _safe_json_loads(m2.group(0))
    return obj2 if isinstance(obj2, dict) else None


def _topic_dir(topic: str) -> Path:
    t = str(topic or "").strip()
    if not t:
        raise ValueError("topic is required")
    p = (TOPICS_DIR / t).resolve()
    if not str(p).startswith(str(TOPICS_DIR.resolve())):
        raise ValueError("invalid topic")
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"topic dir not found: {p}")
    return p


def _validate_topic_name(name: str) -> str:
    t = str(name or "").strip()
    if not t:
        raise ValueError("topic name is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", t):
        raise ValueError("invalid topic name (use letters/digits/_/- , max 32 chars)")
    return t


def _create_topic(topic: str) -> Path:
    """
    Create a new topic directory with a minimal config.json.
    Default work_dir follows your remote convention: /home/jb/topics/<TOPIC>/work
    """
    t = _validate_topic_name(topic)
    td = (TOPICS_DIR / t).resolve()
    if not str(td).startswith(str(TOPICS_DIR.resolve())):
        raise ValueError("invalid topic path")
    if td.exists():
        raise FileExistsError(f"topic already exists: {t}")
    td.mkdir(parents=True, exist_ok=False)
    cfg = {
        "topic_name": t,
        "goal": "",
        "build": {
            "work_dir": f"/home/jb/topics/{t}/work",
            "base_llm": "Qwen/Qwen2.5-7B-Instruct",
            "retrieval_encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "layers": [0, 1, 2, 3],
        },
    }
    (td / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return td


def _delete_topic(topic: str) -> Path:
    t = _validate_topic_name(topic)
    td = (TOPICS_DIR / t).resolve()
    if not str(td).startswith(str(TOPICS_DIR.resolve())):
        raise ValueError("invalid topic path")
    if not td.exists() or not td.is_dir():
        raise FileNotFoundError(f"topic not found: {t}")
    shutil.rmtree(td)
    return td


def _load_topic_config(topic: str) -> Dict[str, Any]:
    td = _topic_dir(topic)
    cfg_path = td / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        obj = json.loads(cfg_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _topic_build_cfg(topic: str) -> Dict[str, Any]:
    cfg = _load_topic_config(topic)
    build = cfg.get("build") if isinstance(cfg.get("build"), dict) else {}
    return build if isinstance(build, dict) else {}


def _topic_evidence_txt_path(topic: str) -> Path:
    return _topic_dir(topic) / "evidence.txt"


def _topic_evidence_sets_dir(topic: str) -> Path:
    """
    Runtime artifacts SHOULD live in topic build.work_dir, not the code directory.
    Fallback to repo topic_dir only if work_dir is not configured.
    """
    build = _topic_build_cfg(topic)
    work_dir_raw = str(build.get("work_dir") or "").strip()
    if work_dir_raw:
        return Path(work_dir_raw).expanduser() / "evidence_sets"
    return _topic_dir(topic) / "evidence_sets"


def _ensure_pattern_contract(*, topic_root: Path, semantic_specs: Dict[str, Any], sentence_mode: bool = False) -> Path:
    """
    Create a minimal pattern_contract.json if missing.
    This is a safe fallback to avoid reject_no_contract in KVI2 runtime.
    """
    path = topic_root / "pattern_contract.json"
    specs = semantic_specs if isinstance(semantic_specs, dict) else {}
    keys = [str(k).strip().lower() for k in specs.keys() if str(k).strip()]
    if not keys:
        keys = ["symptom", "drug", "mechanism", "location"]

    def _needs_regen() -> bool:
        if not path.exists():
            return True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return True
        patterns = payload.get("patterns") if isinstance(payload, dict) else None
        if not isinstance(patterns, list):
            return True
        # Sentence-KVBank: ensure schema slots match semantic specs (avoid stale clinical_features).
        if sentence_mode:
            want = {f"schema:{k}" for k in keys}
            have = {str(p.get("pattern_id") or "") for p in patterns if isinstance(p, dict)}
            if not want.issubset(have):
                return True
            if any("clinical_features" in pid for pid in have):
                return True
        return False

    if not _needs_regen():
        return path
    patterns = []
    for k in keys:
        if k == "symptom":
            forms = ["X 的临床症状有哪些", "X 的临床表现有哪些", "X 有哪些症状"]
        elif k == "drug":
            forms = ["X 的治疗方法有哪些", "X 的用药建议是什么", "X 的临床用药推荐"]
        elif k == "mechanism":
            forms = ["X 的发病机制是什么", "X 的作用机制是什么"]
        elif k == "location":
            forms = ["X 的流行地区有哪些", "X 的地理分布是什么"]
        else:
            forms = [f"X 的{k}有哪些"]
        inference_level = "soft" if sentence_mode else "schema"
        patterns.append(
            {
                "pattern_id": f"schema:{k}",
                "question_skeleton": {"intent": "ask_schema_list", "surface_forms": forms},
                "slots": {
                    k: {
                        "type": "string",
                        "required": False,
                        "evidence_type": ["schema"],
                        "min_evidence": 1,
                        "inference_level": inference_level,
                        "semantic_type": k,
                    }
                },
                "answer_style": "factual",
            }
        )
    payload = {"topic": str(topic_root.name), "patterns": patterns}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _evidence_manifest_path(topic: str) -> Path:
    return _topic_evidence_sets_dir(topic) / "manifest.json"


def _safe_now_iso() -> str:
    # local time is fine for UI traces; keep ISO-like formatting
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_sentence_id(*, topic: str, claim: str, source_id: str) -> str:
    h = hashlib.sha1()
    h.update(str(topic or "").encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(source_id or "").encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(claim or "").strip().encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_topic_tokenizer(topic: str) -> Optional[Any]:
    """
    Load/cache the HF tokenizer for a topic's base LLM (used for token budget splitting).
    Returns None if base_llm is missing or tokenizer load fails.
    """
    build = _topic_build_cfg(topic)
    base_llm = str(build.get("base_llm") or "").strip()
    if not base_llm:
        return None
    if base_llm in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[base_llm]
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(base_llm, use_fast=True, trust_remote_code=True)
        _TOKENIZER_CACHE[base_llm] = tok
        return tok
    except Exception:
        return None


def _token_len(tok: Any, text: str) -> int:
    try:
        ids = tok.encode(str(text or ""), add_special_tokens=False)
        return int(len(ids))
    except Exception:
        # fallback heuristic: ~2 chars per token for zh/mixture
        s = str(text or "")
        return int(max(0, len(s) // 2))


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,，;；、])\s*")


def _split_claim_to_budget(*, tok: Any, claim: str, max_sentence_tokens: int, max_chunks: int = 4) -> Tuple[List[str], Dict[str, Any]]:
    """
    Split a long claim into 2–4 more atomic sentences within token budget.
    If still too long/too many, keep first max_chunks and mark as truncated.
    Returns (chunks, stats).
    """
    c = str(claim or "").strip()
    stats: Dict[str, Any] = {"was_split": False, "was_truncated": False, "orig_tokens": 0, "chunks": 0}
    if not c:
        return [], stats
    orig_tokens = _token_len(tok, c)
    stats["orig_tokens"] = int(orig_tokens)
    if orig_tokens <= int(max_sentence_tokens):
        stats["chunks"] = 1
        return [c], stats

    parts = [p.strip() for p in _SENT_SPLIT_RE.split(c) if p.strip()]
    if len(parts) <= 1:
        parts = [c]

    fine_parts: List[str] = []
    for p in parts:
        if _token_len(tok, p) <= int(max_sentence_tokens):
            fine_parts.append(p)
            continue
        subs = [x.strip() for x in _CLAUSE_SPLIT_RE.split(p) if x.strip()]
        fine_parts.extend(subs if subs else [p])

    # Greedy pack into <=max_sentence_tokens chunks
    chunks: List[str] = []
    buf = ""
    for seg in fine_parts:
        if not buf:
            buf = seg
            continue
        cand = (buf + " " + seg).strip()
        if _token_len(tok, cand) <= int(max_sentence_tokens):
            buf = cand
        else:
            chunks.append(buf)
            buf = seg
    if buf:
        chunks.append(buf)

    if len(chunks) > int(max_chunks):
        chunks = chunks[: int(max_chunks)]
        stats["was_truncated"] = True

    # If we still ended up with a single overlong chunk, mark truncated (builder will also truncate)
    if len(chunks) == 1 and _token_len(tok, chunks[0]) > int(max_sentence_tokens):
        stats["was_truncated"] = True

    stats["was_split"] = True
    stats["chunks"] = int(len(chunks))
    return chunks, stats


def _load_manifest(topic: str) -> Dict[str, Any]:
    p = _evidence_manifest_path(topic)
    if not p.exists():
        return {"version": 1, "sets": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj.setdefault("version", 1)
            obj.setdefault("sets", {})
            if not isinstance(obj.get("sets"), dict):
                obj["sets"] = {}
            return obj
    except Exception:
        pass
    return {"version": 1, "sets": {}}


def _save_manifest(topic: str, manifest: Dict[str, Any]) -> None:
    d = _topic_evidence_sets_dir(topic)
    d.mkdir(parents=True, exist_ok=True)
    p = _evidence_manifest_path(topic)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_migrated_txt_to_sets(topic: str) -> None:
    """
    One-time migration: if evidence_sets is empty but evidence.txt exists,
    create evidence_000_legacy.jsonl with doi=null.
    """
    d = _topic_evidence_sets_dir(topic)
    d.mkdir(parents=True, exist_ok=True)
    has_any = any(x.is_file() and x.suffix == ".jsonl" for x in d.iterdir())
    if has_any:
        return
    # Migration source is still the legacy file in code dir.
    txt_path = _topic_evidence_txt_path(topic)
    if not txt_path.exists():
        return
    lines = [ln.strip() for ln in _read_text_file(txt_path).splitlines() if ln.strip()]
    if not lines:
        return
    out_path = d / "evidence_000_legacy.jsonl"
    now = _safe_now_iso()
    recs: List[Dict[str, Any]] = []
    for ln in lines:
        recs.append(
            {
                "id": _stable_sentence_id(topic=topic, claim=ln, source_id=""),
                "topic": topic,
                "claim": ln,
                "source_id": None,
                "source_ref": {"doi": None, "title": None},
                "created_at": now,
                "updated_at": now,
                "author": None,
                "tags": [],
            }
        )
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mf = _load_manifest(topic)
    mf.setdefault("sets", {})
    mf["sets"][out_path.name] = {"enabled": True, "note": "migrated_from_evidence_txt", "created_at": now}
    _save_manifest(topic, mf)


def _list_evidence_sets(topic: str) -> List[Dict[str, Any]]:
    _ensure_migrated_txt_to_sets(topic)
    d = _topic_evidence_sets_dir(topic)
    d.mkdir(parents=True, exist_ok=True)
    mf = _load_manifest(topic)
    sets_cfg = mf.get("sets") if isinstance(mf.get("sets"), dict) else {}
    items: List[Dict[str, Any]] = []
    for fp in sorted([x for x in d.iterdir() if x.is_file() and x.suffix == ".jsonl"]):
        cfg = sets_cfg.get(fp.name) if isinstance(sets_cfg, dict) else None
        enabled = True if not isinstance(cfg, dict) else bool(cfg.get("enabled", True))
        # quick count
        cnt = 0
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cnt += 1
        except Exception:
            cnt = 0
        items.append(
            {
                "name": fp.name,
                "path": str(fp),
                "enabled": bool(enabled),
                "count": int(cnt),
                "note": (cfg.get("note") if isinstance(cfg, dict) else None),
            }
        )
    return items


def _read_sentence_set(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = _safe_json_loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _write_sentence_set(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in (records or []):
            if not isinstance(r, dict):
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _collect_compiled_claims(topic: str, enabled_only: bool = True) -> Tuple[List[str], Dict[str, Any]]:
    """
    Collect claims from enabled evidence sets.
    Returns (claims, stats).
    """
    items = _list_evidence_sets(topic)
    claims: List[str] = []
    stats: Dict[str, Any] = {"sets": [], "total": 0}
    for it in items:
        if enabled_only and not bool(it.get("enabled")):
            continue
        fp = Path(str(it.get("path") or ""))
        recs = _read_sentence_set(fp)
        added = 0
        for r in recs:
            c = str(r.get("claim") or "").strip()
            if not c:
                continue
            claims.append(c)
            added += 1
        stats["sets"].append({"name": it.get("name"), "enabled": bool(it.get("enabled")), "count": int(added)})
        stats["total"] = int(stats.get("total") or 0) + int(added)
    # dedupe keep order
    seen: set[str] = set()
    dedup: List[str] = []
    for c in claims:
        if c in seen:
            continue
        seen.add(c)
        dedup.append(c)
    stats["dedup_total"] = len(dedup)
    return dedup, stats


def _collect_enabled_sentence_records(topic: str, enabled_only: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect full sentence records from enabled evidence sets.
    Returns (records, stats).
    """
    items = _list_evidence_sets(topic)
    out: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"sets": [], "total": 0}
    for it in items:
        if enabled_only and not bool(it.get("enabled")):
            continue
        fp = Path(str(it.get("path") or ""))
        recs = _read_sentence_set(fp)
        added = 0
        for r in recs:
            if not isinstance(r, dict):
                continue
            claim = str(r.get("claim") or "").strip()
            if not claim:
                continue
            out.append(r)
            added += 1
        stats["sets"].append({"name": it.get("name"), "enabled": bool(it.get("enabled")), "count": int(added)})
        stats["total"] = int(stats.get("total") or 0) + int(added)

    # best-effort dedupe by (claim, source_id)
    seen: set[Tuple[str, str]] = set()
    dedup: List[Dict[str, Any]] = []
    for r in out:
        claim = str(r.get("claim") or "").strip()
        src = str(r.get("source_id") or "").strip()
        key = (claim, src)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    stats["dedup_total"] = int(len(dedup))
    return dedup, stats


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content or ""), encoding="utf-8")


def _load_doc_approvals(work_dir: Path) -> Dict[str, bool]:
    p = work_dir / "docs.approvals.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {str(k): bool(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _save_doc_approvals(work_dir: Path, approvals: Dict[str, bool]) -> None:
    p = work_dir / "docs.approvals.json"
    p.write_text(json.dumps(approvals, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_doc_details(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    p = work_dir / "docs.details.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    out[str(k)] = v
            return out
    except Exception:
        pass
    return {}


def _save_doc_details(work_dir: Path, data: Dict[str, Dict[str, Any]]) -> None:
    p = work_dir / "docs.details.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_topic_pipeline_lock(topic: str) -> threading.Lock:
    with _PIPELINE_STATE_LOCK:
        lk = _PIPELINE_TOPIC_LOCKS.get(topic)
        if lk is None:
            lk = threading.Lock()
            _PIPELINE_TOPIC_LOCKS[topic] = lk
        return lk


def _pipeline_set(topic: str, **kwargs: Any) -> None:
    with _PIPELINE_STATE_LOCK:
        cur = _PIPELINE_STATE.get(topic, {"topic": topic, "running": False, "logs": [], "updated_at": _safe_now_iso()})
        cur.update(kwargs)
        cur["updated_at"] = _safe_now_iso()
        _PIPELINE_STATE[topic] = cur


def _pipeline_log(topic: str, line: str) -> None:
    with _PIPELINE_STATE_LOCK:
        cur = _PIPELINE_STATE.get(topic, {"topic": topic, "running": False, "logs": [], "updated_at": _safe_now_iso()})
        logs = list(cur.get("logs") or [])
        logs.append(str(line))
        # Keep last 300 lines
        cur["logs"] = logs[-300:]
        cur["updated_at"] = _safe_now_iso()
        _PIPELINE_STATE[topic] = cur


def _pipeline_get(topic: str) -> Dict[str, Any]:
    with _PIPELINE_STATE_LOCK:
        cur = _PIPELINE_STATE.get(topic)
        if not cur:
            return {"topic": topic, "running": False, "logs": [], "updated_at": _safe_now_iso()}
        out = {
            "topic": topic,
            "running": bool(cur.get("running")),
            "logs": list(cur.get("logs") or []),
            "updated_at": cur.get("updated_at"),
            "last_error": cur.get("last_error"),
            "last_result_ok": cur.get("last_result_ok"),
        }
        if cur.get("last_result") is not None:
            out["last_result"] = cur["last_result"]
        return out


def _run_subprocess_with_log_stream(
    topic: str,
    cmd: List[str],
    cwd: str,
    timeout_s: int,
    log_prefix: str = "step4",
    heartbeat_interval_s: int = 60,
) -> Tuple[int, str, str]:
    """Run cmd with Popen; stream stderr to _pipeline_log; log heartbeat every heartbeat_interval_s while running. Returns (returncode, stdout, stderr)."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: List[str] = []
    start = time.time()
    last_heartbeat = start

    def read_stderr() -> None:
        for line in iter(proc.stderr.readline, ""):
            if line:
                line = line.rstrip("\n\r")
                stderr_lines.append(line)
                _pipeline_log(topic, f"[{log_prefix}] {line}")

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval_s:
                last_heartbeat = now
                elapsed_min = int((now - start) / 60)
                _pipeline_log(topic, f"[{log_prefix}] DeepSeek extraction in progress... ({elapsed_min} min)")
            if (now - start) > timeout_s:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout_s)
        stdout_data = (proc.stdout.read() if proc.stdout else "") or ""
        reader.join(timeout=5)
    except subprocess.TimeoutExpired:
        raise
    finally:
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
    return proc.returncode or 0, stdout_data, "\n".join(stderr_lines)


def _run_build_full_pipeline_background(topic: str, obj: Dict[str, Any], topic_lock: threading.Lock) -> None:
    """Run full pipeline in background; set last_result and release topic_lock when done."""
    def _finish(ok: Optional[bool], err: Optional[str], last_result: Optional[Dict[str, Any]] = None) -> None:
        _pipeline_set(topic, running=False, last_result_ok=ok, last_error=err, **({"last_result": last_result} if last_result is not None else {}))
        try:
            topic_lock.release()
        except Exception:
            pass

    topic_cfg = _load_topic_config(topic)
    build = _topic_build_cfg(topic)
    base_llm = str(build.get("base_llm") or "").strip()
    encoder = str(build.get("retrieval_encoder_model") or "").strip()
    topic_dir = _topic_dir(topic)
    work_dir_raw = str(build.get("work_dir") or "").strip()
    out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_base_url = str(topic_cfg.get("deepseek_base_url") or "https://api.deepseek.com").strip()
    ds_model = str(topic_cfg.get("deepseek_model") or "deepseek-chat").strip()
    ds_api_key_env = str(topic_cfg.get("deepseek_api_key_env") or "DEEPSEEK_API_KEY").strip()
    use_deepseek = bool(ds_model and ds_api_key_env)
    filter_doc_id = str(obj.get("doc_id") or "").strip()
    use_base_llm_extraction = bool(obj.get("use_base_llm_extraction")) or bool(filter_doc_id)
    if use_base_llm_extraction:
        use_deepseek = False
        if not base_llm:
            _finish(False, "missing_config", {"ok": False, "error": "bad_request", "message": "Single-doc / base_llm extraction requires build.base_llm in topic config.json"})
            return
    elif not use_deepseek and not base_llm:
        _finish(False, "missing_config", {"ok": False, "error": "bad_request", "message": "Missing DeepSeek config or build.base_llm in topic config.json"})
        return
    if not encoder:
        encoder = ""
    device = str(obj.get("device") or "").strip()
    timeout_s = int(obj.get("timeout_s") or 600)
    _pipeline_log(topic, f"[cfg] timeout_s={timeout_s} device={device or '(auto)'} deepseek={use_deepseek} doc_id={filter_doc_id or '(all)'} use_base_llm_extraction={use_base_llm_extraction}")

    blocks_candidates = ["blocks.evidence.jsonl", "blocks.enriched.jsonl", "blocks.jsonl"]
    recs: List[Dict[str, Any]] = []
    sentence_source = ""
    for bname in blocks_candidates:
        bpath = out_dir / bname
        if bpath.exists():
            blocks = _read_jsonl(bpath)
            for b in blocks:
                if filter_doc_id:
                    b_doc = str(b.get("doc_id") or b.get("source_id") or "").strip()
                    if b_doc != filter_doc_id:
                        continue
                text = str(b.get("text") or b.get("claim") or "").strip()
                if text:
                    recs.append({
                        "id": str(b.get("block_id") or b.get("id") or ""),
                        "claim": text,
                        "source_id": str(b.get("doc_id") or b.get("source_id") or ""),
                        "source_ref": {"doi": None, "title": None},
                        "author": None,
                        "tags": [],
                    })
            if recs:
                sentence_source = f"{bname} ({len(recs)} blocks)" + (f" [doc_id={filter_doc_id}]" if filter_doc_id else " [all docs]")
                _pipeline_log(topic, f"[step1] source={sentence_source}")
                break
    if not recs and not filter_doc_id:
        recs, _ = _collect_enabled_sentence_records(topic, enabled_only=True)
        sentence_source = f"evidence_sets ({len(recs)} records)"
    if not recs:
        msg = f"No evidence for doc_id={filter_doc_id}" if filter_doc_id else f"No evidence in work_dir={out_dir}"
        _finish(False, "no_evidence", {"ok": False, "error": "bad_request", "message": msg})
        return

    sentences_jsonl = out_dir / "sentences.jsonl"
    seen_ids: set = set()
    written = 0
    with sentences_jsonl.open("w", encoding="utf-8") as f:
        for r in recs:
            if not isinstance(r, dict):
                continue
            claim = str(r.get("claim") or "").strip()
            if not claim:
                continue
            src = str(r.get("source_id") or "").strip()
            sid = str(r.get("id") or "").strip() or _stable_sentence_id(topic=topic, claim=claim, source_id=src)
            bid = f"sent_{sid}"
            if bid in seen_ids:
                continue
            seen_ids.add(bid)
            meta = {
                "kind": "sentence",
                "topic": topic,
                "source_ref": r.get("source_ref") if isinstance(r.get("source_ref"), dict) else {"doi": None, "title": None},
                "author": r.get("author"),
                "tags": r.get("tags") if isinstance(r.get("tags"), list) else [],
                "evidence_id": sid,
            }
            f.write(json.dumps({"block_id": bid, "text": claim, "source_id": (src if src else None), "doc_id": (src if src else None), "metadata": meta}, ensure_ascii=False) + "\n")
            written += 1
    pipeline_status: Dict[str, Any] = {"sentences_compiled": written, "sentence_source": sentence_source}
    _pipeline_log(topic, f"[step1] compiled_sentences={written}")

    sentences_tagged = out_dir / "sentences.tagged.jsonl"
    sentences_src = sentences_jsonl
    if encoder:
        specs_path = out_dir / "semantic_type_specs.json"
        if not specs_path.exists():
            specs_path.write_text(json.dumps(_DEFAULT_SEMANTIC_TYPE_SPECS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd_tag = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "annotate_sentences_semantic_tags.py"),
            "--in_jsonl", str(sentences_jsonl),
            "--out_jsonl", str(sentences_tagged),
            "--domain_encoder_model", encoder,
            "--semantic_type_specs", str(specs_path),
        ]
        r_tag = subprocess.run(cmd_tag, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=120)
        if r_tag.returncode == 0 and sentences_tagged.exists():
            sentences_src = sentences_tagged
            pipeline_status["sentences_tagged"] = True
            _pipeline_log(topic, "[step2] tagging ok")
        else:
            pipeline_status["sentences_tagged"] = False
            pipeline_status["tag_error"] = (r_tag.stderr or "")[-500:]
            _pipeline_log(topic, f"[step2] tagging failed: {(r_tag.stderr or '')[-200:]}")
    else:
        pipeline_status["sentences_tagged"] = False
        pipeline_status["tag_note"] = "no encoder configured; skipping tagging"
        _pipeline_log(topic, "[step2] tagging skipped")

    aliases_jsonl = out_dir / "aliases.jsonl"
    aliases_data = obj.get("aliases")
    if isinstance(aliases_data, list) and aliases_data:
        with aliases_jsonl.open("w", encoding="utf-8") as af:
            for alias_rec in aliases_data:
                if isinstance(alias_rec, dict):
                    af.write(json.dumps(alias_rec, ensure_ascii=False) + "\n")
        _pipeline_log(topic, f"[step3] aliases={len(aliases_data)}")

    triples_jsonl = out_dir / "triples.jsonl"
    extract_batch_size = "8" if (use_base_llm_extraction or not use_deepseek) else "1"
    cmd_extract = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_triples.py"),
        "--sentences_jsonl", str(sentences_src),
        "--out_triples", str(triples_jsonl),
        "--batch_size", extract_batch_size,
    ]
    if use_deepseek:
        cmd_extract.extend([
            "--use_deepseek",
            "--deepseek_base_url", ds_base_url,
            "--deepseek_model", ds_model,
            "--deepseek_api_key_env", ds_api_key_env,
        ])
    else:
        cmd_extract.extend(["--model", base_llm])
        if device:
            cmd_extract.extend(["--device", device])
    _pipeline_log(topic, f"[step4] extract backend={'deepseek' if use_deepseek else 'local'}")

    try:
        rc_extract, out_extract, err_extract = _run_subprocess_with_log_stream(
            topic, cmd_extract, str(PROJECT_ROOT), timeout_s, log_prefix="step4", heartbeat_interval_s=60
        )
    except subprocess.TimeoutExpired:
        _pipeline_log(topic, f"[step4] extract timeout after {timeout_s}s")
        _finish(False, "extract_timeout", {"ok": False, "error": "timeout", "message": f"Triple extraction timed out after {timeout_s}s."})
        return
    if rc_extract != 0:
        _pipeline_log(topic, f"[step4] extract failed rc={rc_extract}")
        _finish(False, "extract_failed", {"ok": False, "topic": topic, "failed_step": "extract_triples", "error": (err_extract or "")[-2000:]})
        return
    triple_count = 0
    if triples_jsonl.exists():
        for ln in triples_jsonl.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                triple_count += 1
    pipeline_status["triples_extracted"] = triple_count
    _pipeline_log(topic, f"[step4] extract ok triples={triple_count}")

    graph_index_json = out_dir / "graph_index.json"
    cmd_build = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_knowledge_graph.py"),
        "--triples_jsonl", str(triples_jsonl),
        "--out_graph", str(graph_index_json),
    ]
    if aliases_jsonl.exists():
        cmd_build.extend(["--aliases_jsonl", str(aliases_jsonl)])
    r_build = subprocess.run(cmd_build, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=120)
    if r_build.returncode != 0:
        _pipeline_log(topic, f"[step5] graph build failed rc={r_build.returncode}")
        _finish(False, "build_graph_failed", {"ok": False, "topic": topic, "failed_step": "build_knowledge_graph", "error": (r_build.stderr or "")[-2000:]})
        return
    _pipeline_log(topic, "[step5] graph build ok")
    graph_summary: Dict[str, Any] = {}
    if graph_index_json.exists():
        try:
            gi = json.loads(graph_index_json.read_text(encoding="utf-8"))
            meta = gi.get("meta") or {}
            graph_summary = {"num_nodes": meta.get("num_nodes", 0), "num_triples": meta.get("num_triples", 0), "num_entity_index_entries": meta.get("num_entity_index_entries", 0)}
        except Exception:
            pass
    pipeline_status["graph"] = graph_summary

    triple_kvbank_dir = out_dir / "triple_kvbank"
    triple_kv_summary: Dict[str, Any] = {}
    triple_kv_items: List[Dict[str, Any]] = []
    if base_llm:
        cmd_kv = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "graph" / "triple_kv_compiler.py"),
            "--graph_index", str(graph_index_json),
            "--model", base_llm,
            "--out_dir", str(triple_kvbank_dir),
        ]
        if device:
            cmd_kv.extend(["--device", device])
        try:
            r_kv = subprocess.run(cmd_kv, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_s)
            if r_kv.returncode != 0:
                pipeline_status["triple_kv_error"] = (r_kv.stderr or "")[-1000:]
                _pipeline_log(topic, f"[step6] triple kv failed rc={r_kv.returncode}")
            else:
                _pipeline_log(topic, "[step6] triple kv ok")
        except subprocess.TimeoutExpired:
            pipeline_status["triple_kv_error"] = "timeout"
            _pipeline_log(topic, f"[step6] triple kv timeout after {timeout_s}s")
        manifest_path = triple_kvbank_dir / "manifest.json"
        if manifest_path.exists():
            try:
                tkv = json.loads(manifest_path.read_text(encoding="utf-8"))
                items_dict = tkv.get("items") or {}
                entity_items = tkv.get("entity_items") or {}
                triple_kv_summary = {"num_items": len(items_dict), "num_entities": len(entity_items), "num_layers": tkv.get("num_layers", 0)}
                for iid, it in items_dict.items():
                    triple_kv_items.append({
                        "item_id": iid,
                        "type": it.get("item_type", ""),
                        "entity": it.get("entity_name", ""),
                        "text": it.get("text", ""),
                        "relation": it.get("relation", ""),
                        "object": it.get("object_name", ""),
                        "layers": f"{it.get('layer_start', '?')}-{it.get('layer_end', '?')}",
                        "tokens": it.get("token_count", 0),
                    })
            except Exception:
                pass
    else:
        pipeline_status["triple_kv_note"] = "no base_llm; skipping KV compilation"
        _pipeline_log(topic, "[step6] triple kv skipped")
    pipeline_status["triple_kv"] = triple_kv_summary

    _pipeline_log(topic, f"[done] pipeline ok triples={pipeline_status.get('triples_extracted', 0)}")
    _finish(True, None, {"ok": True, "topic": topic, "pipeline_status": pipeline_status, "triple_kv_items": triple_kv_items})


def _append_lines_to_evidence_set(topic: str, source_id: str, lines: List[str]) -> Dict[str, Any]:
    # Import into the first enabled evidence set; if none exist, create one.
    sets = _list_evidence_sets(topic)
    target: Optional[Dict[str, Any]] = None
    for s in sets:
        if bool(s.get("enabled")):
            target = s
            break
    if target is None:
        d = _topic_evidence_sets_dir(topic)
        d.mkdir(parents=True, exist_ok=True)
        name = "evidence_001.jsonl"
        fp_new = d / name
        fp_new.write_text("", encoding="utf-8")
        mf = _load_manifest(topic)
        mf.setdefault("sets", {})
        mf["sets"][name] = {"enabled": True, "note": "auto_created_for_doc_import", "created_at": _safe_now_iso()}
        _save_manifest(topic, mf)
        target = {"name": name, "path": str(fp_new), "enabled": True}

    fp = Path(str(target.get("path") or ""))
    existing = _read_sentence_set(fp)
    seen_claims = {str(r.get("claim") or "").strip() for r in existing if str(r.get("claim") or "").strip()}
    now = _safe_now_iso()
    appended = 0
    for ln in lines:
        t = str(ln or "").strip()
        if not t or t in seen_claims:
            continue
        seen_claims.add(t)
        existing.append(
            {
                "id": _stable_sentence_id(topic=topic, claim=t, source_id=source_id),
                "topic": topic,
                "claim": t,
                "source_id": source_id,
                "source_ref": {"doi": None, "title": None},
                "created_at": now,
                "updated_at": now,
                "author": None,
                "tags": [],
            }
        )
        appended += 1
    _write_sentence_set(fp, existing)
    return {"evidence_set": str(target.get("name")), "appended": int(appended)}


_REL_OPS: set[str] = {">", ">=", "<", "<=", "=", "!=", "range"}


def _validate_normalize_relation(r: Dict[str, Any]) -> Optional[Dict[str, str]]:
    var = str(r.get("variable") or "").strip()
    op = str(r.get("operator") or "").strip()
    val = str(r.get("value") or "").strip()
    if not (var or op or val):
        return None
    if not var:
        raise ValueError("relation variable is required")
    if op not in _REL_OPS:
        raise ValueError(f"invalid relation operator: {op}")
    if not val:
        raise ValueError("relation value is required")
    if op == "range":
        parts = [x.strip() for x in val.split(",") if str(x).strip()]
        if len(parts) != 2:
            raise ValueError("range value must be 'min,max'")
        return {"variable": var, "operator": op, "value": f"{parts[0]},{parts[1]}"}
    return {"variable": var, "operator": op, "value": val}


def _relation_to_claim_text(r: Dict[str, str]) -> str:
    var = str(r.get("variable") or "").strip()
    op = str(r.get("operator") or "").strip()
    val = str(r.get("value") or "").strip()
    if not (var and op and val):
        return ""
    if op == "range":
        parts = [x.strip() for x in val.split(",") if str(x).strip()]
        if len(parts) == 2:
            return f"临床变量{var}取值范围在{parts[0]}到{parts[1]}之间。"
    op_zh = {
        ">": "大于",
        ">=": "大于等于",
        "<": "小于",
        "<=": "小于等于",
        "=": "等于",
        "!=": "不等于",
    }.get(op, op)
    return f"临床变量{var}{op_zh}{val}。"


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(code))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, code: int, body: str, *, content_type: str) -> None:
    data = (body or "").encode("utf-8")
    handler.send_response(int(code))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    # Prevent browser caching of static/HTML files during development
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.end_headers()
    handler.wfile.write(data)


def _read_body_json(handler: BaseHTTPRequestHandler) -> Tuple[Optional[Any], Optional[str]]:
    try:
        n = int(handler.headers.get("Content-Length") or "0")
    except Exception:
        n = 0
    raw = handler.rfile.read(max(0, n)).decode("utf-8", errors="replace")
    obj = _safe_json_loads(raw)
    if obj is None:
        return None, "Invalid JSON body"
    return obj, None

class KVIHandler(BaseHTTPRequestHandler):
    """
    KVI UI server (Evidence Sets only).
    """

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 (shadow builtin)
        # Keep console output minimal for MVP
        return

    def do_HEAD(self) -> None:  # noqa: N802
        """
        Support HEAD requests (curl -I, some proxies/browsers).
        Mirrors GET routing but returns headers only.
        """
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if path == "/" or path == "/index.html":
            p = STATIC_DIR / "index.html"
            body = p.read_text(encoding="utf-8") if p.exists() else _INDEX_FALLBACK_HTML
            data = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            fp = (STATIC_DIR / rel).resolve()
            if not str(fp).startswith(str(STATIC_DIR.resolve())) or (not fp.exists()) or fp.is_dir():
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            ctype = "text/plain; charset=utf-8"
            if fp.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif fp.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif fp.suffix == ".svg":
                ctype = "image/svg+xml"
            data = fp.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            return

        if path == "/api/health":
            # json body would be {"ok": true}
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        qs = parse_qs(parsed.query or "")

        # Browsers often request favicon.ico; avoid confusing 404s.
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if path == "/" or path == "/index.html":
            p = STATIC_DIR / "index.html"
            if not p.exists():
                _text_response(self, HTTPStatus.OK, _INDEX_FALLBACK_HTML, content_type="text/html; charset=utf-8")
                return
            _text_response(self, HTTPStatus.OK, p.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            fp = (STATIC_DIR / rel).resolve()
            if not str(fp).startswith(str(STATIC_DIR.resolve())) or not fp.exists() or fp.is_dir():
                _text_response(self, HTTPStatus.NOT_FOUND, "not found", content_type="text/plain; charset=utf-8")
                return
            ctype = "text/plain; charset=utf-8"
            if fp.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif fp.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif fp.suffix == ".svg":
                ctype = "image/svg+xml"
            _text_response(self, HTTPStatus.OK, fp.read_text(encoding="utf-8"), content_type=ctype)
            return

        if path == "/api/health":
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return

        if path == "/api/config":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "project_root": str(PROJECT_ROOT),
                },
            )
            return

        # -----------------------------
        # KVI Simple UI APIs (new)
        # -----------------------------
        if path == "/api/kvi/topics":
            items: List[Dict[str, Any]] = []
            if TOPICS_DIR.exists():
                for d in sorted([x for x in TOPICS_DIR.iterdir() if x.is_dir()]):
                    topic = d.name
                    cfg = _load_topic_config(topic)
                    build = cfg.get("build") if isinstance(cfg.get("build"), dict) else {}
                    items.append(
                        {
                            "topic": topic,
                            "goal": cfg.get("goal"),
                            "evidence_txt": str(d / "evidence.txt"),
                            "evidence_sets_dir": str(_topic_evidence_sets_dir(topic)),
                            "config_json": str(d / "config.json"),
                            "base_llm": (build.get("base_llm") if isinstance(build, dict) else None),
                            "domain_encoder_model": (build.get("retrieval_encoder_model") if isinstance(build, dict) else None),
                            "work_dir": (build.get("work_dir") if isinstance(build, dict) else None),
                        }
                    )
            _json_response(self, HTTPStatus.OK, {"items": items, "count": len(items)})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_sets"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_sets")].strip("/"))
            items = _list_evidence_sets(topic)
            _json_response(self, HTTPStatus.OK, {"topic": topic, "items": items, "count": len(items), "manifest": str(_evidence_manifest_path(topic))})
            return

        if path.startswith("/api/kvi/topic/") and "/evidence_set/" in path:
            # /api/kvi/topic/<topic>/evidence_set/<name>
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/evidence_set/")
            name = unquote(tail)
            if not name.endswith(".jsonl"):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "evidence_set name must end with .jsonl"})
                return
            d = _topic_evidence_sets_dir(topic)
            fp = (d / name).resolve()
            if not str(fp).startswith(str(d.resolve())):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "invalid evidence_set path"})
                return
            recs = _read_sentence_set(fp)
            # group counts by source_id for doc view
            by_source: Dict[str, int] = {}
            for r in recs:
                sid = r.get("source_id")
                sid = str(sid) if sid is not None else "null"
                by_source[sid] = int(by_source.get(sid, 0)) + 1
            mf = _load_manifest(topic)
            cfg = (mf.get("sets") or {}).get(name) if isinstance(mf.get("sets"), dict) else None
            enabled = True if not isinstance(cfg, dict) else bool(cfg.get("enabled", True))
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "topic": topic,
                    "name": name,
                    "path": str(fp),
                    "enabled": bool(enabled),
                    "records": recs,
                    "count": len(recs),
                    "by_source": by_source,
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_txt"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_txt")].strip("/"))
            pth = _topic_evidence_txt_path(topic)
            txt = _read_text_file(pth)
            _json_response(
                self,
                HTTPStatus.OK,
                {"topic": topic, "path": str(pth), "content": txt, "lines": int(len([ln for ln in txt.splitlines() if ln.strip()]))},
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/docs"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/docs")].strip("/"))
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()

            # Primary: docs.meta.jsonl. Fallback: blocks files (evidence > enriched > jsonl).
            docs_meta = work_dir / "docs.meta.jsonl"
            approvals = _load_doc_approvals(work_dir)

            # Find the best available blocks file
            blocks_path = None
            for bname in ["blocks.evidence.jsonl", "blocks.enriched.jsonl", "blocks.jsonl"]:
                bp = work_dir / bname
                if bp.exists():
                    blocks_path = bp
                    break

            if docs_meta.exists():
                docs = _read_jsonl(docs_meta)
                items: List[Dict[str, Any]] = []
                # Pre-read blocks for counts
                block_counts: Dict[str, int] = {}
                if blocks_path:
                    for b in _read_jsonl(blocks_path):
                        did_b = str(b.get("doc_id") or "").strip()
                        if did_b:
                            block_counts[did_b] = block_counts.get(did_b, 0) + 1

                # Pre-read raw_chunks for title enrichment (original PDF text, not evidence blocks)
                _DOI_LIKE_RE = re.compile(r"^10\.\d{4,}")
                _NOISE_LINE_RE = re.compile(
                    r"^(\s*\d+\s*$|.*\b(vol\.?|issue|pages?|proceedings)\b"
                    r"|.*\b(received|accepted|published|copyright)\b"
                    r"|.*@.*\.(com|edu|org|cn)|\s*https?://\S+\s*$)",
                    re.IGNORECASE,
                )
                raw_chunks_path = work_dir / "raw_chunks.jsonl"
                first_pdf_title: Dict[str, str] = {}
                if raw_chunks_path.exists():
                    for rc in _read_jsonl(raw_chunks_path):
                        rc_did = str(rc.get("doc_id") or "").strip()
                        if rc_did and rc_did not in first_pdf_title:
                            # Extract first meaningful line from original PDF text as title
                            raw_text = str(rc.get("text") or "")
                            candidate = ""
                            for raw_line in raw_text.split("\n"):
                                ln = raw_line.strip()
                                if not ln or len(ln) < 10:
                                    continue
                                if re.search(r"\b10\.\d{4,9}/\S+", ln):
                                    continue
                                if _NOISE_LINE_RE.match(ln):
                                    continue
                                if 10 <= len(ln) <= 300:
                                    candidate = ln
                                    break
                            if candidate:
                                first_pdf_title[rc_did] = candidate

                for d in docs:
                    did = str(d.get("doc_id") or "").strip()
                    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
                    pdf_name = str(d.get("source_uri") or "").split("/")[-1]
                    fallback_title = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
                    title = (meta.get("title") if isinstance(meta, dict) else None) or fallback_title or did

                    # Enrich: if title looks like a DOI/filename, use first line from raw PDF text
                    if _DOI_LIKE_RE.match(title) or title == did or title == fallback_title:
                        pdf_title = first_pdf_title.get(did, "")
                        if pdf_title:
                            title = pdf_title

                    items.append({
                        "doc_id": did,
                        "pdf_name": pdf_name,
                        "source_uri": d.get("source_uri"),
                        "title": title,
                        "doi": meta.get("doi") if isinstance(meta, dict) else None,
                        "publication_year": meta.get("publication_year") if isinstance(meta, dict) else None,
                        "approved": bool(approvals.get(did, False)),
                        "block_count": block_counts.get(did, 0),
                    })
                _json_response(self, HTTPStatus.OK, {"topic": topic, "items": items, "count": len(items), "source": "docs.meta.jsonl"})
                return

            # Fallback: derive doc list from blocks file
            if not blocks_path:
                _json_response(self, HTTPStatus.NOT_FOUND, {
                    "error": "not_found",
                    "message": f"No docs.meta.jsonl or blocks files found in work_dir={work_dir}. "
                               f"Checked: blocks.evidence.jsonl, blocks.enriched.jsonl, blocks.jsonl. "
                               f"Run PDF ingestion first.",
                })
                return

            # Group blocks by doc_id
            doc_blocks: Dict[str, List[Dict[str, Any]]] = {}
            for b in _read_jsonl(blocks_path):
                did = str(b.get("doc_id") or "").strip()
                if not did:
                    did = "(unknown)"
                if did not in doc_blocks:
                    doc_blocks[did] = []
                doc_blocks[did].append(b)

            items_fb: List[Dict[str, Any]] = []
            for did, blks in doc_blocks.items():
                first = blks[0]
                source_uri = str(first.get("source_uri") or "")
                pdf_name = source_uri.split("/")[-1] if source_uri else did
                items_fb.append({
                    "doc_id": did,
                    "pdf_name": pdf_name,
                    "source_uri": source_uri or None,
                    "title": pdf_name,
                    "doi": None,
                    "publication_year": None,
                    "approved": bool(approvals.get(did, False)),
                    "block_count": len(blks),
                })
            _json_response(self, HTTPStatus.OK, {"topic": topic, "items": items_fb, "count": len(items_fb), "source": str(blocks_path.name)})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/build_full_pipeline/status"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/build_full_pipeline/status")].strip("/"))
            _json_response(self, HTTPStatus.OK, {"ok": True, **_pipeline_get(topic)})
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/blocks"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/blocks")])
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            blocks_path = work_dir / "blocks.evidence.jsonl"
            if not blocks_path.exists():
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found", "message": f"blocks.evidence.jsonl not found: {blocks_path}"})
                return
            blocks = [b for b in _read_jsonl(blocks_path) if str(b.get("doc_id") or "").strip() == str(doc_id)]
            items = [{"block_id": b.get("block_id"), "block_type": b.get("block_type") or "paragraph_summary", "claim": (b.get("text") or "")} for b in blocks]
            _json_response(self, HTTPStatus.OK, {"topic": topic, "doc_id": doc_id, "items": items, "count": len(items), "blocks_jsonl": str(blocks_path)})
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/details"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/details")])
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            details_map = _load_doc_details(work_dir)
            details = details_map.get(str(doc_id), {})

            # Prefill from blocks when manual details are absent
            if not isinstance(details, dict) or not details:
                blocks_path = work_dir / "blocks.evidence.jsonl"
                abstract = ""
                key_notes: List[str] = []
                if blocks_path.exists():
                    for b in _read_jsonl(blocks_path):
                        if str(b.get("doc_id") or "").strip() != str(doc_id):
                            continue
                        bt = str(b.get("block_type") or "").strip().lower()
                        txt = str(b.get("text") or "").strip()
                        if not txt:
                            continue
                        if bt == "abstract" and not abstract:
                            abstract = txt
                        elif bt in {"results", "discussion", "conclusion", "introduction", "paragraph_summary"}:
                            if len(key_notes) < 10:
                                key_notes.append(txt)
                details = {
                    "doc_id": str(doc_id),
                    "abstract": abstract,
                    "key_notes": key_notes,
                    "relations": [],
                    "updated_at": None,
                    "source": "prefill_from_blocks",
                }

            _json_response(self, HTTPStatus.OK, {"topic": topic, "doc_id": doc_id, "details": details})
            return

        _text_response(self, HTTPStatus.NOT_FOUND, "not found", content_type="text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        # -----------------------------
        # KVI Simple UI APIs (new)
        # -----------------------------
        if path == "/api/kvi/topics/create":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            name = str(obj.get("topic") or obj.get("name") or "").strip()
            try:
                td = _create_topic(name)
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": name, "topic_dir": str(td), "config_json": str(td / "config.json")})
            return

        if path == "/api/kvi/topics/delete":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            name = str(obj.get("topic") or obj.get("name") or "").strip()
            if not name:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "topic required"})
                return
            try:
                td = _delete_topic(name)
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": name, "topic_dir": str(td)})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_txt"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_txt")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            content = str(obj.get("content") or "")
            pth = _topic_evidence_txt_path(topic)
            _write_text_file(pth, content)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "path": str(pth)})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_sets/create"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_sets/create")].strip("/"))
            d = _topic_evidence_sets_dir(topic)
            d.mkdir(parents=True, exist_ok=True)
            obj, _ = _read_body_json(self)
            obj = obj if isinstance(obj, dict) else {}
            base = str(obj.get("base_name") or "evidence").strip() or "evidence"
            note = str(obj.get("note") or "").strip() or None
            # find next index
            used = {p.name for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl"}
            idx = 1
            while True:
                name = f"{base}_{idx:03d}.jsonl"
                if name not in used:
                    break
                idx += 1
                if idx > 9999:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "too many evidence sets"})
                    return
            fp = d / name
            fp.write_text("", encoding="utf-8")
            mf = _load_manifest(topic)
            mf.setdefault("sets", {})
            mf["sets"][name] = {"enabled": True, "note": note, "created_at": _safe_now_iso()}
            _save_manifest(topic, mf)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "name": name, "path": str(fp), "enabled": True})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_set/save"):
            # POST /api/kvi/topic/<topic>/evidence_set/save {name, enabled?, records:[...]}
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_set/save")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            name = str(obj.get("name") or "").strip()
            if not name.endswith(".jsonl"):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "name must end with .jsonl"})
                return
            records = obj.get("records")
            if not isinstance(records, list):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "records must be a list"})
                return
            d = _topic_evidence_sets_dir(topic)
            d.mkdir(parents=True, exist_ok=True)
            fp = (d / name).resolve()
            if not str(fp).startswith(str(d.resolve())):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "invalid evidence_set path"})
                return
            # Token-budget split settings (server-side).
            max_sentence_tokens = int(obj.get("max_sentence_tokens") or DEFAULT_MAX_SENTENCE_TOKENS)
            if max_sentence_tokens <= 0:
                max_sentence_tokens = DEFAULT_MAX_SENTENCE_TOKENS
            max_sentence_tokens = max(16, min(256, int(max_sentence_tokens)))
            tok = _get_topic_tokenizer(topic)
            now = _safe_now_iso()
            cleaned: List[Dict[str, Any]] = []
            split_stats: Dict[str, Any] = {
                "max_sentence_tokens": int(max_sentence_tokens),
                "split_records": 0,
                "generated_records": 0,
                "truncated_records": 0,
                "tokenizer_loaded": bool(tok is not None),
            }
            for r in records:
                if not isinstance(r, dict):
                    continue
                claim = str(r.get("claim") or "").strip()
                if not claim:
                    continue
                src_id = r.get("source_id")
                src_id_s = str(src_id).strip() if src_id is not None and str(src_id).strip() else ""
                created_at = str(r.get("created_at") or "").strip() or now
                updated_at = now
                chunks = [claim]
                ch_meta: Dict[str, Any] = {"was_split": False}
                if tok is not None and _token_len(tok, claim) > int(max_sentence_tokens):
                    chunks, ch_meta = _split_claim_to_budget(
                        tok=tok,
                        claim=claim,
                        max_sentence_tokens=int(max_sentence_tokens),
                        max_chunks=4,
                    )
                    if bool(ch_meta.get("was_split")):
                        split_stats["split_records"] = int(split_stats["split_records"]) + 1
                    if bool(ch_meta.get("was_truncated")):
                        split_stats["truncated_records"] = int(split_stats["truncated_records"]) + 1
                for ch in chunks:
                    ch = str(ch or "").strip()
                    if not ch:
                        continue
                    sid = _stable_sentence_id(topic=topic, claim=ch, source_id=src_id_s)
                    cleaned.append(
                        {
                            "id": sid,
                            "topic": topic,
                            "claim": ch,
                            "source_id": (src_id_s if src_id_s else None),
                            "source_ref": r.get("source_ref") if isinstance(r.get("source_ref"), dict) else {"doi": None, "title": None},
                            "created_at": created_at,
                            "updated_at": updated_at,
                            "author": r.get("author"),
                            "tags": r.get("tags") if isinstance(r.get("tags"), list) else [],
                            "split_from_id": (str(r.get("id") or "").strip() or None) if bool(ch_meta.get("was_split")) else None,
                        }
                    )
                    split_stats["generated_records"] = int(split_stats["generated_records"]) + 1
            _write_sentence_set(fp, cleaned)
            mf = _load_manifest(topic)
            mf.setdefault("sets", {})
            if name not in mf["sets"]:
                mf["sets"][name] = {"enabled": True, "created_at": now}
            if "enabled" in obj:
                mf["sets"][name]["enabled"] = bool(obj.get("enabled"))
            _save_manifest(topic, mf)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "name": name,
                    "path": str(fp),
                    "count": len(cleaned),
                    "enabled": bool((mf["sets"][name] or {}).get("enabled", True)),
                    "split_stats": split_stats,
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/evidence_set/set_enabled"):
            # POST /api/kvi/topic/<topic>/evidence_set/set_enabled {name, enabled}
            topic = unquote(path[len("/api/kvi/topic/") : -len("/evidence_set/set_enabled")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            name = str(obj.get("name") or "").strip()
            enabled = bool(obj.get("enabled", True))
            if not name.endswith(".jsonl"):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "name must end with .jsonl"})
                return
            mf = _load_manifest(topic)
            mf.setdefault("sets", {})
            mf["sets"].setdefault(name, {})
            mf["sets"][name]["enabled"] = bool(enabled)
            _save_manifest(topic, mf)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "name": name, "enabled": bool(enabled)})
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/compile_simple"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/compile_simple")].strip("/"))
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return

            # Optional overrides
            obj, _ = _read_body_json(self)
            obj = obj if isinstance(obj, dict) else {}
            device = str(obj.get("device") or "").strip()
            dtype = str(obj.get("dtype") or "").strip()
            max_sentence_tokens = int(obj.get("max_sentence_tokens") or DEFAULT_MAX_SENTENCE_TOKENS)
            if max_sentence_tokens <= 0:
                max_sentence_tokens = DEFAULT_MAX_SENTENCE_TOKENS
            max_sentence_tokens = max(16, min(256, int(max_sentence_tokens)))

            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            # Sentence-KVBank build input
            recs, rec_stats = _collect_enabled_sentence_records(topic, enabled_only=True)
            if not recs:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "No claims found in enabled evidence sets. Create/enable an evidence set first."},
                )
                return

            sentences_jsonl = out_dir / "sentences.jsonl"
            compiled_txt = out_dir / "evidence.compiled.txt"
            seen_ids: set[str] = set()
            written = 0
            with sentences_jsonl.open("w", encoding="utf-8") as f:
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    claim = str(r.get("claim") or "").strip()
                    if not claim:
                        continue
                    src = str(r.get("source_id") or "").strip()
                    sid = str(r.get("id") or "").strip() or _stable_sentence_id(topic=topic, claim=claim, source_id=src)
                    bid = f"sent_{sid}"
                    if bid in seen_ids:
                        continue
                    seen_ids.add(bid)
                    meta = {
                        "kind": "sentence",
                        "topic": topic,
                        "source_ref": r.get("source_ref") if isinstance(r.get("source_ref"), dict) else {"doi": None, "title": None},
                        "author": r.get("author"),
                        "tags": r.get("tags") if isinstance(r.get("tags"), list) else [],
                        "evidence_id": sid,
                    }
                    f.write(
                        json.dumps(
                            {
                                "block_id": bid,
                                "text": claim,
                                "source_id": (src if src else None),
                                "doc_id": (src if src else None),
                                "metadata": meta,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

            # Human-readable compiled txt (one sentence per line)
            compiled_txt.write_text("\n".join([str(r.get("claim") or "").strip() for r in recs if str(r.get("claim") or "").strip()]) + "\n", encoding="utf-8")

            kv_dir = out_dir / "kvbank_sentences"
            kv_dir.mkdir(parents=True, exist_ok=True)

            # Ensure semantic_type_specs.json exists in work_dir (config-driven intent taxonomy).
            specs_path = out_dir / "semantic_type_specs.json"
            if not specs_path.exists():
                specs_path.write_text(json.dumps(_DEFAULT_SEMANTIC_TYPE_SPECS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            # Offline tag sentences with semantic intent (no base LLM).
            sentences_tagged = out_dir / "sentences.tagged.jsonl"
            use_llm_intent = bool(obj.get("use_llm_intent", False))
            cmd_tag = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "annotate_sentences_semantic_tags.py"),
                "--in_jsonl",
                str(sentences_jsonl),
                "--out_jsonl",
                str(sentences_tagged),
                "--domain_encoder_model",
                encoder,
                "--semantic_type_specs",
                str(specs_path),
            ]
            if use_llm_intent:
                cmd_tag.extend(["--llm_intent_enable", "--llm_intent_model", base_llm])
            r0 = subprocess.run(cmd_tag, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            logs: List[Dict[str, Any]] = [
                {
                    "cmd": " ".join(cmd_tag),
                    "returncode": int(r0.returncode),
                    "stdout": (r0.stdout or "")[-8000:],
                    "stderr": (r0.stderr or "")[-8000:],
                }
            ]
            if r0.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "topic": topic, "failed_cmd": " ".join(cmd_tag), "logs": logs})
                return

            cmd_kv = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_kvbank_from_blocks_jsonl.py"),
                "--blocks_jsonl",
                str(sentences_tagged),
                "--disable_enriched",
                "--out_dir",
                str(kv_dir),
                "--base_llm",
                base_llm,
                "--domain_encoder_model",
                encoder,
                "--layers",
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                "--block_tokens",
                str(int(max_sentence_tokens)),
                "--shard_size",
                "1024",
            ]
            if device:
                cmd_kv.extend(["--device", device])
            if dtype:
                cmd_kv.extend(["--dtype", dtype])

            r1 = subprocess.run(cmd_kv, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            logs.append(
                {
                    "cmd": " ".join(cmd_kv),
                    "returncode": int(r1.returncode),
                    "stdout": (r1.stdout or "")[-8000:],
                    "stderr": (r1.stderr or "")[-8000:],
                }
            )
            if r1.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "topic": topic, "failed_cmd": " ".join(cmd_kv), "logs": logs})
                return

            # -- Entity priming KV bank (complementary injection) --
            entity_priming_jsonl = out_dir / "entity_priming.jsonl"
            priming_kv_dir = out_dir / "kvbank_entity_priming"
            if entity_priming_jsonl.exists():
                cmd_priming = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_kvbank_from_blocks_jsonl.py"),
                    "--blocks_jsonl",
                    str(entity_priming_jsonl),
                    "--disable_enriched",
                    "--out_dir",
                    str(priming_kv_dir),
                    "--base_llm",
                    base_llm,
                    "--domain_encoder_model",
                    encoder,
                    "--layers",
                    "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                    "--block_tokens",
                    str(int(max_sentence_tokens)),
                    "--shard_size",
                    "1024",
                ]
                if device:
                    cmd_priming.extend(["--device", device])
                if dtype:
                    cmd_priming.extend(["--dtype", dtype])
                r_priming = subprocess.run(cmd_priming, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
                logs.append(
                    {
                        "step": "entity_priming_kvbank",
                        "cmd": " ".join(cmd_priming),
                        "returncode": int(r_priming.returncode),
                        "stdout": (r_priming.stdout or "")[-4000:],
                        "stderr": (r_priming.stderr or "")[-4000:],
                    }
                )
                # Non-fatal: priming is optional; log but don't fail compile

            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "out_dir": str(out_dir),
                    "max_sentence_tokens": int(max_sentence_tokens),
                    "compiled_evidence_txt": str(compiled_txt),
                    "sentences_jsonl": str(sentences_jsonl),
                    "sentences_tagged_jsonl": str(sentences_tagged),
                    "semantic_type_specs": str(specs_path),
                    "compiled_stats": rec_stats,
                    "written_sentences": int(written),
                    "kv_dir": str(kv_dir),
                    "entity_priming_kv_dir": str(priming_kv_dir) if entity_priming_jsonl.exists() else None,
                    "logs": logs,
                },
            )
            return

        # ==================== Scheme C: Build Knowledge Graph ====================
        if path.startswith("/api/kvi/topic/") and path.endswith("/compile_graph"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/compile_graph")].strip("/"))
            topic_cfg = _load_topic_config(topic)
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            # DeepSeek config from topic config.json (top-level)
            ds_base_url = str(topic_cfg.get("deepseek_base_url") or "https://api.deepseek.com").strip()
            ds_model = str(topic_cfg.get("deepseek_model") or "deepseek-chat").strip()
            ds_api_key_env = str(topic_cfg.get("deepseek_api_key_env") or "DEEPSEEK_API_KEY").strip()
            use_deepseek = bool(ds_model and ds_api_key_env)

            if not use_deepseek and not base_llm:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing DeepSeek config or build.base_llm in topic config.json"})
                _finish_pipeline(False, "missing_config")
                return

            # Use tagged sentences if available, else raw sentences
            sentences_src = out_dir / "sentences.tagged.jsonl"
            if not sentences_src.exists():
                sentences_src = out_dir / "sentences.jsonl"
            if not sentences_src.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"sentences.jsonl not found: {out_dir}. Run Compile KVBank first."})
                return

            obj, _ = _read_body_json(self)
            obj = obj if isinstance(obj, dict) else {}
            device = str(obj.get("device") or "").strip()
            timeout_s = int(obj.get("timeout_s") or 600)
            graph_logs: List[Dict[str, Any]] = []

            # -- Write aliases.jsonl from request body (if provided) --
            aliases_jsonl = out_dir / "aliases.jsonl"
            aliases_data = obj.get("aliases")
            if isinstance(aliases_data, list) and aliases_data:
                with aliases_jsonl.open("w", encoding="utf-8") as af:
                    for alias_rec in aliases_data:
                        if isinstance(alias_rec, dict):
                            af.write(json.dumps(alias_rec, ensure_ascii=False) + "\n")
                graph_logs.append({"step": "write_aliases", "count": len(aliases_data), "path": str(aliases_jsonl)})

            # Step 1: Extract triples
            triples_jsonl = out_dir / "triples.jsonl"
            cmd_extract = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "extract_triples.py"),
                "--sentences_jsonl", str(sentences_src),
                "--out_triples", str(triples_jsonl),
                "--batch_size", "1",
            ]
            if use_deepseek:
                cmd_extract.extend([
                    "--use_deepseek",
                    "--deepseek_base_url", ds_base_url,
                    "--deepseek_model", ds_model,
                    "--deepseek_api_key_env", ds_api_key_env,
                ])
            else:
                cmd_extract.extend(["--model", base_llm])
                if device:
                    cmd_extract.extend(["--device", device])

            try:
                r_extract = subprocess.run(cmd_extract, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "timeout", "message": f"Triple extraction timed out after {timeout_s}s.", "cmd": " ".join(cmd_extract)})
                return
            graph_logs.append({
                "step": "extract_triples",
                "backend": "deepseek" if use_deepseek else "local_llm",
                "cmd": " ".join(cmd_extract),
                "returncode": int(r_extract.returncode),
                "stdout": (r_extract.stdout or "")[-4000:],
                "stderr": (r_extract.stderr or "")[-4000:],
            })
            if r_extract.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "topic": topic, "failed_step": "extract_triples", "logs": graph_logs})
                return

            # Step 2: Build knowledge graph
            graph_index_json = out_dir / "graph_index.json"
            cmd_build = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_knowledge_graph.py"),
                "--triples_jsonl", str(triples_jsonl),
                "--out_graph", str(graph_index_json),
            ]
            if aliases_jsonl.exists():
                cmd_build.extend(["--aliases_jsonl", str(aliases_jsonl)])
            r_build = subprocess.run(cmd_build, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=120)
            graph_logs.append({
                "step": "build_knowledge_graph",
                "cmd": " ".join(cmd_build),
                "returncode": int(r_build.returncode),
                "stdout": (r_build.stdout or "")[-4000:],
                "stderr": (r_build.stderr or "")[-4000:],
            })
            if r_build.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "topic": topic, "failed_step": "build_knowledge_graph", "logs": graph_logs})
                return

            # Step 3: Compile Triple KV Bank (三元 KVI)
            triple_kvbank_dir = out_dir / "triple_kvbank"
            cmd_triple_kv = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "graph" / "triple_kv_compiler.py"),
                "--graph_index", str(graph_index_json),
                "--model", base_llm,
                "--out_dir", str(triple_kvbank_dir),
            ]
            if device:
                cmd_triple_kv.extend(["--device", device])
            try:
                r_triple_kv = subprocess.run(
                    cmd_triple_kv, cwd=str(PROJECT_ROOT),
                    capture_output=True, text=True, check=False, timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                r_triple_kv = None
            if r_triple_kv is not None:
                graph_logs.append({
                    "step": "compile_triple_kvbank",
                    "cmd": " ".join(cmd_triple_kv),
                    "returncode": int(r_triple_kv.returncode),
                    "stdout": (r_triple_kv.stdout or "")[-4000:],
                    "stderr": (r_triple_kv.stderr or "")[-4000:],
                })
                # Non-fatal: triple KV is optional enhancement
                if r_triple_kv.returncode != 0:
                    print(f"[compile_graph] triple_kv compilation failed (non-fatal)", flush=True)
            else:
                graph_logs.append({
                    "step": "compile_triple_kvbank",
                    "error": "timeout",
                })

            # Summary
            graph_summary: Dict[str, Any] = {}
            if graph_index_json.exists():
                try:
                    gi = json.loads(graph_index_json.read_text(encoding="utf-8"))
                    meta = gi.get("meta") or {}
                    graph_summary = {
                        "num_nodes": meta.get("num_nodes", 0),
                        "num_triples": meta.get("num_triples", 0),
                        "num_entity_index_entries": meta.get("num_entity_index_entries", 0),
                    }
                except Exception:
                    pass

            triple_kv_summary: Dict[str, Any] = {}
            triple_kv_manifest_path = triple_kvbank_dir / "manifest.json"
            if triple_kv_manifest_path.exists():
                try:
                    tkv = json.loads(triple_kv_manifest_path.read_text(encoding="utf-8"))
                    triple_kv_summary = {
                        "num_items": len(tkv.get("items") or {}),
                        "num_entities": len(tkv.get("entity_items") or {}),
                        "num_layers": tkv.get("num_layers", 0),
                    }
                except Exception:
                    pass

            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "extraction_backend": "deepseek" if use_deepseek else "local_llm",
                    "graph_index": str(graph_index_json),
                    "triples_jsonl": str(triples_jsonl),
                    "sentences_source": str(sentences_src),
                    "graph_summary": graph_summary,
                    "triple_kvbank_dir": str(triple_kvbank_dir),
                    "triple_kv_summary": triple_kv_summary,
                    "logs": graph_logs,
                },
            )
            return

        # ==================== Full Pipeline: Sentences → Triples → Graph → KV (async: POST returns 202, run in background) ====================
        if path.startswith("/api/kvi/topic/") and path.endswith("/build_full_pipeline"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/build_full_pipeline")].strip("/"))
            obj, body_err = _read_body_json(self)
            if body_err:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": body_err})
                return
            obj = obj if isinstance(obj, dict) else {}
            topic_lock = _get_topic_pipeline_lock(topic)
            if not topic_lock.acquire(blocking=False):
                _json_response(
                    self,
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": "running", "message": f"build_full_pipeline already running for topic={topic}"},
                )
                return
            _pipeline_set(topic, running=True, logs=[f"[{_safe_now_iso()}] build_full_pipeline started"], last_error=None, last_result_ok=None, last_result=None)
            def run_bg() -> None:
                _run_build_full_pipeline_background(topic, obj, topic_lock)
            t = threading.Thread(target=run_bg, daemon=True)
            t.start()
            _json_response(self, HTTPStatus.ACCEPTED, {
                "ok": True,
                "status": "started",
                "message": "Pipeline running in background. Poll GET .../build_full_pipeline/status for logs; when running=false, use last_result.",
            })
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/run_simple"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/run_simple")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            pattern_index_dir = out_dir / "pattern_sidecar"
            kv_dir = out_dir / "kvbank_sentences"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_sentences not found: {kv_dir}. Click 编译 first."})
                return

            top_k = int(obj.get("top_k") or 8)
            show_baseline = bool(obj.get("show_baseline", True))
            simple_max_steps = int(obj.get("simple_max_steps") or 1)
            simple_step_new_tokens = int(obj.get("simple_step_new_tokens") or 192)
            simple_max_blocks_per_step = int(obj.get("simple_max_blocks_per_step") or HARD_MAX_INJECTED_SENTENCES_PER_STEP)
            simple_max_blocks_per_step = max(1, min(HARD_MAX_INJECTED_SENTENCES_PER_STEP, int(simple_max_blocks_per_step)))
            max_sentence_tokens = int(obj.get("max_sentence_tokens") or DEFAULT_MAX_SENTENCE_TOKENS)
            max_sentence_tokens = max(16, min(256, int(max_sentence_tokens)))
            max_total_injected_tokens = int(obj.get("max_total_injected_tokens") or HARD_MAX_TOTAL_INJECTED_TOKENS)
            max_total_injected_tokens = max(64, min(2048, int(max_total_injected_tokens)))
            regen_on_violation = bool(obj.get("regen_on_violation", False))
            max_regen_rounds = int(obj.get("max_regen_rounds") or 1)
            max_regen_rounds = max(0, min(2, int(max_regen_rounds)))

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kvi2_runtime_test.py"),
                "--pipeline",
                "simple",
                "--model",
                base_llm,
                "--prompt",
                prompt,
                "--kv_dir",
                str(kv_dir),
                "--sentences_jsonl",
                str(out_dir / "sentences.tagged.jsonl"),
                "--semantic_type_specs",
                str(out_dir / "semantic_type_specs.json"),
                "--pattern_index_dir",
                str(pattern_index_dir),
                "--sidecar_dir",
                str(out_dir),
                "--domain_encoder_model",
                encoder,
                "--use_chat_template",
                "--top_k",
                str(top_k),
                "--simple_max_steps",
                str(simple_max_steps),
                "--simple_step_new_tokens",
                str(simple_step_new_tokens),
                "--simple_max_blocks_per_step",
                str(simple_max_blocks_per_step),
                "--simple_max_sentence_tokens",
                str(int(max_sentence_tokens)),
                "--simple_max_total_injected_tokens",
                str(int(max_total_injected_tokens)),
            ]
            if regen_on_violation:
                cmd.append("--simple_regen_on_violation")
                cmd.extend(["--simple_max_regen_rounds", str(int(max_regen_rounds))])
            if show_baseline:
                cmd.append("--show_baseline")
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "kv_dir": str(kv_dir),
                    "pattern_index_dir": str(pattern_index_dir),
                    "sidecar_dir": str(out_dir),
                    "result": out,
                    "stderr_tail": (r.stderr or "")[-2000:],
                },
            )
            return

        # -----------------------------
        # Evidence Routing Modes (A/B)
        # -----------------------------
        if path.startswith("/api/kvi/topic/") and path.endswith("/route"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/route")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            kv_dir = out_dir / "kvbank_sentences"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_sentences not found: {kv_dir}. Click 编译 first."})
                return
            # Ensure pattern_contract.json exists (kvi2 requires contracts).
            # For sentence KVBank, place contract under work/ to match loader inference.
            is_sentence_bank = kv_dir.name in {"kvbank_sentences", "kvbank_sentences_v2"}
            topic_root = out_dir if is_sentence_bank else (out_dir.parent if out_dir.name == "work" else out_dir)
            specs_path = out_dir / "semantic_type_specs.json"
            specs = _DEFAULT_SEMANTIC_TYPE_SPECS
            try:
                if specs_path.exists():
                    specs = json.loads(specs_path.read_text(encoding="utf-8"))
            except Exception:
                specs = _DEFAULT_SEMANTIC_TYPE_SPECS
            _ensure_pattern_contract(topic_root=topic_root, semantic_specs=specs, sentence_mode=is_sentence_bank)
            top_k = int(obj.get("top_k") or 8)
            w_ann = float(obj.get("route_w_ann")) if obj.get("route_w_ann") is not None else 1.0
            w_intent = float(obj.get("route_w_intent")) if obj.get("route_w_intent") is not None else 0.6
            w_quality = float(obj.get("route_w_quality")) if obj.get("route_w_quality") is not None else 0.2
            rerank_wo_ann = bool(obj.get("route_rerank_without_ann", False))
            llm_intent = bool(obj.get("route_llm_intent_enable", False))
            trace_text = str(obj.get("route_trace_text") or "").strip()
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kvi2_runtime_test.py"),
                "--pipeline",
                "route",
                "--model",
                base_llm,
                "--prompt",
                prompt,
                "--kv_dir",
                str(kv_dir),
                "--sentences_jsonl",
                str(out_dir / "sentences.tagged.jsonl"),
                "--semantic_type_specs",
                str(out_dir / "semantic_type_specs.json"),
                "--pattern_index_dir",
                str(out_dir / "pattern_sidecar"),
                "--sidecar_dir",
                str(out_dir),
                "--domain_encoder_model",
                encoder,
                "--use_chat_template",
                "--local_files_only",
                "--top_k",
                str(top_k),
                "--route_w_ann",
                str(w_ann),
                "--route_w_intent",
                str(w_intent),
                "--route_w_quality",
                str(w_quality),
            ]
            if rerank_wo_ann:
                cmd.append("--route_rerank_without_ann")
            if llm_intent:
                cmd.append("--route_llm_intent_enable")
            if trace_text:
                cmd.extend(["--route_trace_text", trace_text])
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "result": out,
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/modeA"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/modeA")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return
            # Fail fast if base_llm is a local path that doesn't exist (avoid hanging on load).
            try:
                b = str(base_llm or "")
                b_exp = Path(b).expanduser()
                if (b.startswith("/") or b.startswith("./") or b.startswith("../") or b.startswith("~")) and not b_exp.exists():
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"base_llm path not found: {b_exp}"})
                    return
            except Exception:
                pass
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            kv_dir = out_dir / "kvbank_sentences"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_sentences not found: {kv_dir}. Click 编译 first."})
                return
            top_k = int(obj.get("top_k") or 8)
            w_ann = float(obj.get("route_w_ann")) if obj.get("route_w_ann") is not None else 1.0
            w_intent = float(obj.get("route_w_intent")) if obj.get("route_w_intent") is not None else 0.6
            w_quality = float(obj.get("route_w_quality")) if obj.get("route_w_quality") is not None else 0.2
            rerank_wo_ann = bool(obj.get("route_rerank_without_ann", False))
            llm_intent = bool(obj.get("route_llm_intent_enable", False))
            trace_text = str(obj.get("route_trace_text") or "").strip()
            timeout_s = int(obj.get("timeout_s") or 180)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kvi2_runtime_test.py"),
                "--pipeline",
                "modeA",
                "--model",
                base_llm,
                "--prompt",
                prompt,
                "--kv_dir",
                str(kv_dir),
                "--sentences_jsonl",
                str(out_dir / "sentences.tagged.jsonl"),
                "--semantic_type_specs",
                str(out_dir / "semantic_type_specs.json"),
                "--pattern_index_dir",
                str(out_dir / "pattern_sidecar"),
                "--sidecar_dir",
                str(out_dir),
                "--domain_encoder_model",
                encoder,
                "--use_chat_template",
                "--local_files_only",
                "--top_k",
                str(top_k),
                "--route_w_ann",
                str(w_ann),
                "--route_w_intent",
                str(w_intent),
                "--route_w_quality",
                str(w_quality),
            ]
            # Entity priming KV bank (complementary injection)
            priming_kv_dir = out_dir / "kvbank_entity_priming"
            _ep_exists = priming_kv_dir.exists()
            print(f"[modeA_cmd] entity_priming check: path={priming_kv_dir} exists={_ep_exists}", flush=True)
            if _ep_exists:
                cmd.extend(["--entity_priming_kv_dir", str(priming_kv_dir)])
            if rerank_wo_ann:
                cmd.append("--route_rerank_without_ann")
            if llm_intent:
                cmd.append("--route_llm_intent_enable")
            if trace_text:
                cmd.extend(["--route_trace_text", trace_text])
            try:
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "timeout",
                        "message": f"Mode A timed out after {timeout_s}s. Model loading or generation is too slow. Use a local model path or increase timeout_s.",
                        "cmd": " ".join(cmd),
                    },
                )
                return
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "result": out,
                    "stderr_tail": (r.stderr or "")[-4000:],
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/modeA_rag"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/modeA_rag")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return
            # Fail fast if base_llm is a local path that doesn't exist (avoid hanging on load).
            try:
                b = str(base_llm or "")
                b_exp = Path(b).expanduser()
                if (b.startswith("/") or b.startswith("./") or b.startswith("../") or b.startswith("~")) and not b_exp.exists():
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"base_llm path not found: {b_exp}"})
                    return
            except Exception:
                pass
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            kv_dir = out_dir / "kvbank_sentences"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_sentences not found: {kv_dir}. Click 编译 first."})
                return
            top_k = int(obj.get("top_k") or 8)
            w_ann = float(obj.get("route_w_ann")) if obj.get("route_w_ann") is not None else 1.0
            w_intent = float(obj.get("route_w_intent")) if obj.get("route_w_intent") is not None else 0.6
            w_quality = float(obj.get("route_w_quality")) if obj.get("route_w_quality") is not None else 0.2
            rerank_wo_ann = bool(obj.get("route_rerank_without_ann", False))
            llm_intent = bool(obj.get("route_llm_intent_enable", False))
            trace_text = str(obj.get("route_trace_text") or "").strip()
            timeout_s = int(obj.get("timeout_s") or 180)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kvi2_runtime_test.py"),
                "--pipeline",
                "modeA_rag",
                "--model",
                base_llm,
                "--prompt",
                prompt,
                "--kv_dir",
                str(kv_dir),
                "--sentences_jsonl",
                str(out_dir / "sentences.tagged.jsonl"),
                "--semantic_type_specs",
                str(out_dir / "semantic_type_specs.json"),
                "--pattern_index_dir",
                str(out_dir / "pattern_sidecar"),
                "--sidecar_dir",
                str(out_dir),
                "--domain_encoder_model",
                encoder,
                "--use_chat_template",
                "--local_files_only",
                "--top_k",
                str(top_k),
                "--route_w_ann",
                str(w_ann),
                "--route_w_intent",
                str(w_intent),
                "--route_w_quality",
                str(w_quality),
            ]
            if rerank_wo_ann:
                cmd.append("--route_rerank_without_ann")
            if llm_intent:
                cmd.append("--route_llm_intent_enable")
            if trace_text:
                cmd.extend(["--route_trace_text", trace_text])
            try:
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "timeout",
                        "message": f"Mode A RAG timed out after {timeout_s}s. Model loading or generation is too slow. Use a local model path or increase timeout_s.",
                        "cmd": " ".join(cmd),
                    },
                )
                return
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "result": out,
                },
            )
            return

        # ==================== Scheme C: Graph Inference ====================
        if path.startswith("/api/kvi/topic/") and path.endswith("/modeA_graph"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/modeA_graph")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            if not base_llm:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm in topic config.json"})
                return
            # Fail fast if base_llm is a local path that doesn't exist.
            try:
                b = str(base_llm or "")
                b_exp = Path(b).expanduser()
                if (b.startswith("/") or b.startswith("./") or b.startswith("../") or b.startswith("~")) and not b_exp.exists():
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"base_llm path not found: {b_exp}"})
                    return
            except Exception:
                pass
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            graph_index_json = out_dir / "graph_index.json"
            if not graph_index_json.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"graph_index.json not found: {graph_index_json}. Click Build Graph first."})
                return
            timeout_s = int(obj.get("timeout_s") or 180)
            triple_kvbank_dir = out_dir / "triple_kvbank"
            enable_kvi = bool(obj.get("enable_kvi", False))
            # Locate sentences file for text search fallback (hybrid retrieval)
            sentences_jsonl = out_dir / "sentences.tagged.jsonl"
            if not sentences_jsonl.exists():
                sentences_jsonl = out_dir / "sentences.jsonl"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_graph_inference.py"),
                "--model", base_llm,
                "--prompt", prompt,
                "--graph_index", str(graph_index_json),
                "--use_chat_template",
                "--local_files_only",
            ]
            if sentences_jsonl.exists():
                cmd.extend(["--sentences_jsonl", str(sentences_jsonl)])
            # KVI is off by default (pure RAG); only enable when explicitly requested
            if enable_kvi and triple_kvbank_dir.exists():
                max_kv_triples = int(obj.get("max_kv_triples") or 3)
                drm_threshold = float(obj.get("drm_threshold") or 0.05)
                top_k_relations = int(obj.get("top_k_relations") or 2)
                cmd.extend([
                    "--enable_kvi",
                    "--triple_kvbank_dir", str(triple_kvbank_dir),
                    "--max_kv_triples", str(max_kv_triples),
                    "--drm_threshold", str(drm_threshold),
                    "--top_k_relations", str(top_k_relations),
                ])
            try:
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "timeout",
                        "message": f"Graph inference timed out after {timeout_s}s.",
                        "cmd": " ".join(cmd),
                    },
                )
                return
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "result": out,
                    "stderr_tail": (r.stderr or "")[-4000:],
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and path.endswith("/modeB"):
            topic = unquote(path[len("/api/kvi/topic/") : -len("/modeB")].strip("/"))
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            prompt = str(obj.get("prompt") or "").strip()
            if not prompt:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "prompt required"})
                return
            build = _topic_build_cfg(topic)
            base_llm = str(build.get("base_llm") or "").strip()
            encoder = str(build.get("retrieval_encoder_model") or "").strip()
            if not base_llm or not encoder:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "Missing build.base_llm or build.retrieval_encoder_model in topic config.json"})
                return
            topic_dir = _topic_dir(topic)
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            kv_dir = out_dir / "kvbank_sentences"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_sentences not found: {kv_dir}. Click 编译 first."})
                return
            top_k = int(obj.get("top_k") or 8)
            w_ann = float(obj.get("route_w_ann")) if obj.get("route_w_ann") is not None else 1.0
            w_intent = float(obj.get("route_w_intent")) if obj.get("route_w_intent") is not None else 0.6
            w_quality = float(obj.get("route_w_quality")) if obj.get("route_w_quality") is not None else 0.2
            rerank_wo_ann = bool(obj.get("route_rerank_without_ann", False))
            llm_intent = bool(obj.get("route_llm_intent_enable", False))
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kvi2_runtime_test.py"),
                "--pipeline",
                "modeB",
                "--model",
                base_llm,
                "--prompt",
                prompt,
                "--kv_dir",
                str(kv_dir),
                "--sentences_jsonl",
                str(out_dir / "sentences.tagged.jsonl"),
                "--semantic_type_specs",
                str(out_dir / "semantic_type_specs.json"),
                "--pattern_index_dir",
                str(out_dir / "pattern_sidecar"),
                "--sidecar_dir",
                str(out_dir),
                "--domain_encoder_model",
                encoder,
                "--top_k",
                str(top_k),
                "--route_w_ann",
                str(w_ann),
                "--route_w_intent",
                str(w_intent),
                "--route_w_quality",
                str(w_quality),
            ]
            if rerank_wo_ann:
                cmd.append("--route_rerank_without_ann")
            if llm_intent:
                cmd.append("--route_llm_intent_enable")
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            if r.returncode != 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "returncode": int(r.returncode), "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-8000:]})
                return
            out = _safe_parse_last_json_obj(r.stdout) or {"raw_stdout": (r.stdout or "")[-8000:]}
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "cmd": " ".join(cmd),
                    "result": out,
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/set_approved"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/set_approved")])
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            approved = bool(obj.get("approved", False))
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            approvals = _load_doc_approvals(work_dir)
            approvals[str(doc_id)] = bool(approved)
            _save_doc_approvals(work_dir, approvals)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "doc_id": doc_id, "approved": approved})
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/details/save"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/details/save")])
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            abstract = str(obj.get("abstract") or "").strip()
            key_notes_in = obj.get("key_notes")
            rel_in = obj.get("relations")
            key_notes: List[str] = []
            if isinstance(key_notes_in, list):
                for x in key_notes_in:
                    t = str(x or "").strip()
                    if t:
                        key_notes.append(t)
            relations: List[Dict[str, str]] = []
            if isinstance(rel_in, list):
                for r in rel_in:
                    if not isinstance(r, dict):
                        continue
                    try:
                        rr = _validate_normalize_relation(r)
                    except Exception as e:
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"invalid relation: {e}"})
                        return
                    if rr:
                        relations.append(rr)
            details = {
                "doc_id": str(doc_id),
                "abstract": abstract,
                "key_notes": key_notes,
                "relations": relations,
                "updated_at": _safe_now_iso(),
                "source": "manual",
            }
            details_map = _load_doc_details(work_dir)
            details_map[str(doc_id)] = details
            _save_doc_details(work_dir, details_map)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "doc_id": doc_id, "details": details})
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/import_details_to_evidence"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/import_details_to_evidence")])
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            approvals = _load_doc_approvals(work_dir)
            if not bool(approvals.get(str(doc_id), False)):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "doc is not approved; approve it first"})
                return
            details_map = _load_doc_details(work_dir)
            d = details_map.get(str(doc_id))
            if not isinstance(d, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "doc details not found; save details first"})
                return
            lines: List[str] = []
            abs_text = str(d.get("abstract") or "").strip()
            if abs_text:
                lines.append(f"文献摘要：{abs_text}")
            for i, n in enumerate(d.get("key_notes") or [], start=1):
                t = str(n or "").strip()
                if t:
                    lines.append(f"关键要点{i}：{t}")
            for r in d.get("relations") or []:
                if not isinstance(r, dict):
                    continue
                try:
                    rr = _validate_normalize_relation(r)
                except Exception:
                    rr = None
                text = _relation_to_claim_text(rr or {})
                if text:
                    lines.append(text)
            if not lines:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "empty details; nothing to import"})
                return
            imp = _append_lines_to_evidence_set(topic=topic, source_id=str(doc_id), lines=lines)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "doc_id": doc_id,
                    "evidence_set": imp.get("evidence_set"),
                    "appended": int(imp.get("appended") or 0),
                    "num_lines": len(lines),
                },
            )
            return

        if path.startswith("/api/kvi/topic/") and "/doc/" in path and path.endswith("/import_to_evidence"):
            rest = path[len("/api/kvi/topic/") :].strip("/")
            topic, _, tail = rest.partition("/doc/")
            doc_id = unquote(tail[: -len("/import_to_evidence")])
            build = _topic_build_cfg(topic)
            work_dir = Path(str(build.get("work_dir") or "")).expanduser()
            blocks_path = work_dir / "blocks.evidence.jsonl"
            if not blocks_path.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"blocks.evidence.jsonl not found: {blocks_path}"})
                return
            approvals = _load_doc_approvals(work_dir)
            if not bool(approvals.get(str(doc_id), False)):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "doc is not approved; approve it first"})
                return
            blocks = [b for b in _read_jsonl(blocks_path) if str(b.get("doc_id") or "").strip() == str(doc_id)]
            lines: List[str] = []
            for b in blocks:
                t = str(b.get("text") or "").strip()
                if t:
                    lines.append(t)
            imp = _append_lines_to_evidence_set(topic=topic, source_id=str(doc_id), lines=lines)
            _json_response(
                self,
                HTTPStatus.OK,
                {"ok": True, "topic": topic, "doc_id": doc_id, "evidence_set": imp.get("evidence_set"), "appended": int(imp.get("appended") or 0)},
            )
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})


def main() -> None:
    ap = argparse.ArgumentParser(description="KVI UI server (no dependencies).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = ThreadingHTTPServer((str(args.host), int(args.port)), KVIHandler)
    print(f"[kvi_ui] Serving on http://{args.host}:{int(args.port)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

