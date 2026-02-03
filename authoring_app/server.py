from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
import re
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

_DEFAULT_SEMANTIC_TYPE_SPECS = {
    "symptom": {
        "description": "临床表现、症状体征、实验室异常、常见表现的枚举或陈述句。",
        "threshold": 0.28,
        # Optional runtime guidance (config-driven; no evidence text in prompt).
        "focus_terms": ["症状", "体征", "临床表现", "实验室异常", "出血", "呕吐", "腹泻", "白细胞", "血小板"],
        "deny_terms": ["机制", "发病机制", "致病机制", "感染", "免疫", "内皮", "通透性", "复制", "通路", "汉滩病毒", "汉坦病毒"],
    },
    "drug": {
        "description": "治疗、用药、药物、获批/批准、疗效、不良反应等相关陈述句。",
        "threshold": 0.28,
        "focus_terms": ["治疗", "用药", "药物", "疗效", "不良反应", "获批", "批准"],
        "deny_terms": ["地区分布", "流行区域", "机制", "发病机制"],
    },
    "location": {
        "description": "地区分布、流行区域、病例报告地点、地理范围等相关陈述句。",
        "threshold": 0.28,
        "focus_terms": ["地区", "分布", "流行", "报告", "病例", "省", "市", "国家", "区域"],
        "deny_terms": ["治疗", "用药", "药物", "机制", "发病机制"],
    },
    "mechanism": {
        "description": "作用机制/发病机制：感染哪些细胞、免疫应答/免疫抑制、炎症反应、病理过程、通透性改变、多器官损伤等。",
        "threshold": 0.26,
        "focus_terms": ["机制", "发病机制", "致病机制", "感染", "免疫", "炎症", "内皮", "通透性", "细胞", "病理过程"],
        "deny_terms": ["临床表现", "症状", "体征", "治疗", "用药", "地区分布", "流行区域"],
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


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(code))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, code: int, body: str, *, content_type: str) -> None:
    data = (body or "").encode("utf-8")
    handler.send_response(int(code))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
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
            docs_meta = work_dir / "docs.meta.jsonl"
            if not docs_meta.exists():
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found", "message": f"docs.meta.jsonl not found: {docs_meta}"})
                return
            approvals = _load_doc_approvals(work_dir)
            docs = _read_jsonl(docs_meta)
            items: List[Dict[str, Any]] = []
            for d in docs:
                did = str(d.get("doc_id") or "").strip()
                meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
                title = (meta.get("title") if isinstance(meta, dict) else None) or did
                items.append(
                    {
                        "doc_id": did,
                        "pdf_name": str(d.get("source_uri") or "").split("/")[-1],
                        "source_uri": d.get("source_uri"),
                        "title": title,
                        "doi": meta.get("doi") if isinstance(meta, dict) else None,
                        "publication_year": meta.get("publication_year") if isinstance(meta, dict) else None,
                        "approved": bool(approvals.get(did, False)),
                    }
                )
            _json_response(self, HTTPStatus.OK, {"topic": topic, "items": items, "count": len(items), "docs_meta_jsonl": str(docs_meta)})
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
                "0,1,2,3",
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
                    "logs": logs,
                },
            )
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
            top_k = int(obj.get("top_k") or 8)
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
                "--top_k",
                str(top_k),
            ]
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
                "--top_k",
                str(top_k),
            ]
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
            ]
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
                if ln in seen_claims:
                    continue
                seen_claims.add(ln)
                existing.append(
                    {
                        "id": _stable_sentence_id(topic=topic, claim=ln, source_id=str(doc_id)),
                        "topic": topic,
                        "claim": ln,
                        "source_id": str(doc_id),
                        "source_ref": {"doi": None, "title": None},
                        "created_at": now,
                        "updated_at": now,
                        "author": None,
                        "tags": [],
                    }
                )
                appended += 1
            _write_sentence_set(fp, existing)
            _json_response(
                self,
                HTTPStatus.OK,
                {"ok": True, "topic": topic, "doc_id": doc_id, "evidence_set": str(target.get("name")), "appended": appended},
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

