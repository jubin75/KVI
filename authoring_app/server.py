from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
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

try:
    from external_kv_injection.src.authoring.models import (  # type: ignore
        EvidenceRejectionCode,
        EvidenceUnit,
        new_evidence_id,
    )
except ModuleNotFoundError:
    from src.authoring.models import EvidenceRejectionCode, EvidenceUnit, new_evidence_id  # type: ignore


STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # external_kv_injection/
TOPICS_DIR = PROJECT_ROOT / "config" / "topics"

_INDEX_FALLBACK_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authoring UI (static missing)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }}
    .box {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; max-width: 980px; }}
    .muted {{ color: #555; }}
  </style>
</head>
<body>
  <div class="box">
    <h2>Authoring UI static files not found</h2>
    <p class="muted">
      The HTTP server is running, but <code>authoring_app/static/index.html</code> was not found on disk.
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
      API is reachable: <a href="/api/health">/api/health</a> · <a href="/api/evidence">/api/evidence</a>
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


class JsonlEvidenceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _read_all(self) -> List[EvidenceUnit]:
        units: List[EvidenceUnit] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = _safe_json_loads(s)
                if not isinstance(obj, dict):
                    continue
                try:
                    units.append(EvidenceUnit.from_dict(obj))
                except Exception:
                    continue
        # IMPORTANT: keep file order (queue-like UX).
        return units

    def _write_all(self, units: List[EvidenceUnit]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def list(self) -> List[EvidenceUnit]:
        return self._read_all()

    def get(self, evidence_id: str) -> Optional[EvidenceUnit]:
        eid = str(evidence_id or "").strip()
        if not eid:
            return None
        for u in self._read_all():
            if str(u.evidence_id) == eid:
                return u
        return None

    def upsert(self, unit: EvidenceUnit) -> EvidenceUnit:
        eid = str(unit.evidence_id or "").strip()
        if not eid:
            unit.evidence_id = new_evidence_id()
            eid = str(unit.evidence_id)
        units = self._read_all()
        out: List[EvidenceUnit] = []
        replaced = False
        for u in units:
            if str(u.evidence_id) == eid:
                out.append(unit)
                replaced = True
            else:
                out.append(u)
        if not replaced:
            out.append(unit)
        out.sort(key=lambda u: str(u.evidence_id))
        self._write_all(out)
        return unit


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


def _filter_units(units: List[EvidenceUnit], *, qs: Dict[str, List[str]]) -> List[EvidenceUnit]:
    status = (qs.get("status", [""])[0] or "").strip().lower()
    # UI uses kb_id; keep schema_id as a backward-compatible alias.
    kb_id = (qs.get("kb_id", [""])[0] or "").strip()
    schema_id = (qs.get("schema_id", [""])[0] or "").strip()
    if (not kb_id) and schema_id:
        kb_id = schema_id
    semantic_type = (qs.get("semantic_type", [""])[0] or "").strip().lower()
    q = (qs.get("q", [""])[0] or "").strip().lower()

    out: List[EvidenceUnit] = []
    for u in units:
        if status and str(u.status).lower() != status:
            continue
        if kb_id and str(u.schema_id) != kb_id:
            continue
        if semantic_type and str(u.semantic_type).lower() != semantic_type:
            continue
        if q:
            hay = " ".join(
                [
                    str(u.evidence_id or ""),
                    str(u.claim or ""),
                    str(u.schema_id or ""),
                    str(u.semantic_type or ""),
                ]
            ).lower()
            if q not in hay:
                continue
        out.append(u)
    return out


class AuthoringHandler(BaseHTTPRequestHandler):
    store: JsonlEvidenceStore
    default_kb_id: str = ""

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

        if path in {"/api/rejection_codes", "/api/evidence"} or path.startswith("/api/evidence/"):
            # For HEAD we only confirm reachability.
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
                    "db_path": str(getattr(self.store, "path", "")),
                    "default_kb_id": str(getattr(self, "default_kb_id", "") or ""),
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

        if path == "/api/rejection_codes":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "codes": [
                        EvidenceRejectionCode.SEMANTIC_TYPE_MISMATCH,
                        EvidenceRejectionCode.SCHEMA_MISMATCH,
                        EvidenceRejectionCode.NON_ENUMERATIVE,
                        EvidenceRejectionCode.MIXED_SEMANTICS,
                        EvidenceRejectionCode.LOW_CONFIDENCE,
                        EvidenceRejectionCode.NOT_APPROVED,
                    ]
                },
            )
            return

        if path == "/api/export/evidence.txt":
            # Legacy/simple pipeline export.
            # Query params:
            #   status=approved|all (default approved)
            status = (qs.get("status", ["approved"])[0] or "approved").strip().lower()
            units = self.store.list()
            lines: List[str] = []
            for u in units:
                if status != "all" and str(u.status).lower() != "approved":
                    continue
                txt = str(u.claim or "").strip()
                if not txt:
                    continue
                lines.append(f"{u.evidence_id}\t{txt}")
            body = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/evidence":
            units_all = _filter_units(self.store.list(), qs=qs)
            try:
                offset = int((qs.get("offset", ["0"])[0] or "0").strip())
            except Exception:
                offset = 0
            try:
                limit = int((qs.get("limit", ["200"])[0] or "200").strip())
            except Exception:
                limit = 200
            offset = max(0, offset)
            limit = min(2000, max(1, limit))
            units = units_all[offset : offset + limit]
            payload = [
                {
                    "evidence_id": u.evidence_id,
                    "semantic_type": u.semantic_type,
                    # Backing field in DB is `schema_id`, but UI speaks kb_id.
                    "kb_id": u.schema_id,
                    "schema_id": u.schema_id,
                    "effective_kb_id": (u.schema_id or str(getattr(self, "default_kb_id", "") or "")),
                    "status": u.status,
                    "polarity": u.polarity,
                    "claim": u.claim,
                    "rejection": u.rejection,
                    "updated_at": u.updated_at,
                }
                for u in units
            ]
            _json_response(
                self,
                HTTPStatus.OK,
                {"items": payload, "count": int(len(payload)), "total": int(len(units_all)), "offset": int(offset), "limit": int(limit)},
            )
            return

        if path.startswith("/api/evidence/"):
            eid_raw = path[len("/api/evidence/") :].strip()
            eid = unquote(eid_raw)
            u = self.store.get(eid)
            if u is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found", "evidence_id": eid})
                return
            _json_response(self, HTTPStatus.OK, u.to_dict())
            return

        _text_response(self, HTTPStatus.NOT_FOUND, "not found", content_type="text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        # -----------------------------
        # KVI Simple UI APIs (new)
        # -----------------------------
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
            now = _safe_now_iso()
            cleaned: List[Dict[str, Any]] = []
            for r in records:
                if not isinstance(r, dict):
                    continue
                claim = str(r.get("claim") or "").strip()
                if not claim:
                    continue
                src_id = r.get("source_id")
                src_id_s = str(src_id).strip() if src_id is not None and str(src_id).strip() else ""
                sid = str(r.get("id") or "").strip() or _stable_sentence_id(topic=topic, claim=claim, source_id=src_id_s)
                created_at = str(r.get("created_at") or "").strip() or now
                updated_at = now
                cleaned.append(
                    {
                        "id": sid,
                        "topic": topic,
                        "claim": claim,
                        "source_id": (src_id_s if src_id_s else None),
                        "source_ref": r.get("source_ref") if isinstance(r.get("source_ref"), dict) else {"doi": None, "title": None},
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "author": r.get("author"),
                        "tags": r.get("tags") if isinstance(r.get("tags"), list) else [],
                    }
                )
            _write_sentence_set(fp, cleaned)
            mf = _load_manifest(topic)
            mf.setdefault("sets", {})
            if name not in mf["sets"]:
                mf["sets"][name] = {"enabled": True, "created_at": now}
            if "enabled" in obj:
                mf["sets"][name]["enabled"] = bool(obj.get("enabled"))
            _save_manifest(topic, mf)
            _json_response(self, HTTPStatus.OK, {"ok": True, "topic": topic, "name": name, "path": str(fp), "count": len(cleaned), "enabled": bool((mf["sets"][name] or {}).get("enabled", True))})
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

            topic_dir = _topic_dir(topic)
            # Build a compiled raw text from enabled evidence sets.
            claims, claim_stats = _collect_compiled_claims(topic, enabled_only=True)
            if not claims:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "No claims found in enabled evidence sets. Create/enable an evidence set first."},
                )
                return

            # Optional overrides
            obj, _ = _read_body_json(self)
            obj = obj if isinstance(obj, dict) else {}
            device = str(obj.get("device") or "").strip()
            dtype = str(obj.get("dtype") or "").strip()

            # Prefer writing artifacts to configured work_dir (matches your CLI usage),
            # but fallback to topic_dir for minimal setups.
            work_dir_raw = str(build.get("work_dir") or "").strip()
            out_dir = Path(work_dir_raw).expanduser() if work_dir_raw else topic_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            pattern_dir = out_dir / "pattern_sidecar"
            pattern_dir.mkdir(parents=True, exist_ok=True)

            compiled_txt = out_dir / "evidence.compiled.txt"
            compiled_txt.write_text("\n".join(claims) + ("\n" if claims else ""), encoding="utf-8")

            blocks_jsonl = out_dir / "blocks.jsonl"
            blocks_enriched = out_dir / "blocks.enriched.jsonl"
            kv_dir = out_dir / "kvbank_blocks"

            cmds: List[List[str]] = []
            cmds.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_blocks_from_raw_text.py"),
                    "--raw_text",
                    str(compiled_txt),
                    "--out",
                    str(blocks_jsonl),
                    "--tokenizer",
                    base_llm,
                    "--chunk_tokens",
                    "4096",
                    "--chunk_overlap",
                    "256",
                    "--block_tokens",
                    "256",
                    "--keep_last_incomplete_block",
                ]
            )
            cmds.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_pattern_index_from_blocks_v2.py"),
                    "--blocks_jsonl_in",
                    str(blocks_jsonl),
                    "--blocks_jsonl_out",
                    str(blocks_enriched),
                    "--pattern_out_dir",
                    str(pattern_dir),
                ]
            )
            cmds.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "pattern_contract_autogen.py"),
                    "--blocks_jsonl_in",
                    str(blocks_enriched),
                    "--out",
                    str(out_dir / "pattern_contract.json"),
                    "--topic",
                    str(topic),
                    "--min_abbr_count",
                    "1",
                    "--min_slot_count",
                    "1",
                    "--max_abbr",
                    "50",
                    "--max_slots",
                    "50",
                ]
            )
            cmd_kv = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_kvbank_from_blocks_jsonl.py"),
                "--blocks_jsonl",
                str(blocks_enriched),
                "--out_dir",
                str(kv_dir),
                "--base_llm",
                base_llm,
                "--domain_encoder_model",
                encoder,
                "--layers",
                "0,1,2,3",
                "--block_tokens",
                "256",
                "--shard_size",
                "1024",
            ]
            if device:
                cmd_kv.extend(["--device", device])
            if dtype:
                cmd_kv.extend(["--dtype", dtype])
            cmds.append(cmd_kv)

            logs: List[Dict[str, Any]] = []
            for c in cmds:
                r = subprocess.run(c, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
                logs.append(
                    {
                        "cmd": " ".join(c),
                        "returncode": int(r.returncode),
                        "stdout": (r.stdout or "")[-8000:],
                        "stderr": (r.stderr or "")[-8000:],
                    }
                )
                if r.returncode != 0:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "topic": topic, "failed_cmd": " ".join(c), "logs": logs})
                    return

            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "topic": topic,
                    "topic_dir": str(topic_dir),
                    "compiled_evidence_txt": str(compiled_txt),
                    "compiled_stats": claim_stats,
                    "out_dir": str(out_dir),
                    "pattern_index_dir": str(pattern_dir),
                    "blocks_jsonl": str(blocks_jsonl),
                    "blocks_enriched_jsonl": str(blocks_enriched),
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
            kv_dir = out_dir / "kvbank_blocks"
            blocks_enriched = out_dir / "blocks.enriched.jsonl"
            if not kv_dir.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"kvbank_blocks not found: {kv_dir}. Click 编译 first."})
                return
            if not blocks_enriched.exists():
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"blocks.enriched.jsonl not found: {blocks_enriched}. Click 编译 first."})
                return

            top_k = int(obj.get("top_k") or 8)
            show_baseline = bool(obj.get("show_baseline", True))
            simple_use_evidence_units = bool(obj.get("simple_use_evidence_units", True))
            simple_require_units = bool(obj.get("simple_require_units", True))
            simple_max_steps = int(obj.get("simple_max_steps") or 1)
            simple_step_new_tokens = int(obj.get("simple_step_new_tokens") or 192)
            simple_max_blocks_per_step = int(obj.get("simple_max_blocks_per_step") or 4)
            simple_max_unit_sentences = int(obj.get("simple_max_unit_sentences") or 6)

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
                "--blocks_jsonl",
                str(blocks_enriched),
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
                "--simple_max_unit_sentences",
                str(simple_max_unit_sentences),
            ]
            if simple_use_evidence_units:
                cmd.append("--simple_use_evidence_units")
            if simple_require_units:
                cmd.append("--simple_require_units")
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
                    "blocks_jsonl": str(blocks_enriched),
                    "pattern_index_dir": str(pattern_index_dir),
                    "sidecar_dir": str(out_dir),
                    "result": out,
                    "stderr_tail": (r.stderr or "")[-2000:],
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

        if path == "/api/doc_bundle/list":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            docs_meta_jsonl = str(obj.get("docs_meta_jsonl") or "").strip()
            blocks_evidence_jsonl = str(obj.get("blocks_evidence_jsonl") or "").strip()
            if not docs_meta_jsonl or not blocks_evidence_jsonl:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "docs_meta_jsonl and blocks_evidence_jsonl required"},
                )
                return
            try:
                docs = _read_jsonl(Path(docs_meta_jsonl))
                blocks = _read_jsonl(Path(blocks_evidence_jsonl))
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return

            meta_by_doc: Dict[str, Dict[str, Any]] = {}
            for d in docs:
                did = str(d.get("doc_id") or "").strip()
                if did:
                    meta_by_doc[did] = d

            counts: Dict[str, int] = {}
            type_counts: Dict[str, Dict[str, int]] = {}
            for b in blocks:
                did = str(b.get("doc_id") or "").strip()
                if not did:
                    continue
                counts[did] = int(counts.get(did, 0)) + 1
                bt = str(b.get("block_type") or "paragraph_summary")
                type_counts.setdefault(did, {})[bt] = int(type_counts[did].get(bt, 0)) + 1

            items: List[Dict[str, Any]] = []
            for did, drec in meta_by_doc.items():
                meta = drec.get("meta") if isinstance(drec.get("meta"), dict) else {}
                items.append(
                    {
                        "doc_id": did,
                        "source_uri": drec.get("source_uri"),
                        "kb_id": drec.get("kb_id"),
                        "title": meta.get("title") or did,
                        "journal": meta.get("journal"),
                        "doi": meta.get("doi"),
                        "publication_year": meta.get("publication_year"),
                        "published_at": meta.get("published_at"),
                        "authors": meta.get("authors") if isinstance(meta.get("authors"), list) else [],
                        "evidence_count": int(counts.get(did, 0)),
                        "block_type_counts": type_counts.get(did, {}),
                    }
                )
            # Sort by evidence_count desc
            items.sort(key=lambda x: int(x.get("evidence_count") or 0), reverse=True)
            _json_response(self, HTTPStatus.OK, {"items": items, "count": len(items)})
            return

        if path == "/api/doc_bundle/doc":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            blocks_evidence_jsonl = str(obj.get("blocks_evidence_jsonl") or "").strip()
            doc_id = str(obj.get("doc_id") or "").strip()
            docs_meta_jsonl = str(obj.get("docs_meta_jsonl") or "").strip()
            if not blocks_evidence_jsonl or not doc_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "blocks_evidence_jsonl and doc_id required"})
                return
            try:
                blocks = _read_jsonl(Path(blocks_evidence_jsonl))
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return
            doc_meta: Optional[Dict[str, Any]] = None
            if docs_meta_jsonl:
                try:
                    docs = _read_jsonl(Path(docs_meta_jsonl))
                    for d in docs:
                        if str(d.get("doc_id") or "").strip() == doc_id:
                            doc_meta = d
                            break
                except Exception:
                    doc_meta = None

            out_blocks: List[Dict[str, Any]] = []
            for b in blocks:
                if str(b.get("doc_id") or "").strip() != doc_id:
                    continue
                out_blocks.append(
                    {
                        "block_id": b.get("block_id"),
                        "block_type": b.get("block_type") or "paragraph_summary",
                        "text": b.get("text") or "",
                        "source_uri": b.get("source_uri"),
                        "metadata": b.get("metadata") if isinstance(b.get("metadata"), dict) else {},
                    }
                )
            _json_response(self, HTTPStatus.OK, {"doc_id": doc_id, "doc_meta": doc_meta, "blocks": out_blocks, "count": len(out_blocks)})
            return

        if path == "/api/import/blocks":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            blocks_jsonl = str(obj.get("blocks_jsonl") or "").strip()
            kb_id = str(obj.get("kb_id") or obj.get("schema_id") or "").strip() or str(getattr(self, "default_kb_id", "") or "").strip()
            default_semantic_type = str(obj.get("default_semantic_type") or "generic").strip()
            evidence_type = str(obj.get("evidence_type") or "pdf_block").strip()
            max_blocks = int(obj.get("max_blocks") or 0)
            if not blocks_jsonl or not kb_id:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "blocks_jsonl required; kb_id is optional if server has --default_kb_id"},
                )
                return
            try:
                try:
                    from external_kv_injection.src.authoring.importers import (  # type: ignore
                        import_blocks_jsonl_to_authoring_db,
                    )
                except ModuleNotFoundError:
                    from src.authoring.importers import import_blocks_jsonl_to_authoring_db  # type: ignore
                stats = import_blocks_jsonl_to_authoring_db(
                    blocks_jsonl=Path(blocks_jsonl),
                    authoring_db_jsonl=self.store.path,
                    schema_id=kb_id,
                    default_semantic_type=default_semantic_type,
                    evidence_type=evidence_type,
                    max_blocks=max_blocks,
                )
                _json_response(self, HTTPStatus.OK, {"ok": True, "stats": stats.__dict__})
                return
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return

        if path == "/api/import/blocks.evidence":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            blocks_evidence_jsonl = str(obj.get("blocks_evidence_jsonl") or obj.get("blocks_jsonl") or "").strip()
            docs_meta_jsonl = str(obj.get("docs_meta_jsonl") or "").strip()
            kb_id = str(obj.get("kb_id") or obj.get("schema_id") or "").strip() or str(getattr(self, "default_kb_id", "") or "").strip()
            default_semantic_type = str(obj.get("default_semantic_type") or "generic").strip()
            evidence_type = str(obj.get("evidence_type") or "extractive_suggestion").strip()
            if not blocks_evidence_jsonl or not kb_id:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "blocks_evidence_jsonl required; kb_id is optional if server has --default_kb_id"},
                )
                return
            try:
                try:
                    from external_kv_injection.src.authoring.importers import (  # type: ignore
                        import_deepseek_blocks_evidence_jsonl_to_authoring_db,
                    )
                except ModuleNotFoundError:
                    from src.authoring.importers import import_deepseek_blocks_evidence_jsonl_to_authoring_db  # type: ignore
                stats = import_deepseek_blocks_evidence_jsonl_to_authoring_db(
                    blocks_evidence_jsonl=Path(blocks_evidence_jsonl),
                    docs_meta_jsonl=(Path(docs_meta_jsonl) if docs_meta_jsonl else None),
                    authoring_db_jsonl=self.store.path,
                    schema_id=kb_id,
                    default_semantic_type=default_semantic_type,
                    evidence_type=evidence_type,
                )
                _json_response(self, HTTPStatus.OK, {"ok": True, "stats": stats.__dict__})
                return
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return

        if path == "/api/evidence":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            # Create / upsert
            try:
                eu = EvidenceUnit.from_dict(obj)
            except Exception as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
                return
            if not str(eu.evidence_id or "").strip():
                eu.evidence_id = new_evidence_id()
            if not str(eu.status or "").strip():
                eu.status = "draft"
            if not str(eu.schema_id or "").strip() and str(getattr(self, "default_kb_id", "") or "").strip():
                eu.schema_id = str(getattr(self, "default_kb_id", "") or "")
            saved = self.store.upsert(eu)
            _json_response(self, HTTPStatus.OK, saved.to_dict())
            return

        if path.startswith("/api/evidence/") and path.endswith("/approve"):
            eid_raw = path[len("/api/evidence/") : -len("/approve")].strip().rstrip("/")
            eid = unquote(eid_raw)
            u = self.store.get(eid)
            if u is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            u.status = "approved"
            u.rejection = None
            if not str(u.schema_id or "").strip() and str(getattr(self, "default_kb_id", "") or "").strip():
                u.schema_id = str(getattr(self, "default_kb_id", "") or "")
            self.store.upsert(u)
            _json_response(self, HTTPStatus.OK, u.to_dict())
            return

        if path.startswith("/api/evidence/") and path.endswith("/reject"):
            eid_raw = path[len("/api/evidence/") : -len("/reject")].strip().rstrip("/")
            eid = unquote(eid_raw)
            u = self.store.get(eid)
            if u is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            rej = obj.get("rejection") if isinstance(obj.get("rejection"), dict) else obj
            code = str(rej.get("code") or "").strip()
            message = str(rej.get("message") or "").strip()
            if not code or not message:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "rejection.code and rejection.message required"})
                return
            # enforce finite set (soft enforcement: unknown codes are rejected)
            allowed = {
                EvidenceRejectionCode.SEMANTIC_TYPE_MISMATCH,
                EvidenceRejectionCode.SCHEMA_MISMATCH,
                EvidenceRejectionCode.NON_ENUMERATIVE,
                EvidenceRejectionCode.MIXED_SEMANTICS,
                EvidenceRejectionCode.LOW_CONFIDENCE,
                EvidenceRejectionCode.NOT_APPROVED,
            }
            if code not in allowed:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"unknown rejection code: {code}"})
                return
            details = rej.get("details") if isinstance(rej.get("details"), dict) else {}
            confidence = rej.get("confidence")
            try:
                conf = float(confidence) if confidence is not None else 0.9
            except Exception:
                conf = 0.9
            u.status = "rejected"
            u.rejection = {"code": code, "message": message, "details": details, "confidence": conf}
            if not str(u.schema_id or "").strip() and str(getattr(self, "default_kb_id", "") or "").strip():
                u.schema_id = str(getattr(self, "default_kb_id", "") or "")
            self.store.upsert(u)
            _json_response(self, HTTPStatus.OK, u.to_dict())
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        if not path.startswith("/api/evidence/"):
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        eid_raw = path[len("/api/evidence/") :].strip().rstrip("/")
        eid = unquote(eid_raw)
        obj, err = _read_body_json(self)
        if err or not isinstance(obj, dict):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
            return
        try:
            eu = EvidenceUnit.from_dict(obj)
        except Exception as e:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": f"{type(e).__name__}: {e}"})
            return
        eu.evidence_id = eid  # path is source of truth
        if not str(eu.schema_id or "").strip() and str(getattr(self, "default_kb_id", "") or "").strip():
            eu.schema_id = str(getattr(self, "default_kb_id", "") or "")
        saved = self.store.upsert(eu)
        _json_response(self, HTTPStatus.OK, saved.to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description="Authoring MVP UI server (no dependencies).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--db",
        default=str((PROJECT_ROOT / "authoring_app" / "authoring_db.jsonl").resolve()),
        help="Path to authoring evidence_units.jsonl (optional; kept for back-compat).",
    )
    ap.add_argument("--default_kb_id", default="", help="Default kb_id used for imports/new drafts (optional).")
    ap.add_argument(
        "--default_schema_id",
        default="",
        help="Deprecated alias of --default_kb_id (kept for back-compat).",
    )
    args = ap.parse_args()

    store = JsonlEvidenceStore(Path(str(args.db)))
    AuthoringHandler.store = store  # type: ignore[assignment]
    dk = str(args.default_kb_id or "").strip()
    if not dk:
        dk = str(args.default_schema_id or "").strip()
    AuthoringHandler.default_kb_id = dk

    server = ThreadingHTTPServer((str(args.host), int(args.port)), AuthoringHandler)
    print(f"[authoring_app] Serving on http://{args.host}:{int(args.port)}  db={args.db}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

