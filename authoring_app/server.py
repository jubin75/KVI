from __future__ import annotations

import argparse
import json
import sys
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
                },
            )
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
    ap.add_argument("--db", required=True, help="Path to authoring evidence_units.jsonl")
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

