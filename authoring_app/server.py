from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


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
        # stable sort by id (human friendly)
        units.sort(key=lambda u: str(u.evidence_id))
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
    schema_id = (qs.get("schema_id", [""])[0] or "").strip()
    semantic_type = (qs.get("semantic_type", [""])[0] or "").strip().lower()
    q = (qs.get("q", [""])[0] or "").strip().lower()

    out: List[EvidenceUnit] = []
    for u in units:
        if status and str(u.status).lower() != status:
            continue
        if schema_id and str(u.schema_id) != schema_id:
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

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 (shadow builtin)
        # Keep console output minimal for MVP
        return

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

        if path == "/api/evidence":
            units = _filter_units(self.store.list(), qs=qs)
            payload = [
                {
                    "evidence_id": u.evidence_id,
                    "semantic_type": u.semantic_type,
                    "schema_id": u.schema_id,
                    "status": u.status,
                    "polarity": u.polarity,
                    "claim": u.claim,
                    "rejection": u.rejection,
                    "updated_at": u.updated_at,
                }
                for u in units
            ]
            _json_response(self, HTTPStatus.OK, {"items": payload, "count": int(len(payload))})
            return

        if path.startswith("/api/evidence/"):
            eid = path[len("/api/evidence/") :].strip()
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

        if path == "/api/import/blocks":
            obj, err = _read_body_json(self)
            if err or not isinstance(obj, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": err or "dict required"})
                return
            blocks_jsonl = str(obj.get("blocks_jsonl") or "").strip()
            schema_id = str(obj.get("schema_id") or "").strip()
            default_semantic_type = str(obj.get("default_semantic_type") or "generic").strip()
            evidence_type = str(obj.get("evidence_type") or "pdf_block").strip()
            max_blocks = int(obj.get("max_blocks") or 0)
            if not blocks_jsonl or not schema_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": "blocks_jsonl and schema_id required"})
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
                    schema_id=schema_id,
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
            schema_id = str(obj.get("schema_id") or "").strip()
            default_semantic_type = str(obj.get("default_semantic_type") or "generic").strip()
            evidence_type = str(obj.get("evidence_type") or "extractive_suggestion").strip()
            if not blocks_evidence_jsonl or not schema_id:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "bad_request", "message": "blocks_evidence_jsonl (or blocks_jsonl) and schema_id required"},
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
                    authoring_db_jsonl=self.store.path,
                    schema_id=schema_id,
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
            saved = self.store.upsert(eu)
            _json_response(self, HTTPStatus.OK, saved.to_dict())
            return

        if path.startswith("/api/evidence/") and path.endswith("/approve"):
            eid = path[len("/api/evidence/") : -len("/approve")].strip().rstrip("/")
            u = self.store.get(eid)
            if u is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            u.status = "approved"
            u.rejection = None
            self.store.upsert(u)
            _json_response(self, HTTPStatus.OK, u.to_dict())
            return

        if path.startswith("/api/evidence/") and path.endswith("/reject"):
            eid = path[len("/api/evidence/") : -len("/reject")].strip().rstrip("/")
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
        eid = path[len("/api/evidence/") :].strip().rstrip("/")
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
        saved = self.store.upsert(eu)
        _json_response(self, HTTPStatus.OK, saved.to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description="Authoring MVP UI server (no dependencies).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", required=True, help="Path to authoring evidence_units.jsonl")
    args = ap.parse_args()

    store = JsonlEvidenceStore(Path(str(args.db)))
    AuthoringHandler.store = store  # type: ignore[assignment]

    server = ThreadingHTTPServer((str(args.host), int(args.port)), AuthoringHandler)
    print(f"[authoring_app] Serving on http://{args.host}:{int(args.port)}  db={args.db}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

