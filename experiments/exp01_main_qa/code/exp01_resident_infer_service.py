#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import runpy
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse


_EXEC_LOCK = threading.Lock()
_TOK_CACHE: Dict[str, Any] = {}
_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_PATCHED = False


def _json_response(handler: BaseHTTPRequestHandler, code: int, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _patch_transformers_cache() -> None:
    global _CACHE_PATCHED
    if _CACHE_PATCHED:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    orig_tok = AutoTokenizer.from_pretrained
    orig_model = AutoModelForCausalLM.from_pretrained

    def cached_tok(model_id: str, *args: Any, **kwargs: Any) -> Any:
        key = json.dumps({"m": str(model_id), "k": kwargs}, sort_keys=True, default=str)
        if key not in _TOK_CACHE:
            _TOK_CACHE[key] = orig_tok(model_id, *args, **kwargs)
        return _TOK_CACHE[key]

    def cached_model(model_id: str, *args: Any, **kwargs: Any) -> Any:
        key = json.dumps({"m": str(model_id), "k": kwargs}, sort_keys=True, default=str)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = orig_model(model_id, *args, **kwargs)
        return _MODEL_CACHE[key]

    AutoTokenizer.from_pretrained = cached_tok  # type: ignore[assignment]
    AutoModelForCausalLM.from_pretrained = cached_model  # type: ignore[assignment]
    _CACHE_PATCHED = True


def _safe_parse_last_json_obj(text: str) -> Dict[str, Any]:
    s = str(text or "").strip()
    if not s:
        return {}
    start = s.rfind("{")
    while start >= 0:
        cand = s[start:]
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = s.rfind("{", 0, start)
    return {}


def _execute_script(script_path: Path, argv: list[str]) -> Tuple[Dict[str, Any], str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + argv
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            try:
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit as e:
                code = int(e.code) if isinstance(e.code, int) else 0
                if code != 0:
                    raise
    finally:
        sys.argv = old_argv
    stdout_txt = out_buf.getvalue()
    stderr_txt = err_buf.getvalue()
    obj = _safe_parse_last_json_obj(stdout_txt)
    return obj, stdout_txt, stderr_txt


class Handler(BaseHTTPRequestHandler):
    repo_root = Path(__file__).resolve().parents[3]
    graph_script = repo_root / "scripts" / "run_graph_inference.py"
    kvi_script = repo_root / "scripts" / "run_kvi2_runtime_test.py"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            _json_response(
                self,
                HTTPStatus.OK,
                {"ok": True, "cached_models": len(_MODEL_CACHE), "cached_tokenizers": len(_TOK_CACHE)},
            )
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
            return

        try:
            with _EXEC_LOCK:
                if parsed.path == "/infer/graph":
                    argv = list(payload.get("argv") or [])
                    obj, stdout_txt, stderr_txt = _execute_script(self.graph_script, argv)
                    _json_response(self, HTTPStatus.OK, {"ok": True, "result": obj, "stderr_tail": stderr_txt[-2000:], "stdout_tail": stdout_txt[-2000:]})
                    return
                if parsed.path == "/infer/kvi":
                    argv = list(payload.get("argv") or [])
                    obj, stdout_txt, stderr_txt = _execute_script(self.kvi_script, argv)
                    _json_response(self, HTTPStatus.OK, {"ok": True, "result": obj, "stderr_tail": stderr_txt[-2000:], "stdout_tail": stdout_txt[-2000:]})
                    return
        except Exception as e:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    p = argparse.ArgumentParser(description="Exp01 resident inference service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18888)
    args = p.parse_args()
    _patch_transformers_cache()
    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(json.dumps({"ok": True, "host": args.host, "port": int(args.port)}, ensure_ascii=False))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
