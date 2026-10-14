"""HTTP mock order API — thin stand-in for sàn/OMS REST.

Hypothesis (order_conn_v3): program→execute + CONFIRM work when side-effects
go through HTTP JSON endpoints (localhost mock), not only in-process stores.
"""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

from .file_store import apply_slot, empty_state


class _OrderAPIHandler(BaseHTTPRequestHandler):
    """In-memory multi-order store behind REST-ish routes."""

    store: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # quiet
        return

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _send(self, code: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        # GET /orders/{id}
        parts = self.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "orders":
            oid = parts[1]
            with self.lock:
                st = self.store.get(oid)
            if st is None:
                self._send(404, {"error": "not_found", "order_id": oid})
            else:
                self._send(200, dict(st))
            return
        self._send(404, {"error": "unknown_route"})

    def do_POST(self):
        # POST /orders/apply  body: {tool, args, order_id?}
        if self.path.rstrip("/") != "/orders/apply":
            self._send(404, {"error": "unknown_route"})
            return
        body = self._read_json()
        tool = str(body.get("tool", ""))
        args = dict(body.get("args") or {})
        oid = str(body.get("order_id") or args.get("order_id") or "")
        with self.lock:
            if tool == "LOOKUP_ORDER" and oid and oid in self.store:
                st = dict(self.store[oid])
            elif oid and oid in self.store:
                st = dict(self.store[oid])
            else:
                st = empty_state()
            try:
                st = apply_slot(tool, args, st)
            except Exception as ex:
                self._send(400, {"error": str(ex), "state": st})
                return
            key = str(st.get("order_id") or oid or "UNKNOWN")
            if st.get("exists"):
                self.store[key] = dict(st)
            self._send(200, dict(st))


def start_server(host: str = "127.0.0.1", port: int = 0) -> Tuple[HTTPServer, str]:
    """Bind ephemeral port; return (server, base_url)."""
    _OrderAPIHandler.store = {}
    server = HTTPServer((host, port), _OrderAPIHandler)
    base = f"http://{server.server_address[0]}:{server.server_address[1]}"
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, base


class HttpOrderClient:
    """Client used by pack — mirrors FileOrderStore / CsvOrderStore API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _post_apply(self, tool: str, args: Dict[str, Any], order_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"tool": tool, "args": args, "order_id": order_id or args.get("order_id", "")}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base + "/orders/apply",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                j = json.loads(body)
            except Exception:
                j = {"error": body}
            if e.code == 400 and "error" in j:
                raise RuntimeError(j["error"])
            raise

    def get(self, order_id: str) -> Dict[str, Any]:
        req = request.Request(self.base + f"/orders/{order_id}", method="GET")
        try:
            with request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            if e.code == 404:
                return empty_state()
            raise

    def apply(self, tool: str, args: Dict[str, Any], order_id: Optional[str] = None) -> Dict[str, Any]:
        return self._post_apply(tool, args, order_id=order_id)
