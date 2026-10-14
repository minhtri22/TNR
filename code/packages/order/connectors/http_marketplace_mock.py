"""HTTP mock that speaks Shopee/Sapo-shaped JSON; maps to internal apply_slot.

Routes:
  POST /v2/orders/apply   lab unified apply (tool, args) → returns shaped response
  GET  /api/v2/orders/{sn}  Shopee-like get
  GET  /admin/orders/{code} Sapo-like get
"""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

from .adapters_marketplace import from_sapo, from_shopee, to_sapo, to_shopee
from .file_store import apply_slot, empty_state


class _Handler(BaseHTTPRequestHandler):
    store: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()
    default_shape: str = "shopee"  # or sapo

    def log_message(self, fmt, *args):
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
        path = self.path.split("?")[0]
        # Shopee-like
        if path.startswith("/api/v2/orders/"):
            sn = path[len("/api/v2/orders/") :].strip("/")
            with self.lock:
                st = self.store.get(sn)
            if st is None:
                self._send(404, {"error": "order_not_found"})
            else:
                self._send(200, to_shopee(st))
            return
        # Sapo-like
        if path.startswith("/admin/orders/"):
            code = path[len("/admin/orders/") :].strip("/")
            with self.lock:
                st = self.store.get(code)
            if st is None:
                self._send(404, {"error": "not_found"})
            else:
                self._send(200, to_sapo(st))
            return
        self._send(404, {"error": "unknown_route"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v2/orders/apply":
            self._send(404, {"error": "unknown_route"})
            return
        body = self._read_json()
        tool = str(body.get("tool", ""))
        args = dict(body.get("args") or {})
        oid = str(body.get("order_id") or args.get("order_id") or "")
        shape = str(body.get("shape") or self.default_shape)
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
                self._send(400, {"error": str(ex)})
                return
            key = str(st.get("order_id") or oid or "UNKNOWN")
            if st.get("exists"):
                self.store[key] = dict(st)
            shaped = to_shopee(st) if shape == "shopee" else to_sapo(st)
            self._send(200, {"shape": shape, "order": shaped, "_internal": dict(st)})


def start_marketplace_server(host: str = "127.0.0.1", port: int = 0) -> Tuple[HTTPServer, str]:
    _Handler.store = {}
    server = HTTPServer((host, port), _Handler)
    base = f"http://{server.server_address[0]}:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, base


class MarketplaceHttpClient:
    def __init__(self, base_url: str, shape: str = "shopee"):
        self.base = base_url.rstrip("/")
        self.shape = shape

    def apply(self, tool: str, args: Dict[str, Any], order_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns internal state (unwrapped from shaped HTTP body)."""
        payload = {
            "tool": tool,
            "args": args,
            "order_id": order_id or args.get("order_id", ""),
            "shape": self.shape,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base + "/v2/orders/apply",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            j = json.loads(e.read().decode("utf-8"))
            raise RuntimeError(j.get("error", str(e)))
        # prefer server _internal for exactness; also verify adapter roundtrip
        return dict(body.get("_internal") or {})

    def get_shaped(self, order_id: str) -> Dict[str, Any]:
        if self.shape == "shopee":
            url = self.base + f"/api/v2/orders/{order_id}"
        else:
            url = self.base + f"/admin/orders/{order_id}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            return json.loads(e.read().decode("utf-8"))

    def get_internal(self, order_id: str) -> Dict[str, Any]:
        shaped = self.get_shaped(order_id)
        if self.shape == "shopee":
            return from_shopee(shaped)
        return from_sapo(shaped)
