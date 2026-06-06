"""HTTP Gateway — 通过 HTTP API 与 Room 交互

零外部依赖（stdlib http.server），开箱即用。

API:
  POST /chat    {"message": "..."}  →  {"responses": [{"sender", "content", "kind"}, ...]}
  GET  /history                     →  [{"sender", "content", "kind", "timestamp"}, ...]
  GET  /members                     →  [{"name", "role"}, ...]
  GET  /                            →  Chat Room Web UI（从 static/chat.html 加载）
"""

from __future__ import annotations
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

from . import BaseGateway, to_json

# 静态文件目录（与当前文件同目录下的 static/）
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 缓存 Web UI 内容，避免每次请求读磁盘
_WEB_UI: str | None = None


def _load_web_ui() -> str:
    global _WEB_UI
    if _WEB_UI is None:
        path = os.path.join(_STATIC_DIR, "chat.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _WEB_UI = f.read()
        except FileNotFoundError:
            _WEB_UI = f"<h1>CollabRoom</h1><p>静态文件丢失: {path}</p>"
    return _WEB_UI


class _Handler(BaseHTTPRequestHandler):
    """每个 HTTP 请求的处理器 — 通过 gateway 引用调用 Room"""

    gateway: "HTTPGateway | None" = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        gw = self.gateway
        if not gw:
            self._json(503, {"error": "Gateway not ready"})
            return

        if path == "" or path == "/":
            self._html(_load_web_ui())
        elif path == "/history":
            tail = int(parsed.query.split("=")[1]) if parsed.query.startswith("tail=") else 50
            self._json(200, gw.get_history(tail=tail))
        elif path == "/members":
            self._json(200, gw.list_members())
        else:
            self._json(404, {"error": f"Unknown path: {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        gw = self.gateway
        if not gw:
            self._json(503, {"error": "Gateway not ready"})
            return

        if parsed.path == "/chat":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode() if length else "{}"
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return

            message = data.get("message", "").strip()
            if not message:
                self._json(400, {"error": "message is required"})
                return

            try:
                responses = gw.handle_message("user", message)
                self._json(200, {"responses": responses})
            except Exception as e:
                self._json(500, {"error": f"处理消息时出错: {e}"})
        else:
            self._json(404, {"error": f"Unknown path: {parsed.path}"})

    def _json(self, status: int, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(to_json(data).encode())

    def _html(self, content: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def log_message(self, format, *args):
        """抑制默认的日志输出（太吵）"""
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器 — 处理长请求时不阻塞其他请求"""
    allow_reuse_address = True
    daemon_threads = True


class HTTPGateway(BaseGateway):
    """HTTP API Gateway

    用法:
      gw = HTTPGateway(room, host="0.0.0.0", port=8765)
      gw.run()  # 阻塞
    """

    def __init__(self, room, host: str = "0.0.0.0",
                 port: int = 8765, name: str = "http"):
        super().__init__(room, name)
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None

    def run(self):
        _Handler.gateway = self
        self._server = ThreadedHTTPServer((self.host, self.port), _Handler)
        print(f"🌐 HTTP Gateway 启动: http://{self.host}:{self.port}")
        print(f"   POST /chat   发送消息")
        print(f"   GET  /history 查看对话历史")
        print(f"   GET  /members 查看成员")
        print(f"   GET  /        简易 Web UI")
        print(f"   Ctrl+C 停止")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if self._server:
            print("\n\U0001f6d1 HTTP Gateway \u505c\u6b62")
            self._server.shutdown()
            self._server = None
