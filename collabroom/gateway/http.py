"""HTTP Gateway — 通过 HTTP API 与 Room 交互

零外部依赖（stdlib http.server），开箱即用。

API:
  POST /chat    {"message": "..."}  →  [{"sender", "content", "kind"}, ...]
  GET  /history                     →  [{"sender", "content", "kind", "timestamp"}, ...]
  GET  /members                     →  [{"name", "role"}, ...]
  GET  /                            →  简易 Web UI
"""

from __future__ import annotations
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from . import BaseGateway, to_json


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
            self._html(_WEB_UI)
        elif path == "/history":
            self._json(200, gw.get_history(tail=50))
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

            responses = gw.handle_message("user", message)
            self._json(200, {"responses": responses})
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
        self._server = HTTPServer((self.host, self.port), _Handler)
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
            print("\n🛑 HTTP Gateway 停止")
            self._server.shutdown()
            self._server = None


# ── 简易 Web UI ──

_WEB_UI = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>CollabRoom</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; max-width: 800px; margin: 0 auto; padding: 20px; }
  h1 { color: #e94560; margin-bottom: 20px; }
  #chat { background: #16213e; border-radius: 12px; padding: 16px; height: 500px; overflow-y: auto; margin-bottom: 16px; }
  .msg { margin: 8px 0; padding: 8px 12px; border-radius: 8px; }
  .msg .sender { font-weight: bold; color: #0f3460; margin-bottom: 4px; }
  .msg .sender.架构师 { color: #e94560; }
  .msg .sender.开发者 { color: #4ecca3; }
  .msg .sender.测试 { color: #ffd369; }
  .msg .sender.user { color: #aaa; }
  .msg.user { background: #0f3460; }
  .msg.agent { background: #1a1a3e; border-left: 3px solid #e94560; }
  .row { display: flex; gap: 8px; }
  input { flex: 1; padding: 12px; border: none; border-radius: 8px; background: #16213e; color: #e0e0e0; font-size: 14px; }
  input:focus { outline: 2px solid #e94560; }
  button { padding: 12px 24px; border: none; border-radius: 8px; background: #e94560; color: white; font-weight: bold; cursor: pointer; }
  button:hover { background: #c73e54; }
  .loading { color: #888; font-style: italic; }
</style>
</head>
<body>
<h1>🏠 CollabRoom</h1>
<div id="chat"></div>
<div class="row">
  <input id="input" placeholder="输入消息..." onkeydown="if(event.key==='Enter') send()">
  <button onclick="send()">发送</button>
</div>
<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');
  function addMsg(sender, content, kind) {
    const div = document.createElement('div');
    div.className = 'msg ' + (kind === 'dm' || sender === 'user' ? 'user' : 'agent');
    div.innerHTML = '<div class="sender ' + sender + '">' + sender + '</div><div>' + content + '</div>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }
  async function send() {
    const msg = input.value.trim();
    if (!msg) return;
    addMsg('user', msg, 'public');
    input.value = '';
    const load = document.createElement('div');
    load.className = 'loading';
    load.textContent = '🤔 Agent 们正在思考…';
    chat.appendChild(load);
    const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: msg}) });
    const data = await res.json();
    load.remove();
    if (data.responses) data.responses.forEach(r => addMsg(r.sender, r.content, r.kind));
    if (data.responses && data.responses.length === 0) addMsg('system', '(没有 Agent 回应)', 'public');
  }
</script>
</body>
</html>"""
