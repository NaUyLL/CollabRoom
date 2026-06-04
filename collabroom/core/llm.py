"""LLM 客户端 — 一条 POST 请求，不加重试/限流/流式"""
from __future__ import annotations
import json, os
from urllib.request import Request, urlopen
from urllib.error import URLError

from .types import LLMResponse, ToolCall, Usage

def _load_api_key() -> str:
    """从环境变量或 .env 读取 DEEPSEEK_API_KEY"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    for p in [os.path.expanduser("~/.env"), "/opt/data/.env"]:
        try:
            with open(p) as f:
                for line in f:
                    if "DEEPSEEK" in line and "=" in line:
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            return key
        except FileNotFoundError:
            continue
    raise RuntimeError("DEEPSEEK_API_KEY 未找到，设环境变量或写 .env")

class LLM:
    """最简 LLM 客户端 — 未来重试/限流加在 _send 层"""

    def __init__(self, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1",
                 api_key: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or _load_api_key()

    def chat(self, messages: list[dict],
             tools: list[dict] | None = None,
             temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        """调 LLM，返回 text + tool_calls"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools

        raw = self._send(body)
        return self._parse(raw)

    # ── 发送层：以后重试/限流就改这里 ──────────────────────

    def _send(self, body: dict) -> dict:
        req = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    # ── 解析层 ──────────────────────────────────────────

    def _parse(self, raw: dict) -> LLMResponse:
        choice = raw.get("choices", [{}])[0]
        msg = choice.get("message", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        usage_raw = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
        )

        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
        )

# ── 消息构造辅助 ─────────────────────────────────────

def system_msg(content: str) -> dict:
    return {"role": "system", "content": content}

def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}

def assistant_msg(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant"}
    if content:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg

def tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

def _tool_calls_to_dicts(tcs: list[ToolCall]) -> list[dict]:
    """ToolCall → OpenAI API 格式"""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }
        for tc in tcs
    ]
