"""LLM 客户端 — 支持重试与限流"""
from __future__ import annotations
import json, os, time, random, logging, threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .types import LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

# ── 重试策略 ────────────────────────────────────────────
MAX_RETRIES = 3
BASE_DELAY = 1.0        # 首次重试等待 1s
MAX_DELAY = 60.0        # 最大等待
JITTER = 0.1            # ±10% 随机抖动

RETRYABLE_HTTP_CODES = {429, 500, 502, 503}

# ── 限流（令牌桶） ──────────────────────────────────────
RPM_LIMIT = 30          # 每分钟最多 30 次请求


class _TokenBucket:
    """线程安全的令牌桶限流器"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # 每秒补充令牌数
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.refill_rate
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self._last_refill = now

    def acquire(self, block: bool = True) -> float | None:
        """获取一个令牌。返回等待秒数，None 表示立即通过。"""
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return None  # 无需等待
            if not block:
                return None  # 非阻塞模式，直接放行
            # 需要等待下一个令牌
            wait = 1.0 / self.refill_rate
            self.tokens = 0  # 清空
            self._last_refill = now = time.monotonic()
            # 记录本次获取，下次 refill 从 now 开始算
            return wait

    def wait_if_needed(self):
        """阻塞直到获取到令牌"""
        wait = self.acquire(block=True)
        if wait is not None and wait > 0:
            logger.debug("限流等待 %.2fs", wait)
            time.sleep(wait)


# 全局令牌桶（所有 LLM 实例共享）
_global_bucket = _TokenBucket(capacity=RPM_LIMIT, refill_rate=RPM_LIMIT / 60.0)


def _exponential_backoff(attempt: int) -> float:
    """指数退避：1s → 2s → 4s → 8s ... 上限 60s"""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER * (random.random() * 2 - 1)  # ±10%
    return delay + jitter


def _is_retryable(err: Exception) -> bool:
    """判断异常是否可重试"""
    if isinstance(err, HTTPError):
        return err.code in RETRYABLE_HTTP_CODES
    if isinstance(err, URLError):
        # 网络级错误（DNS 失败、连接超时、重置等）可重试
        return True
    return False


def _request_with_retry(url: str, data: bytes, headers: dict) -> dict:
    """带重试的 HTTP POST 请求"""
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            req = Request(url, data=data, headers=headers)
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except (HTTPError, URLError) as e:
            last_exc = e
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = _exponential_backoff(attempt)
                code = getattr(e, "code", "?")
                logger.warning(
                    "LLM 请求失败 (attempt %d/%d, code=%s): %s. "
                    "%.1fs 后重试...",
                    attempt + 1, MAX_RETRIES + 1, code, e, delay,
                )
                time.sleep(delay)
            else:
                # 不可重试或已达最大次数
                break
        except Exception as e:
            # 非网络异常（如 JSON 解析失败）不重试
            last_exc = e
            break

    # 所有重试都失败
    if isinstance(last_exc, HTTPError):
        try:
            detail = last_exc.read().decode()
        except Exception:
            detail = str(last_exc)
        raise RuntimeError(
            f"LLM API 错误 (HTTP {last_exc.code}): {detail}"
        ) from last_exc
    raise RuntimeError(f"LLM 请求失败: {last_exc}") from last_exc


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
    """LLM 客户端 — 带重试与限流"""

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

        logger.info("LLM chat | model=%s | messages=%d | tools=%s",
                     self.model, len(messages), len(tools) if tools else 0)

        t0 = time.monotonic()
        raw = self._send(body)
        elapsed = time.monotonic() - t0

        usage = raw.get("usage", {})
        logger.info("LLM response | model=%s | %.2fs | prompt=%d | completion=%d | finish=%s",
                     self.model, elapsed,
                     usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0),
                     raw.get("choices", [{}])[0].get("finish_reason", ""))

        return self._parse(raw)

    # ── 发送层：带重试 ──────────────────────────────────

    def _send(self, body: dict) -> dict:
        # 限流：获取令牌，等待直到放行
        _global_bucket.wait_if_needed()

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.debug("LLM _send | url=%s | body_size=%d", url, len(data))
        return _request_with_retry(url, data, headers)

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
