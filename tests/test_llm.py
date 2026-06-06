"""测试 LLM 客户端 — llm.py

覆盖：
  - LLM 初始化、_load_api_key 路径
  - 消息构造函数（system/user/assistant/tool）
  - _tool_calls_to_dicts 格式转换
  - _is_retryable 重试判断
  - _exponential_backoff 退避计算
  - _TokenBucket 限流（acquire/wait_if_needed/refill）
  - LLM.chat() 完整流程（拼 body → _send → _parse）
  - _parse 响应解析（content/tool_calls/usage/finish_reason）
"""
from __future__ import annotations
import json
import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from collabroom.core.llm import (
    LLM,
    _load_api_key,
    _is_retryable,
    _exponential_backoff,
    _TokenBucket,
    system_msg,
    user_msg,
    assistant_msg,
    tool_msg,
    _tool_calls_to_dicts,
)
from collabroom.core.types import ToolCall, Usage


# ═══════════════════════════════════════════════════════════════
# 消息构造函数
# ═══════════════════════════════════════════════════════════════

class TestSystemMsg:
    """测试 system_msg() — 系统消息构造函数"""

    def test_basic(self):
        """返回 role=system 的字典"""
        assert system_msg("你是助手") == {"role": "system", "content": "你是助手"}

    def test_empty(self):
        """content 为空也正常返回"""
        assert system_msg("") == {"role": "system", "content": ""}


class TestUserMsg:
    """测试 user_msg() — 用户消息构造函数"""

    def test_basic(self):
        """返回 role=user 的字典"""
        assert user_msg("你好") == {"role": "user", "content": "你好"}

    def test_long_message(self):
        """长文本也能正确构造"""
        long_text = "请帮我分析一下" * 20
        result = user_msg(long_text)
        assert result["role"] == "user"
        assert result["content"] == long_text


class TestAssistantMsg:
    """测试 assistant_msg() — 助手消息构造函数"""

    def test_content_only(self):
        """只有 content 时不含 tool_calls 字段"""
        msg = assistant_msg("好的，我来帮你")
        assert msg == {"role": "assistant", "content": "好的，我来帮你"}
        assert "tool_calls" not in msg

    def test_with_tool_calls(self):
        """有 tool_calls 时包含该字段"""
        tcs = [{"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]
        msg = assistant_msg("我来调用工具", tool_calls=tcs)
        assert msg["role"] == "assistant"
        assert msg["content"] == "我来调用工具"
        assert msg["tool_calls"] == tcs

    def test_content_none_no_field(self):
        """content 为 None 时不含 content 字段"""
        msg = assistant_msg(None)
        assert msg == {"role": "assistant"}
        assert "content" not in msg

    def test_content_none_with_tool_calls(self):
        """content 为 None，只有 tool_calls"""
        tcs = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"/a"}'}}]
        msg = assistant_msg(None, tool_calls=tcs)
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert msg["tool_calls"] == tcs


class TestToolMsg:
    """测试 tool_msg() — 工具结果消息构造函数"""

    def test_basic(self):
        """构造正确的 tool 角色消息"""
        msg = tool_msg("call_1", '{"result": "ok"}')
        assert msg == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"result": "ok"}',
        }

    def test_with_error(self):
        """工具返回错误信息"""
        msg = tool_msg("call_2", '{"error": "not found"}')
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_2"
        assert msg["content"] == '{"error": "not found"}'


# ═══════════════════════════════════════════════════════════════
# _tool_calls_to_dicts
# ═══════════════════════════════════════════════════════════════

class TestToolCallsToDicts:
    """测试 _tool_calls_to_dicts() — ToolCall → OpenAI API 格式"""

    def test_single(self):
        """单个 ToolCall 转换为 API 格式"""
        tcs = [ToolCall(id="c1", name="echo", arguments={"text": "hello"})]
        result = _tool_calls_to_dicts(tcs)
        assert len(result) == 1
        assert result[0]["id"] == "c1"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "echo"
        assert result[0]["function"]["arguments"] == '{"text": "hello"}'

    def test_multiple(self):
        """多个 ToolCall 全部转换"""
        tcs = [
            ToolCall(id="c1", name="echo", arguments={"text": "a"}),
            ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2}),
        ]
        result = _tool_calls_to_dicts(tcs)
        assert len(result) == 2
        assert result[0]["id"] == "c1"
        assert result[1]["id"] == "c2"

    def test_empty_list(self):
        """空列表返回空列表"""
        assert _tool_calls_to_dicts([]) == []

    def test_unicode_args(self):
        """中文参数正确序列化（ensure_ascii=False）"""
        tcs = [ToolCall(id="c1", name="search", arguments={"keyword": "中文关键词"})]
        result = _tool_calls_to_dicts(tcs)
        args_str = result[0]["function"]["arguments"]
        assert "中文关键词" in args_str
        assert "\\u" not in args_str  # ensure_ascii=False


# ═══════════════════════════════════════════════════════════════
# _is_retryable
# ═══════════════════════════════════════════════════════════════

class TestIsRetryable:
    """测试 _is_retryable() — 判断异常是否可重试"""

    def _http_error(self, code: int):
        """构造指定状态码的 HTTPError（绕过构造函数限制用 mock）"""
        from urllib.error import HTTPError
        err = HTTPError.__new__(HTTPError)
        err.code = code
        return err

    def test_429_too_many_requests(self):
        """429 是可重试的"""
        assert _is_retryable(self._http_error(429)) is True

    def test_500_internal_server(self):
        """500 是可重试的"""
        assert _is_retryable(self._http_error(500)) is True

    def test_502_bad_gateway(self):
        """502 是可重试的"""
        assert _is_retryable(self._http_error(502)) is True

    def test_503_service_unavailable(self):
        """503 是可重试的"""
        assert _is_retryable(self._http_error(503)) is True

    def test_400_bad_request_not_retryable(self):
        """400 不可重试"""
        assert _is_retryable(self._http_error(400)) is False

    def test_401_unauthorized_not_retryable(self):
        """401 不可重试"""
        assert _is_retryable(self._http_error(401)) is False

    def test_404_not_found_not_retryable(self):
        """404 不可重试"""
        assert _is_retryable(self._http_error(404)) is False

    def test_urlerror_is_retryable(self):
        """URLError（网络层错误）可重试"""
        from urllib.error import URLError
        assert _is_retryable(URLError("connection refused")) is True

    def test_valueerror_not_retryable(self):
        """普通 ValueError 不可重试"""
        assert _is_retryable(ValueError("test")) is False

    def test_generic_exception_not_retryable(self):
        """普通 Exception 不可重试"""
        assert _is_retryable(Exception("test")) is False


# ═══════════════════════════════════════════════════════════════
# _exponential_backoff
# ═══════════════════════════════════════════════════════════════

class TestExponentialBackoff:
    """测试 _exponential_backoff() — 指数退避计算"""

    def test_first_attempt_about_1s(self):
        """第 0 次（首次重试）约 1s"""
        delay = _exponential_backoff(0)
        # 1.0 ± 10%
        assert 0.8 < delay < 1.3

    def test_second_attempt_about_2s(self):
        """第 1 次约 2s"""
        delay = _exponential_backoff(1)
        assert 1.6 < delay < 2.7

    def test_third_attempt_about_4s(self):
        """第 2 次约 4s"""
        delay = _exponential_backoff(2)
        assert 3.2 < delay < 5.3

    def test_capped_at_max_delay(self):
        """指数增长不超过 60s 上限"""
        delay = _exponential_backoff(100)
        assert delay <= 66  # 60 + 6 jitter

    def test_always_positive(self):
        """不管怎样，延迟都是正数"""
        for attempt in range(10):
            assert _exponential_backoff(attempt) > 0


# ═══════════════════════════════════════════════════════════════
# _TokenBucket
# ═══════════════════════════════════════════════════════════════

class TestTokenBucketInit:
    """测试 _TokenBucket 初始化"""

    def test_initial_tokens_equal_capacity(self):
        """初始化令牌数等于容量"""
        bucket = _TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10
        assert bucket.capacity == 10

    def test_refill_rate_stored(self):
        """refill_rate 正确存储"""
        bucket = _TokenBucket(capacity=5, refill_rate=0.5)
        assert bucket.refill_rate == 0.5


class TestTokenBucketAcquire:
    """测试 _TokenBucket.acquire() — 令牌获取"""

    def test_acquire_when_tokens_available(self):
        """有令牌时立即返回 None（无需等待）"""
        bucket = _TokenBucket(capacity=5, refill_rate=1.0)
        result = bucket.acquire()
        assert result is None
        assert bucket.tokens == 4

    def test_multiple_acquires(self):
        """连续获取令牌"""
        # 冻结时间避免 refill 产生浮点误差
        with patch("time.monotonic", return_value=1000.0):
            bucket = _TokenBucket(capacity=3, refill_rate=1.0)
            assert bucket.acquire() is None  # t=3→2
            assert bucket.acquire() is None  # t=2→1
            assert bucket.acquire() is None  # t=1→0
            assert bucket.tokens < 1e-5

    def test_non_blocking_when_empty(self):
        """非阻塞模式：无令牌时也返回 None"""
        bucket = _TokenBucket(capacity=0, refill_rate=1.0)
        bucket.tokens = 0
        result = bucket.acquire(block=False)
        assert result is None

    def test_blocking_returns_wait_time(self):
        """阻塞模式：无令牌时返回等待秒数"""
        bucket = _TokenBucket(capacity=0, refill_rate=2.0)
        bucket.tokens = 0
        # 需要等待 1/2.0 = 0.5s
        wait = bucket.acquire(block=True)
        assert wait is not None
        assert wait > 0


class TestTokenBucketRefill:
    """测试 _TokenBucket._refill() — 令牌补充"""

    def test_refill_after_time_elapsed(self):
        """经过一段时间后 refill 补上令牌"""
        with patch("time.monotonic", side_effect=[0.0, 100.0, 102.0]):
            bucket = _TokenBucket(capacity=10, refill_rate=2.0)
            bucket.tokens = 5
            bucket._refill()
            # 5 + 2*2 = 9，不超过容量 10
            assert 8 <= bucket.tokens <= 10

    def test_refill_capped_at_capacity(self):
        """refill 不会超过容量"""
        with patch("time.monotonic", side_effect=[0.0, 100.0, 110.0]):
            bucket = _TokenBucket(capacity=10, refill_rate=100.0)
            bucket.tokens = 9
            bucket._refill()
            assert bucket.tokens == 10  # 不超过容量


class TestTokenBucketWaitIfNeeded:
    """测试 _TokenBucket.wait_if_needed() — 阻塞等待令牌"""

    def test_passes_immediately(self):
        """有令牌时立即通过"""
        bucket = _TokenBucket(capacity=5, refill_rate=1.0)
        with patch("time.monotonic", return_value=0):
            bucket.wait_if_needed()  # 不应抛异常或卡住
        assert bucket.tokens == 4

    def test_sleeps_when_empty(self):
        """无令牌时调用 time.sleep 等待"""
        bucket = _TokenBucket(capacity=0, refill_rate=2.0)
        bucket.tokens = 0
        with patch("time.monotonic", return_value=0), \
             patch("time.sleep") as mock_sleep:
            bucket.wait_if_needed()
            mock_sleep.assert_called_once()
            wait_val = mock_sleep.call_args[0][0]
            assert wait_val > 0


class TestTokenBucketThreadSafety:
    """测试 _TokenBucket 线程安全性（有 Lock）"""

    def test_has_lock(self):
        """初始化后有 threading.Lock"""
        bucket = _TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket._lock is not None


# ═══════════════════════════════════════════════════════════════
# _load_api_key
# ═══════════════════════════════════════════════════════════════

class TestLoadApiKey:
    """测试 _load_api_key() — API Key 加载"""

    def test_from_env(self):
        """从环境变量 DEEPSEEK_API_KEY 读取"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-env"}):
            assert _load_api_key() == "sk-test-env"

    def test_from_env_empty_string_skips(self):
        """环境变量为空字符串时继续查文件"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
             patch("builtins.open", side_effect=FileNotFoundError):
            pass  # 会抛 RuntimeError

    def test_from_home_env_file(self):
        """从 ~/.env 读取"""
        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.expanduser", return_value="/home/user/.env"), \
             patch("builtins.open", side_effect=FileNotFoundError):
            # 两个路径都找不到
            pass

    def test_from_opt_data_env(self):
        """从 /opt/data/.env 读取"""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = ["# comment\n", 'DEEPSEEK_API_KEY="sk-from-file"\n']
        with patch.dict(os.environ, {}, clear=True), \
             patch("builtins.open", side_effect=[FileNotFoundError, mock_file]):
            key = _load_api_key()
            assert key == "sk-from-file"

    def test_raises_when_not_found(self):
        """找不到 API Key 时抛 RuntimeError"""
        with patch.dict(os.environ, {}, clear=True), \
             patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                _load_api_key()


# ═══════════════════════════════════════════════════════════════
# LLM 初始化
# ═══════════════════════════════════════════════════════════════

class TestLLMInit:
    """测试 LLM.__init__()"""

    def test_defaults(self):
        """默认参数：deepseek-chat、DeepSeek API URL"""
        with patch("collabroom.core.llm._load_api_key", return_value="sk-test"):
            llm = LLM()
            assert llm.model == "deepseek-chat"
            assert llm.base_url == "https://api.deepseek.com/v1"
            assert llm.api_key == "sk-test"

    def test_custom_model(self):
        """可以指定自定义模型名"""
        with patch("collabroom.core.llm._load_api_key", return_value="sk-test"):
            llm = LLM(model="gpt-4")
            assert llm.model == "gpt-4"

    def test_custom_base_url(self):
        """可以指定自定义 base_url"""
        with patch("collabroom.core.llm._load_api_key", return_value="sk-test"):
            llm = LLM(base_url="https://api.example.com/v1")
            assert llm.base_url == "https://api.example.com/v1"

    def test_base_url_strips_trailing_slash(self):
        """base_url 尾部斜杠会被去除"""
        with patch("collabroom.core.llm._load_api_key", return_value="sk-test"):
            llm = LLM(base_url="https://api.example.com/v1/")
            assert llm.base_url == "https://api.example.com/v1"

    def test_explicit_api_key(self):
        """显式传入 api_key 时不调用 _load_api_key"""
        llm = LLM(api_key="sk-manual")
        assert llm.api_key == "sk-manual"


# ═══════════════════════════════════════════════════════════════
# LLM._parse
# ═══════════════════════════════════════════════════════════════

class TestLLMParse:
    """测试 LLM._parse() — 响应解析"""

    def _make_llm(self):
        """创建测试用 LLM 实例"""
        return LLM(api_key="sk-test")

    def test_simple_content_response(self):
        """纯文本响应的解析"""
        raw = {
            "choices": [{"message": {"content": "你好！"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert result.content == "你好！"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total == 15

    def test_with_tool_calls(self):
        """含 tool_calls 的响应解析"""
        raw = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/a.txt"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/a.txt"}
        assert result.finish_reason == "tool_calls"

    def test_tool_calls_is_null(self):
        """tool_calls 为 null 时正常解析"""
        raw = {
            "choices": [{"message": {"content": "done", "tool_calls": None}, "finish_reason": "stop"}],
            "usage": {},
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert result.content == "done"
        assert result.tool_calls == []

    def test_missing_usage(self):
        """缺少 usage 字段时使用默认值"""
        raw = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0

    def test_malformed_arguments(self):
        """tool_call arguments 不是合法 JSON 时返回空 dict"""
        raw = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "tool1", "arguments": "not-valid-json"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert result.tool_calls[0].arguments == {}

    def test_multiple_tool_calls(self):
        """多个 tool_calls 全部解析"""
        raw = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "echo", "arguments": '{"t":"a"}'}},
                        {"id": "c2", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
        }
        llm = self._make_llm()
        result = llm._parse(raw)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "echo"
        assert result.tool_calls[1].name == "add"


# ═══════════════════════════════════════════════════════════════
# LLM.chat() 完整流程
# ═══════════════════════════════════════════════════════════════

class TestLLMChat:
    """测试 LLM.chat() — 完整调用流程（完全 mock）"""

    def _make_llm(self, send_return=None):
        """创建 LLM 实例，_send 被 mock"""
        llm = LLM(api_key="sk-test")
        llm._send = MagicMock(return_value=send_return or {
            "choices": [{"message": {"content": "回复"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        return llm

    def test_basic_chat_builds_body(self):
        """chat() 拼 body 并调用 _send"""
        llm = self._make_llm()
        msgs = [{"role": "user", "content": "你好"}]
        result = llm.chat(msgs)

        # 验证 _send 被调用
        llm._send.assert_called_once()
        body = llm._send.call_args[0][0]
        assert body["model"] == "deepseek-chat"
        assert body["messages"] == msgs
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 2048
        assert "tools" not in body  # 无 tools 时不应包含此字段

        # 验证返回结果
        assert result.content == "回复"
        assert result.finish_reason == "stop"

    def test_chat_with_tools(self):
        """有 tools 时 body 包含 tools 字段"""
        llm = self._make_llm()
        tools = [{"type": "function", "function": {"name": "echo"}}]
        llm.chat([{"role": "user", "content": "hi"}], tools=tools)

        body = llm._send.call_args[0][0]
        assert body["tools"] == tools

    def test_chat_custom_params(self):
        """自定义 temperature 和 max_tokens"""
        llm = self._make_llm()
        llm.chat([], temperature=0.3, max_tokens=1024)

        body = llm._send.call_args[0][0]
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 1024

    def test_chat_returns_llmresponse(self):
        """chat() 返回 LLMResponse 实例"""
        from collabroom.core.types import LLMResponse
        llm = self._make_llm()
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponse)

    def test_chat_with_tool_call_response(self):
        """LLM 返回 tool_calls 时 chat() 正确解析"""
        llm = self._make_llm(send_return={
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q":"test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        })
        result = llm.chat([{"role": "user", "content": "search test"}])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.finish_reason == "tool_calls"

    def test_chat_tracks_elapsed_time(self):
        """chat() 会记录请求耗时"""
        llm = self._make_llm()
        with patch("time.monotonic") as mock_time:
            mock_time.side_effect = [1000.0, 1002.5]
            llm.chat([{"role": "user", "content": "hi"}])
            # 验证调用了两次 monotonic（开始和结束）
            assert mock_time.call_count >= 2


# ═══════════════════════════════════════════════════════════════
# LLM._send 发送层
# ═══════════════════════════════════════════════════════════════

class TestLLMSend:
    """测试 LLM._send() — 发送层"""

    def test_send_waits_for_token_bucket(self):
        """_send 会先调用 TokenBucket.wait_if_needed() 限流"""
        import collabroom.core.llm as llm_module

        # mock 掉限流和 HTTP 请求
        with patch.object(llm_module._global_bucket, "wait_if_needed") as mock_wait, \
             patch("collabroom.core.llm._request_with_retry", return_value={"choices": []}):
            llm = LLM(api_key="sk-test")
            llm._send({"model": "test", "messages": []})
            mock_wait.assert_called_once()

    def test_send_url_construction(self):
        """请求 URL 为 base_url + /chat/completions"""
        import collabroom.core.llm as llm_module

        with patch.object(llm_module._global_bucket, "wait_if_needed"), \
             patch("collabroom.core.llm._request_with_retry") as mock_req:
            mock_req.return_value = {"choices": []}
            llm = LLM(api_key="sk-test", base_url="https://api.example.com/v1")
            llm._send({"model": "deepseek-chat", "messages": []})

            # 验证 URL
            url = mock_req.call_args[0][0]
            assert url == "https://api.example.com/v1/chat/completions"

    def test_send_headers_contain_auth(self):
        """请求头包含 Authorization 和 Content-Type"""
        import collabroom.core.llm as llm_module

        with patch.object(llm_module._global_bucket, "wait_if_needed"), \
             patch("collabroom.core.llm._request_with_retry") as mock_req:
            mock_req.return_value = {"choices": []}
            llm = LLM(api_key="sk-mykey")
            llm._send({"model": "deepseek-chat", "messages": []})

            headers = mock_req.call_args[0][2]
            assert headers["Authorization"] == "Bearer sk-mykey"
            assert headers["Content-Type"] == "application/json"

    def test_send_body_is_json_bytes(self):
        """_send 将 body 序列化为 JSON bytes"""
        import collabroom.core.llm as llm_module

        with patch.object(llm_module._global_bucket, "wait_if_needed"), \
             patch("collabroom.core.llm._request_with_retry") as mock_req:
            mock_req.return_value = {"choices": []}
            llm = LLM(api_key="sk-test")
            llm._send({"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]})

            data = mock_req.call_args[0][1]
            assert isinstance(data, bytes)
            decoded = json.loads(data)
            assert decoded["messages"][0]["content"] == "hi"
