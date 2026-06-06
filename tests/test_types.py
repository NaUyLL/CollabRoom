"""测试核心数据结构 — types.py"""
from __future__ import annotations
from collabroom.core.types import ToolCall, Usage, LLMResponse, Step, AgentResult


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/tmp/a.txt"})
        assert tc.id == "call_1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/a.txt"}

    def test_default_arguments(self):
        tc = ToolCall(id="c1", name="echo")
        assert tc.arguments == {}


class TestUsage:
    def test_total(self):
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert u.total == 150

    def test_default_zero(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total == 0


class TestLLMResponse:
    def test_defaults(self):
        resp = LLMResponse()
        assert resp.content is None
        assert resp.tool_calls == []
        assert resp.usage.prompt_tokens == 0
        assert resp.finish_reason == ""

    def test_with_content(self):
        resp = LLMResponse(content="Hello", usage=Usage(prompt_tokens=10, completion_tokens=5))
        assert resp.content == "Hello"
        assert resp.usage.total == 15

    def test_with_tool_calls(self):
        tcs = [ToolCall(id="c1", name="echo", arguments={"text": "hi"})]
        resp = LLMResponse(content=None, tool_calls=tcs)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "echo"


class TestStep:
    def test_basic(self):
        step = Step(role="think", content="我需要查一下资料")
        assert step.role == "think"
        assert step.content == "我需要查一下资料"
        assert step.tool_name == ""
        assert step.tool_args == {}
        assert step.tool_result == ""

    def test_act_step(self):
        step = Step(
            role="act",
            tool_name="read_file",
            tool_args={"path": "/tmp/x.txt"},
        )
        assert step.role == "act"
        assert step.tool_name == "read_file"

    def test_observe_step(self):
        step = Step(role="observe", tool_result='{"ok": true}')
        assert step.tool_result == '{"ok": true}'


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult()
        assert r.final_answer == ""
        assert r.steps == []
        assert r.total_tokens == 0
        assert r.total_tool_calls == 0
        assert r.elapsed_ms == 0

    def test_with_data(self):
        steps = [Step(role="think", content="思考中"), Step(role="done", content="完成")]
        r = AgentResult(
            final_answer="完成",
            steps=steps,
            total_tokens=150,
            total_tool_calls=2,
            elapsed_ms=1234.5,
        )
        assert r.final_answer == "完成"
        assert len(r.steps) == 2
        assert r.total_tokens == 150
        assert r.total_tool_calls == 2
        assert r.elapsed_ms == 1234.5
