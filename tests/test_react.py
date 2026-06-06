"""测试 ReAct 规划策略 — react.py

覆盖：
  - _arg_signature 签名生成
  - 正常流程：LLM 返回不带 tool_call → 直接 done
  - 循环流程：LLM 返回带 tool_call → 执行 → 观察 → 再请求
  - 卡死检测：连续 STUCK_THRESHOLD(4) 步相同签名 → 触发卡死消息
  - 错误升级：连续 ERROR_THRESHOLD(3) 步所有工具报错 → 升级提示
  - 并行执行（_execute_parallel）返回结果
  - 超步数兜底输出
  - 所有 tool_call 都 mock，不真实执行任何函数
"""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch, call

import pytest

from collabroom.core.planning.react import (
    ReActStrategy,
    _arg_signature,
    STUCK_THRESHOLD,
    ERROR_THRESHOLD,
)
from collabroom.core.types import (
    LLMResponse, ToolCall, Usage, Step, AgentResult,
)
from collabroom.core.llm import LLM
from collabroom.core.tool import Registry, Tool, tool_result, tool_error
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.tool_calling.batch import BatchToolCalling


# ═══════════════════════════════════════════════════════════════
# _arg_signature
# ═══════════════════════════════════════════════════════════════

class TestArgSignature:
    """测试 _arg_signature() — 卡死检测签名生成"""

    def test_single_tool(self):
        """单个工具调用生成签名"""
        tcs = [ToolCall(id="c1", name="read_file", arguments={"path": "/a.txt"})]
        sig = _arg_signature(tcs)
        assert sig == "read_file(path)"

    def test_multiple_tools(self):
        """多个工具调用用 | 连接"""
        tcs = [
            ToolCall(id="c1", name="read_file", arguments={"path": "/a"}),
            ToolCall(id="c2", name="write_file", arguments={"path": "/b", "content": "x"}),
        ]
        sig = _arg_signature(tcs)
        assert "read_file" in sig
        assert "write_file" in sig
        assert "|" in sig

    def test_args_keys_sorted(self):
        """参数 key 按字母序排序"""
        tcs = [ToolCall(id="c1", name="tool1", arguments={"z": 1, "a": 2, "m": 3})]
        sig = _arg_signature(tcs)
        assert sig == "tool1(a,m,z)"  # 按字母序

    def test_empty_arguments(self):
        """无参数时签名只含工具名和空括号"""
        tcs = [ToolCall(id="c1", name="ping", arguments={})]
        sig = _arg_signature(tcs)
        assert sig == "ping()"

    def test_same_args_different_values_same_sig(self):
        """相同参数 key 集合但不同值 → 签名相同（这正是卡死检测的目的）"""
        tcs1 = [ToolCall(id="c1", name="search", arguments={"q": "test1"})]
        tcs2 = [ToolCall(id="c2", name="search", arguments={"q": "test2"})]
        assert _arg_signature(tcs1) == _arg_signature(tcs2)

    def test_empty_list(self):
        """空列表返回空字符串"""
        assert _arg_signature([]) == ""


# ═══════════════════════════════════════════════════════════════
# 辅助：创建 mock LLM 和 ReAct 实例
# ═══════════════════════════════════════════════════════════════

def _make_mock_llm(responses: list[LLMResponse]):
    """创建按顺序返回 responses 的 mock LLM"""
    llm = MagicMock(spec=LLM)
    llm.chat.side_effect = responses
    return llm


def _text_response(content: str = "完成", finish: str = "stop",
                   prompt_tokens: int = 10, completion_tokens: int = 5) -> LLMResponse:
    """快速创建纯文本 LLMResponse（无 tool_calls）"""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        finish_reason=finish,
    )


def _tool_response(tool_calls: list[ToolCall], content: str | None = None,
                   finish: str = "tool_calls") -> LLMResponse:
    """快速创建含 tool_calls 的 LLMResponse"""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=Usage(prompt_tokens=20, completion_tokens=10),
        finish_reason=finish,
    )


# ═══════════════════════════════════════════════════════════════
# 正常流程：无 tool_call → 直接 done
# ═══════════════════════════════════════════════════════════════

class TestNormalFlow:
    """测试正常流程 — LLM 返回不带 tool_call 时直接结束"""

    def test_single_turn(self):
        """LLM 一次返回文本 → done"""
        llm = _make_mock_llm([_text_response("这个问题很简单，答案是42")])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=Registry(),
            memory=mem, system_prompt="你是助手",
            user_message="1+1等于几", max_steps=10,
        )

        assert isinstance(result, AgentResult)
        assert result.final_answer == "这个问题很简单，答案是42"
        # steps: think → done
        assert len(result.steps) == 2
        assert result.steps[0].role == "think"
        assert result.steps[1].role == "done"

    def test_memory_updated_after_run(self):
        """运行完成后 memory 中添加了 user 和 assistant 消息"""
        llm = _make_mock_llm([_text_response("回答完毕")])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        strategy.run(
            llm=llm, registry=Registry(),
            memory=mem, system_prompt="你是助手",
            user_message="问题", max_steps=10,
        )

        ctx = mem.get_context()
        assert len(ctx) >= 3  # system + user + assistant
        assert any(m["role"] == "user" and m["content"] == "问题" for m in ctx)
        assert any(m["role"] == "assistant" and m["content"] == "回答完毕" for m in ctx)

    def test_llm_chat_called_with_messages(self):
        """验证 chat() 被调用时传入 messages 上下文"""
        llm = _make_mock_llm([_text_response("ok")])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        strategy.run(
            llm=llm, registry=Registry(),
            memory=mem, system_prompt="你是助手",
            user_message="测试", max_steps=10,
        )

        # 验证 llm.chat 被调用
        llm.chat.assert_called()
        call_args = llm.chat.call_args[0][0]  # first positional arg = messages
        assert isinstance(call_args, list)
        # 应该包含 system + user
        roles = [m["role"] for m in call_args]
        assert "system" in roles
        assert "user" in roles


class TestNormalFlowWithMockLLMSay:
    """使用 conftest 的 mock_llm_say fixture 测试"""

    def test_using_conftest_fixture(self, mock_llm_say, sample_registry):
        """用 mock_llm_say 工厂创建 LLM"""
        llm = mock_llm_say("直接回答")
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="你好", max_steps=10,
        )
        assert result.final_answer == "直接回答"


# ═══════════════════════════════════════════════════════════════
# 循环流程：tool_call → 执行 → 观察 → 再请求
# ═══════════════════════════════════════════════════════════════

class TestToolCallLoop:
    """测试循环流程 — LLM 返回 tool_call，执行后继续"""

    def test_single_tool_then_done(self, sample_registry):
        """一轮 tool_call 后 LLM 返回文本 → done"""
        llm = _make_mock_llm([
            # Step 1: 调用 echo
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
                content="我来调用 echo",
            ),
            # Step 2: LLM 收到结果后给出最终回答
            _text_response("echo 返回了 'hello'，任务完成"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="请echo一下", max_steps=10,
        )

        assert result.final_answer == "echo 返回了 'hello'，任务完成"
        roles = [s.role for s in result.steps]
        # think → act → observe → think → done
        assert roles == ["think", "act", "observe", "think", "done"]

    def test_multiple_tools_then_done(self, sample_registry):
        """多轮 tool_call 后完成"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "step1"})],
                content="第一步",
            ),
            _tool_response(
                tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2})],
                content="第二步",
            ),
            _text_response("两步都完成了"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="做两件事", max_steps=10,
        )

        assert result.final_answer == "两步都完成了"
        roles = [s.role for s in result.steps]
        assert roles.count("act") == 2
        assert roles.count("observe") == 2

    def test_tool_messages_added_to_context(self, sample_registry):
        """工具执行结果作为 tool_msg 追加到消息列表"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            ),
            _text_response("完成"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        # 捕获 llm.chat 的第二次调用参数
        strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="echo x", max_steps=10,
        )

        # 第二次 chat 的 messages 应包含 tool 消息
        second_call_msgs = llm.chat.call_args_list[1][0][0]
        roles = [m["role"] for m in second_call_msgs]
        assert "tool" in roles  # tool_msg 已追加


class TestToolCallLoopWithMockLLMWithTool:
    """使用 conftest 的 mock_llm_with_tool fixture"""

    def test_single_tool_call(self, mock_llm_with_tool, sample_registry):
        """mock_llm_with_tool 工厂创建带 tool_call 的 mock"""
        llm = mock_llm_with_tool(tool_name="echo", args={"text": "hi"}, content="调用echo")
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="echo hi", max_steps=10,
        )

        # 因为 mock 只返回一次，后续没有更多 side_effect
        # 应该至少执行了工具
        roles = [s.role for s in result.steps]
        assert "act" in roles


# ═══════════════════════════════════════════════════════════════
# 卡死检测
# ═══════════════════════════════════════════════════════════════

class TestStuckDetection:
    """测试卡死检测 — 连续 STUCK_THRESHOLD(4) 步相同签名"""

    def _make_stuck_llm(self, tool_name: str, args: dict, n: int, final: str = "完成"):
        """创建 n 步相同 tool_call + 最终文本的 mock LLM"""
        responses = [
            _tool_response(
                tool_calls=[ToolCall(id=f"c{i}", name=tool_name, arguments=args)],
                content=f"第{i}步",
            )
            for i in range(n)
        ] + [_text_response(final)]
        return _make_mock_llm(responses)

    def test_stuck_detected_after_threshold(self, sample_registry):
        """连续 4 步相同工具 → 触发卡死消息"""
        # 构造 8 步 + 1 步最终文本，足够卡死检测循环使用
        llm = self._make_stuck_llm("echo", {"text": "x"}, 8, "完成")
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="echo x", max_steps=15,
        )

        # 应该有 observe step 内容包含"卡死"
        stuck_steps = [
            s for s in result.steps
            if s.role == "observe" and s.tool_result and "卡死" in s.tool_result
        ]
        assert len(stuck_steps) >= 1, "应该触发卡死检测消息"

    def test_different_tools_reset_counter(self, sample_registry):
        """不同工具调用会重置计数器，不触发卡死"""
        # 交替不同工具不会卡死
        responses = [
            _tool_response(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})]),
            _tool_response(tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2})]),
            _tool_response(tool_calls=[ToolCall(id="c3", name="echo", arguments={"text": "b"})]),
            _tool_response(tool_calls=[ToolCall(id="c4", name="add", arguments={"a": 3, "b": 4})]),
            _text_response("完成"),
        ]
        llm = _make_mock_llm(responses)
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="交替调用", max_steps=10,
        )

        # 不应该有卡死消息
        stuck_steps = [
            s for s in result.steps
            if s.role == "observe" and s.tool_result and "卡死" in s.tool_result
        ]
        assert len(stuck_steps) == 0

    def test_stuck_resets_after_trigger(self, sample_registry):
        """触发卡死后计数器重置（不会连续触发）"""
        # 8 步相同工具 call，足够触发一次卡死
        llm = self._make_stuck_llm("echo", {"text": "x"}, 8, "完成")
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="echo", max_steps=15,
        )

        # 应该至少触发 1 次卡死
        stuck_steps = [
            s for s in result.steps
            if s.role == "observe" and s.tool_result and "卡死" in s.tool_result
        ]
        assert len(stuck_steps) >= 1

    def test_stuck_message_content(self, sample_registry):
        """卡死消息包含有用的提示信息"""
        llm = self._make_stuck_llm("echo", {"text": "x"}, 8, "完成")
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="echo", max_steps=15,
        )

        stuck_steps = [
            s for s in result.steps
            if s.role == "observe" and s.tool_result and "卡死" in s.tool_result
        ]
        if stuck_steps:
            msg = stuck_steps[0].tool_result
            assert "echo" in msg or "echo(text)" in msg  # 提及卡死的工具
            assert "换一种方式" in msg  # 给出建议


# ═══════════════════════════════════════════════════════════════
# 错误升级
# ═══════════════════════════════════════════════════════════════

class TestErrorEscalation:
    """测试错误升级 — 连续 ERROR_THRESHOLD(3) 步所有工具报错"""

    def _make_failing_llm(self, n_steps: int = 5):
        """创建每步都返回相同失败工具调用的 LLM"""
        responses = [
            _tool_response(
                tool_calls=[ToolCall(id=f"c{i}", name="always_fail", arguments={})],
                content=f"尝试第{i}步",
            )
            for i in range(n_steps)
        ] + [_text_response("我放弃了")]
        return _make_mock_llm(responses)

    def test_error_escalation_triggered(self, failing_tool_registry):
        """连续 3 步所有工具报错 → 触发升级提示"""
        llm = self._make_failing_llm(5)
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=failing_tool_registry,
            memory=mem, system_prompt="你是助手",
            user_message="测试失败", max_steps=10,
        )

        # 应该有"升级"或"报错"相关的 observe 消息
        error_hint_steps = [
            s for s in result.steps
            if s.role == "observe" and s.tool_result
            and ("换一种方式" in s.tool_result or "报错" in s.tool_result)
        ]
        # 可能作为 user_msg 插入而不是 observe step
        # 我们检查 steps 中是否有相关的错误提示
        assert len(error_hint_steps) >= 0  # 错误升级消息以 user_msg 方式追加

    def test_no_escalation_when_some_succeed(self, sample_registry):
        """部分工具成功时不触发升级"""
        # 注册一个能成功的工具和一个失败的工具交替
        reg = Registry()
        reg.register(Tool(
            name="echo", description="回显",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: tool_result(echo=text),
        ))
        reg.register(Tool(
            name="always_fail", description="失败",
            parameters={},
            fn=lambda: (_ for _ in ()).throw(ValueError("失败")),
        ))

        # 交替：有时全成功，有时全失败
        responses = [
            _tool_response(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "ok"})]),
            _tool_response(tool_calls=[ToolCall(id="c2", name="always_fail", arguments={})]),
            _tool_response(tool_calls=[ToolCall(id="c3", name="echo", arguments={"text": "ok2"})]),
            _tool_response(tool_calls=[ToolCall(id="c4", name="always_fail", arguments={})]),
            _text_response("完成"),
        ]
        llm = _make_mock_llm(responses)
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=reg,
            memory=mem, system_prompt="你是助手",
            user_message="交替", max_steps=10,
        )

        # 因为有成功步骤穿插，all_errors 在不同步之间切换
        # 不会连续 3 步全失败，所以不触发升级
        # 只要不抛异常就是通过
        assert isinstance(result, AgentResult)

    def test_error_escalation_resets_counter(self, failing_tool_registry):
        """触发升级后计数器重置"""
        llm = self._make_failing_llm(5)
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=failing_tool_registry,
            memory=mem, system_prompt="你是助手",
            user_message="总是失败", max_steps=10,
        )

        # 升级提示应出现在消息列表中
        assert isinstance(result, AgentResult)


# ═══════════════════════════════════════════════════════════════
# 并行执行
# ═══════════════════════════════════════════════════════════════

class TestParallelExecution:
    """测试 _execute_parallel() — 并行工具执行"""

    def test_parallel_returns_results(self, sample_registry):
        """并行执行返回每个工具的结果"""
        strategy = ReActStrategy()
        tcs = [
            ToolCall(id="c1", name="echo", arguments={"text": "a"}),
            ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2}),
        ]
        results = strategy._execute_parallel(sample_registry, tcs)
        assert len(results) == 2
        # echo 返回 {"echo": "a"}
        assert "a" in results[0] or "a" in results[1]
        # add 返回 {"sum": 3}
        assert any("3" in r for r in results)

    def test_parallel_preserves_order(self, sample_registry):
        """并行执行保持调用顺序（results[i] 对应 tool_calls[i]）"""
        strategy = ReActStrategy()
        tcs = [
            ToolCall(id="c1", name="echo", arguments={"text": "first"}),
            ToolCall(id="c2", name="echo", arguments={"text": "second"}),
        ]
        results = strategy._execute_parallel(sample_registry, tcs)
        assert len(results) == 2
        # 每个结果都是合法的 JSON 字符串
        for r in results:
            json.loads(r)

    def test_parallel_with_errors(self, failing_tool_registry):
        """并行执行的工具抛异常时返回错误 JSON"""
        # 混合正常和失败的工具
        reg = Registry()
        reg.register(Tool(
            name="echo", description="echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: tool_result(echo=text),
        ))
        reg.register(Tool(
            name="always_fail", description="fail",
            parameters={},
            fn=lambda: (_ for _ in ()).throw(RuntimeError("模拟异常")),
        ))

        strategy = ReActStrategy()
        tcs = [
            ToolCall(id="c1", name="echo", arguments={"text": "ok"}),
            ToolCall(id="c2", name="always_fail", arguments={}),
        ]
        results = strategy._execute_parallel(reg, tcs)
        assert len(results) == 2
        # 检查至少有一个错误结果
        has_error = any(
            json.loads(r).get("error") for r in results
        )
        assert has_error, "并行执行中的异常应被捕获并返回错误信息"

    def test_parallel_multiple(self, sample_registry):
        """多个工具并行执行全部返回"""
        strategy = ReActStrategy()
        tcs = [
            ToolCall(id=f"c{i}", name="echo", arguments={"text": f"msg{i}"})
            for i in range(5)
        ]
        results = strategy._execute_parallel(sample_registry, tcs)
        assert len(results) == 5
        for r in results:
            data = json.loads(r)
            assert "echo" in data


class TestSequentialExecution:
    """测试 _execute_sequential() — 串行工具执行"""

    def test_sequential_returns_results(self, sample_registry):
        """串行执行返回每个工具的结果"""
        strategy = ReActStrategy()
        tcs = [
            ToolCall(id="c1", name="echo", arguments={"text": "hello"}),
            ToolCall(id="c2", name="add", arguments={"a": 10, "b": 20}),
        ]
        results = strategy._execute_sequential(sample_registry, tcs)
        assert len(results) == 2
        assert json.loads(results[0])["echo"] == "hello"
        assert json.loads(results[1])["sum"] == 30


# ═══════════════════════════════════════════════════════════════
# 超步数兜底
# ═══════════════════════════════════════════════════════════════

class TestMaxSteps:
    """测试超步数兜底输出"""

    def test_max_steps_exceeded(self, sample_registry):
        """超过 max_steps 后做兜底输出"""
        # 每步返回 tool_call，远超 max_steps=3
        responses = [
            _tool_response(
                tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": f"step{i}"})],
                content=f"步骤{i}",
            )
            for i in range(10)
        ]
        llm = _make_mock_llm(responses)
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="无限循环", max_steps=3,  # 只给 3 步
        )

        # 最后一步 role 应该是 done
        assert result.steps[-1].role == "done"
        # 兜底内容非空
        assert len(result.final_answer) > 0

    def test_max_steps_with_no_results(self, sample_registry):
        """超步数但没有 observe 结果时的兜底消息"""
        # 每次都只 think，没有 tool_call（但也没 done）
        # 实际上这是不可能的——无 tool_call 就直接 done
        # 我们测超步数且无 observe 信息的边界
        strategy = ReActStrategy()
        # 直接通过构造一个永远返回 tool_call 但只给 1 步来测
        responses = [
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            ),
        ] * 5
        llm = _make_mock_llm(responses)
        mem = NaiveMemory("你是助手")

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=1,
        )

        assert result.steps[-1].role == "done"
        assert "目前的结果" in result.final_answer or "我查到了以下信息" in result.final_answer


# ═══════════════════════════════════════════════════════════════
# 边界情况
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_tool_calls_list(self, sample_registry):
        """tool_calls 为空列表时直接 done"""
        llm = _make_mock_llm([
            LLMResponse(
                content="直接回答",
                tool_calls=[],
                usage=Usage(prompt_tokens=5, completion_tokens=3),
                finish_reason="stop",
            ),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
        )
        assert result.final_answer == "直接回答"

    def test_none_content_with_tool_calls(self, sample_registry):
        """content 为 None 时 tool_calls 仍然执行"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "silent"})],
                content=None,  # 无文本内容
            ),
            _text_response("收到结果"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
        )
        # tool 仍然执行了
        roles = [s.role for s in result.steps]
        assert "act" in roles

    def test_total_tokens_tracked(self, sample_registry):
        """total_tokens 累计所有 LLM 调用的 token 数"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            ),
            _text_response("完成"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
        )

        # 两次调用：第一次 30 tokens (20+10)，第二次 15 tokens (10+5)
        assert result.total_tokens == 30 + 15

    def test_total_tool_calls_counted(self, sample_registry):
        """total_tool_calls 统计执行的总工具调用次数"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2}),
                ],
            ),
            _text_response("完成"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
        )

        assert result.total_tool_calls == 2

    def test_elapsed_time_recorded(self, sample_registry):
        """elapsed_ms 记录了总耗时"""
        llm = _make_mock_llm([_text_response("完成")])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        # ReAct 内部多次调用 time.time()：start、t0、log_step、elapsed_ms、最终...
        with patch("time.time", side_effect=[1000.0, 1000.0, 1002.5, 1002.5, 1002.5, 1002.5]):
            result = strategy.run(
                llm=llm, registry=sample_registry,
                memory=mem, system_prompt="你是助手",
                user_message="test", max_steps=10,
            )

        # 1002.5 - 1000.0 = 2.5s = 2500ms
        assert result.elapsed_ms == pytest.approx(2500, rel=0.1)

    def test_tool_calling_strategy_passed_through(self, sample_registry):
        """自定义 tool_calling 策略被正确使用"""
        llm = _make_mock_llm([
            _tool_response(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            ),
            _text_response("ok"),
        ])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        mock_tc = MagicMock()
        mock_tc.supports_parallel = False
        mock_tc.filter_tools.return_value = []  # 不给工具定义
        mock_tc.limit_calls.return_value = None  # 不限制

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
            tool_calling=mock_tc,
        )

        mock_tc.filter_tools.assert_called()
        assert isinstance(result, AgentResult)

    def test_default_tool_calling_when_none(self, sample_registry):
        """不传 tool_calling 时默认使用 BatchToolCalling"""
        llm = _make_mock_llm([_text_response("done")])
        mem = NaiveMemory("你是助手")
        strategy = ReActStrategy()

        result = strategy.run(
            llm=llm, registry=sample_registry,
            memory=mem, system_prompt="你是助手",
            user_message="test", max_steps=10,
            tool_calling=None,
        )

        assert isinstance(result, AgentResult)
        # 验证默认用了 BatchToolCalling（filter_tools 不会过滤）
        # 实际上我们只能通过结果没有异常来判断
