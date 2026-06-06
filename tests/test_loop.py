"""测试 Agent 核心循环 — loop.py

覆盖：
  - Agent 初始化（默认 memory/planning/tool_calling）
  - Agent.run() 调用 planning.run() 并传递参数
  - memory/planning/tool_calling 可插拔
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from collabroom.core.loop import Agent
from collabroom.core.llm import LLM
from collabroom.core.tool import Registry
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.memory.tiered import TieredMemory
from collabroom.core.planning.react import ReActStrategy
from collabroom.core.tool_calling.batch import BatchToolCalling
from collabroom.core.types import AgentResult, Step, Usage


# ═══════════════════════════════════════════════════════════════
# Agent 初始化
# ═══════════════════════════════════════════════════════════════

class TestAgentInit:
    """测试 Agent.__init__() — 初始化与默认值"""

    def test_basic_init(self, mock_llm, empty_registry):
        """基本初始化：传入 llm 和 registry"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="你是一个测试助手",
        )
        assert agent.llm is mock_llm
        assert agent.registry is empty_registry
        assert agent.system_prompt == "你是一个测试助手"
        assert agent.max_steps == 10  # 默认值

    def test_default_memory_is_naive(self, mock_llm, empty_registry):
        """默认使用 NaiveMemory"""
        agent = Agent(llm=mock_llm, registry=empty_registry, system_prompt="test")
        assert isinstance(agent.memory, NaiveMemory)
        # NaiveMemory 内部已包含 system prompt
        ctx = agent.memory.get_context()
        assert ctx[0]["role"] == "system"
        assert ctx[0]["content"] == "test"

    def test_default_planning_is_react(self, mock_llm, empty_registry):
        """默认使用 ReActStrategy"""
        agent = Agent(llm=mock_llm, registry=empty_registry, system_prompt="test")
        assert isinstance(agent.planning, ReActStrategy)

    def test_default_tool_calling_is_batch(self, mock_llm, empty_registry):
        """默认使用 BatchToolCalling"""
        agent = Agent(llm=mock_llm, registry=empty_registry, system_prompt="test")
        assert isinstance(agent.tool_calling, BatchToolCalling)

    def test_custom_max_steps(self, mock_llm, empty_registry):
        """可以自定义 max_steps"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            max_steps=25,
        )
        assert agent.max_steps == 25


# ═══════════════════════════════════════════════════════════════
# Agent.run() 委托 planning.run()
# ═══════════════════════════════════════════════════════════════

class TestAgentRun:
    """测试 Agent.run() — 委托给 planning.run()"""

    def test_run_delegates_to_planning(self, mock_llm, empty_registry):
        """run() 调用 planning.run() 并传递所有必要参数"""
        # 创建 mock planning，验证调用
        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult(final_answer="done")

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="你是助手",
            max_steps=5,
            planning=mock_planning,
            tool_calling=BatchToolCalling(),
        )
        result = agent.run("请帮我做一件事")

        # 验证 planning.run 被调用
        mock_planning.run.assert_called_once()
        kwargs = mock_planning.run.call_args[1]

        assert kwargs["llm"] is mock_llm
        assert kwargs["registry"] is empty_registry
        assert kwargs["system_prompt"] == "你是助手"
        assert kwargs["user_message"] == "请帮我做一件事"
        assert kwargs["max_steps"] == 5
        assert isinstance(kwargs["tool_calling"], BatchToolCalling)
        # memory 应该是 NaiveMemory（默认）
        assert isinstance(kwargs["memory"], NaiveMemory)

        # 返回 planning 的结果
        assert result.final_answer == "done"

    def test_run_returns_agent_result(self, mock_llm, empty_registry):
        """run() 返回 AgentResult"""
        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult(
            final_answer="完成",
            total_tokens=100,
            total_tool_calls=3,
        )
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            planning=mock_planning,
        )
        result = agent.run("test message")
        assert isinstance(result, AgentResult)
        assert result.final_answer == "完成"
        assert result.total_tokens == 100
        assert result.total_tool_calls == 3

    def test_run_custom_memory_passed_to_planning(self, mock_llm, empty_registry):
        """自定义 memory 会被传给 planning.run()"""
        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult()
        custom_mem = NaiveMemory("custom prompt")

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="agent prompt",
            planning=mock_planning,
        )
        agent.run("hi", memory=custom_mem)

        kwargs = mock_planning.run.call_args[1]
        assert kwargs["memory"] is custom_mem

    def test_run_passes_tool_calling(self, mock_llm, empty_registry):
        """run() 将 tool_calling 策略传递给 planning.run()"""
        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult()
        tc_strategy = BatchToolCalling()

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            planning=mock_planning,
            tool_calling=tc_strategy,
        )
        agent.run("hi")

        kwargs = mock_planning.run.call_args[1]
        assert kwargs["tool_calling"] is tc_strategy


# ═══════════════════════════════════════════════════════════════
# 可插拔策略
# ═══════════════════════════════════════════════════════════════

class TestPluggableMemory:
    """测试 memory 可插拔 — 不同记忆策略"""

    def test_naive_memory(self, mock_llm, empty_registry):
        """使用 NaiveMemory"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            memory=NaiveMemory("你好"),
        )
        assert isinstance(agent.memory, NaiveMemory)
        ctx = agent.memory.get_context()
        assert ctx[0]["content"] == "你好"

    def test_tiered_memory(self, mock_llm, empty_registry):
        """使用 TieredMemory"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            memory=TieredMemory("tiered prompt", working_window=5),
        )
        assert isinstance(agent.memory, TieredMemory)
        ctx = agent.memory.get_context()
        assert ctx[0]["content"] == "tiered prompt"

    def test_custom_memory_mock(self, mock_llm, empty_registry):
        """可以注入任意实现了 Memory 接口的 mock"""
        mock_memory = MagicMock()
        mock_memory.get_context.return_value = [
            {"role": "system", "content": "mock system"},
        ]

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="unused",
            memory=mock_memory,
        )
        assert agent.memory is mock_memory


class TestPluggablePlanning:
    """测试 planning 可插拔 — 不同规划策略"""

    def test_react_strategy(self, mock_llm, empty_registry):
        """使用 ReActStrategy"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            planning=ReActStrategy(),
        )
        assert isinstance(agent.planning, ReActStrategy)

    def test_mock_planning_strategy(self, mock_llm, empty_registry):
        """注入 mock PlanningStrategy"""
        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult(final_answer="mock result")

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            planning=mock_planning,
        )
        result = agent.run("question")
        assert result.final_answer == "mock result"

        # 验证 run 被调用时会传入 llm 和 registry
        kwargs = mock_planning.run.call_args[1]
        assert kwargs["llm"] is mock_llm
        assert kwargs["registry"] is empty_registry
        assert kwargs["user_message"] == "question"


class TestPluggableToolCalling:
    """测试 tool_calling 可插拔 — 不同工具调用策略"""

    def test_batch_tool_calling(self, mock_llm, empty_registry):
        """使用 BatchToolCalling"""
        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            tool_calling=BatchToolCalling(),
        )
        assert isinstance(agent.tool_calling, BatchToolCalling)
        assert agent.tool_calling.supports_parallel is True

    def test_mock_tool_calling_strategy(self, mock_llm, empty_registry):
        """注入 mock ToolCallingStrategy"""
        mock_tc = MagicMock()
        mock_tc.supports_parallel = False
        mock_tc.filter_tools.return_value = []
        mock_tc.limit_calls.return_value = []

        mock_planning = MagicMock()
        mock_planning.run.return_value = AgentResult()

        agent = Agent(
            llm=mock_llm,
            registry=empty_registry,
            system_prompt="test",
            planning=mock_planning,
            tool_calling=mock_tc,
        )
        agent.run("test")

        # 验证 tool_calling 被传递到 planning
        kwargs = mock_planning.run.call_args[1]
        assert kwargs["tool_calling"] is mock_tc


class TestAgentIntegration:
    """测试 Agent 与真实策略的集成（planning 用 mock LLM）"""

    def test_full_flow_with_mock_llm(self, mock_llm_say, empty_registry):
        """集成测试：mock LLM 返回文本 → Agent.run() 完整流程"""
        # mock_llm_say 工厂可以创建返回特定内容的 LLM
        llm = mock_llm_say("直接回答，无需工具")

        agent = Agent(
            llm=llm,
            registry=empty_registry,
            system_prompt="你是测试助手",
            max_steps=5,
        )
        result = agent.run("你好")

        # LLM 返回文本，没有 tool_calls，直接 done
        # 通过 mock_llm_say，chat() 返回 content="直接回答，无需工具"
        assert isinstance(result, AgentResult)
        assert len(result.steps) > 0
        # 应该有 think + done 两步
        roles = [s.role for s in result.steps]
        assert "think" in roles
        assert "done" in roles

    def test_with_tool_calls_single_step(self, mock_llm_with_tool, sample_registry):
        """mock LLM 返回 tool_call → 执行 → 观察"""
        # 第一次返回 tool_call，第二次返回文本
        from collabroom.core.types import LLMResponse, ToolCall, Usage

        llm = MagicMock(spec=LLM)
        llm.chat.side_effect = [
            # 第一次：返回 tool_call echo("hello")
            LLMResponse(
                content="我来调用 echo",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                finish_reason="tool_calls",
            ),
            # 第二次：返回最终文本
            LLMResponse(
                content="echo 返回了 hello，任务完成",
                tool_calls=[],
                usage=Usage(prompt_tokens=20, completion_tokens=10),
                finish_reason="stop",
            ),
        ]

        agent = Agent(
            llm=llm,
            registry=sample_registry,
            system_prompt="你是测试助手",
            max_steps=5,
        )
        result = agent.run("echo 一下")

        # 验证至少有一次 think + act + observe + done
        assert isinstance(result, AgentResult)
        roles = [s.role for s in result.steps]
        assert "act" in roles  # 执行了工具
        assert "observe" in roles  # 有观察结果
        assert "done" in roles
        assert result.final_answer == "echo 返回了 hello，任务完成"
