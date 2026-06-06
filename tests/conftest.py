"""pytest 共享 fixtures"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest

from collabroom.core.types import LLMResponse, Usage, ToolCall, AgentResult, Step
from collabroom.core.tool import Tool, Registry, tool_result, tool_error
from collabroom.core.llm import LLM
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.memory.tiered import TieredMemory
from collabroom.core.loop import Agent as CoreAgent
from collabroom.room import Room, AgentMember, RoomMessage


# ── Mock LLM ──────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """返回一个 MagicMock LLM，默认返回空内容"""
    llm = MagicMock(spec=LLM)
    llm.chat.return_value = LLMResponse(content="", usage=Usage())
    return llm


@pytest.fixture
def mock_llm_yes():
    """返回一个总是回答 YES 的 mock LLM（用于举手测试）"""
    llm = MagicMock(spec=LLM)
    llm.chat.return_value = LLMResponse(content="YES", usage=Usage())
    return llm


@pytest.fixture
def mock_llm_no():
    """返回一个总是回答 NO 的 mock LLM（用于不举手测试）"""
    llm = MagicMock(spec=LLM)
    llm.chat.return_value = LLMResponse(content="NO", usage=Usage())
    return llm


@pytest.fixture
def mock_llm_say(say: str = "你好，我是测试Agent"):
    """返回一个固定回复的 mock LLM"""
    def _make(say_text: str = "你好，我是测试Agent"):
        llm = MagicMock(spec=LLM)
        llm.chat.return_value = LLMResponse(content=say_text, usage=Usage())
        return llm
    return _make


# ── Registry ──────────────────────────────────────────────

@pytest.fixture
def empty_registry():
    return Registry()


@pytest.fixture
def sample_registry():
    reg = Registry()
    reg.register(Tool(
        name="echo",
        description="回显输入",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文字"},
            },
            "required": ["text"],
        },
        fn=lambda text: tool_result(echo=text),
    ))
    reg.register(Tool(
        name="add",
        description="两个数相加",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        fn=lambda a, b: tool_result(sum=a + b),
    ))
    return reg


# ── Memory ────────────────────────────────────────────────

@pytest.fixture
def naive_mem():
    return NaiveMemory("你是测试助手")


@pytest.fixture
def tiered_mem():
    return TieredMemory("你是测试助手", working_window=3)


# ── AgentMember & Room ────────────────────────────────────

@pytest.fixture
def core_agent(mock_llm, empty_registry):
    return CoreAgent(llm=mock_llm, registry=empty_registry,
                     system_prompt="测试Agent")


@pytest.fixture
def core_agent_yes(mock_llm_yes, empty_registry):
    return CoreAgent(llm=mock_llm_yes, registry=empty_registry,
                     system_prompt="测试Agent")


@pytest.fixture
def core_agent_no(mock_llm_no, empty_registry):
    return CoreAgent(llm=mock_llm_no, registry=empty_registry,
                     system_prompt="测试Agent")


@pytest.fixture
def room(core_agent, core_agent_yes):
    room = Room("测试会议室")
    room.register(AgentMember("Alice", "测试助手", core_agent_yes))
    room.register(AgentMember("Bob", "测试助手", core_agent))
    return room


# ── RoomMessage ───────────────────────────────────────────

@pytest.fixture
def sample_messages():
    return [
        RoomMessage(sender="user", content="大家好", timestamp=100.0),
        RoomMessage(sender="Alice", content="你好", timestamp=101.0),
        RoomMessage(sender="Bob", content="嗨", timestamp=102.0, kind="dm"),
    ]
