"""
pytest 共享 fixtures

组织：
  - Mock LLM 相关
  - Registry / Tool 相关
  - Memory 相关
  - Agent / Room 相关
  - 临时目录 / 文件系统 相关
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
import tempfile
import os

from collabroom.core.types import LLMResponse, Usage, ToolCall, AgentResult, Step
from collabroom.core.tool import Tool, Registry, tool_result, tool_error
from collabroom.core.llm import LLM
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.memory.tiered import TieredMemory
from collabroom.core.loop import Agent as CoreAgent
from collabroom.room import Room, AgentMember, RoomMessage


# ═══════════════════════════════════════════════════════════════
# Mock LLM
# ═══════════════════════════════════════════════════════════════

def _make_mock_llm(content: str = "", tool_calls: list | None = None,
                   usage: Usage | None = None, finish_reason: str = "stop"):
    """快速创建一个 mock LLM，返回指定 content + tool_calls"""
    llm = MagicMock(spec=LLM)
    llm.chat.return_value = LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or Usage(),
        finish_reason=finish_reason,
    )
    return llm


@pytest.fixture
def mock_llm():
    """默认 mock LLM，返回空内容"""
    return _make_mock_llm()


@pytest.fixture
def mock_llm_yes():
    """总是回答 YES（用于举手测试）"""
    return _make_mock_llm(content="YES")


@pytest.fixture
def mock_llm_no():
    """总是回答 NO"""
    return _make_mock_llm(content="NO")


@pytest.fixture
def mock_llm_say():
    """固定回复的工厂 fixture"""
    def _make(say_text: str = "你好"):
        return _make_mock_llm(content=say_text)
    return _make


@pytest.fixture
def mock_llm_with_tool():
    """返回 tool_calls 的 mock LLM 工厂"""
    def _make(tool_name: str = "echo", args: dict | None = None,
              content: str | None = None):
        tcs = [ToolCall(id="call_1", name=tool_name, arguments=args or {})]
        return _make_mock_llm(content=content, tool_calls=tcs)
    return _make


@pytest.fixture
def mock_token_bucket():
    """控制 TokenBucket 超时的辅助（patch time.monotonic）"""
    with patch("time.monotonic") as mock_monotonic:
        mock_monotonic.side_effect = [i * 0.1 for i in range(100)]
        yield mock_monotonic


# ═══════════════════════════════════════════════════════════════
# Registry / Tool
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def empty_registry():
    return Registry()


@pytest.fixture
def sample_registry():
    """含 echo 和 add 工具的注册表"""
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


@pytest.fixture
def failing_tool_registry():
    """含一个始终失败的测试工具"""
    reg = Registry()
    reg.register(Tool(
        name="always_fail",
        description="一定失败",
        parameters={},
        fn=lambda: (_ for _ in ()).throw(ValueError("模拟失败")),
    ))
    return reg


# ═══════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def naive_mem():
    return NaiveMemory("你是测试助手")


@pytest.fixture
def filled_naive_mem(naive_mem):
    """已有 3 轮对话的 NaiveMemory"""
    for i in range(3):
        naive_mem.add("user", f"用户问题{i}")
        naive_mem.add("assistant", f"助手回答{i}")
    return naive_mem


@pytest.fixture
def tiered_mem():
    return TieredMemory("你是测试助手", working_window=3)


@pytest.fixture
def filled_tiered_mem():
    """已有溢出消息的 TieredMemory（w=3, 已填6轮，3轮溢出到摘要）"""
    mem = TieredMemory("你是测试助手", working_window=3)
    for i in range(6):
        mem.add("user", f"问题{i}")
        mem.add("assistant", f"回答{i}")
    return mem


# ═══════════════════════════════════════════════════════════════
# Agent / Room
# ═══════════════════════════════════════════════════════════════

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
    """含 Alice(举手) 和 Bob(沉默) 的 2 人 Room"""
    room = Room("测试会议室")
    room.register(AgentMember("Alice", "测试助手", core_agent_yes))
    room.register(AgentMember("Bob", "测试助手", core_agent))
    return room


@pytest.fixture
def room_3agents(core_agent_yes, core_agent, core_agent_no):
    """含 Alice(YES) / Bob(空) / Charlie(NO) 的 3 人 Room"""
    room = Room("三人测试室")
    room.register(AgentMember("Alice", "活跃成员", core_agent_yes))
    room.register(AgentMember("Bob", "中性成员", core_agent))
    room.register(AgentMember("Charlie", "沉默成员", core_agent_no))
    return room


@pytest.fixture
def room_with_dm():
    """用于私信测试的 Room（带 mock member）"""
    room = Room("私信测试室")
    alice_agent = MagicMock(spec=CoreAgent)
    alice_agent.system_prompt = "测试"
    alice_agent.run.return_value = AgentResult(final_answer="收到私信，明白")
    alice_member = AgentMember("Alice", "测试助手", alice_agent, on_pass="PASS")

    bob_agent = MagicMock(spec=CoreAgent)
    bob_agent.system_prompt = "测试"
    bob_agent.run.return_value = AgentResult(final_answer="已收到 Alice 的消息")
    bob_member = AgentMember("Bob", "测试助手", bob_agent, on_pass="PASS")

    room.register(alice_member)
    room.register(bob_member)
    return room


# ═══════════════════════════════════════════════════════════════
# RoomMessage
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_messages():
    return [
        RoomMessage(sender="user", content="大家好", timestamp=100.0),
        RoomMessage(sender="Alice", content="你好", timestamp=101.0),
        RoomMessage(sender="Bob", content="嗨", timestamp=102.0, kind="dm"),
    ]


# ═══════════════════════════════════════════════════════════════
# 临时文件系统
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_workspace():
    """创建临时工作目录，测试完自动清理"""
    with tempfile.TemporaryDirectory(prefix="collabroom_test_") as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(orig_cwd)


@pytest.fixture
def sample_text_file(tmp_workspace):
    """在临时目录中创建一个测试文本文件"""
    path = os.path.join(tmp_workspace, "test.txt")
    with open(path, "w") as f:
        f.write("第1行: hello world\n第2行: foo bar\n第3行: test data\n")
    return path
