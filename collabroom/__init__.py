"""CollabRoom — 多 Agent 协作框架

让多个 AI Agent 在一个房间里协作交流。
每个 Agent 是独立的个体，各有不同的角色、性格、工具和记忆。

核心用法：
    from collabroom import Room, AgentMember, CoreAgent, LLM

    agent = CoreAgent(llm=LLM(), registry=..., system_prompt="...")
    member = AgentMember("架构师", "系统架构师", agent)
    room = Room("设计讨论室")
    room.register(member)

    responses = room.round("帮我设计一个系统")
"""

__version__ = "0.1.0"

from .room import Room, AgentMember, RoomMessage
from .core.loop import Agent as CoreAgent
from .core.llm import LLM
from .core.tool import Tool, Registry
from .core.types import AgentResult
from .gateway.cli import run_gateway
