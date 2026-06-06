"""Agent 核心循环 — 不知道 memory/planning/tool_calling 是谁，只调接口"""
from __future__ import annotations
from .types import AgentResult
from .llm import LLM
from .tool import Registry
from .memory import Memory
from .memory.naive import NaiveMemory
from .memory.tiered import TieredMemory
from .planning import PlanningStrategy
from .planning.react import ReActStrategy
from .tool_calling import ToolCallingStrategy
from .tool_calling.batch import BatchToolCalling

class Agent:
    """Agent — memory、planning、tool_calling 都可插拔"""

    def __init__(self, llm: LLM, registry: Registry,
                 system_prompt: str, max_steps: int = 10,
                 memory: Memory | None = None,
                 planning: PlanningStrategy | None = None,
                 tool_calling: ToolCallingStrategy | None = None):
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.memory = memory or NaiveMemory(system_prompt)
        self.planning = planning or ReActStrategy()
        self.tool_calling = tool_calling or BatchToolCalling()

        # ── 自动注入 LLM 到 TieredMemory（方向 C） ──
        if isinstance(self.memory, TieredMemory):
            self.memory.set_summary_llm(llm)

    def run(self, user_message: str,
            memory: Memory | None = None) -> AgentResult:
        mem = memory or self.memory
        result = self.planning.run(
            llm=self.llm,
            registry=self.registry,
            memory=mem,
            system_prompt=self.system_prompt,
            user_message=user_message,
            max_steps=self.max_steps,
            tool_calling=self.tool_calling,
        )
        return result
