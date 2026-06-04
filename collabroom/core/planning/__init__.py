"""Planning 策略 — 控制 Agent 的思考与行动编排方式"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import AgentResult
    from ..llm import LLM
    from ..tool import Registry
    from ..memory import Memory

class PlanningStrategy(ABC):
    """规划策略接口 — 每种策略实现自己的 run() 方法"""

    system_prompt_prefix: str = ""  # 注入到 system prompt 开头的指令

    @abstractmethod
    def run(
        self,
        llm: LLM,
        registry: Registry,
        memory: Memory,
        system_prompt: str,
        user_message: str,
        max_steps: int,
        tool_calling: ToolCallingStrategy | None = None,
    ) -> AgentResult:
        ...

from ..tool_calling import ToolCallingStrategy  # noqa: E402 — 放最后避免循环导入
