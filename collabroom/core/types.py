"""共用的数据结构 — 足够跑通实验，不多不少"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generator

@dataclass
class ToolCall:
    """LLM 决定调用的一个工具"""
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass
class LLMResponse:
    """一次 LLM 调用的结果（非流式）"""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""

@dataclass
class LLMStreamChunk:
    """流式返回的一个 chunk"""
    content: str = ""
    tool_call_builder: dict | None = None  # 累积中的 tool_call
    finish_reason: str = ""

@dataclass
class Step:
    """ReAct 循环中的一步"""
    role: str                    # "think" | "act" | "observe" | "done"
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    token_usage: Usage = field(default_factory=Usage)
    elapsed_ms: float = 0

@dataclass
class AgentResult:
    """一次 chat() 的完整结果"""
    final_answer: str = ""
    steps: list[Step] = field(default_factory=list)
    total_tokens: int = 0
    total_tool_calls: int = 0
    elapsed_ms: float = 0
