"""Tool Calling 策略 — 控制工具调用的并行度和 schema 粒度"""

from __future__ import annotations
from abc import ABC, abstractmethod

class ToolCallingStrategy(ABC):
    """工具调用策略接口

    filter_tools: 控制每步给 LLM 看哪些工具定义
    limit_calls: 控制一次最多执行几个工具调用
    """

    name: str = ""

    @property
    def supports_parallel(self) -> bool:
        """此策略支持并行执行多个 tool_call 吗？"""
        return False

    def __init__(self, verbosity: str = "long"):
        assert verbosity in ("short", "long"), f"verbosity 只能是 short 或 long: {verbosity}"
        self.verbosity = verbosity

    @abstractmethod
    def filter_tools(self, tool_defs: list[dict],
                     last_result: str | None = None) -> list[dict]:
        """返回本轮应该暴露给 LLM 的工具定义"""
        ...

    @abstractmethod
    def limit_calls(self, tool_calls: list,
                    tool_defs: list[dict]) -> list:
        """限制本轮实际执行的 tool_calls 数量"""
        ...

    def _apply_verbosity(self, defs: list[dict]) -> list[dict]:
        """根据 verbosity 修短 description"""
        if self.verbosity == "long":
            return defs
        result = []
        for d in defs:
            d = d.copy()
            func = dict(d.get("function", {}))
            if self.verbosity == "short":
                # 只保留 15 字以内的 description
                desc = func.get("description", "")
                func["description"] = desc[:15] + "..." if len(desc) > 15 else desc
                # 参数 description 清空
                params = func.get("parameters", {})
                if isinstance(params, dict):
                    props = params.get("properties", {})
                    for p_name, p_schema in props.items():
                        if isinstance(p_schema, dict):
                            p_schema.pop("description", None)
                    params["properties"] = props
                func["parameters"] = params
            d["function"] = func
            result.append(d)
        return result
