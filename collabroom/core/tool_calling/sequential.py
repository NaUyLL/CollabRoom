"""SequentialToolCalling — 每步只执行一个工具调用，强制串行"""

from . import ToolCallingStrategy

class SequentialToolCalling(ToolCallingStrategy):
    """串行模式：LLM 仍然可以看到所有工具，但每步只执行第一个调用"""

    name = "sequential"

    def filter_tools(self, tool_defs: list[dict],
                     last_result: str | None = None) -> list[dict]:
        return self._apply_verbosity(tool_defs)

    def limit_calls(self, tool_calls: list,
                    tool_defs: list[dict]) -> list:
        """只取第一个 tool_call 执行"""
        if tool_calls:
            return [tool_calls[0]]
        return []
