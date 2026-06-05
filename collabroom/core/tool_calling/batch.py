"""BatchToolCalling — 全量 tools 一起暴露，LLM 可一次调多个

parallel=True：多个 tool_call 会被并行执行（Codex CLI 风格），而非串行。
"""

from . import ToolCallingStrategy

class BatchToolCalling(ToolCallingStrategy):
    """批量模式：所有工具一次给 LLM，LLM 可并行调用多个"""

    name = "batch"

    @property
    def supports_parallel(self) -> bool:
        """Batch 模式天然适合并行"""
        return True

    def filter_tools(self, tool_defs: list[dict],
                     last_result: str | None = None) -> list[dict]:
        return self._apply_verbosity(tool_defs)

    def limit_calls(self, tool_calls: list,
                    tool_defs: list[dict]) -> list:
        return tool_calls  # 不限量
