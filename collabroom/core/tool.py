"""工具注册表 — dict 存 + 函数执行，不搞继承"""
from __future__ import annotations
import json, traceback
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class Tool:
    """一个可被 LLM 调用的工具"""
    name: str
    description: str
    parameters: dict          # JSON Schema
    fn: Callable              # fn(**kwargs) -> str

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

# ── 工具返回辅助函数 ──────────────────────────────
# 消除全仓库散落的 json.dumps({"error":...}) 样板代码

def tool_error(message: str, **extra) -> str:
    """返回 JSON 错误字符串

    >>> tool_error("file not found")
    '{"error": "file not found"}'
    """
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)

def tool_result(data: dict | None = None, **kwargs) -> str:
    """返回 JSON 成功字符串

    >>> tool_result(success=True, chars=42)
    '{"success": true, "chars": 42}'
    >>> tool_result({"key": "value"})
    '{"key": "value"}'
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


class Registry:
    """工具注册表：注册 → 出 schema → 执行"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = tool.fn(**arguments)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, indent=2)
            return result
        except Exception as e:
            return json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
