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
