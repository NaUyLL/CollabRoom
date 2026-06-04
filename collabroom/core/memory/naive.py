"""NaiveMemory — 永不裁剪，全量保留"""
from __future__ import annotations
from copy import deepcopy
from . import Memory

class NaiveMemory(Memory):
    """最简单的记忆：所有消息依次追加，不做任何裁剪"""

    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)
        self._messages: list[dict] = [self._system]

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        return deepcopy(self._messages)

    def token_estimate(self) -> int:
        total = 0
        for m in self._messages:
            content = m.get("content", "") or ""
            total += len(content) // 2  # 中文约 1.5-2 chars/token
            total += 10  # role + overhead
        return total
