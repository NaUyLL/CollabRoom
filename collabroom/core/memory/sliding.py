"""SlidingMemory — 只保留最近 N 轮对话"""
from __future__ import annotations
from copy import deepcopy
from . import Memory

class SlidingMemory(Memory):
    """滑动窗口记忆：只保留最近 max_rounds 轮对话(一轮=user+assistant)"""

    def __init__(self, system_prompt: str, max_rounds: int = 5):
        super().__init__(system_prompt)
        self._messages: list[dict] = [self._system]
        self._rounds: int = 0           # 已记录的用户轮次
        self._max_rounds = max_rounds

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if role == "user":
            self._rounds += 1

        # 超出窗口：移除最早的一轮（user + assistant）
        while self._rounds > self._max_rounds:
            # 找到第一条 user 消息（跳过 system）
            for i, m in enumerate(self._messages):
                if m["role"] == "system":
                    continue
                if m["role"] == "user":
                    # 移除这条 user 和紧随其后的 assistant（如果有）
                    remove_end = i + 2 if (i + 1 < len(self._messages)
                                           and self._messages[i + 1]["role"] == "assistant") else i + 1
                    self._messages = [self._messages[0]] + self._messages[remove_end:]
                    self._rounds -= 1
                    break

    def get_context(self) -> list[dict]:
        return deepcopy(self._messages)

    def token_estimate(self) -> int:
        total = 0
        for m in self._messages:
            content = m.get("content", "") or ""
            total += len(content) // 2
            total += 10
        return total
