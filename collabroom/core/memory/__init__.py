"""Memory 接口 — 所有记忆策略都实现这 3 个方法"""
from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy

class Memory(ABC):
    """记忆策略的抽象接口"""

    def __init__(self, system_prompt: str):
        self._system = {"role": "system", "content": system_prompt}

    @abstractmethod
    def add(self, role: str, content: str):
        """追加一条消息。role: 'user' 或 'assistant'"""
        ...

    def get_context(self) -> list[dict]:
        """返回当前完整消息列表的副本，供 loop 使用"""
        raise NotImplementedError

    @abstractmethod
    def token_estimate(self) -> int:
        """粗略估计当前占用的 token 数（用于统计对比）"""
        ...

    def summary(self) -> dict:
        """返回当前记忆的统计摘要"""
        msgs = self.get_context()
        n_user = sum(1 for m in msgs if m["role"] == "user")
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        return {
            "total_messages": len(msgs),
            "user_turns": n_user,
            "assistant_turns": n_asst,
            "estimated_tokens": self.token_estimate(),
        }

    # ── 序列化（子类可覆盖） ─────────────────────

    def to_dict(self) -> dict:
        """序列化 memory 状态。子类必须覆盖。"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict, system_prompt: str) -> Memory:
        """从字典恢复 memory。子类必须覆盖。"""
        raise NotImplementedError
