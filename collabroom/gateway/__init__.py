"""Gateway 层 — 一个 Room 可以对接多种传输层

架构：
  BaseGateway  ←  CLIGateway（终端交互）
              ←  HTTPGateway（HTTP API）

每个 Gateway 只负责"收发消息"，不碰 Room 核心逻辑。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..room import Room


class BaseGateway(ABC):
    """Gateway 抽象基类 — 所有传输层的共同接口"""

    def __init__(self, room: Room, name: str = "gateway"):
        self.room = room
        self.name = name

    @abstractmethod
    def run(self):
        """启动 Gateway（阻塞）。各子类实现自己的事件循环。"""
        ...

    @abstractmethod
    def stop(self):
        """停止 Gateway。"""
        ...

    # ── 子类可继承的公共方法 ──

    def handle_message(self, sender: str, text: str) -> list[dict]:
        """处理一条用户消息 → 返回结构化响应列表

        返回格式：
          [{"sender": "架构师", "content": "...", "kind": "public"}, ...]
        """
        raw = self.room.round(text)
        return [
            {"sender": name, "content": reply, "kind": "public"}
            for name, reply in raw
        ]

    def get_history(self, tail: int = 20) -> list[dict]:
        """获取对话历史"""
        msgs = self.room.history[-tail:]
        return [
            {
                "sender": m.sender,
                "content": m.content[:1000],
                "kind": m.kind,
                "timestamp": m.timestamp,
            }
            for m in msgs
        ]

    def list_members(self) -> list[dict]:
        """列出所有成员"""
        return [
            {"name": name, "role": m.role_desc[:200]}
            for name, m in self.room.members.items()
        ]


# ── 序列化辅助 ──

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
