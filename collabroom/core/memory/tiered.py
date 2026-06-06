"""多层级记忆 — 工作记忆 + 摘要 + 长时记忆

架构：
  TieredMemory
    ├── WorkingLayer    — 最近 N 轮对话，完整保留
    ├── SummaryLayer    — 早期对话压缩为摘要
    └── FactLayer       — 对话中提取的关键事实

用法：
  mem = TieredMemory(system_prompt, working_window=10)
  mem.add("user", "今天聊什么")
  mem.add("assistant", "聊足球吧")
  ctx = mem.get_context()  # 完整消息列表
"""

from __future__ import annotations
import json
import time
from copy import deepcopy
from typing import Any

from . import Memory


# ── 常量 ──
_DEFAULT_WORKING_WINDOW = 10       # 保留最近的 N 轮对话
_MAX_SUMMARY_TOKENS = 2000         # 摘要文本最大长度（估计）


# ═══════════════════════════════════════════════════════
# 工作记忆层
# ═══════════════════════════════════════════════════════

class WorkingLayer:
    """最近 N 轮对话，完整保留"""

    def __init__(self, window: int = _DEFAULT_WORKING_WINDOW):
        self._messages: list[dict] = []
        self.window = window

    def add(self, msg: dict):
        self._messages.append(msg)

    def get_messages(self) -> list[dict]:
        """返回窗口内的消息（最新的 window 条 user+assistant 对）"""
        # 统计 user 和 assistant 消息
        pairs = 0
        for m in reversed(self._messages):
            if m["role"] in ("user", "assistant"):
                pairs += 1
        # 取最近的 window 条非 system 消息
        count = 0
        result = []
        for m in reversed(self._messages):
            if m["role"] in ("user", "assistant"):
                count += 1
            result.insert(0, m)
            if count >= self.window * 2:
                break
        return result

    def get_overflow(self) -> list[dict]:
        """返回被窗口裁剪掉的消息，用于生成摘要"""
        keep = self.get_messages()
        if len(self._messages) <= len(keep):
            return []
        # 保留的都在后面，溢出在前面
        return self._messages[:len(self._messages) - len(keep)]

    def reset(self):
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)


# ═══════════════════════════════════════════════════════
# 摘要层
# ═══════════════════════════════════════════════════════

class SummaryLayer:
    """早期对话的摘要"""

    def __init__(self):
        self._summary: str = ""

    def update(self, texts: list[str]):
        """追加文本到摘要"""
        if not texts:
            return
        new = "\n".join(texts)
        if self._summary:
            self._summary += "\n" + new
        else:
            self._summary = new

    def set(self, summary: str):
        self._summary = summary

    def get(self) -> str:
        return self._summary

    def reset(self):
        self._summary = ""


# ═══════════════════════════════════════════════════════
# 长时事实层
# ═══════════════════════════════════════════════════════

class FactLayer:
    """对话中提取的关键事实（姓名、偏好、决定等）"""

    def __init__(self):
        self._facts: list[dict] = []  # [{text, timestamp, source}]

    def add_fact(self, text: str, source: str = ""):
        self._facts.append({
            "text": text,
            "timestamp": time.time(),
            "source": source,
        })

    def get_facts(self, limit: int = 10) -> list[str]:
        return [f["text"] for f in self._facts[-limit:]]

    def get_formatted(self) -> str:
        if not self._facts:
            return ""
        lines = ["之前记住的信息："]
        for f in self._facts[-10:]:
            lines.append(f"- {f['text']}")
        return "\n".join(lines)

    def reset(self):
        self._facts = []


# ═══════════════════════════════════════════════════════
# TieredMemory — 三层组合
# ═══════════════════════════════════════════════════════

class TieredMemory(Memory):
    """多层记忆：工作记忆 + 摘要 + 长时事实

    参数:
      system_prompt: 系统提示
      working_window: 保留最近多少轮对话（默认 10）
      auto_summarize: 是否自动摘要溢出内容（需要 LLM，默认 False）
    """

    def __init__(self, system_prompt: str,
                 working_window: int = _DEFAULT_WORKING_WINDOW,
                 auto_summarize: bool = False):
        super().__init__(system_prompt)
        self.working = WorkingLayer(window=working_window)
        self.summary_layer = SummaryLayer()
        self.facts = FactLayer()
        self._auto_summarize = auto_summarize
        self._llm_for_summary = None  # 初始化后注入

    def set_summary_llm(self, llm):
        """注入用于摘要的 LLM 实例"""
        self._llm_for_summary = llm

    # ── Memory 接口 ──────────────────────────────

    def add(self, role: str, content: str):
        """追加一条消息到工作记忆"""
        msg = {"role": role, "content": content}
        self.working.add(msg)

        # 检查是否需要裁剪
        if self.working.window > 0:
            self._maybe_trim()

    def get_context(self) -> list[dict]:
        """返回完整消息列表（system + 摘要 + 工作记忆 + 事实）"""
        msgs = [self._system]

        # 摘要层（如果存在）
        summary_text = self.summary_layer.get()
        if summary_text:
            msgs.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{summary_text}",
            })

        # 长时事实
        facts_text = self.facts.get_formatted()
        if facts_text:
            msgs.append({
                "role": "system",
                "content": facts_text,
            })

        # 工作记忆
        msgs.extend(self.working.get_messages())

        return msgs

    def token_estimate(self) -> int:
        total = 0
        ctx = self.get_context()
        for m in ctx:
            content = m.get("content", "") or ""
            total += len(content) // 2
            total += 10
        return total

    def summary(self) -> dict:
        ctx = self.get_context()
        n_user = sum(1 for m in ctx if m["role"] == "user")
        n_asst = sum(1 for m in ctx if m["role"] == "assistant")
        return {
            "total_messages": len(ctx),
            "working_window": self.working.window,
            "working_messages": len(self.working._messages),
            "has_summary": bool(self.summary_layer.get()),
            "facts_count": len(self.facts._facts),
            "user_turns": n_user,
            "assistant_turns": n_asst,
            "estimated_tokens": self.token_estimate(),
        }

    # ── 内部裁剪 ───────────────────────────────

    def _maybe_trim(self):
        """当工作记忆超过窗口限制时，将溢出消息移入摘要"""
        overflow = self.working.get_overflow()
        if not overflow:
            return

        # 提取溢出消息的文本
        texts = [m["content"] for m in overflow if m.get("content")]
        if not texts:
            return

        # 移入摘要
        self.summary_layer.update(texts)

        # 从工作记忆中删除溢出消息
        keep_count = len(self.working._messages) - len(overflow)
        self.working._messages = self.working._messages[-keep_count:] if keep_count > 0 else []

    # ── LLM-assisted 摘要生成 ─────────────────

    def _llm_summarize(self, texts: list[str]) -> str:
        """用 LLM 将多条消息压缩为一条摘要"""
        if not self._llm_for_summary:
            # 无 LLM 时，直接拼接最后几条
            return texts[-1][:500] if texts else ""

        prompt = (
            "将以下对话压缩为一段 2-3 句话的摘要，保留关键信息（话题、观点、决定）：\n\n"
            + "\n".join(texts)
        )
        try:
            resp = self._llm_for_summary.chat(
                [{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return (resp.content or "")[:500]
        except Exception:
            return texts[-1][:300] if texts else ""

    # ── 序列化 ────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": "TieredMemory",
            "working_window": self.working.window,
            "messages": self.working._messages,
            "summary": self.summary_layer._summary,
            "facts": self.facts._facts,
        }

    @classmethod
    def from_dict(cls, data: dict, system_prompt: str) -> TieredMemory:
        mem = cls(system_prompt,
                  working_window=data.get("working_window", 10))
        mem.working._messages = data.get("messages", [])
        mem.summary_layer._summary = data.get("summary", "")
        mem.facts._facts = data.get("facts", [])
        return mem
