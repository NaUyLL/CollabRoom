"""多层级记忆 — 工作记忆 + 摘要 + 长时记忆，带上下文窗口管理

架构：
  TieredMemory
    ├── WorkingLayer    — 最近 N 轮对话，完整保留
    ├── SummaryLayer    — 早期对话压缩为摘要（文本拼接 或 LLM 摘要）
    └── FactLayer       — 对话中提取的关键事实

窗口管理（方向 C）：
  - soft_limit: token 估计超过阈值时触发裁剪/log warning
  - hard_limit: 超限时主动压缩旧内容到摘要
  - LLM 摘要自动注入：由 Agent loop 调用 set_summary_llm() 注入

用法：
  mem = TieredMemory(system_prompt, working_window=10,
                     soft_limit_tokens=8000, hard_limit_tokens=16000)
  mem.add("user", "今天聊什么")
  mem.add("assistant", "聊足球吧")
  ctx = mem.get_context()  # 自动检查 token 上限
"""

from __future__ import annotations
import json
import logging
import time
from copy import deepcopy
from typing import Any

from . import Memory


# ── 常量 ──
_DEFAULT_WORKING_WINDOW = 10        # 保留最近的 N 轮对话
_SOFT_LIMIT_TOKENS = 8000           # soft limit: log warning + 自动裁剪
_HARD_LIMIT_TOKENS = 16000          # hard limit: 强制压缩旧内容
_MAX_SUMMARY_TOKENS = 2000          # 摘要文本最大长度（估计）

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 工作记忆层
# ═══════════════════════════════════════════════════════════

class WorkingLayer:
    """最近 N 轮对话，完整保留"""

    def __init__(self, window: int = _DEFAULT_WORKING_WINDOW):
        self._messages: list[dict] = []
        self.window = window

    def add(self, msg: dict):
        self._messages.append(msg)

    def get_messages(self) -> list[dict]:
        """返回窗口内的消息（最新的 window 条 user+assistant 对）"""
        pairs = 0
        for m in reversed(self._messages):
            if m["role"] in ("user", "assistant"):
                pairs += 1
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
        return self._messages[:len(self._messages) - len(keep)]

    def reset(self):
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)


# ═══════════════════════════════════════════════════════════
# 摘要层
# ═══════════════════════════════════════════════════════════

class SummaryLayer:
    """早期对话的摘要"""

    def __init__(self):
        self._summary: str = ""

    def update(self, texts: list[str]):
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


# ═══════════════════════════════════════════════════════════
# 长时事实层
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# TieredMemory — 三层组合，带上下文窗口管理
# ═══════════════════════════════════════════════════════════

class TieredMemory(Memory):
    """多层记忆：工作记忆 + 摘要 + 长时事实，带上下文窗口管理

    参数:
      system_prompt: 系统提示
      working_window: 保留最近多少轮对话（默认 10）
      auto_summarize: 是否自动摘要溢出内容（需要 LLM，默认 True）
      soft_limit_tokens: token 软上限（默认 8000），超限触发裁剪+warning
      hard_limit_tokens: token 硬上限（默认 16000），超限强制压缩
    """

    def __init__(self, system_prompt: str,
                 working_window: int = _DEFAULT_WORKING_WINDOW,
                 auto_summarize: bool = True,
                 soft_limit_tokens: int = _SOFT_LIMIT_TOKENS,
                 hard_limit_tokens: int = _HARD_LIMIT_TOKENS):
        super().__init__(system_prompt)
        self.working = WorkingLayer(window=working_window)
        self.summary_layer = SummaryLayer()
        self.facts = FactLayer()
        self._auto_summarize = auto_summarize
        self._llm_for_summary = None

        # 上下文窗口管理：确保 hard >= soft
        self.soft_limit_tokens = max(1, soft_limit_tokens)
        self.hard_limit_tokens = max(self.soft_limit_tokens, hard_limit_tokens)
        self._last_warned = False  # 避免重复 warning

    def set_summary_llm(self, llm):
        """注入用于摘要的 LLM 实例"""
        self._llm_for_summary = llm

    # ── Memory 接口 ──────────────────────────────

    def add(self, role: str, content: str):
        """追加一条消息到工作记忆"""
        msg = {"role": role, "content": content}
        self.working.add(msg)

        # 窗口裁剪（按轮数）
        if self.working.window > 0:
            self._maybe_trim()

    def get_context(self) -> list[dict]:
        """返回完整消息列表（system + 摘要 + 事实 + 工作记忆）"""
        msgs = [self._system]

        # 摘要层
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

        # ── 上下文窗口管理 ──
        estimated = self._estimate_tokens(msgs)

        if estimated >= self.hard_limit_tokens:
            # 超硬上限：强制压缩工作记忆
            logger.warning(
                "Context 超硬上限 (%d >= %d)，强制压缩",
                estimated, self.hard_limit_tokens,
            )
            self._force_compress()
            # 重新构建
            msgs = [self._system]
            if summary_text:
                msgs.append({"role": "system", "content": f"[对话历史摘要]\n{summary_text}"})
            if facts_text:
                msgs.append({"role": "system", "content": facts_text})
            msgs.extend(self.working.get_messages())
            estimated = self._estimate_tokens(msgs)

        elif estimated >= self.soft_limit_tokens:
            # 超软上限：warning + 尝试裁剪
            if not self._last_warned:
                logger.warning(
                    "Context 接近上限 (%d/%d token)",
                    estimated, self.soft_limit_tokens,
                )
                self._last_warned = True
            # 尝试裁剪工作记忆
            self._maybe_trim()
            # 重新获取消息（裁剪后可能 summary 变了）
            msgs = [self._system]
            summary_text = self.summary_layer.get()
            if summary_text:
                msgs.append({"role": "system", "content": f"[对话历史摘要]\n{summary_text}"})
            if facts_text:
                msgs.append({"role": "system", "content": facts_text})
            msgs.extend(self.working.get_messages())
        else:
            self._last_warned = False

        return msgs

    def token_estimate(self) -> int:
        return self._estimate_tokens(self.working.get_messages())

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

    # ── token 估算 ─────────────────────────────

    def _estimate_tokens(self, msgs: list[dict]) -> int:
        """粗略估计消息列表的 token 数"""
        total = 0
        for m in msgs:
            content = m.get("content", "") or ""
            total += len(content) // 2  # 中文约 1.5-2 chars/token
            total += 10  # role + overhead
        return total

    # ── 裁剪逻辑 ───────────────────────────────

    def _maybe_trim(self):
        """当工作记忆超过窗口限制时，将溢出消息移入摘要"""
        overflow = self.working.get_overflow()
        if not overflow:
            return

        texts = [m["content"] for m in overflow if m.get("content")]
        if not texts:
            return

        # 摘要策略：有 LLM 则用 LLM 摘要，否则直接拼接
        if self._auto_summarize and self._llm_for_summary:
            summary = self._llm_summarize(texts)
        else:
            summary = self._summarize_concat(texts)

        self.summary_layer.update([summary])

        # 从工作记忆中删除溢出消息
        keep_count = len(self.working._messages) - len(overflow)
        self.working._messages = self.working._messages[-keep_count:] if keep_count > 0 else []

    def _force_compress(self):
        """硬压缩：将工作记忆中最旧的一半移入摘要"""
        msgs = self.working._messages
        if len(msgs) <= 2:
            return

        mid = len(msgs) // 2
        overflow = msgs[:mid]
        texts = [m["content"] for m in overflow if m.get("content")]
        if not texts:
            return

        if self._auto_summarize and self._llm_for_summary:
            summary = self._llm_summarize(texts, max_tokens=300)
        else:
            summary = self._summarize_concat(texts)

        self.summary_layer.update([summary])

        # 保留后半部分
        self.working._messages = msgs[mid:]

        logger.info("强制压缩：%d 条 → 摘要 (%d chars)", len(overflow), len(summary))

    # ── 摘要策略 ───────────────────────────────

    def _llm_summarize(self, texts: list[str], max_tokens: int = 200) -> str:
        """用 LLM 将多条消息压缩为一条摘要"""
        if not self._llm_for_summary:
            return self._summarize_concat(texts)

        prompt = (
            "将以下对话压缩为一段 2-3 句话的摘要，保留关键信息（话题、观点、决定）：\n\n"
            + "\n".join(texts)
        )
        try:
            resp = self._llm_for_summary.chat(
                [{"role": "system", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            result = (resp.content or "")
            return result[:500]
        except Exception as e:
            logger.warning("LLM 摘要失败 (%s)，降级为拼接", e)
            return self._summarize_concat(texts)

    def _summarize_concat(self, texts: list[str]) -> str:
        """降级策略：拼接最后几条消息的末尾"""
        if not texts:
            return ""
        # 取最后一条消息的末尾
        last = texts[-1]
        if len(last) <= 300:
            return last
        return last[-300:]

    # ── 序列化 ────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": "TieredMemory",
            "working_window": self.working.window,
            "messages": self.working._messages,
            "summary": self.summary_layer._summary,
            "facts": self.facts._facts,
            "soft_limit_tokens": self.soft_limit_tokens,
            "hard_limit_tokens": self.hard_limit_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict, system_prompt: str) -> TieredMemory:
        mem = cls(
            system_prompt,
            working_window=data.get("working_window", 10),
            soft_limit_tokens=data.get("soft_limit_tokens", _SOFT_LIMIT_TOKENS),
            hard_limit_tokens=data.get("hard_limit_tokens", _HARD_LIMIT_TOKENS),
        )
        mem.working._messages = data.get("messages", [])
        mem.summary_layer._summary = data.get("summary", "")
        mem.facts._facts = data.get("facts", [])
        return mem
