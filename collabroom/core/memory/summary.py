"""SummaryMemory — 超限后把旧轮压缩成摘要，保留最新 N 轮细节"""
from __future__ import annotations
from copy import deepcopy
from . import Memory
from ..llm import LLM, system_msg, user_msg, assistant_msg

class SummaryMemory(Memory):
    """摘要记忆：保留最近 max_rounds 轮完整消息，更早的压缩成一段摘要

    摘要也是用 LLM 生成的——这是 Memory 策略需要 LLM 依赖的唯一场景。
    """

    def __init__(self, system_prompt: str, llm: LLM, max_rounds: int = 5):
        super().__init__(system_prompt)
        self._llm = llm
        self._max_rounds = max_rounds
        self._summary: str | None = None          # 已压缩的早期对话摘要
        self._messages: list[dict] = [self._system]  # 最近 N 轮 + system
        self._rounds: int = 0

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if role == "user":
            self._rounds += 1

        if self._rounds > self._max_rounds:
            self._compress()

    def get_context(self) -> list[dict]:
        msgs = deepcopy(self._messages)
        if self._summary:
            # 把摘要拼在 system prompt 之后、最新消息之前
            msgs.insert(1, {
                "role": "system",
                "content": f"[对话历史摘要]\n{self._summary}",
            })
        return msgs

    def token_estimate(self) -> int:
        total = 0
        for m in self._messages:
            content = m.get("content", "") or ""
            total += len(content) // 2 + 10
        if self._summary:
            total += len(self._summary) // 2 + 20
        return total

    # ── 内部 ───────────────────────────────────

    def _compress(self):
        """将最早的一轮用户对话压缩进摘要"""
        # 找到第一条 user 消息
        user_idx = None
        for i, m in enumerate(self._messages):
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                user_idx = i
                break

        if user_idx is None:
            return  # 没有可压缩的

        # 取出要压缩的轮 (user + 其后的 assistant)
        asst_idx = user_idx + 1 if (user_idx + 1 < len(self._messages)
                                    and self._messages[user_idx + 1]["role"] == "assistant") else None
        to_compress_user = self._messages[user_idx]["content"]
        to_compress_asst = self._messages[asst_idx]["content"] if asst_idx else ""

        # 调 LLM 做摘要
        prompt = f"请用一句话概括以下对话的核心内容：\n用户：{to_compress_user}\n助手：{to_compress_asst}"
        resp = self._llm.chat([user_msg(prompt)], temperature=0.3, max_tokens=128)
        compressed = (resp.content or "").strip()

        # 合并到已有摘要
        if self._summary:
            self._summary = f"{self._summary}\n{compressed}"
        else:
            self._summary = compressed

        # 移除被压缩的消息
        remove_end = asst_idx + 1 if asst_idx else user_idx + 1
        self._messages = [self._messages[0]] + self._messages[remove_end:]
        self._rounds -= 1
