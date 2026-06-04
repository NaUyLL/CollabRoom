"""Room — 会议室：管理多个 Agent 实例，协调消息路由与多轮对话

交互机制：
  用户发言
    ① 举手阶段 — 每个 Agent 轻量决策 YES/NO
        无人举手 → 兜底强制选一个
    ② 发言阶段 — 举手者依次发言
        发言中 @某Agent → 强制该 Agent 回应（防循环）
    ③ 用户停止词 → 本轮中断
"""

from __future__ import annotations
import re, time
from dataclasses import dataclass, field

from collabroom.core.llm import LLM, system_msg, user_msg
from collabroom.core.loop import Agent as CoreAgent

# ── 常量 ──
STOP_WORDS = {"停止", "够了", "停", "别说了", "结束", "到此为止",
              "stop", "enough", "halt", "打住"}
MAX_AUTO_DEPTH = 5       # 一轮用户发言内最大自动交互次数
MAX_PAIR_LOOPS = 2       # 同一对 Agent 来回 @ 的最大次数
MENTION_RE = re.compile(r'@(\S+?)（?\s*(?=[:：，。！？\s\n]|$)')

@dataclass
class RoomMessage:
    """房间中的一条消息"""
    sender: str
    content: str
    timestamp: float = 0.0
    kind: str = "public"  # "public" | "dm"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

class AgentMember:
    """一个 Agent 成员 — 包装 core.Agent，有自己的名字和记忆"""

    def __init__(self, name: str, role_desc: str,
                 core_agent: CoreAgent,
                 on_pass: str | None = "PASS"):
        self.name = name
        self.role_desc = role_desc
        self.agent = core_agent
        self.on_pass = on_pass
        self._system_prompt = (
            f"你叫 {name}，你的角色是：{role_desc}\n"
            f"{core_agent.system_prompt}"
        )

    def decide(self, context: str) -> bool:
        """轻量决策：根据当前上下文，判断自己是否要发言。"""
        prompt = (
            f"【角色】{self.name}，{self.role_desc}\n\n"
            f"【当前对话】\n{context}\n\n"
            f"你会对此话题发言吗？只回复 YES 或 NO，不要其他文字。"
        )
        try:
            msgs = [system_msg(prompt)]
            resp = self.agent.llm.chat(msgs, tools=None,
                                         max_tokens=10, temperature=0.2)
            answer = (resp.content or "").strip().upper()
            result = answer.startswith("YES")
            return result
        except Exception:
            return False
    def chat(self, context: str, mention_context: str | None = None,
             force_reply: bool = False) -> str:
        """Agent 发言。force_reply=True 时绕过 PASS 提示（兜底场景）。"""
        if force_reply:
            msg = (
                f"【房间对话上下文】\n{context}\n\n"
                f"现在轮到 {self.name} 发言。作为团队的一员，请根据你的角色给出回应。"
            )
        else:
            msg = f"【房间对话上下文】\n{context}\n\n现在轮到 {self.name} 发言。"
            if mention_context:
                msg = f"【房间对话上下文】\n{context}\n\n{mention_context}\n\n现在轮到 {self.name} 回应。"
            if self.on_pass is not None:
                msg += f" 如果你觉得没什么可说的，只回复「{self.on_pass}」。"
        result = self.agent.run(msg)
        return result.final_answer

    @property
    def memory(self):
        return self.agent.memory

class Room:
    """会议室 — 管理一组 Agent，协调多人对话"""

    def __init__(self, name: str = "会议室"):
        self.name = name
        self.members: dict[str, AgentMember] = {}
        self.history: list[RoomMessage] = []
        self._order: list[str] = []

    def register(self, member: AgentMember, position: int | None = None):
        self.members[member.name] = member
        if position is not None:
            self._order.insert(position, member.name)
        else:
            self._order.append(member.name)

    def say(self, sender: str, content: str,
            kind: str = "public") -> RoomMessage:
        msg = RoomMessage(sender=sender, content=content, kind=kind)
        self.history.append(msg)
        return msg

    def format_history(self, tail: int = 10,
                       include_kind: set[str] | None = None) -> str:
        msgs = self.history
        if include_kind:
            msgs = [m for m in msgs if m.kind in include_kind]
        lines = []
        for m in msgs[-tail:]:
            tag = f"@{m.sender}" if m.kind == "dm" else m.sender
            # 截断长内容，保留关键信息
            content = m.content[:250]
            if len(m.content) > 250:
                content += "…"
            lines.append(f"{tag}: {content}")
        return "\n".join(lines)

    # ── 核心交互 ──────────────────────────────────────

    def round(self, user_message: str) -> list[tuple[str, str]]:
        """用户发言 → 举手 → 发言 → @mention 链式回应 → 停止检测"""
        # 停止词检测
        if self._is_stop(user_message):
            return []

        self.say("user", user_message)
        responses: list[tuple[str, str]] = []

        context = self.format_history(tail=15)

        # ── Phase 1: 举手 ──
        volunteers = self._volunteer_round(context)

        # 用户 @某Agent → 强制加入举手名单
        user_mentions = self._parse_mentions(user_message)
        for name in user_mentions:
            if name in self.members and name not in volunteers:
                volunteers.append(name)
        had_volunteers = bool(volunteers)

        # 兜底：无人举手 → 强制选一个
        if not volunteers:
            fallback = self._pick_fallback()
            if fallback:
                volunteers = [fallback]

        # ── Phase 2: 发言 + @mention 链式回应 ──
        pair_counts: dict[tuple[str, str], int] = {}
        queue = list(volunteers)
        no_volunteers = not had_volunteers
        depth = 0

        while queue and depth < MAX_AUTO_DEPTH:
            speaker = queue.pop(0)
            member = self.members.get(speaker)
            if not member:
                continue

            context = self.format_history(tail=15)
            is_fallback = (no_volunteers and depth == 0)
            reply = member.chat(context, force_reply=is_fallback)
            pass_reply = member.on_pass or "PASS"

            if not is_fallback and reply.strip() == pass_reply.strip():
                continue
            if not reply.strip():
                continue

            self.say(speaker, reply)
            responses.append((speaker, reply))
            depth += 1

            # 检查 @mention
            mentions = self._parse_mentions(reply)
            for target in mentions:
                if target not in self.members or target == speaker:
                    continue

                pair = tuple(sorted([speaker, target]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                if pair_counts[pair] > MAX_PAIR_LOOPS:
                    continue

                if target not in queue:
                    queue.insert(0, target)

        return responses

    def dm(self, from_name: str, to_name: str, content: str) -> str | None:
        """Agent 之间私信"""
        member = self.members.get(to_name)
        if not member:
            return None

        self.say(from_name, content, kind="dm")
        context = self.format_history(tail=5, include_kind={"public", "dm"})

        mention_context = f"【私信】@{from_name} 对你说：{content}"
        reply = member.chat(context, mention_context=mention_context)

        if member.on_pass and reply.strip() == member.on_pass.strip():
            return None

        self.say(to_name, reply, kind="dm")
        return reply

    def list_members(self) -> list[str]:
        return list(self.members.keys())

    # ── 内部方法 ────────────────────────────────────

    def _is_stop(self, msg: str) -> bool:
        """检测用户停止词"""
        return any(w in msg for w in STOP_WORDS)

    def _volunteer_round(self, context: str) -> list[str]:
        """并行举手：每个 Agent 决定是否要发言"""
        volunteers = []
        for name in self._order:
            member = self.members.get(name)
            if not member:
                continue
            try:
                if member.decide(context):
                    volunteers.append(name)
            except Exception:
                pass
        return volunteers

    def _pick_fallback(self) -> str | None:
        """兜底：无人举手时强制选第一个 Agent"""
        return self._order[0] if self._order else None

    def _parse_mentions(self, text: str) -> list[str]:
        """解析 @某Agent 提及"""
        found = MENTION_RE.findall(text)
        # 去重、只保留已注册的成员
        seen = set()
        result = []
        for name in found:
            name = name.strip("@").strip()
            if name in self.members and name not in seen:
                seen.add(name)
                result.append(name)
        return result
