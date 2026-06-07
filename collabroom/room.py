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
import json, re, time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from collabroom.core.llm import LLM, system_msg, user_msg
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.memory.tiered import TieredMemory
from collabroom.core.logger import get_logger, get_trace_id

logger = get_logger("room")

# ── 常量 ──
STOP_WORDS = {"停止", "够了", "停", "别说了", "结束", "到此为止",
              "stop", "enough", "halt", "打住"}
# _is_stop 使用完整匹配（非子串匹配），避免误触发
STOP_MATCH_WHOLE_WORD = True
MAX_AUTO_DEPTH = 5       # 一轮用户发言内最大自动交互次数
MAX_PAIR_LOOPS = 2       # 同一对 Agent 来回 @ 的最大次数
# 支持中英文 @mention：@名字 后跟空格、标点或结尾
MENTION_RE = re.compile(r'@(\S+?)(?=[\s:：，。！？、；\n]|$)')

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
        msg = f"【房间对话上下文】\n{context}\n\n"
        if mention_context:
            msg += f"{mention_context}\n\n"
        if force_reply:
            msg += (
                f"现在轮到 {self.name} 切实行动：\n"
                f"1. 需要讨论分析就直接说出来\n"
                f"2. 需要执行任务（修改代码、运行命令等）就直接用你的工具去执行\n"
                f"完成后给出最终回复。"
            )
        else:
            msg += f"现在轮到 {self.name} 发言。"
            if self.on_pass is not None:
                msg += f" 如果你觉得没什么可说的，只回复「{self.on_pass}」。"
        result = self.agent.run(msg)
        return result.final_answer

    @property
    def memory(self):
        return self.agent.memory

    # ── 序列化 ────────────────────────────────────

    MEMORY_TYPE_MAP: dict[str, type] = {
        "NaiveMemory": NaiveMemory,
        "TieredMemory": TieredMemory,
    }

    def to_dict(self) -> dict:
        """序列化 AgentMember 状态（不包括 LLM/Registry — 运行时重建）"""
        return {
            "name": self.name,
            "role_desc": self.role_desc,
            "on_pass": self.on_pass,
            "system_prompt": self._system_prompt,
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, core_agent: CoreAgent) -> AgentMember:
        """从字典重建 AgentMember，注入 CoreAgent（含 LLM/Registry）

        根据保存的 memory type 分发到正确的 from_dict，确保跨 memory 类型正确恢复。
        """
        member = cls(
            name=data["name"],
            role_desc=data["role_desc"],
            core_agent=core_agent,
            on_pass=data.get("on_pass"),
        )
        # 统一使用保存的 system_prompt，避免 make_agent 回调传入不同值
        sp = data.get("system_prompt", core_agent.system_prompt)
        member._system_prompt = sp

        # 根据保存的 type 恢复 memory
        mem_data = data.get("memory")
        if mem_data:
            mem_type = mem_data.get("type", "")
            mem_cls = cls.MEMORY_TYPE_MAP.get(mem_type)
            if mem_cls is None:
                logger.warning("unknown_memory_type", mem_type=mem_type)
            else:
                try:
                    restored = mem_cls.from_dict(mem_data, sp)
                    core_agent.memory = restored
                except Exception as e:
                    logger.warning("memory_restore_failed", mem_type=mem_type, error=str(e))
        return member

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
            logger.info("room_stop", text=user_message[:100])
            return []

        self.say("user", user_message)
        responses: list[tuple[str, str]] = []

        context = self.format_history(tail=15)

        # ── Phase 1: 举手 ──
        volunteers = self._volunteer_round(context)

        # 用户 @某Agent → 强制加入举手名单，排最前面，走执行模式
        user_mentions = self._parse_mentions(user_message)
        user_mentioned = set(user_mentions)
        for name in self._order:  # 按注册顺序
            if name in user_mentioned and name in self.members:
                # 已在 volunteers 里的移到最前，不在的插入最前
                if name in volunteers:
                    volunteers.remove(name)
                volunteers.insert(0, name)
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
            # 用户 @mention → 执行模式（force_reply + 含用户指令）
            is_user_direct = (speaker in user_mentioned and depth == 0)
            if is_user_direct:
                reply = member.chat(
                    context,
                    mention_context=f"【用户指定】用户 @了你并说：{user_message}",
                    force_reply=True,
                )
            else:
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
        """检测用户停止词 — 使用完整匹配，避免「停下来」误触「停」"""
        if STOP_MATCH_WHOLE_WORD:
            # 按空格/标点分词后精确匹配
            tokens = re.split(r'[\s，。！？、；：,\.!?;:\n]+', msg.strip())
            return any(w in tokens for w in STOP_WORDS)
        return any(w in msg for w in STOP_WORDS)

    def _volunteer_round(self, context: str) -> list[str]:
        """并行举手：每个 Agent 决定是否要发言

        使用 dict 暂存结果避免线程安全问题，最后按注册顺序过滤。
        """
        with ThreadPoolExecutor(max_workers=len(self._order) or 1) as executor:
            future_map = {
                executor.submit(self._decide_one, name, context): name
                for name in self._order
            }
            # 用 dict 暂存，key=name, value=是否举手
            decided: dict[str, bool] = {}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    decided[name] = future.result()
                except Exception as exc:
                    decided[name] = False
                    logger.warning("decision_error", member=name, error=str(exc))
        # 按注册顺序过滤出举手者，保持可预测性
        return [name for name in self._order if decided.get(name)]

    def _decide_one(self, name: str, context: str) -> bool:
        """单个 Agent 举手决策（可被子类重写）"""
        member = self.members.get(name)
        if not member:
            return False
        return member.decide(context)

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

    # ── 序列化（Session 持久化，方向 B） ─────────────

    def save(self, path: str) -> str:
        """将 Room 完整状态序列化到 JSON 文件

        保存内容：
          - 版本号
          - Room 名称
          - 所有成员的 name/role/memory/system_prompt
          - 对话历史

        Args:
            path: 输出文件路径

        Returns:
            JSON 字符串（同时写入文件）
        """
        data = {
            "version": 1,
            "name": self.name,
            "members": {
                name: member.to_dict()
                for name, member in self.members.items()
            },
            "history": [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "kind": m.kind,
                }
                for m in self.history
            ],
        }
        text = json.dumps(data, ensure_ascii=False, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("room_saved", path=path, members=len(self.members),
                     history=len(self.history))
        return text

    @classmethod
    def load(cls, path: str, llm: LLM,
             make_agent: Callable[[str, str, str, LLM], CoreAgent]) -> Room:
        """从 JSON 文件恢复 Room

        Args:
            path: JSON 文件路径
            llm: LLM 实例（用于重建 Agent）
            make_agent: 函数 (name, role_desc, system_prompt, llm) -> CoreAgent
                        每次调用创建 CoreAgent（含 Registry）

        Returns:
            Room 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 0)
        if version < 1:
            raise ValueError(f"不支持的 Session 版本: {version}")

        room = cls(name=data.get("name", "会议室"))

        for name, member_data in data.get("members", {}).items():
            role_desc = member_data.get("role_desc", "")
            system_prompt = member_data.get("system_prompt", name)
            core_agent = make_agent(name, role_desc, system_prompt, llm)

            member = AgentMember.from_dict(member_data, core_agent)
            room.register(member)

        # 恢复对话历史
        for h in data.get("history", []):
            msg = RoomMessage(
                sender=h["sender"],
                content=h["content"],
                timestamp=h.get("timestamp", 0.0),
                kind=h.get("kind", "public"),
            )
            room.history.append(msg)

        logger.info("room_loaded", path=path, members=len(room.members),
                     history=len(room.history))
        return room
