"""Prompt Builder — 结构化分层 system prompt 构建器

按以下层次组合：
  # 身份层 — 你是谁
  # 规则层 — 怎么说话、怎么用工具
  # 技能层 — role-specific 知识注入
  # 工具层 — 可用工具说明
  # 记忆层 — 之前对话的上下文参考

用法:
  builder = PromptBuilder(name="老球迷")
  builder.add_identity("你是一个资深足球迷...")
  builder.add_rule("用 @名字 提及他人")
  builder.add_skill(knowledge_skill)
  prompt = builder.build()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import datetime


@dataclass
class Skill:
    """一个可复用的技能/知识包"""
    name: str
    description: str
    instructions: str = ""          # 注入到 system prompt 的指令
    tool_defs: list[dict] = field(default_factory=list)  # 额外工具定义（JSON Schema）
    constraints: list[str] = field(default_factory=list)  # 约束规则

    def to_prompt_block(self) -> str:
        """渲染为提示文本块"""
        parts = [f"[技能: {self.name}] {self.description}"]
        if self.instructions:
            parts.append(self.instructions)
        if self.constraints:
            parts.extend(f"- {c}" for c in self.constraints)
        if self.tool_defs:
            names = [t.get("function", {}).get("name", "?") for t in self.tool_defs]
            parts.append(f"额外工具: {', '.join(names)}")
        return "\n".join(parts)


class PromptBuilder:
    """system prompt 构建器 — 分段组合，自动注入上下文"""

    def __init__(self, name: str = ""):
        self.name = name
        self._identities: list[str] = []
        self._rules: list[str] = []
        self._skills: list[Skill] = []
        self._context_hints: list[str] = []

        # 默认规则
        self._rules.extend([
            "不要用 @符号，直接自然说话，像真人聊天一样",
            "想对某个人说话时，直接说他的名字，例如「TechLead，你觉得呢？」",
            "不要称呼自己，也不要称呼用户",
            "没想法就说 PASS，不要强行接话",
            "需要实时信息时用工具查询，把搜索结果详细告诉房间里的其他人",
            "引用搜索结果时尽可能详细、完整，让对方获得充足信息",
            "如果其他 agent 已经回答了用户的同一个问题，你不需要重复相同的内容",
            "但如果是用户提出的新问题或新的话题，可以自由回答或补充",
        ])

    def add_identity(self, text: str):
        """添加身份描述"""
        self._identities.append(text)

    def add_rule(self, rule: str):
        """添加规则"""
        self._rules.append(rule)

    def add_skill(self, skill: Skill):
        """添加技能包"""
        self._skills.append(skill)

    def add_context_hint(self, hint: str):
        """添加上下文提示（时间、场景等动态信息）"""
        self._context_hints.append(hint)

    # ── 技能工具定义 ──────────────────────────────────

    def get_extra_tool_defs(self) -> list[dict]:
        """获取所有技能声明的额外工具定义"""
        defs = []
        seen = set()
        for skill in self._skills:
            for d in skill.tool_defs:
                name = d.get("function", {}).get("name", "")
                if name not in seen:
                    seen.add(name)
                    defs.append(d)
        return defs

    # ── 构建 ──────────────────────────────────────────

    def build(self) -> str:
        """组合所有层，返回完整 system prompt"""
        sections = []

        # ── 身份 ──
        if self._identities:
            sections.append("# 你的身份")
            sections.extend(self._identities)

        # ── 动态上下文 ──
        if self._context_hints:
            sections.append("# 当前状态")
            sections.extend(self._context_hints)

        # ── 规则 ──
        if self._rules:
            sections.append("# 对话规则")
            for r in self._rules:
                sections.append(f"- {r}")

        # ── 技能 ──
        if self._skills:
            sections.append("# 你的技能")
            for skill in self._skills:
                sections.append(skill.to_prompt_block())

        return "\n\n".join(sections)

    def build_for_reply(self, context: str, followup: bool = False,
                        force_reply: bool = False) -> str:
        """构建一轮对话的用户消息 prompt"""
        parts = [f"聊天内容：\n{context}"]

        if force_reply:
            parts.append("\n大家都在等你说话，随便说两句吧。")
        elif followup:
            parts.append("\n看了别人说的，有什么想补充的吗？")
        else:
            parts.append("\n你有什么想说的？")

        return "\n".join(parts)


# ── 默认技能工厂 ─────────────────────────────────────

def time_aware_skill() -> Skill:
    """时间感知技能 — 让 agent 知道当前时间"""
    now = datetime.datetime.now()
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    return Skill(
        name="time_awareness",
        description="当前时间感知",
        instructions=(
            f"当前时间：{now.year}年{now.month}月{now.day}日 "
            f"星期{weekday_cn} {now.strftime('%H:%M')}"
        ),
    )


def web_search_skill() -> Skill:
    """联网搜索技能"""
    return Skill(
        name="web_search",
        description="实时信息查询",
        instructions=(
            "需要查文档、技术方案、实时信息时，使用 search_web 工具。\n"
            "查询时用中文关键词，结果会从中文搜索引擎获取。\n"
            "得到搜索结果后，把**完整信息**告诉房间里的人，包括关键细节。\n"
            "如果搜索结果有多个要点，逐一列出。"
        ),
    )


def mention_skill() -> Skill:
    """@提及技能 — 让 agent 理解被 @了要回复"""
    return Skill(
        name="mention_awareness",
        description="提及感知",
        instructions=(
            "当有人直接叫你的名字时，表示在叫你回复。\n"
            "被点名了不论有没有想法都应该接话。"
        ),
    )


