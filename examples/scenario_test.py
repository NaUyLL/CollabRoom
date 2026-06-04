#!/usr/bin/env python3
"""多 Agent 团队场景测试 — 不同角色组合的对话演示"""
import sys, json

from collabroom.core.llm import LLM
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.room import Room, AgentMember
from collabroom.core.tool_calling.batch import BatchToolCalling

# ── 一个空的工具注册表（纯对话场景） ──
class EmptyRegistry:
    def get_definitions(self): return []
    def execute(self, name, args): return "{}"

EMPTY = EmptyRegistry()

def make_agent(name: str, role_desc: str, system_prompt: str,
               max_steps: int = 3) -> AgentMember:
    """创建一个纯对话 AgentMember（无工具，只聊天）"""
    full_prompt = f"【角色设定】\n{system_prompt}\n\n你是 {name}。你的说话风格：{role_desc}"
    core = CoreAgent(
        llm=LLM(),
        registry=EMPTY,
        system_prompt=full_prompt,
        memory=NaiveMemory(full_prompt),
        tool_calling=BatchToolCalling(verbosity="short"),
        max_steps=max_steps,
    )
    return AgentMember(name=name, role_desc=role_desc, core_agent=core, on_pass="PASS")

def show_round(room: Room, user_msg: str):
    """执行一轮对话并打印完整过程"""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"👤 用户: {user_msg}")
    print(f"{sep}")
    responses = room.round(user_msg)
    for name, reply in responses:
        print(f"\n【{name}】")
        for line in reply.strip().split("\n"):
            print(f"  {line}")
    print()
    return responses

# ═══════════════════════════════════════════
# 场景 1: 技术决策 — 微服务 vs 单体
# ═══════════════════════════════════════════

def scenario_1():
    print("\n" + "█" * 60)
    print("  ╔══════════════════════════════════════╗")
    print("  ║  场景 1: 架构之争 — 各执己见        ║")
    print("  ║  成员: 微服务派 vs 单体派 vs 运维    ║")
    print("  ╚══════════════════════════════════════╝")
    print("█" * 60)

    room = Room("架构评审会")
    room.register(make_agent(
        "微服务架构师", "喜欢微服务，说话带架构术语",
        "你是坚定的微服务拥护者。\"任何问题都可以通过加一层微服务解决\"是你的信条。"
        "接口要用 gRPC，服务要容器化，必须上 K8s。喜欢拉数据说话。"
    ))
    room.register(make_agent(
        "单体派老兵", "反对过度设计，说话接地气",
        "你在这个行业干了 15 年。你见过太多微服务项目搞砸的案例。"
        "\"能用单体解决的问题就不要搞复杂\"是你的原则。说话直接，喜欢举反例。"
    ))
    room.register(make_agent(
        "运维负责人", "实用主义，关注成本和稳定性",
        "你负责上线和运维。谁方案好你站谁，但你最关心：部署成本、监控、排查难度。"
        "经常问\"出了问题谁背锅\"。务实，讨厌画饼。"
    ))

    show_round(room,
        "公司想做一个内部工单系统，我有点纠结该用微服务还是单体架构。大家有什么看法？")
    return room

# ═══════════════════════════════════════════
# 场景 2: 三位不同性格的产品讨论
# ═══════════════════════════════════════════

def scenario_2():
    print("\n" + "█" * 60)
    print("  ╔══════════════════════════════════════╗")
    print("  ║  场景 2: 产品讨论会 — 性格碰撞      ║")
    print("  ║  成员: 产品经理 vs 技术总监 vs CEO   ║")
    print("  ╚══════════════════════════════════════╝")
    print("█" * 60)

    room = Room("产品评审会")
    room.register(make_agent(
        "产品经理", "满脑子新功能，永远在加需求",
        "你是典型的产品经理。脑子里有无数新功能点子，每次开会都能想出 10 个新需求。"
        "觉得\"这个功能很简单\"是口头禅。重视用户体验和竞品对标。精力旺盛。"
    ))
    room.register(make_agent(
        "技术总监", "现实主义，总在掐灭产品经理的幻想",
        "你是技术总监，负责评估可行性。产品经理说的每个功能你都要反问："
        "\"后端改多少？\n\"\"QA 要测多久？\n\"\"历史数据兼容吗？\""
        "你务实、直接，遇到不合理的需求会怼回去。但内心其实想把产品做好。"
    ))
    room.register(make_agent(
        "CEO", "结果导向，催进度，喜欢画大饼",
        "你是公司的创始人兼 CEO。你不懂技术细节，但看重结果和时间节点。"
        "口头禅是\"下个月能上线吗\"和\"竞品已经有了\"。喜欢鼓励大家\"All in\"。"
        "说话有感染力但有时候不切实际。"
    ))

    show_round(room,
        "我们新版本要做一个 AI 聊天助手功能，大家讨论一下怎么做？")
    return room

# ═══════════════════════════════════════════
# 场景 3: 挖坑三人组
# ═══════════════════════════════════════════

def scenario_3():
    print("\n" + "█" * 60)
    print("  ╔══════════════════════════════════════╗")
    print("  ║  场景 3: 挖坑三人组 — 谁都不靠谱    ║")
    print("  ║  成员: 画饼侠 vs 撤退哥 vs 吃瓜群众  ║")
    print("  ╚══════════════════════════════════════╝")
    print("█" * 60)

    room = Room("项目启动会")
    room.register(make_agent(
        "画饼侠", "永远说'没问题'，实际上什么都没想清楚",
        "你是一个永远说\"没问题\"的项目负责人。不管什么需求你都先接下来说能做，"
        "工期永远说\"两周\"。口头禅：\"这个问题不大\"、\"我们团队经验丰富\"。"
        "先答应再说的类型。说话充满迷之自信。"
    ))
    room.register(make_agent(
        "撤退哥", "永远在说风险和撤退方案",
        "你是一个永远看到风险的 QA 负责人。任何项目你都能找到撤退的理由。"
        "\"这个风险太大\"是你开头的惯用语。你擅长从各种角度指出为什么这个事情做不成。"
        "口头禅：\"要不我们先调研一下？\"、\"我觉得时机还不成熟\"。"
    ))
    room.register(make_agent(
        "吃瓜群众", "中立，但偶尔补刀",
        "你是一个旁观者，对任何事情都保持中立。你不站队，但偶尔会补一刀让人尴尬。"
        "说话简短，经常用\"嗯\"\"确实\"\"有道理\"开头。"
        "有时候会问一些直击灵魂的问题让两边都下不来台。"
    ))

    show_round(room,
        "老板说我们要三个月做一个比肩抖音的产品，大家觉得怎么样？")
    return room

# ═══════════════════════════════════════════
# 运行所有场景
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  Mason 多 Agent 团队 — 场景测试")
    print("  3 个场景 | 每个 3 个 Agent | 纯对话无工具")
    print("█" * 60)

    rooms = []
    for i, fn in enumerate([scenario_1, scenario_2, scenario_3], 1):
        print(f"\n{'='*60}")
        print(f"  开始场景 {i}...")
        print(f"{'='*60}")
        r = fn()
        rooms.append(r)

    # 全局统计
    print(f"\n{'='*60}")
    print(f"  📊 全局统计")
    print(f"{'='*60}")
    for i, r in enumerate(rooms, 1):
        print(f"  场景 {i}: {r.name} — {len(r.history)} 条消息")
        for m in r.history:
            tag = f"DM" if m.kind == "dm" else m.sender
            print(f"    [{tag[:8]}]: {m.content[:60]}...")
    print()
