#!/usr/bin/env python3
"""系统工具团队演示 — Agent 通过真实 OS 工具协作开发

三个 Agent 拥有真实的文件系统/终端工具：
  架构师   → read_file, write_file, search_files
  开发者   → read_file, write_file, patch, terminal, execute_code
  测试     → read_file, write_file, terminal, execute_code, search_files

比原本的虚工具（design_diagram/write_spec/implement）强在：
  ✓ 真的能读项目代码
  ✓ 真的能写文件到磁盘
  ✓ 真的能跑测试命令
  ✓ 真的能执行 Python 验证
"""

import sys
sys.path.insert(0, "/opt/data/collabroom")

from collabroom.core.llm import LLM
from collabroom.core.tool import Registry
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.system_tools import register_all, register_defaults
from collabroom.room import Room, AgentMember
from collabroom.gateway.cli import run_gateway
from collabroom.core.tool_calling.batch import BatchToolCalling

# ═══════════════════════════════════════════
# 工具注册（每个角色有不同工具集）
# ═══════════════════════════════════════════

def make_architect_tools() -> Registry:
    """架构师：设计相关，不需要跑代码"""
    r = Registry()
    register_defaults(r, exclude={"terminal", "execute_code", "patch"})
    return r

def make_developer_tools() -> Registry:
    """开发者：全栈工具"""
    r = Registry()
    register_all(r)
    return r

def make_tester_tools() -> Registry:
    """测试：读、写、跑、搜"""
    r = Registry()
    register_defaults(r, exclude={"patch"})
    return r

# ═══════════════════════════════════════════
# Agent 配置
# ═══════════════════════════════════════════

LLM_INSTANCE = LLM()

AGENTS = [
    {
        "name": "架构师",
        "role": "你是团队的系统架构师。严谨、有条理，擅长全局设计。",
        "style": (
            "用户提出需求后，先用 search_files 了解现有项目结构，"
            "然后写设计方案到文件（write_file），再 @开发者 安排实现。"
            "设计的文件放到 design/ 目录下。"
        ),
        "tools": make_architect_tools(),
        "max_steps": 5,
    },
    {
        "name": "开发者",
        "role": "你是团队的后端开发者。踏实、务实，注重代码质量。",
        "style": (
            "收到架构师（@架构师）的设计后，用 read_file 看设计文档，"
            "然后用 write_file 实现代码，写完用 patch 修改，"
            "最后用 terminal 或 execute_code 做快速验证。"
            "遇到不清楚的地方 @架构师 追问，完成后 @测试 通知测试。"
        ),
        "tools": make_developer_tools(),
        "max_steps": 8,
    },
    {
        "name": "测试",
        "role": "你是团队的测试工程师。严谨、挑剔，不放过任何边界条件。",
        "style": (
            "收到开发完成的消息后，用 read_file 看代码，"
            "用 terminal 跑测试命令或用 execute_code 执行验证脚本。"
            "发现 bug 就 write_file 写入测试报告，"
            "然后 @开发者 指出问题。"
        ),
        "tools": make_tester_tools(),
        "max_steps": 6,
    },
]

# ═══════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════

def main():
    room = Room(name="CollabRoom 开发团队（真实工具版）")

    for cfg in AGENTS:
        system_prompt = (
            f"【核心身份】\n"
            f"{cfg['role']}\n\n"
            f"【工作方式】\n"
            f"{cfg['style']}\n\n"
            f"【团队协作】\n"
            f"这是团队协作场景。房间里有架构师、开发者、测试三个角色。\n"
            f"你的工具有 read_file/write_file/search_files/terminal/execute_code/patch。\n"
            f"这些工具操作的是真实的文件系统 —— 你写的内容会实际保存到磁盘。\n"
            f"你可以 @名字 直接对话其他成员。\n"
            f"每次只调用 1-2 个工具，调用完给出结论。"
        )

        core = CoreAgent(
            llm=LLM_INSTANCE,
            registry=cfg["tools"],
            system_prompt=system_prompt,
            max_steps=cfg["max_steps"],
            memory=NaiveMemory(system_prompt),
            tool_calling=BatchToolCalling(verbosity="short"),
        )
        member = AgentMember(
            name=cfg["name"],
            role_desc=cfg["role"],
            core_agent=core,
            on_pass="PASS",
        )
        room.register(member)

    print(f"\n🚀 启动团队协作 — 所有工具操作真实文件系统！")
    print(f"   试试说：帮我在当前目录做一个 Python 小项目\n")
    run_gateway(room)


if __name__ == "__main__":
    main()
