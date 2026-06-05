#!/usr/bin/env python3
"""HTTP Demo — 启动一个可通过 HTTP 访问的 CollabRoom

用法:
  python3 examples/http_demo.py
  # 浏览器打开 http://localhost:8765
  # 或 curl:
  #   curl -X POST http://localhost:8765/chat \\
  #     -H 'Content-Type: application/json' \\
  #     -d '{"message": "你好，Agent 们"}'
"""

import sys
sys.path.insert(0, "/opt/data/collabroom")

from collabroom.core.llm import LLM
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.system_tools import register_all, register_defaults
from collabroom.room import Room, AgentMember
from collabroom.core.tool_calling.batch import BatchToolCalling
from collabroom.gateway.http import HTTPGateway
from collabroom.core.tool import Registry

# ── 创建 LLM 和工具 ──
llm = LLM()

arch_tools = Registry()
register_defaults(arch_tools, exclude={"terminal", "execute_code", "patch"})

dev_tools = Registry()
register_all(dev_tools)

test_tools = Registry()
register_defaults(test_tools, exclude={"patch"})

# ── Agent 配置 ──
AGENTS = [
    {
        "name": "架构师",
        "role": "你是团队的系统架构师。严谨、有条理，擅长全局设计和技术选型。",
        "style": "先分析需求，阅读现有代码了解架构，然后设计方案。可搜索代码、写设计文档。",
        "tools": arch_tools,
        "max_steps": 6,
    },
    {
        "name": "开发者",
        "role": "你是团队的后端开发者。踏实、务实，关注实现细节和可维护性。",
        "style": "阅读设计文档，实现代码，快速验证。可读写文件、跑命令。",
        "tools": dev_tools,
        "max_steps": 8,
    },
    {
        "name": "测试",
        "role": "你是团队的测试工程师。严谨、挑剔，关注质量、边界情况。",
        "style": "评估方案的可测试性，识别风险。可跑代码验证、搜索文件。",
        "tools": test_tools,
        "max_steps": 6,
    },
]

# ── 搭建 Room ──
room = Room(name="CollabRoom HTTP Demo")

for cfg in AGENTS:
    system_prompt = (
        f"【核心身份】\n{cfg['role']}\n\n"
        f"【工作方式】\n{cfg['style']}\n\n"
        f"【团队协作】\n"
        f"房间里有架构师、开发者、测试三个角色。\n"
        f"你可以 @名字 直接对话其他成员。\n"
        f"每次只调用 1-2 个工具。\n"
        f"如果你觉得没什么可说的，回复 PASS。"
    )
    core = CoreAgent(
        llm=llm,
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

# ── 启动 HTTP Gateway ──
if __name__ == "__main__":
    gw = HTTPGateway(room, host="0.0.0.0", port=8765)
    gw.run()
