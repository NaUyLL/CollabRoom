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
        "max_steps": 12,
    },
    {
        "name": "开发者",
        "role": "你是团队的后端开发者。踏实、务实，关注实现细节和可维护性。",
        "style": "阅读设计文档，实现代码，快速验证。可读写文件、跑命令。修改后记得用 git commit 并提 PR。",
        "tools": dev_tools,
        "max_steps": 15,
    },
    {
        "name": "测试",
        "role": "你是团队的测试工程师。严谨、挑剔，关注质量、边界情况。",
        "style": "评估方案的可测试性，识别风险。可跑代码验证、搜索文件。",
        "tools": test_tools,
        "max_steps": 10,
    },
]

# ── 搭建 Room ──
room = Room(name="CollabRoom HTTP Demo")

for cfg in AGENTS:
    system_prompt = (
        f"【核心身份】\n{cfg['role']}\n\n"
        f"【工作方式】\n{cfg['style']}\n\n"
        f"【团队协作】\n"
        f"房间里有架构师、开发者、测试三个角色，按举手→发言的顺序协作。\n"
        f"你可以 @名字 直接对话其他成员（例如 @开发者 看看你的方案）。\n"
        f"每次只调用 1-2 个工具，别一次全调。\n"
        f"如果你觉得没什么可说的，回复 PASS。\n"
        f"看到别人说了什么之后再补充更有价值的观点，不要重复。\n\n"
        f"【工具说明】\n"
        f"- read_file/patch/write_file — 读写和修改项目源码\n"
        f"- search_files — 搜索文件内容和文件名\n"
        f"- terminal — 运行 shell 命令，包括 git add/commit/push、gh pr create --title \"标题\" --body \"说明\"（gh pr create 必加 --title/--body）\n"
        f"- execute_code — 运行 Python 代码做快速验证\n"
        f"修改代码后请提交 git commit 和 PR，让同事审查。"
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
