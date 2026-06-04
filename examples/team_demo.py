#!/usr/bin/env python3
"""团队演示：架构师 + 开发者 + 测试工程师 在一个房间里讨论"""

import sys, json

from collabroom.core.llm import LLM
from collabroom.core.tool import Tool, Registry
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.room import Room, AgentMember
from collabroom.gateway.cli import run_gateway
from collabroom.core.tool_calling.batch import BatchToolCalling

# ═══════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════

def design_diagram(title: str, content: str) -> str:
    return json.dumps({"diagram": f"【{title}】\n{content}", "status": "已绘制"}, ensure_ascii=False)

def write_spec(module: str, content: str) -> str:
    return json.dumps({"spec": f"《{module} 技术规格》\n{content}", "status": "已归档"}, ensure_ascii=False)

def implement(module: str, code: str) -> str:
    return json.dumps({"module": module, "code": code, "status": "已实现"}, ensure_ascii=False)

def review_code(code: str) -> str:
    issues = []
    if "print" in code: issues.append("有 print 调试语句")
    if len(code) > 300: issues.append("函数过长")
    if "TODO" in code: issues.append("有未完成的 TODO")
    if not issues: issues.append("代码质量良好")
    return json.dumps({"issues": issues}, ensure_ascii=False)

def write_testcase(module: str, cases: str) -> str:
    return json.dumps({"module": module, "test_cases": cases, "status": "已编写"}, ensure_ascii=False)

def run_test(cases: str) -> str:
    return json.dumps({"passed": 8, "failed": 1, "total": 9, "details": "1 个边界条件失败"}, ensure_ascii=False)

def check_edge_cases(module: str, description: str) -> str:
    return json.dumps({
        "module": module, "risks": ["输入为空", "高并发冲突"],
        "recommendation": "需要加防御性校验",
    }, ensure_ascii=False)

def make_all_tools() -> Registry:
    r = Registry()
    for t in [
        Tool("design_diagram", "绘制系统架构图", {"type":"object","properties":{"title":{"type":"string"},"content":{"type":"string"}},"required":["title","content"]}, design_diagram),
        Tool("write_spec", "编写技术规格文档", {"type":"object","properties":{"module":{"type":"string"},"content":{"type":"string"}},"required":["module","content"]}, write_spec),
        Tool("implement", "实现模块代码", {"type":"object","properties":{"module":{"type":"string"},"code":{"type":"string"}},"required":["module","code"]}, implement),
        Tool("review_code", "审查代码质量", {"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}, review_code),
        Tool("write_testcase", "编写测试用例", {"type":"object","properties":{"module":{"type":"string"},"cases":{"type":"string"}},"required":["module","cases"]}, write_testcase),
        Tool("run_test", "运行测试并获取结果", {"type":"object","properties":{"cases":{"type":"string"}},"required":["cases"]}, run_test),
        Tool("check_edge_cases", "检查边界条件风险", {"type":"object","properties":{"module":{"type":"string"},"description":{"type":"string"}},"required":["module","description"]}, check_edge_cases),
    ]:
        r.register(t)
    return r

# ═══════════════════════════════════════════
# 角色定义
# ═══════════════════════════════════════════

LLM = LLM()
ALL_TOOLS = make_all_tools()

AGENTS = [
    {
        "name": "架构师",
        "role": (
            "你是团队的系统架构师。你严谨、有条理，擅长全局设计。\n"
            "用户提出需求时，你先画架构图、拆模块、写技术规格。\n"
            "工作方式是先说设计方案，然后@开发者 安排实现，@测试 要求验证。\n"
        ),
        "prompt": "你是一个资深的系统架构师。你是团队的引领者：用户提出需求后，你先用 design_diagram 设计架构，用 write_spec 写规格文档，然后安排开发者去实现。",
    },
    {
        "name": "开发者",
        "role": (
            "你是团队的后端开发者。你踏实、务实，注重代码质量。\n"
            "架构师给出设计后，你用 implement 实现代码。\n"
            "写完后用 review_code 做自审。\n"
            "如果架构师的设计有不清楚的地方，@架构师 追问。\n"
            "实现完后 @测试 通知可以开始测试了。\n"
        ),
        "prompt": "你是一个经验丰富的后端开发者。看到架构师的设计后，你用 implement 实现代码，写完用 review_code 自审。可以@架构师 追问细节，写完后 @测试 说一声。",
    },
    {
        "name": "测试",
        "role": (
            "你是团队的测试工程师。你严谨、挑剔，不放过任何一个边界条件。\n"
            "开发者交付代码后，你用 write_testcase 写测试用例，用 run_test 跑测试，用 check_edge_cases 查边界。\n"
            "发现 bug 或风险时，@开发者 指出问题并要求修复。\n"
            "如果觉得架构设计有隐患，也可以 @架构师 质疑。\n"
        ),
        "prompt": "你是一位严谨的测试工程师。看到设计和代码后，你用 write_testcase 写用例，用 run_test 跑测试，用 check_edge_cases 检查边界。发现问题就 @开发者 要求修复。",
    },
]

def main():
    room = Room(name="mason 研发团队")

    for cfg in AGENTS:
        system = (
            f"{cfg['prompt']}\n\n"
            f"【你的风格】\n{cfg['role']}\n\n"
            f"这是团队协作场景，房间里有架构师、开发者、测试。"
            f"你可以 @名字 直接对话其他成员。"
        )
        core = CoreAgent(
            llm=LLM,
            registry=ALL_TOOLS,
            system_prompt=system,
            max_steps=3,
            memory=NaiveMemory(system),
            tool_calling=BatchToolCalling(verbosity="short"),
        )
        member = AgentMember(
            name=cfg["name"],
            role_desc=cfg["role"],
            core_agent=core,
            on_pass="PASS",
        )
        room.register(member)

    run_gateway(room)

if __name__ == "__main__":
    main()
