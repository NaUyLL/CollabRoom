"""Plan-then — 先出计划，再逐条执行，观察预规划能否减少无效 tool_call"""

from __future__ import annotations
import json, time
from typing import TYPE_CHECKING

from ..types import AgentResult, Step
from ..llm import system_msg, user_msg, assistant_msg, tool_msg, _tool_calls_to_dicts
from ..tool_calling import ToolCallingStrategy
from ..tool_calling.batch import BatchToolCalling
from . import PlanningStrategy

if TYPE_CHECKING:
    from ..llm import LLM
    from ..tool import Registry
    from ..memory import Memory

PLAN_PROMPT = """你是一个分步骤规划器。

在调用任何工具之前，请先制定一个详细的执行计划。
以 JSON 数组形式输出计划，数组每个元素是一个对象：
{"step": 步骤序号, "description": "这一步要做什么", "tool": "要用的工具名（如果不需工具则为null）", "expected": "预期结果"}

输出完计划后，不要执行任何工具，只需要输出计划本身。
用户将根据你的计划逐步骤执行。"""

class PlanThenStrategy(PlanningStrategy):
    """Plan-then：先由 LLM 出一个 JSON 步骤计划，再逐条执行"""

    system_prompt_prefix = "当用户提出问题时，先制定计划再执行。"

    def run(
        self,
        llm: LLM,
        registry: Registry,
        memory: Memory,
        system_prompt: str,
        user_message: str,
        max_steps: int,
        tool_calling: ToolCallingStrategy | None = None,
    ) -> AgentResult:
        tc = tool_calling or BatchToolCalling()
        steps: list[Step] = []
        start = time.time()
        total_tokens = 0
        total_calls = 0
        all_tool_defs = registry.get_definitions()

        # ── Phase 1: 制定计划（不启用工具） ──
        t0 = time.time()
        plan_msgs = [
            system_msg(system_prompt + "\n\n" + PLAN_PROMPT),
            user_msg(user_message),
        ]
        plan_resp = llm.chat(plan_msgs, tools=None)
        total_tokens += plan_resp.usage.total

        plan_text = plan_resp.content or ""
        steps.append(Step(
            role="think",
            content=f"[计划阶段]\n{plan_text[:500]}",
            token_usage=plan_resp.usage,
            elapsed_ms=(time.time() - t0) * 1000,
        ))

        plan_steps = self._parse_plan(plan_text)
        if not plan_steps:
            plan_steps = [{"step": 1, "description": user_message, "tool": None}]

        # ── Phase 2: 逐条执行 ──
        ctx_msgs = memory.get_context()
        ctx_msgs.append(user_msg(
            f"[任务] {user_message}\n"
            f"[执行计划]\n" + json.dumps(plan_steps, ensure_ascii=False, indent=2)
        ))

        for plan_step in plan_steps:
            step_desc = plan_step.get("description", f"步骤 {plan_step.get('step', '?')}")
            step_index = plan_step.get("step", plan_steps.index(plan_step) + 1)

            ctx_msgs.append(user_msg(
                f"\n--- 开始执行步骤 {step_index}: {step_desc} ---"
            ))

            for inner_n in range(1, max_steps + 1):
                t1 = time.time()
                tools_def = tc.filter_tools(all_tool_defs)
                resp = llm.chat(ctx_msgs, tools=tools_def or None)
                total_tokens += resp.usage.total

                steps.append(Step(
                    role="think",
                    content=f"[步骤{step_index}] {resp.content or ''}",
                    token_usage=resp.usage,
                    elapsed_ms=(time.time() - t1) * 1000,
                ))

                if not resp.tool_calls:
                    ctx_msgs.append(assistant_msg(resp.content or "(步骤完成)"))
                    break

                limited = tc.limit_calls(resp.tool_calls, all_tool_defs)
                calls_to_exec = limited if limited else resp.tool_calls[:1]

                ctx_msgs.append(assistant_msg(
                    resp.content,
                    tool_calls=_tool_calls_to_dicts(calls_to_exec),
                ))

                for tci in calls_to_exec:
                    total_calls += 1
                    t2 = time.time()
                    result = registry.execute(tci.name, tci.arguments)

                    steps.append(Step(role="act", tool_name=tci.name, tool_args=tci.arguments))
                    steps.append(Step(
                        role="observe",
                        tool_result=result,
                        elapsed_ms=(time.time() - t2) * 1000,
                    ))
                    ctx_msgs.append(tool_msg(tci.id, result))
            else:
                steps.append(Step(role="done", content=f"(步骤{step_index}达到最大步数限制)"))

        # ── Phase 3: 汇总 ──
        ctx_msgs.append(user_msg("所有步骤执行完毕。请给出最终答案。"))
        t3 = time.time()
        final_resp = llm.chat(ctx_msgs, tools=None)
        total_tokens += final_resp.usage.total
        final_answer = final_resp.content or ""

        steps.append(Step(
            role="think",
            content=f"[汇总] {final_answer[:300]}",
            token_usage=final_resp.usage,
            elapsed_ms=(time.time() - t3) * 1000,
        ))
        steps.append(Step(role="done", content=final_answer))

        memory.add("user", user_message)
        memory.add("assistant", final_answer)

        return AgentResult(
            final_answer=final_answer,
            steps=steps,
            total_tokens=total_tokens,
            total_tool_calls=total_calls,
            elapsed_ms=(time.time() - start) * 1000,
        )

    def _parse_plan(self, text: str) -> list[dict]:
        import re
        for pattern in [
            lambda t: json.loads(t) if isinstance(t, str) else None,
            lambda t: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL).group(1))
            if re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL) else None,
            lambda t: json.loads(re.search(r'(\[\s*\{.*\}\s*\])', t, re.DOTALL).group(1))
            if re.search(r'(\[\s*\{.*\}\s*\])', t, re.DOTALL) else None,
        ]:
            try:
                data = pattern(text)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        return []
