"""Reflect — 每步后自省"这一步正确吗"，观察自省能否提高准确率"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING

from ..types import AgentResult, Step
from ..llm import user_msg, assistant_msg, tool_msg, _tool_calls_to_dicts
from ..tool_calling import ToolCallingStrategy
from ..tool_calling.batch import BatchToolCalling
from . import PlanningStrategy

if TYPE_CHECKING:
    from ..llm import LLM
    from ..tool import Registry
    from ..memory import Memory

REFLECT_PROMPT = """请检视上一步的执行结果：

1. 工具调用是否正确（参数、返回值）？
2. 结果是否符合预期？
3. 如果正确，回答 "✓ 继续执行"
4. 如果发现问题，请指出问题并给出修正方案"""

class ReflectStrategy(PlanningStrategy):
    """Reflect：每个 act→observe 后加一步自省检查"""

    system_prompt_prefix = "每执行一步工具后，要检视自己的执行结果是否准确。"

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
        msgs = memory.get_context()
        msgs.append(user_msg(user_message))

        steps: list[Step] = []
        start = time.time()
        total_tokens = 0
        total_calls = 0
        reflect_calls = 0
        reflect_tokens = 0
        corrections = 0
        all_tool_defs = registry.get_definitions()
        last_result: str | None = None

        for step_n in range(1, max_steps + 1):
            t0 = time.time()

            # ── THINK ──
            tools_def = tc.filter_tools(all_tool_defs, last_result)
            resp = llm.chat(msgs, tools=tools_def or None)
            total_tokens += resp.usage.total

            steps.append(Step(
                role="think",
                content=resp.content or "",
                token_usage=resp.usage,
                elapsed_ms=(time.time() - t0) * 1000,
            ))

            if not resp.tool_calls:
                steps.append(Step(role="done", content=resp.content or ""))
                break

            # ── ACT + OBSERVE ──
            limited = tc.limit_calls(resp.tool_calls, all_tool_defs)
            calls_to_exec = limited if limited else resp.tool_calls[:1]

            msgs.append(assistant_msg(
                resp.content,
                tool_calls=_tool_calls_to_dicts(calls_to_exec),
            ))

            for tci in calls_to_exec:
                total_calls += 1
                t1 = time.time()
                result = registry.execute(tci.name, tci.arguments)
                last_result = result

                steps.append(Step(role="act", tool_name=tci.name, tool_args=tci.arguments))
                steps.append(Step(
                    role="observe",
                    tool_result=result,
                    elapsed_ms=(time.time() - t1) * 1000,
                ))
                msgs.append(tool_msg(tci.id, result))

            # ── REFLECT ──
            t2 = time.time()
            reflect_msgs = msgs.copy()
            reflect_msgs.append(user_msg(REFLECT_PROMPT))
            reflect_resp = llm.chat(reflect_msgs, tools=None)
            total_tokens += reflect_resp.usage.total
            reflect_tokens += reflect_resp.usage.total
            reflect_calls += 1

            reflect_content = reflect_resp.content or ""
            steps.append(Step(
                role="think",
                content=f"[反省] {reflect_content[:200]}",
                token_usage=reflect_resp.usage,
                elapsed_ms=(time.time() - t2) * 1000,
            ))

            if "错误" in reflect_content or "不对" in reflect_content or "修正" in reflect_content:
                corrections += 1
                msgs.append(user_msg(
                    "你刚刚反馈了问题。请根据你的分析修正执行。"
                ))
        else:
            steps.append(Step(role="done", content="(达到最大步数限制)"))

        final_answer = steps[-1].content if steps else ""

        memory.add("user", user_message)
        memory.add("assistant", final_answer)

        return AgentResult(
            final_answer=final_answer,
            steps=steps,
            total_tokens=total_tokens,
            total_tool_calls=total_calls,
            elapsed_ms=(time.time() - start) * 1000,
        )
