"""ReAct — 想到就干，标准的 think→act→observe 循环，是 baseline"""

from __future__ import annotations
from typing import TYPE_CHECKING
import time

from ..types import AgentResult, Step
from ..llm import user_msg, assistant_msg, tool_msg, _tool_calls_to_dicts
from ..tool_calling import ToolCallingStrategy
from ..tool_calling.batch import BatchToolCalling
from . import PlanningStrategy

if TYPE_CHECKING:
    from ..llm import LLM
    from ..tool import Registry
    from ..memory import Memory

class ReActStrategy(PlanningStrategy):
    """ReAct：标准 think → act → observe 循环 — tool_calling 可插拔"""

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
        all_tool_defs = registry.get_definitions()
        last_result: str | None = None

        for step_n in range(1, max_steps + 1):
            t0 = time.time()

            # ── ToolCalling: 筛选本轮的工具定义 ──
            tools_def = tc.filter_tools(all_tool_defs, last_result)

            # ── THINK ──
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

            # ── ToolCalling: 限制实际执行的调用数 ──
            limited = tc.limit_calls(resp.tool_calls, all_tool_defs)
            calls_to_exec = limited if limited else resp.tool_calls[:1]
            extra = len(resp.tool_calls) - len(calls_to_exec)

            # ── ACT + OBSERVE ──
            msgs.append(assistant_msg(
                resp.content,
                tool_calls=_tool_calls_to_dicts(calls_to_exec),
            ))

            for tc_item in calls_to_exec:
                total_calls += 1
                t1 = time.time()
                result = registry.execute(tc_item.name, tc_item.arguments)
                last_result = result

                steps.append(Step(
                    role="act",
                    tool_name=tc_item.name,
                    tool_args=tc_item.arguments,
                ))
                steps.append(Step(
                    role="observe",
                    tool_result=result,
                    elapsed_ms=(time.time() - t1) * 1000,
                ))
                msgs.append(tool_msg(tc_item.id, result))

            # 如果有被截断的调用，告诉 LLM 继续
            if extra > 0:
                dropped_names = ", ".join(
                    tc_item.name for tc_item in resp.tool_calls[len(calls_to_exec):]
                )
                msgs.append(user_msg(
                    f"(还有 {extra} 个未执行的调用: {dropped_names}。如有需要请继续。)"
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
