"""ReAct — 标准 think→act→observe 循环

优化（参照 Hermes + Codex CLI）：
  1. 并行工具执行 — 多个 tool_call 同时跑（Codex 风格）
  2. 卡死检测 — 连续 N 步相同的工具+参数，自动打断
  3. 错误升级 — 工具持续报错时告诉 LLM 换策略
  4. 更大的 max_steps — 默认 25（Hermes 90 太多，25 够用）
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import time, json, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..types import AgentResult, Step
from ..llm import user_msg, assistant_msg, tool_msg, _tool_calls_to_dicts
from ..tool_calling import ToolCallingStrategy
from ..tool_calling.batch import BatchToolCalling
from . import PlanningStrategy

if TYPE_CHECKING:
    from ..llm import LLM
    from ..tool import Registry
    from ..memory import Memory

# ── 卡死检测 ──
STUCK_THRESHOLD = 4        # 连续 N 步相同工具+摘要参数 → 判定卡死
ERROR_THRESHOLD = 3        # 连续 N 步所有工具都报错 → 升级提示

# ── 工具摘要：取参数 key 做对比用，不去比较完整参数值 ──
def _arg_signature(tool_calls: list) -> str:
    """生成工具调用的签名用于卡死检测（只对比工具名和参数 key 集合）"""
    parts = []
    for tc in tool_calls:
        args_keys = sorted(tc.arguments.keys()) if hasattr(tc, 'arguments') else []
        parts.append(f"{tc.name}({','.join(args_keys)})")
    return "|".join(parts)


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

        # ── 卡死 / 错误追踪 ──
        last_sig: str | None = None       # 上一步的工具调用签名
        consecutive_same = 0               # 连续相同签名的步数
        consecutive_errors = 0             # 连续所有工具都报错的步数

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

            # ── 卡死检测 ──
            sig = _arg_signature(calls_to_exec)
            if sig == last_sig:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_sig = sig

            if consecutive_same >= STUCK_THRESHOLD:
                # 告诉 LLM 卡死了，让它换策略
                stuck_msg = (
                    f"⚠️ 检测到卡死：你已连续 {consecutive_same} 步调用相同的工具：{sig}。"
                    f"结果是一样的。请换一种方式完成目标，不要重复同样的调用。"
                )
                msgs.append(assistant_msg(resp.content, tool_calls=_tool_calls_to_dicts(calls_to_exec)))
                msgs.append(user_msg(stuck_msg))
                steps.append(Step(role="observe", tool_result=stuck_msg))
                # 重置计数器，避免连续触发
                consecutive_same = 0
                continue

            # ── ACT + OBSERVE ──
            msgs.append(assistant_msg(
                resp.content,
                tool_calls=_tool_calls_to_dicts(calls_to_exec),
            ))

            # ── 并行执行工具 ──
            if len(calls_to_exec) > 1 and tc.supports_parallel:
                # Codex 风格：并行跑所有 tool_call
                results = self._execute_parallel(registry, calls_to_exec)
            else:
                # 逐个执行（传统 ReAct 风格）
                results = self._execute_sequential(registry, calls_to_exec)

            total_calls += len(calls_to_exec)

            # ── 处理结果 ──
            all_errors = True
            for tc_item, result in zip(calls_to_exec, results):
                is_error = result.startswith('{"error"') or result.startswith("{\"error\"")
                if not is_error:
                    all_errors = False

                steps.append(Step(
                    role="act",
                    tool_name=tc_item.name,
                    tool_args=tc_item.arguments,
                ))
                steps.append(Step(
                    role="observe",
                    tool_result=result[:500],
                    elapsed_ms=0,
                ))
                msgs.append(tool_msg(tc_item.id, result))

            # ── 错误升级 ──
            if all_errors:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            if consecutive_errors >= ERROR_THRESHOLD:
                hint = (
                    f"⚠️ 工具连续 {consecutive_errors} 步全部报错。"
                    f"请检查参数是否正确，换一种方式操作，不要重复之前的调用。"
                )
                msgs.append(user_msg(hint))
                consecutive_errors = 0

            last_result = results[-1] if results else None

            # 如果有被截断的调用，告诉 LLM 继续
            if extra > 0:
                dropped_names = ", ".join(
                    tc_item.name for tc_item in resp.tool_calls[len(calls_to_exec):]
                )
                msgs.append(user_msg(
                    f"(还有 {extra} 个未执行的调用: {dropped_names}。如有需要请继续。)"
                ))
        else:
            # 超步数时，把已有结果拼成回复，不输出生硬提示
            final_parts = []
            for s in steps:
                if s.role == "observe" and s.tool_result:
                    final_parts.append(s.tool_result[:200])
            if final_parts:
                summary = "我查到了以下信息：\n" + "\n".join(final_parts)
            else:
                summary = "要处理的内容比较多，时间不太够，我先输出目前的结果。"
            steps.append(Step(role="done", content=summary))

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

    # ── 并行执行 ──

    def _execute_parallel(self, registry: Registry,
                          tool_calls: list) -> list[str]:
        """并行执行多个 tool_call（ThreadPoolExecutor）"""
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {
                pool.submit(registry.execute, tc.name, tc.arguments): i
                for i, tc in enumerate(tool_calls)
            }
            results: list[str] = [""] * len(tool_calls)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = json.dumps({
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }, ensure_ascii=False)
            return results

    def _execute_sequential(self, registry: Registry,
                            tool_calls: list) -> list[str]:
        """逐个执行（传统模式）"""
        results = []
        for tc_item in tool_calls:
            result = registry.execute(tc_item.name, tc_item.arguments)
            results.append(result)
        return results
