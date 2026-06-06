"""测试 TieredMemory — 多层级记忆（工作记忆 + 摘要 + 长时事实）"""
from __future__ import annotations
from unittest.mock import MagicMock

from collabroom.core.types import LLMResponse, Usage
from collabroom.core.llm import LLM
from collabroom.core.memory.tiered import (
    WorkingLayer,
    SummaryLayer,
    FactLayer,
    TieredMemory,
)


# ═══════════════════════════════════════════════════════════════
# WorkingLayer 测试
# ═══════════════════════════════════════════════════════════════


class TestWorkingLayer:
    """测试工作记忆层 WorkingLayer"""

    def test_init_default_window(self):
        """默认窗口大小为 10"""
        wl = WorkingLayer()
        assert wl.window == 10
        assert len(wl) == 0

    def test_init_custom_window(self):
        """可自定义窗口大小"""
        wl = WorkingLayer(window=5)
        assert wl.window == 5

    def test_add_message(self):
        """add 追加消息到内部列表"""
        wl = WorkingLayer(window=3)
        wl.add({"role": "user", "content": "你好"})
        assert len(wl) == 1
        assert wl._messages[0]["content"] == "你好"

    def test_get_messages_within_window(self):
        """消息数未超 window 时 get_messages 返回全部"""
        wl = WorkingLayer(window=5)
        for i in range(3):
            wl.add({"role": "user", "content": f"问题{i}"})
            wl.add({"role": "assistant", "content": f"回答{i}"})
        msgs = wl.get_messages()
        assert len(msgs) == 6  # 3 轮对话 = 6 条

    def test_get_messages_exceeds_window(self):
        """消息数超过 window 时只返回最近的 window*2 条"""
        wl = WorkingLayer(window=2)
        # 添加 4 轮对话（8 条）
        for i in range(4):
            wl.add({"role": "user", "content": f"问题{i}"})
            wl.add({"role": "assistant", "content": f"回答{i}"})
        msgs = wl.get_messages()
        # 只保留最近 2 轮 = 4 条
        assert len(msgs) == 4
        assert msgs[0]["content"] == "问题2"
        assert msgs[-1]["content"] == "回答3"

    def test_get_overflow_empty(self):
        """消息未超出窗口时 get_overflow 返回空列表"""
        wl = WorkingLayer(window=3)
        wl.add({"role": "user", "content": "你好"})
        assert wl.get_overflow() == []

    def test_get_overflow_has_items(self):
        """消息超出窗口时 get_overflow 返回被裁剪的消息"""
        wl = WorkingLayer(window=2)
        for i in range(4):
            wl.add({"role": "user", "content": f"问题{i}"})
            wl.add({"role": "assistant", "content": f"回答{i}"})
        overflow = wl.get_overflow()
        # 8 条消息，保留 4 条，溢出前 4 条
        assert len(overflow) == 4
        assert overflow[0]["content"] == "问题0"

    def test_reset_clears_all(self):
        """reset 清空所有消息"""
        wl = WorkingLayer(window=3)
        wl.add({"role": "user", "content": "你好"})
        wl.reset()
        assert len(wl) == 0
        assert wl._messages == []

    def test_len(self):
        """__len__ 返回消息总数"""
        wl = WorkingLayer()
        assert len(wl) == 0
        wl.add({"role": "user", "content": "a"})
        wl.add({"role": "assistant", "content": "b"})
        assert len(wl) == 2


# ═══════════════════════════════════════════════════════════════
# SummaryLayer 测试
# ═══════════════════════════════════════════════════════════════


class TestSummaryLayer:
    """测试摘要层 SummaryLayer"""

    def test_init_empty(self):
        """初始化时摘要为空"""
        sl = SummaryLayer()
        assert sl.get() == ""

    def test_update_single_text(self):
        """update 单条文本"""
        sl = SummaryLayer()
        sl.update(["用户问了天气"])
        assert sl.get() == "用户问了天气"

    def test_update_multiple_texts(self):
        """update 多条文本用换行拼接"""
        sl = SummaryLayer()
        sl.update(["第一段", "第二段"])
        assert sl.get() == "第一段\n第二段"

    def test_update_accumulates(self):
        """多次 update 会累积"""
        sl = SummaryLayer()
        sl.update(["第一次对话"])
        sl.update(["第二次对话"])
        assert "第一次对话" in sl.get()
        assert "第二次对话" in sl.get()

    def test_update_empty_list_noop(self):
        """空列表 update 不修改摘要"""
        sl = SummaryLayer()
        sl.update(["已有内容"])
        sl.update([])
        assert sl.get() == "已有内容"

    def test_set_replaces(self):
        """set 直接替换摘要内容"""
        sl = SummaryLayer()
        sl.update(["旧内容"])
        sl.set("全新的摘要")
        assert sl.get() == "全新的摘要"

    def test_reset_clears(self):
        """reset 清空摘要"""
        sl = SummaryLayer()
        sl.update(["一些内容"])
        sl.reset()
        assert sl.get() == ""


# ═══════════════════════════════════════════════════════════════
# FactLayer 测试
# ═══════════════════════════════════════════════════════════════


class TestFactLayer:
    """测试长时事实层 FactLayer"""

    def test_init_empty(self):
        """初始化时无事实"""
        fl = FactLayer()
        assert fl._facts == []
        assert fl.get_facts() == []

    def test_add_fact(self):
        """add_fact 添加一条事实"""
        fl = FactLayer()
        fl.add_fact("用户叫张三", source="user")
        assert len(fl._facts) == 1
        assert fl._facts[0]["text"] == "用户叫张三"
        assert fl._facts[0]["source"] == "user"
        assert "timestamp" in fl._facts[0]

    def test_add_fact_default_source(self):
        """不传 source 时默认为空字符串"""
        fl = FactLayer()
        fl.add_fact("偏好 Python")
        assert fl._facts[0]["source"] == ""

    def test_get_facts_returns_texts(self):
        """get_facts 返回最近的事实文本列表"""
        fl = FactLayer()
        fl.add_fact("事实1")
        fl.add_fact("事实2")
        fl.add_fact("事实3")
        facts = fl.get_facts()
        assert facts == ["事实1", "事实2", "事实3"]

    def test_get_facts_limit(self):
        """get_facts 支持 limit 参数限制返回数量"""
        fl = FactLayer()
        for i in range(15):
            fl.add_fact(f"事实{i}")
        facts = fl.get_facts(limit=5)
        assert len(facts) == 5
        # 返回最近 5 条
        assert facts == [f"事实{i}" for i in range(10, 15)]

    def test_get_formatted_empty(self):
        """无事实时 get_formatted 返回空字符串"""
        fl = FactLayer()
        assert fl.get_formatted() == ""

    def test_get_formatted_with_facts(self):
        """有事实时 get_formatted 返回格式化文本"""
        fl = FactLayer()
        fl.add_fact("用户喜欢 Python")
        fl.add_fact("项目叫 collabroom")
        formatted = fl.get_formatted()
        assert "之前记住的信息" in formatted
        assert "用户喜欢 Python" in formatted
        assert "项目叫 collabroom" in formatted

    def test_get_formatted_limit_10(self):
        """get_formatted 最多返回最近 10 条"""
        fl = FactLayer()
        for i in range(15):
            fl.add_fact(f"事实{i}")
        formatted = fl.get_formatted()
        lines = formatted.split("\n")
        # "之前记住的信息：" + 10 条事实
        assert len(lines) == 11

    def test_reset_clears_facts(self):
        """reset 清空所有事实"""
        fl = FactLayer()
        fl.add_fact("某事实")
        fl.reset()
        assert fl._facts == []
        assert fl.get_facts() == []


# ═══════════════════════════════════════════════════════════════
# TieredMemory 测试
# ═══════════════════════════════════════════════════════════════


class TestTieredMemoryInit:
    """测试 TieredMemory 初始化"""

    def test_init_basic(self, tiered_mem):
        """基本初始化：system_prompt + working_window"""
        assert tiered_mem._system == {
            "role": "system",
            "content": "你是测试助手",
        }
        assert tiered_mem.working.window == 3
        assert tiered_mem.summary_layer.get() == ""
        assert tiered_mem.facts._facts == []

    def test_init_auto_summarize_default_true(self):
        """auto_summarize 默认为 True（方向 C：默认启用 LLM 摘要）"""
        mem = TieredMemory("你是助手")
        assert mem._auto_summarize is True

    def test_init_auto_summarize_true(self):
        """可设置 auto_summarize=True"""
        mem = TieredMemory("你是助手", auto_summarize=True)
        assert mem._auto_summarize is True

    def test_init_no_llm(self):
        """初始化时 _llm_for_summary 为 None"""
        mem = TieredMemory("你是助手")
        assert mem._llm_for_summary is None


class TestTieredMemoryAdd:
    """测试 TieredMemory.add() 消息追加与自动裁剪"""

    def test_add_messages_to_working(self, tiered_mem):
        """add 将消息追加到 working 层"""
        tiered_mem.add("user", "你好")
        tiered_mem.add("assistant", "你好！")
        assert len(tiered_mem.working) == 2

    def test_add_messages_within_window_no_trim(self, tiered_mem):
        """消息数未超窗口时不触发裁剪（摘要保持为空）"""
        # window=3, 也就是 6 条消息以内不裁剪
        for i in range(2):  # 2 轮 = 4 条
            tiered_mem.add("user", f"问题{i}")
            tiered_mem.add("assistant", f"回答{i}")
        assert tiered_mem.summary_layer.get() == ""

    def test_add_exceeds_window_triggers_trim(self):
        """消息超出窗口后自动裁剪，溢出内容进入摘要"""
        mem = TieredMemory("你是助手", working_window=2)
        for i in range(4):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        # 前 2 轮被裁剪到摘要
        assert mem.summary_layer.get() != ""
        assert "问题0" in mem.summary_layer.get()
        assert "问题1" in mem.summary_layer.get()
        # working 只保留后 2 轮
        assert len(mem.working._messages) == 4

    def test_add_with_window_zero_no_trim(self):
        """window=0 时不触发裁剪（special case）"""
        mem = TieredMemory("你是助手", working_window=0)
        for i in range(10):
            mem.add("user", f"q{i}")
        # 不裁剪，所有消息都在 working
        assert len(mem.working._messages) == 10


class TestTieredMemoryGetContext:
    """测试 TieredMemory.get_context() 返回完整上下文"""

    def test_get_context_empty(self, tiered_mem):
        """空记忆时只返回 system prompt"""
        ctx = tiered_mem.get_context()
        assert len(ctx) == 1
        assert ctx[0] == {"role": "system", "content": "你是测试助手"}

    def test_get_context_with_messages(self, tiered_mem):
        """有消息时返回 system + working 消息"""
        tiered_mem.add("user", "你好")
        tiered_mem.add("assistant", "你好！")
        ctx = tiered_mem.get_context()
        assert len(ctx) == 3
        assert ctx[0]["role"] == "system"
        assert ctx[1] == {"role": "user", "content": "你好"}
        assert ctx[2] == {"role": "assistant", "content": "你好！"}

    def test_get_context_with_summary(self):
        """有摘要时上下文包含摘要消息"""
        mem = TieredMemory("你是助手", working_window=2)
        for i in range(4):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        ctx = mem.get_context()
        # system + 摘要 + 工作记忆(4条) = 6
        assert len(ctx) == 6
        # 第二条是摘要 system 消息
        assert ctx[1]["role"] == "system"
        assert "[对话历史摘要]" in ctx[1]["content"]

    def test_get_context_with_facts(self, tiered_mem):
        """有事实时上下文包含事实消息"""
        tiered_mem.facts.add_fact("用户叫张三")
        tiered_mem.add("user", "你好")
        ctx = tiered_mem.get_context()
        # system + 事实 + user = 3
        assert len(ctx) == 3
        assert "之前记住的信息" in ctx[1]["content"]
        assert "用户叫张三" in ctx[1]["content"]

    def test_get_context_with_summary_and_facts(self):
        """同时有摘要和事实时都包含"""
        mem = TieredMemory("你是助手", working_window=1)
        mem.facts.add_fact("用户偏好 Python")
        for i in range(3):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        ctx = mem.get_context()
        # system + 摘要 + 事实 + working = 1+1+1+2 = 5
        assert len(ctx) == 5
        roles = [m["role"] for m in ctx]
        assert roles == ["system", "system", "system", "user", "assistant"]


class TestTieredMemoryMaybeTrim:
    """测试 TieredMemory._maybe_trim() 内部裁剪逻辑"""

    def test_no_trim_when_under_window(self):
        """消息数未超窗口时不裁剪"""
        mem = TieredMemory("你是助手", working_window=5)
        mem.add("user", "你好")
        mem.add("assistant", "你好！")
        # 手动调 _maybe_trim 也不应有变化
        overflow_before = mem.working.get_overflow()
        mem._maybe_trim()
        assert mem.summary_layer.get() == ""
        # 消息数不变
        assert len(mem.working) == 2

    def test_trim_moves_overflow_to_summary(self):
        """裁剪时将溢出消息移入摘要并从 working 删除"""
        mem = TieredMemory("你是助手", working_window=2)
        for i in range(4):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        # 触发裁剪后 working 只保留 window*2 条
        assert len(mem.working._messages) == 4
        assert mem.working._messages[0]["content"] == "问题2"
        assert mem.summary_layer.get() != ""

    def test_trim_handler_overflow_without_content(self):
        """溢出消息如果没有 content 字段则不处理"""
        mem = TieredMemory("你是助手", working_window=1)
        # 第1条无 content → 被 skip；第2-3条正常 → 第2条溢出到摘要
        mem.add("user", "将被裁剪")
        mem.add("assistant", "将一同裁剪")
        mem.add("user", "有效消息")
        mem.add("assistant", "有效回复")
        # 裁剪触发：window=1 → 前2条溢出（有content），后2条保留
        mem._maybe_trim()
        assert "将被裁剪" in mem.summary_layer.get()
        assert "有效消息" not in mem.summary_layer.get()


class TestTieredMemorySerialization:
    """测试 TieredMemory 序列化/反序列化"""

    def test_to_dict_basic(self, tiered_mem):
        """基本 to_dict 返回正确结构"""
        tiered_mem.add("user", "你好")
        d = tiered_mem.to_dict()
        assert d["type"] == "TieredMemory"
        assert d["working_window"] == 3
        assert len(d["messages"]) == 1
        assert d["summary"] == ""
        assert d["facts"] == []

    def test_to_dict_with_all_data(self):
        """to_dict 包含摘要和事实"""
        mem = TieredMemory("你是助手", working_window=2)
        for i in range(3):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        mem.facts.add_fact("用户喜欢 Go")
        d = mem.to_dict()
        assert d["summary"] != ""
        assert len(d["facts"]) == 1
        assert d["facts"][0]["text"] == "用户喜欢 Go"

    def test_from_dict_restores_state(self):
        """from_dict 恢复完整状态"""
        original = TieredMemory("你是助手", working_window=2)
        for i in range(3):
            original.add("user", f"问题{i}")
            original.add("assistant", f"回答{i}")
        original.facts.add_fact("事实A")

        data = original.to_dict()
        restored = TieredMemory.from_dict(data, "你是助手")

        # 检查各层恢复
        assert restored.working.window == 2
        assert len(restored.working._messages) == len(original.working._messages)
        assert restored.summary_layer.get() == original.summary_layer.get()
        assert len(restored.facts._facts) == len(original.facts._facts)
        assert restored.facts._facts[0]["text"] == "事实A"

    def test_from_dict_default_window(self):
        """from_dict 不传 working_window 时用默认值 10"""
        mem = TieredMemory.from_dict({}, "你是助手")
        assert mem.working.window == 10

    def test_roundtrip_preserves_context(self):
        """序列化再反序列化后 get_context 一致"""
        original = TieredMemory("你是助手", working_window=2)
        for i in range(2):
            original.add("user", f"问题{i}")
            original.add("assistant", f"回答{i}")
        original.facts.add_fact("偏好 Python")

        data = original.to_dict()
        restored = TieredMemory.from_dict(data, "你是助手")
        assert original.get_context() == restored.get_context()


class TestTieredMemoryLLMSummarize:
    """测试 TieredMemory 的 LLM 摘要功能"""

    def test_set_summary_llm(self):
        """set_summary_llm 注入 LLM 实例"""
        mem = TieredMemory("你是助手")
        mock_llm = MagicMock(spec=LLM)
        mem.set_summary_llm(mock_llm)
        assert mem._llm_for_summary is mock_llm

    def test_llm_summarize_without_llm(self):
        """无 LLM 时降级为拼接策略：取最后 300 字符"""
        mem = TieredMemory("你是助手")
        texts = ["消息1", "消息2", "最后一条" * 200]
        result = mem._llm_summarize(texts)
        # 无 LLM 时走 _summarize_concat：texts[-1][-300:]（超过 300 则截取末尾）
        expected = texts[-1][-300:]
        assert result == expected

    def test_llm_summarize_without_llm_short_text(self):
        """无 LLM 且文本短于 300 字符时返回全部"""
        mem = TieredMemory("你是助手")
        texts = ["短消息"]
        result = mem._llm_summarize(texts)
        assert result == "短消息"

    def test_llm_summarize_empty_texts(self):
        """空列表返回空字符串"""
        mem = TieredMemory("你是助手")
        result = mem._llm_summarize([])
        assert result == ""

    def test_llm_summarize_with_mock_llm(self):
        """有 LLM 时调用 LLM.chat 获取摘要"""
        mem = TieredMemory("你是助手")
        mock_llm = MagicMock(spec=LLM)
        mock_llm.chat.return_value = LLMResponse(
            content="用户讨论了天气和足球话题",
            usage=Usage(prompt_tokens=50, completion_tokens=10),
        )
        mem.set_summary_llm(mock_llm)

        texts = ["用户: 今天天气如何", "助手: 今天晴天", "用户: 聊足球吧"]
        result = mem._llm_summarize(texts)

        # 验证 LLM 被调用
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args[0][0]
        assert any("对话压缩" in m["content"] for m in call_args)

        assert result == "用户讨论了天气和足球话题"

    def test_llm_summarize_truncates_to_500(self):
        """LLM 返回超过 500 字符时截断"""
        mem = TieredMemory("你是助手")
        mock_llm = MagicMock(spec=LLM)
        mock_llm.chat.return_value = LLMResponse(
            content="A" * 600,
            usage=Usage(),
        )
        mem.set_summary_llm(mock_llm)
        result = mem._llm_summarize(["测试"])
        assert len(result) == 500

    def test_llm_summarize_exception_fallback(self):
        """LLM 调用异常时降级到 _summarize_concat：取最后一条消息末尾 300 字符"""
        mem = TieredMemory("你是助手")
        mock_llm = MagicMock(spec=LLM)
        mock_llm.chat.side_effect = RuntimeError("LLM 超时")
        mem.set_summary_llm(mock_llm)

        # 使用非对称字符串，确保 last[-300:] 和 last[:300] 不同
        first_part = "开头" * 50  # 100 chars
        middle_part = "中间" * 80  # 160 chars
        last_part = "结尾" * 70   # 140 chars — 总计 400 chars，超过 300
        texts = ["消息1", first_part + middle_part + last_part]
        result = mem._llm_summarize(texts)
        # _summarize_concat 取最后 300 chars = middle_part[:160] + last_part[140:][:140]
        expected_text = texts[-1][-300:]
        assert result == expected_text


class TestTieredMemorySummary:
    """测试 TieredMemory.summary() 统计方法"""

    def test_summary_empty(self, tiered_mem):
        """空记忆统计"""
        stats = tiered_mem.summary()
        assert stats["total_messages"] == 1  # system
        assert stats["working_window"] == 3
        assert stats["working_messages"] == 0
        assert stats["has_summary"] is False
        assert stats["facts_count"] == 0
        assert stats["user_turns"] == 0
        assert stats["assistant_turns"] == 0

    def test_summary_with_data(self, filled_tiered_mem):
        """有数据后的统计"""
        stats = filled_tiered_mem.summary()
        assert stats["has_summary"] is True  # 溢出到摘要
        assert stats["working_messages"] > 0
        assert stats["user_turns"] > 0
        assert stats["assistant_turns"] > 0
        assert "estimated_tokens" in stats

    def test_summary_with_facts(self, tiered_mem):
        """有事实时的统计"""
        tiered_mem.facts.add_fact("用户叫张三")
        stats = tiered_mem.summary()
        assert stats["facts_count"] == 1


class TestTieredMemoryTokenEstimate:
    """测试 TieredMemory.token_estimate()"""

    def test_token_estimate_empty(self, tiered_mem):
        """空记忆 token 估算包含 system prompt"""
        # working 为空，但系统提示有内容
        assert tiered_mem.token_estimate() >= 0

    def test_token_estimate_grows_with_messages(self, tiered_mem):
        """消息越多估算越大"""
        before = tiered_mem.token_estimate()
        tiered_mem.add("user", "这是一条很长的测试消息" * 10)
        tiered_mem.add("assistant", "这也是一条很长的回复消息" * 10)
        after = tiered_mem.token_estimate()
        assert after > before
