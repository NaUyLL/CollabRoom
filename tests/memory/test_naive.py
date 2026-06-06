"""测试 NaiveMemory — 全量保留、永不裁剪的简单记忆"""
from __future__ import annotations

from collabroom.core.memory.naive import NaiveMemory


class TestNaiveMemoryInit:
    """测试 NaiveMemory 初始化"""

    def test_init_stores_system_prompt(self):
        """初始化后 _system 字段包含正确的 system prompt"""
        mem = NaiveMemory("你是测试助手")
        assert mem._system == {"role": "system", "content": "你是测试助手"}

    def test_init_messages_contains_system(self):
        """初始化后 _messages 列表已包含 system 消息"""
        mem = NaiveMemory("你是助手")
        assert len(mem._messages) == 1
        assert mem._messages[0]["role"] == "system"
        assert mem._messages[0]["content"] == "你是助手"


class TestNaiveMemoryAdd:
    """测试 NaiveMemory.add() 消息追加"""

    def test_add_user_message(self):
        """添加一条 user 消息后 _messages 末尾为此消息"""
        mem = NaiveMemory("你是测试助手")
        mem.add("user", "你好")
        assert len(mem._messages) == 2
        assert mem._messages[-1] == {"role": "user", "content": "你好"}

    def test_add_assistant_message(self):
        """添加一条 assistant 消息"""
        mem = NaiveMemory("你是测试助手")
        mem.add("assistant", "你好，有什么可以帮你？")
        assert mem._messages[-1] == {
            "role": "assistant",
            "content": "你好，有什么可以帮你？",
        }

    def test_add_multiple_messages(self):
        """多次 add 后消息按顺序追加"""
        mem = NaiveMemory("你是测试助手")
        mem.add("user", "问题1")
        mem.add("assistant", "回答1")
        mem.add("user", "问题2")
        mem.add("assistant", "回答2")
        # system + 4 条消息
        assert len(mem._messages) == 5
        roles = [m["role"] for m in mem._messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]


class TestNaiveMemoryGetContext:
    """测试 NaiveMemory.get_context() 返回完整消息列表"""

    def test_get_context_returns_deepcopy(self, naive_mem):
        """get_context 返回的是深拷贝，修改不影响内部状态"""
        ctx = naive_mem.get_context()
        assert isinstance(ctx, list)
        # 修改返回的列表不影响内部状态
        ctx.append({"role": "user", "content": "不应该出现"})
        assert len(naive_mem._messages) == 1  # 只有 system

    def test_get_context_after_add(self):
        """add 消息后 get_context 包含所有消息"""
        mem = NaiveMemory("你是助手")
        mem.add("user", "你好")
        mem.add("assistant", "你好！")
        ctx = mem.get_context()
        assert len(ctx) == 3
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        assert ctx[2]["role"] == "assistant"

    def test_get_context_with_many_rounds(self, filled_naive_mem):
        """多次对话后 get_context 包含 system + 所有历史消息"""
        ctx = filled_naive_mem.get_context()
        # system + 3 轮(user+assistant) = 7 条
        assert len(ctx) == 7
        assert ctx[0] == {"role": "system", "content": "你是测试助手"}
        for i in range(3):
            assert ctx[1 + i * 2] == {
                "role": "user",
                "content": f"用户问题{i}",
            }
            assert ctx[2 + i * 2] == {
                "role": "assistant",
                "content": f"助手回答{i}",
            }


class TestNaiveMemorySummary:
    """测试 NaiveMemory.summary() 统计信息"""

    def test_summary_empty(self, naive_mem):
        """空记忆（仅 system prompt）的统计"""
        stats = naive_mem.summary()
        assert stats["total_messages"] == 1
        assert stats["user_turns"] == 0
        assert stats["assistant_turns"] == 0

    def test_summary_after_conversation(self, filled_naive_mem):
        """3 轮对话后的统计"""
        stats = filled_naive_mem.summary()
        assert stats["total_messages"] == 7  # system + 6
        assert stats["user_turns"] == 3
        assert stats["assistant_turns"] == 3

    def test_summary_has_estimated_tokens(self, naive_mem):
        """summary 包含 estimated_tokens 字段"""
        naive_mem.add("user", "你好")
        stats = naive_mem.summary()
        assert "estimated_tokens" in stats
        assert isinstance(stats["estimated_tokens"], int)


class TestNaiveMemoryTokenEstimate:
    """测试 NaiveMemory.token_estimate()"""

    def test_token_estimate_increases_with_messages(self, naive_mem):
        """消息越多 token 估算值越大"""
        before = naive_mem.token_estimate()
        naive_mem.add("user", "这是一条比较长的消息用来测试 token 估算")
        after = naive_mem.token_estimate()
        assert after > before

    def test_token_estimate_always_positive(self, naive_mem):
        """即使只有 system prompt，估算值也大于 0"""
        assert naive_mem.token_estimate() > 0
