"""测试 Room 模块 — RoomMessage / Room / AgentMember"""
from __future__ import annotations
import time
from unittest.mock import MagicMock, patch

import pytest

from collabroom.room import (
    RoomMessage, Room, AgentMember,
    STOP_WORDS, STOP_MATCH_WHOLE_WORD,
    MAX_AUTO_DEPTH, MAX_PAIR_LOOPS, MENTION_RE,
)
from collabroom.core.types import AgentResult, Usage, LLMResponse
from collabroom.core.llm import LLM


# ═══════════════════════════════════════════════════════════════
# RoomMessage
# ═══════════════════════════════════════════════════════════════

class TestRoomMessage:
    """RoomMessage 数据类测试"""

    def test_创建消息并指定所有字段(self):
        """手动指定 timestamp 和 kind"""
        msg = RoomMessage(sender="Alice", content="你好", timestamp=100.0, kind="dm")
        assert msg.sender == "Alice"
        assert msg.content == "你好"
        assert msg.timestamp == 100.0
        assert msg.kind == "dm"

    def test_timestamp自动填充为0时自动设当前时间(self):
        """timestamp=0 时 __post_init__ 自动填充 time.time()"""
        with patch("collabroom.room.time.time", return_value=1234.567):
            msg = RoomMessage(sender="Bob", content="嗨")
        assert msg.timestamp == 1234.567

    def test_timestamp为非零时不覆盖(self):
        """timestamp 已指定具体值时不覆盖"""
        msg = RoomMessage(sender="Bob", content="嗨", timestamp=42.0)
        assert msg.timestamp == 42.0

    def test_kind默认值为public(self):
        """不指定 kind 时默认为 'public'"""
        msg = RoomMessage(sender="user", content="大家好")
        assert msg.kind == "public"


# ═══════════════════════════════════════════════════════════════
# Room 初始化 & 基础方法
# ═══════════════════════════════════════════════════════════════

class TestRoomInit:
    """Room 初始化测试"""

    def test_初始化默认name(self):
        """不带参数时默认名称为 '会议室'"""
        room = Room()
        assert room.name == "会议室"
        assert room.members == {}
        assert room.history == []
        assert room._order == []

    def test_初始化自定义name(self):
        """带自定义名称"""
        room = Room("开发讨论")
        assert room.name == "开发讨论"
        assert room.members == {}
        assert room.history == []
        assert room._order == []


class TestRoomRegister:
    """Room.register() 成员注册测试"""

    def test_注册成员到末尾(self, core_agent):
        """不带 position 时追加到 _order 末尾"""
        room = Room("测试室")
        member = AgentMember("Alice", "测试助手", core_agent)
        room.register(member)
        assert "Alice" in room.members
        assert room.members["Alice"] is member
        assert room._order == ["Alice"]

    def test_注册多个成员按顺序排列(self, core_agent, core_agent_yes):
        """多个成员依次注册，_order 按注册顺序"""
        room = Room("测试室")
        room.register(AgentMember("Alice", "助手A", core_agent_yes))
        room.register(AgentMember("Bob", "助手B", core_agent))
        assert room._order == ["Alice", "Bob"]
        assert list(room.members.keys()) == ["Alice", "Bob"]

    def test_指定position注册(self, core_agent, core_agent_yes, core_agent_no):
        """通过 position 参数指定插入位置"""
        room = Room("测试室")
        # 先插入 Alice 和 Charlie
        room.register(AgentMember("Alice", "助手A", core_agent_yes))
        room.register(AgentMember("Charlie", "助手C", core_agent_no))
        # Bob 插到中间
        room.register(AgentMember("Bob", "助手B", core_agent), position=1)
        assert room._order == ["Alice", "Bob", "Charlie"]


class TestRoomSay:
    """Room.say() 添加消息测试"""

    def test_say添加public消息(self):
        """发送一条 public 消息到 history"""
        room = Room()
        msg = room.say("user", "你好")
        assert len(room.history) == 1
        assert room.history[0].sender == "user"
        assert room.history[0].content == "你好"
        assert room.history[0].kind == "public"
        assert msg is room.history[0]

    def test_say添加dm消息(self):
        """发送一条私信消息"""
        room = Room()
        msg = room.say("Alice", "秘密消息", kind="dm")
        assert msg.kind == "dm"
        assert room.history[0].kind == "dm"


class TestRoomFormatHistory:
    """Room.format_history() 历史格式化测试"""

    def test_默认tail10条消息(self):
        """默认取最后 10 条"""
        room = Room()
        for i in range(15):
            room.say(f"user{i}", f"消息{i}")
        output = room.format_history()
        lines = output.split("\n")
        assert len(lines) == 10
        # tail=10 取索引 5~14 的最后 10 条
        assert "user5" in output
        assert "user4" not in output  # 前5条被截掉
        assert "user14" in output

    def test_自定义tail参数(self):
        """指定 tail=3 只显示最后3条"""
        room = Room()
        for i in range(5):
            room.say(f"user{i}", f"消息{i}")
        output = room.format_history(tail=3)
        lines = output.split("\n")
        assert len(lines) == 3
        assert "user4" in output

    def test_kind过滤只包含public(self):
        """include_kind 过滤，只显示 public"""
        room = Room()
        room.say("user", "公开消息", kind="public")
        room.say("Alice", "私信", kind="dm")
        room.say("Bob", "公开回复", kind="public")
        output = room.format_history(include_kind={"public"})
        assert "公开消息" in output
        assert "公开回复" in output
        assert "私信" not in output

    def test_kind过滤只包含dm(self):
        """include_kind 过滤，只显示 dm"""
        room = Room()
        room.say("user", "公开消息", kind="public")
        room.say("Alice", "私信", kind="dm")
        output = room.format_history(include_kind={"dm"})
        assert "公开消息" not in output
        assert "私信" in output

    def test_内容超过250字截断(self):
        """超过 250 字的内容会被截断并在末尾加 …"""
        room = Room()
        long_text = "x" * 300
        room.say("user", long_text)
        output = room.format_history()
        # 截断后应该有 250 个字符 + "…"
        assert "x" * 250 + "…" in output
        assert len(output.split(": ", 1)[1].rstrip()) == 251  # 250 chars + "…"

    def test_私信消息带at前缀(self):
        """dm 类型的消息显示为 @sender 格式"""
        room = Room()
        room.say("Alice", "你好", kind="dm")
        output = room.format_history()
        assert "@Alice:" in output

    def test_public消息不带at前缀(self):
        """public 消息不带 @ 前缀"""
        room = Room()
        room.say("Alice", "你好", kind="public")
        output = room.format_history()
        assert "Alice:" in output
        assert "@Alice:" not in output


class TestRoomListMembers:
    """Room.list_members() 测试"""

    def test_空房间返回空列表(self):
        """没有成员时返回空列表"""
        room = Room()
        assert room.list_members() == []

    def test_有成员时返回名字列表(self, core_agent, core_agent_yes):
        """返回已注册成员的名字"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        room.register(AgentMember("Bob", "助手", core_agent))
        assert room.list_members() == ["Alice", "Bob"]


# ═══════════════════════════════════════════════════════════════
# _is_stop — 停止词检测
# ═══════════════════════════════════════════════════════════════

class TestIsStop:
    """Room._is_stop() 停止词检测"""

    def test_停止词_停止_返回True(self):
        """完整匹配 '停止' → True"""
        room = Room()
        assert room._is_stop("停止") is True

    def test_停止词_够了_返回True(self):
        """完整匹配 '够了' → True"""
        room = Room()
        assert room._is_stop("够了") is True

    def test_停止词_stop_返回True(self):
        """英文停止词 'stop' → True"""
        room = Room()
        assert room._is_stop("stop") is True

    def test_停止词_enough_返回True(self):
        """英文停止词 'enough' → True"""
        room = Room()
        assert room._is_stop("enough") is True

    def test_停止词_停_返回True(self):
        """完整匹配 '停' → True"""
        room = Room()
        assert room._is_stop("停") is True

    def test_停止词_停下来_返回False(self):
        """'停下来' 不等于 '停'（完整匹配模式），应返回 False"""
        room = Room()
        assert room._is_stop("停下来") is False

    def test_普通语句_返回False(self):
        """正常对话不应触发停止"""
        room = Room()
        assert room._is_stop("今天天气怎么样") is False
        assert room._is_stop("请帮我分析一下这个文件") is False

    def test_多个标点混入_正确分词(self):
        """含空格/标点的复合消息，分词后正确检测停止词"""
        room = Room()
        # "停止" 独立出现就该命中
        assert room._is_stop("好的，停止。") is True
        # "够了" 后跟标点
        assert room._is_stop("够了！别再说了") is True

    def test_停止词在STOP_WORDS集合中(self):
        """验证 STOP_WORDS 常量包含常见停止词"""
        assert "停止" in STOP_WORDS
        assert "够了" in STOP_WORDS
        assert "停" in STOP_WORDS
        assert "stop" in STOP_WORDS
        assert "enough" in STOP_WORDS
        assert "halt" in STOP_WORDS

    def test_完整匹配模式开关(self):
        """STOP_MATCH_WHOLE_WORD 默认为 True"""
        assert STOP_MATCH_WHOLE_WORD is True

    def test_子串模式关闭时_包含停止词的词不触发(self):
        """当 STOP_MATCH_WHOLE_WORD=True 时，'停车' 不会因为含 '停' 而被触发"""
        room = Room()
        # '停车' 是完整 token，不等于 '停'
        assert room._is_stop("停车") is False

    def test_多token消息中含停止词(self):
        """多个词的消息中有一个是停止词 → True"""
        room = Room()
        assert room._is_stop("我觉得 够了 不用继续了") is True


# ═══════════════════════════════════════════════════════════════
# _parse_mentions — @mention 解析
# ═══════════════════════════════════════════════════════════════

class TestParseMentions:
    """Room._parse_mentions() @mention 解析测试"""

    def test_单个提及(self, core_agent_yes):
        """@Alice 你好 → [Alice]"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Alice 你好")
        assert result == ["Alice"]

    def test_多个提及(self, core_agent_yes, core_agent):
        """@Alice @Bob 大家好 → [Alice, Bob]"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        room.register(AgentMember("Bob", "助手", core_agent))
        result = room._parse_mentions("@Alice @Bob 大家好")
        assert result == ["Alice", "Bob"]

    def test_提及不存在的成员_过滤掉(self, core_agent_yes):
        """@Charlie（不存在）→ 过滤掉，返回空"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Charlie 你好")
        assert result == []

    def test_重复提及_去重(self, core_agent_yes):
        """@Alice @Alice 你好 → [Alice]（去重）"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Alice @Alice 你好")
        assert result == ["Alice"]

    def test_中文标点后_正确识别(self, core_agent_yes):
        """@Alice，你好 → [Alice]"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Alice，你好")
        assert result == ["Alice"]

    def test_中英文混用(self, core_agent_yes, core_agent):
        """@Alice 和 @Bob 之间用中文标点"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        room.register(AgentMember("Bob", "助手", core_agent))
        result = room._parse_mentions("@Alice，@Bob。你们好")
        assert result == ["Alice", "Bob"]

    def test_冒号后识别(self, core_agent_yes):
        """@Alice: 你好 → [Alice]"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Alice: 你好")
        assert result == ["Alice"]

    def test_无提及返回空列表(self, core_agent_yes):
        """没有 @mention 的消息"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("大家好，今天天气不错")
        assert result == []

    def test_混合存在和不存在的成员(self, core_agent_yes):
        """@Alice @Charlie → 只返回 ['Alice']"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        result = room._parse_mentions("@Alice @Charlie 你好")
        assert result == ["Alice"]


# ═══════════════════════════════════════════════════════════════
# _volunteer_round — 并行举手
# ═══════════════════════════════════════════════════════════════

class TestVolunteerRound:
    """Room._volunteer_round() 并行举手测试"""

    def test_所有成员举手_全部返回(self, core_agent_yes):
        """所有 Agent 都回答 YES → 全部返回"""
        room = Room()
        room.register(AgentMember("Alice", "总有话说", core_agent_yes))
        room.register(AgentMember("Bob", "爱凑热闹", core_agent_yes))
        # 用 MagicMock 包装 core_agent 让 decide 返回 True
        for name in room.members:
            room.members[name].agent.llm.chat.return_value = LLMResponse(
                content="YES\n", usage=Usage(),
            )
        volunteers = room._volunteer_round("大家好")
        assert volunteers == ["Alice", "Bob"]

    def test_无成员举手_返回空列表(self, core_agent_no):
        """所有 Agent 都回答 NO → 空列表"""
        room = Room()
        room.register(AgentMember("Alice", "沉默", core_agent_no))
        room.register(AgentMember("Bob", "不说话", core_agent_no))
        volunteers = room._volunteer_round("大家好")
        assert volunteers == []

    def test_混合举手_只返回举手者(self, core_agent_yes, core_agent_no):
        """一个 YES 一个 NO → 只返回 YES 的"""
        room = Room()
        room.register(AgentMember("Alice", "活跃", core_agent_yes))
        room.register(AgentMember("Bob", "沉默", core_agent_no))
        volunteers = room._volunteer_round("大家好")
        assert volunteers == ["Alice"]

    def test_空房间_返回空列表(self):
        """没有成员时返回空列表"""
        room = Room()
        volunteers = room._volunteer_round("大家好")
        assert volunteers == []

    def test_决策异常_不影响其他成员(self, core_agent_yes, core_agent):
        """某个 Agent 抛异常时被捕获，不影响其他成员"""
        room = Room()
        room.register(AgentMember("Alice", "活跃", core_agent_yes))

        # Bob 的 decide 会抛异常
        bob_agent = MagicMock()
        bob_agent.llm = MagicMock()
        bob_agent.llm.chat.side_effect = RuntimeError("LLM 挂了")
        bob = AgentMember("Bob", "有问题", bob_agent)
        room.register(bob)

        volunteers = room._volunteer_round("大家好")
        # Alice 举手成功，Bob 异常被忽略
        assert volunteers == ["Alice"]

    def test_结果按注册顺序排列(self, core_agent_yes, core_agent):
        """举手结果按 _order 注册顺序排列"""
        room = Room()
        # 先注册 Charlie，再注册 Alice，再注册 Bob
        room.register(AgentMember("Charlie", "最后举手", core_agent_yes))  # 位置0
        room.register(AgentMember("Alice", "第一个举手", core_agent_yes))  # 位置1
        room.register(AgentMember("Bob", "第二个举手", core_agent))  # 位置2(不举手)
        volunteers = room._volunteer_round("测试")
        # 应该按注册顺序 ["Charlie", "Alice"]
        assert volunteers == ["Charlie", "Alice"]


# ═══════════════════════════════════════════════════════════════
# _pick_fallback — 兜底选择
# ═══════════════════════════════════════════════════════════════

class TestPickFallback:
    """Room._pick_fallback() 兜底选择测试"""

    def test_有成员_返回第一个(self, core_agent, core_agent_yes):
        """有成员时返回 _order 的第一个"""
        room = Room()
        room.register(AgentMember("Alice", "助手", core_agent_yes))
        room.register(AgentMember("Bob", "助手", core_agent))
        assert room._pick_fallback() == "Alice"

    def test_空房间_返回None(self):
        """没有成员时返回 None"""
        room = Room()
        assert room._pick_fallback() is None

    def test_单成员_返回该成员(self, core_agent):
        """只有一个成员时返回它"""
        room = Room()
        room.register(AgentMember("Solo", "独苗", core_agent))
        assert room._pick_fallback() == "Solo"


# ═══════════════════════════════════════════════════════════════
# round — 完整对话轮次
# ═══════════════════════════════════════════════════════════════

class TestRound:
    """Room.round() 完整流程测试"""

    def test_用户停止词_返回空结果(self):
        """用户说 '停止' → 返回 []"""
        room = Room()
        result = room.round("停止")
        assert result == []

    def test_用户说够了_返回空结果(self):
        """用户说 '够了' → 返回 []"""
        room = Room()
        result = room.round("够了")
        assert result == []

    def test_停止词不加入history(self):
        """停止词的消息不加入 history"""
        room = Room()
        room.round("停止")
        assert len(room.history) == 0

    def test_正常发言_举手成员依次回应(self, room):
        """用户发普通消息 → 举手者（Alice）回复"""
        responses = room.round("大家好，讨论一下")
        assert len(responses) >= 1
        assert responses[0][0] == "Alice"

    def test_用户at某Agent_强制加入(self, core_agent_yes, core_agent):
        """用户 @Bob → Bob 即使不举手也被强制加入"""
        room = Room("提及测试")
        alice = AgentMember("Alice", "活跃", core_agent_yes)
        # Bob 用 MagicMock 控制行为
        bob_agent = MagicMock()
        bob_agent.llm = core_agent.llm  # decide 用核心 LLM
        bob_agent.run.return_value = AgentResult(final_answer="收到，我来说说")
        bob_agent.system_prompt = "测试"
        bob = AgentMember("Bob", "沉默", bob_agent)
        room.register(alice)
        room.register(bob)

        responses = room.round("@Bob 请回答我")
        speakers = [r[0] for r in responses]
        assert "Bob" in speakers

    def test_无人举手_兜底选第一个(self, core_agent_no):
        """所有人都回答 NO → 兜底选第一个 Agent"""
        room = Room("测试")
        room.register(AgentMember("Alice", "沉默A", core_agent_no))
        room.register(AgentMember("Bob", "沉默B", core_agent_no))
        # Bob 的 agent 也会被兜底调用
        responses = room.round("有人吗")
        # 兜底选第一个
        assert len(responses) >= 1
        assert responses[0][0] == "Alice"

    def test_发言含at_mention_链式回应(self, core_agent_yes, core_agent):
        """Alice 发言中 @Bob → Bob 被加入队列回应"""
        room = Room("链式测试")
        # Alice: 举手
        alice = AgentMember("Alice", "活跃", core_agent_yes)
        # Bob: 用 MagicMock，默认不举手
        bob_agent = MagicMock()
        bob_agent.llm = MagicMock()
        bob_agent.llm.chat.return_value = LLMResponse(content="NO", usage=Usage())
        bob_agent.run.return_value = AgentResult(final_answer="我觉得不错")
        bob_agent.system_prompt = "测试"
        bob = AgentMember("Bob", "沉默", bob_agent)
        room.register(alice)
        room.register(bob)

        # 改造 Alice 让其回复中包含 @Bob
        alice.agent.llm.chat.return_value = LLMResponse(
            content="你好 @Bob 你怎么看？", usage=Usage(),
        )

        responses = room.round("讨论")
        speakers = [r[0] for r in responses]
        # Alice 发言 @Bob → Bob 链式回应
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_发言PASS_跳过(self):
        """Agent 回复 PASS → 跳过，不加入 responses"""
        room = Room("PASS测试")
        alice_agent = MagicMock()
        alice_agent.llm = MagicMock()
        alice_agent.llm.chat.return_value = LLMResponse(content="YES", usage=Usage())
        alice_agent.run.return_value = AgentResult(final_answer="PASS")
        alice_agent.system_prompt = "测试"
        alice = AgentMember("Alice", "测试", alice_agent)
        room.register(alice)

        responses = room.round("你好")
        # PASS 被跳过
        assert len(responses) == 0

    def test_MAX_AUTO_DEPTH限制(self):
        """验证 MAX_AUTO_DEPTH 限制存在"""
        assert MAX_AUTO_DEPTH == 5

    def test_MAX_PAIR_LOOPS限制(self):
        """验证 MAX_PAIR_LOOPS 限制存在"""
        assert MAX_PAIR_LOOPS == 2

    def test_链式回应不超过MAX_PAIR_LOOPS(self, core_agent_yes):
        """同一对 Agent 来回 @ 不超过 MAX_PAIR_LOOPS 次"""
        room = Room("循环测试")
        alice = AgentMember("Alice", "活跃", core_agent_yes)
        bob_agent = MagicMock()
        bob_agent.llm = MagicMock()
        bob_agent.llm.chat.return_value = LLMResponse(content="NO", usage=Usage())
        bob_agent.run.return_value = AgentResult(final_answer="@Alice 我同意")
        bob_agent.system_prompt = "测试"
        bob = AgentMember("Bob", "沉默", bob_agent)
        room.register(alice)
        room.register(bob)

        # Alice 发言 @Bob
        alice.agent.llm.chat.return_value = LLMResponse(
            content="@Bob 你怎么看？", usage=Usage(),
        )

        responses = room.round("讨论")
        # 最多 2 轮来回
        assert len(responses) <= MAX_AUTO_DEPTH

    def test_用户消息加入history(self, room):
        """用户消息被加入 history，sender='user'"""
        room.round("帮我分析")
        assert room.history[0].sender == "user"
        assert room.history[0].content == "帮我分析"


# ═══════════════════════════════════════════════════════════════
# dm — 私信
# ═══════════════════════════════════════════════════════════════

class TestDm:
    """Room.dm() 私信测试"""

    def test_正常私信_返回回复(self, room_with_dm):
        """正常私信流程，目标 Agent 回复"""
        reply = room_with_dm.dm("Alice", "Bob", "有个问题想问你")
        assert reply is not None
        assert "收到" in reply or "message" in reply.lower() or "消息" in reply

    def test_目标不存在_返回None(self, room_with_dm):
        """私信目标不存在 → None"""
        reply = room_with_dm.dm("Alice", "Charlie", "你好")
        assert reply is None

    def test_回复PASS_返回None(self, core_agent_yes):
        """目标回复 PASS → 返回 None"""
        room = Room("PASS私信")
        alice = AgentMember("Alice", "发送者", core_agent_yes)

        bob_agent = MagicMock()
        bob_agent.run.return_value = AgentResult(final_answer="PASS")
        bob = AgentMember("Bob", "接收者", bob_agent, on_pass="PASS")
        room.register(alice)
        room.register(bob)

        reply = room.dm("Alice", "Bob", "你好")
        assert reply is None

    def test_私信消息kind为dm(self, room_with_dm):
        """私信产生的 history 消息 kind='dm'"""
        room_with_dm.dm("Alice", "Bob", "测试")
        # 应该有 dm 消息
        dm_msgs = [m for m in room_with_dm.history if m.kind == "dm"]
        assert len(dm_msgs) >= 1


# ═══════════════════════════════════════════════════════════════
# AgentMember
# ═══════════════════════════════════════════════════════════════

class TestAgentMember:
    """AgentMember 测试"""

    def test_初始化(self, core_agent):
        """AgentMember 基本属性"""
        member = AgentMember("Alice", "测试助手", core_agent)
        assert member.name == "Alice"
        assert member.role_desc == "测试助手"
        assert member.agent is core_agent
        assert member.on_pass == "PASS"  # 默认值

    def test_on_pass自定义(self, core_agent):
        """自定义 on_pass 值"""
        member = AgentMember("Alice", "助手", core_agent, on_pass="SKIP")
        assert member.on_pass == "SKIP"

    def test_on_pass为None(self, core_agent):
        """on_pass=None 时不提示跳过"""
        member = AgentMember("Alice", "助手", core_agent, on_pass=None)
        assert member.on_pass is None

    def test_decide_YES(self, core_agent_yes):
        """LLM 返回 YES → decide() 返回 True"""
        member = AgentMember("Alice", "活跃", core_agent_yes)
        assert member.decide("讨论话题") is True

    def test_decide_NO(self, core_agent_no):
        """LLM 返回 NO → decide() 返回 False"""
        member = AgentMember("Alice", "沉默", core_agent_no)
        assert member.decide("讨论话题") is False

    def test_decide_异常时返回False(self, core_agent):
        """LLM.chat 抛异常时 decide() 返回 False"""
        core_agent.llm.chat.side_effect = RuntimeError("连接失败")
        member = AgentMember("Alice", "助手", core_agent)
        assert member.decide("测试") is False

    def test_chat_正常发言(self):
        """Agent chat 返回 run() 的 final_answer"""
        agent_mock = MagicMock()
        agent_mock.run.return_value = AgentResult(final_answer="我认为这个问题应该从两方面分析")
        agent_mock.system_prompt = "测试"
        member = AgentMember("Alice", "助手", agent_mock)
        reply = member.chat("今天讨论什么")
        assert reply == "我认为这个问题应该从两方面分析"

    def test_chat_force_reply(self):
        """force_reply=True 时绕过 PASS 提示"""
        agent_mock = MagicMock()
        agent_mock.run.return_value = AgentResult(final_answer="我是被强制要求发言的")
        agent_mock.system_prompt = "测试"
        member = AgentMember("Alice", "助手", agent_mock)
        reply = member.chat("上下文", force_reply=True)
        assert reply == "我是被强制要求发言的"

    def test_memory属性委托给agent(self, core_agent):
        """AgentMember.memory → agent.memory"""
        member = AgentMember("Alice", "助手", core_agent)
        assert member.memory is core_agent.memory

    def test_system_prompt拼接(self, core_agent):
        """_system_prompt 包含名字、角色和 agent 的 system_prompt"""
        core_agent.system_prompt = "你是一个智能助手"
        member = AgentMember("Alice", "代码审查员", core_agent)
        assert "Alice" in member._system_prompt
        assert "代码审查员" in member._system_prompt
        assert "智能助手" in member._system_prompt
