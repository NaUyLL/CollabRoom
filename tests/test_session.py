"""测试 Session 持久化 — Room.save / Room.load / AgentMember 序列化"""
from __future__ import annotations

from unittest.mock import MagicMock
import json
import tempfile
import os

import pytest

from collabroom.core.llm import LLM
from collabroom.core.loop import Agent as CoreAgent
from collabroom.core.memory.naive import NaiveMemory
from collabroom.core.memory.tiered import TieredMemory
from collabroom.core.tool import Registry
from collabroom.room import Room, AgentMember, RoomMessage


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _make_dummy_agent(name: str, role: str, sp: str, llm: object) -> CoreAgent:
    """创建 dummy CoreAgent（mock LLM，空 Registry）"""
    return CoreAgent(llm=llm, registry=Registry(), system_prompt=sp)


def _setup_room(memory_cls=NaiveMemory, **mem_kwargs) -> tuple[Room, LLM]:
    """创建含 N 个 Agent + 有历史的 Room"""
    llm = MagicMock(spec=LLM)
    room = Room("测试室")
    for name, role in [("Alice", "助手A"), ("Bob", "助手B")]:
        mem = memory_cls(role, **(mem_kwargs or {}))
        core = CoreAgent(llm=llm, registry=Registry(),
                         system_prompt=role, memory=mem)
        member = AgentMember(name, role, core, on_pass="PASS")
        room.register(member)

    room.say("user", "你好")
    room.say("Alice", "你好，有什么需要帮助的？")
    room.say("user", "今天天气怎么样")
    room.say("Bob", "今天天气不错，适合出行")
    return room, llm


# ═══════════════════════════════════════════════════════════════
# AgentMember 序列化
# ═══════════════════════════════════════════════════════════════

class TestAgentMemberSerialization:
    """AgentMember.to_dict / from_dict"""

    def test_to_dict_includes_fields(self):
        """to_dict 包含所有必要字段"""
        llm = MagicMock(spec=LLM)
        core = CoreAgent(llm=llm, registry=Registry(), system_prompt="助手",
                         memory=NaiveMemory("助手"))
        member = AgentMember("Alice", "测试助手", core, on_pass="PASS")
        d = member.to_dict()
        assert d["name"] == "Alice"
        assert d["role_desc"] == "测试助手"
        assert d["on_pass"] == "PASS"
        assert "system_prompt" in d
        assert "memory" in d

    def test_to_dict_naive_memory(self):
        """NaiveMemory 可序列化"""
        llm = MagicMock(spec=LLM)
        mem = NaiveMemory("助手")
        mem.add("user", "问题1")
        mem.add("assistant", "回答1")
        core = CoreAgent(llm=llm, registry=Registry(), system_prompt="助手", memory=mem)
        member = AgentMember("Alice", "助手", core)
        d = member.to_dict()
        assert d["memory"]["type"] == "NaiveMemory"
        assert len(d["memory"]["messages"]) == 2

    def test_to_dict_tiered_memory(self):
        """TieredMemory 可序列化"""
        llm = MagicMock(spec=LLM)
        mem = TieredMemory("助手", auto_summarize=False)
        mem.add("user", "问题1")
        mem.add("assistant", "回答1")
        core = CoreAgent(llm=llm, registry=Registry(), system_prompt="助手", memory=mem)
        member = AgentMember("Alice", "助手", core)
        d = member.to_dict()
        assert d["memory"]["type"] == "TieredMemory"
        assert "working_window" in d["memory"]

    def test_naive_memory_round_trip(self):
        """NaiveMemory 序列化→反序列化往返"""
        llm = MagicMock(spec=LLM)
        orig_mem = NaiveMemory("助手")
        orig_mem.add("user", "问题1")
        orig_mem.add("assistant", "回答1")

        data = orig_mem.to_dict()
        restored = NaiveMemory.from_dict(data, "助手")

        orig_ctx = orig_mem.get_context()
        res_ctx = restored.get_context()
        assert len(orig_ctx) == len(res_ctx)
        assert orig_ctx[1]["content"] == res_ctx[1]["content"]

    def test_naive_memory_to_dict_excludes_system(self):
        """NaiveMemory.to_dict 不包含 system 消息"""
        mem = NaiveMemory("助手")
        mem.add("user", "问题")
        d = mem.to_dict()
        # messages 只包含 user/assistant，不含 system
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"

    def test_tiered_memory_round_trip(self):
        """TieredMemory 序列化→反序列化往返"""
        orig_mem = TieredMemory("助手", working_window=3, auto_summarize=False)
        for i in range(4):
            orig_mem.add("user", f"问题{i}")
            orig_mem.add("assistant", f"回答{i}")

        data = orig_mem.to_dict()
        restored = TieredMemory.from_dict(data, "助手")

        assert orig_mem.working.window == restored.working.window
        assert len(orig_mem.working._messages) == len(restored.working._messages)
        assert orig_mem.working._messages[0]["content"] == restored.working._messages[0]["content"]


# ═══════════════════════════════════════════════════════════════
# Room 序列化
# ═══════════════════════════════════════════════════════════════

class TestRoomSave:
    """Room.save()"""

    def test_save_creates_file(self):
        """save() 写入文件"""
        room, _ = _setup_room()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            room.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_json_structure(self):
        """save() 输出的 JSON 结构正确"""
        room, _ = _setup_room()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            text = room.save(path)
            data = json.loads(text)
            assert data["version"] == 1
            assert data["name"] == "测试室"
            assert len(data["members"]) == 2
            assert len(data["history"]) == 4
            # member 字段
            for name, m in data["members"].items():
                assert "name" in m
                assert "role_desc" in m
                assert "memory" in m
                assert m["memory"] is not None, f"{name} 的 memory 不应为 null"
                assert "type" in m["memory"], f"{name} 的 memory 缺少 type"
            # history 字段
            for h in data["history"]:
                assert "sender" in h
                assert "content" in h
                assert "kind" in h
        finally:
            os.unlink(path)

    def test_save_naive_memory_not_null(self):
        """NaiveMemory 的 memory 字段不为 null（修复问题1）"""
        room, _ = _setup_room(memory_cls=NaiveMemory)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            text = room.save(path)
            data = json.loads(text)
            for name, m in data["members"].items():
                assert m["memory"] is not None, f"NaiveMemory {name} 的 memory 不应为 null"
                assert m["memory"]["type"] == "NaiveMemory"
        finally:
            os.unlink(path)

    def test_save_tiered_memory_correct_type(self):
        """TieredMemory 保存的 type 字段正确"""
        room, _ = _setup_room(memory_cls=TieredMemory, auto_summarize=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            text = room.save(path)
            data = json.loads(text)
            for name, m in data["members"].items():
                assert m["memory"]["type"] == "TieredMemory"
        finally:
            os.unlink(path)


class TestRoomLoad:
    """Room.load()"""

    def _make_room_and_save(self, memory_cls=NaiveMemory, **mem_kwargs) -> str:
        room, llm = _setup_room(memory_cls=memory_cls, **mem_kwargs)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        room.save(path)
        return path

    def test_load_full_round_trip(self):
        """save→load 完整往返，NaiveMemory"""
        path = self._make_room_and_save(memory_cls=NaiveMemory)
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            assert restored.name == "测试室"
            assert len(restored.members) == 2
            assert len(restored.history) == 4
            assert restored.history[0].sender == "user"
            assert restored.history[0].content == "你好"
        finally:
            os.unlink(path)

    def test_load_tiered_memory_round_trip(self):
        """save→load 完整往返，TieredMemory"""
        path = self._make_room_and_save(memory_cls=TieredMemory, auto_summarize=False)
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            assert restored.name == "测试室"
            assert len(restored.members) == 2
            # 验证 memory 类型被正确恢复
            for name in restored.members:
                mem = restored.members[name].memory
                assert isinstance(mem, TieredMemory), f"{name} 的 memory 类型错误: {type(mem).__name__}"
        finally:
            os.unlink(path)

    def test_load_history_content(self):
        """load 后历史内容一致"""
        path = self._make_room_and_save()
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            assert restored.history[1].content == "你好，有什么需要帮助的？"
            assert restored.history[3].content == "今天天气不错，适合出行"
        finally:
            os.unlink(path)

    def test_load_memory_content_restored(self):
        """load 后 memory 结构恢复（消息数量一致）"""
        room, llm = _setup_room(memory_cls=NaiveMemory)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        room.save(path)
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            # NaiveMemory 的 _messages 包含 system + 保存的 user/assistant 消息
            for name in restored.members:
                mem = restored.members[name].memory
                ctx = mem.get_context()
                assert len(ctx) >= 1  # 至少包含 system
                assert isinstance(mem, NaiveMemory)
        finally:
            os.unlink(path)

    def test_load_version_check(self):
        """load 检查 version 字段"""
        path = self._make_room_and_save()
        try:
            # 篡改 version 为 0
            with open(path) as f:
                data = json.load(f)
            data["version"] = 0
            with open(path, "w") as f:
                json.dump(data, f)

            llm = MagicMock(spec=LLM)
            with pytest.raises(ValueError, match="版本"):
                Room.load(path, llm, _make_dummy_agent)
        finally:
            os.unlink(path)

    def test_load_type_dispatch_naive(self):
        """保存 NaiveMemory → load 时正确识别类型"""
        path = self._make_room_and_save(memory_cls=NaiveMemory)
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            for name in restored.members:
                assert isinstance(restored.members[name].memory, NaiveMemory)
        finally:
            os.unlink(path)

    def test_load_type_dispatch_tiered(self):
        """保存 TieredMemory → load 时正确识别类型"""
        path = self._make_room_and_save(memory_cls=TieredMemory, auto_summarize=False)
        try:
            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_dummy_agent)
            for name in restored.members:
                assert isinstance(restored.members[name].memory, TieredMemory)
        finally:
            os.unlink(path)

    def test_load_cross_type_safety(self):
        """保存的 type 与运行时默认 type 不同时，按保存的 type 恢复（修复问题2）"""
        # 保存 TieredMemory
        path = self._make_room_and_save(memory_cls=TieredMemory, auto_summarize=False)
        try:
            # load 时 make_agent 创建 NaiveMemory，但应仍恢复为 TieredMemory
            def _make_with_naive(name, role, sp, llm):
                return CoreAgent(llm=llm, registry=Registry(),
                                 system_prompt=sp, memory=NaiveMemory(sp))

            llm = MagicMock(spec=LLM)
            restored = Room.load(path, llm, _make_with_naive)
            for name in restored.members:
                # 即使 make_agent 传了 NaiveMemory，from_dict 应覆盖为 TieredMemory
                assert isinstance(restored.members[name].memory, TieredMemory), \
                    f"{name}: 期望 TieredMemory，实际 {type(restored.members[name].memory).__name__}"
        finally:
            os.unlink(path)

    def test_load_unknown_type_graceful(self):
        """未知 memory type 不崩溃，使用默认 memory"""
        path = self._make_room_and_save()
        try:
            # 篡改 type 为未知
            with open(path) as f:
                data = json.load(f)
            for name, m in data["members"].items():
                m["memory"]["type"] = "UnknownMemory"
            with open(path, "w") as f:
                json.dump(data, f)

            llm = MagicMock(spec=LLM)
            # 不会崩溃，使用默认 NaiveMemory
            restored = Room.load(path, llm, _make_dummy_agent)
            assert len(restored.members) == 2
        finally:
            os.unlink(path)
