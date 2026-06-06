"""测试工具注册表 — tool.py"""
from __future__ import annotations
import json
from collabroom.core.tool import Tool, Registry, tool_result, tool_error


class TestTool:
    def test_basic(self):
        t = Tool(
            name="echo",
            description="回显输入",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: tool_result(echo=text),
        )
        assert t.name == "echo"
        assert t.description == "回显输入"

    def test_to_openai_schema(self):
        t = Tool(
            name="read_file",
            description="读取文件",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
            fn=lambda path: "",
        )
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert "parameters" in schema["function"]


class TestToolResult:
    def test_tool_result_kwargs(self):
        r = tool_result(success=True, chars=42)
        data = json.loads(r)
        assert data["success"] is True
        assert data["chars"] == 42

    def test_tool_result_dict(self):
        r = tool_result({"key": "value", "num": 1})
        data = json.loads(r)
        assert data["key"] == "value"
        assert data["num"] == 1

    def test_tool_result_empty(self):
        r = tool_result()
        data = json.loads(r)
        assert data == {}

    def test_tool_error_string(self):
        r = tool_error("file not found")
        data = json.loads(r)
        assert data["error"] == "file not found"

    def test_tool_error_with_extra(self):
        r = tool_error("权限不足", code=403)
        data = json.loads(r)
        assert data["error"] == "权限不足"
        assert data["code"] == 403


class TestRegistry:
    def test_register_and_list(self, empty_registry):
        assert empty_registry.list_tools() == []
        t = Tool(name="echo", description="回显", parameters={}, fn=lambda: "")
        empty_registry.register(t)
        assert empty_registry.list_tools() == ["echo"]

    def test_get_definitions(self, sample_registry):
        defs = sample_registry.get_definitions()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert "echo" in names
        assert "add" in names

    def test_execute_success(self, sample_registry):
        result = sample_registry.execute("echo", {"text": "hello"})
        data = json.loads(result)
        assert data["echo"] == "hello"

    def test_execute_with_number_args(self, sample_registry):
        result = sample_registry.execute("add", {"a": 3, "b": 4})
        data = json.loads(result)
        assert data["sum"] == 7

    def test_execute_unknown_tool(self, sample_registry):
        result = sample_registry.execute("nonexistent", {})
        data = json.loads(result)
        assert "error" in data
        assert "未知工具" in data["error"]

    def test_execute_error_handling(self, sample_registry):
        """工具内部抛异常时返回结构化错误"""
        # 注册一个会抛异常的工具
        def failing_fn(x):
            raise ValueError("测试错误")
        sample_registry.register(Tool(
            name="fail",
            description="必失败",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            fn=failing_fn,
        ))
        result = sample_registry.execute("fail", {"x": "test"})
        data = json.loads(result)
        assert "error" in data
        assert "ValueError" in data["error"]

    def test_execute_non_string_return(self, sample_registry):
        """工具返回非字符串时自动 json.dumps"""
        sample_registry.register(Tool(
            name="get_list",
            description="返回列表",
            parameters={},
            fn=lambda: [1, 2, 3],
        ))
        result = sample_registry.execute("get_list", {})
        data = json.loads(result)
        assert data == [1, 2, 3]
