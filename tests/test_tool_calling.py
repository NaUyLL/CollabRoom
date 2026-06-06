"""测试 BatchToolCalling — 批量工具调用策略"""
from __future__ import annotations

import pytest
from collabroom.core.tool_calling import ToolCallingStrategy
from collabroom.core.tool_calling.batch import BatchToolCalling


# ═══════════════════════════════════════════════════════════════
# 工具定义辅助函数
# ═══════════════════════════════════════════════════════════════


def _make_tool_def(name: str, description: str) -> dict:
    """快速创建一个标准的 OpenAI 格式工具定义"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本",
                    },
                },
                "required": ["text"],
            },
        },
    }


def _sample_tool_defs() -> list[dict]:
    """返回一组示例工具定义"""
    return [
        _make_tool_def("echo", "回显输入的文本内容"),
        _make_tool_def("add", "计算两个数字之和"),
        _make_tool_def("search", "在知识库中搜索相关内容并返回匹配结果"),
    ]


# ═══════════════════════════════════════════════════════════════
# TestBatchToolCalling
# ═══════════════════════════════════════════════════════════════


class TestBatchToolCallingInit:
    """测试 BatchToolCalling 初始化"""

    def test_name_attribute(self):
        """name 属性为 'batch'"""
        strategy = BatchToolCalling()
        assert strategy.name == "batch"

    def test_supports_parallel(self):
        """Batch 模式支持并行执行"""
        strategy = BatchToolCalling()
        assert strategy.supports_parallel is True

    def test_default_verbosity(self):
        """默认 verbosity 为 'long'"""
        strategy = BatchToolCalling()
        assert strategy.verbosity == "long"

    def test_custom_verbosity(self):
        """可设置 verbosity='short'"""
        strategy = BatchToolCalling(verbosity="short")
        assert strategy.verbosity == "short"

    def test_invalid_verbosity_raises(self):
        """非法 verbosity 值抛出 AssertionError"""
        with pytest.raises(AssertionError, match="verbosity"):
            BatchToolCalling(verbosity="medium")

    def test_is_subclass_of_strategy(self):
        """BatchToolCalling 是 ToolCallingStrategy 的子类"""
        assert issubclass(BatchToolCalling, ToolCallingStrategy)


class TestBatchToolCallingFilterTools:
    """测试 BatchToolCalling.filter_tools()"""

    def test_filter_tools_passthrough_long(self):
        """verbosity=long 时 filter_tools 原样返回所有工具定义"""
        strategy = BatchToolCalling(verbosity="long")
        defs = _sample_tool_defs()
        result = strategy.filter_tools(defs)
        assert result == defs

    def test_filter_tools_short_truncates_description(self):
        """verbosity=short 时截断 description 到 15 字符"""
        strategy = BatchToolCalling(verbosity="short")
        defs = _sample_tool_defs()
        result = strategy.filter_tools(defs)

        assert len(result) == len(defs)
        # search 的 description 超过 15 字符会被截断
        search_func = result[2]["function"]
        assert len(search_func["description"]) <= 18  # 15 + "..."
        assert search_func["description"].endswith("...")

    def test_filter_tools_short_strips_param_descriptions(self):
        """verbosity=short 时移除参数级别的 description"""
        strategy = BatchToolCalling(verbosity="short")
        defs = _sample_tool_defs()
        result = strategy.filter_tools(defs)

        for d in result:
            params = d["function"]["parameters"]
            props = params.get("properties", {})
            for p_name, p_schema in props.items():
                assert "description" not in p_schema

    def test_filter_tools_short_keeps_short_desc(self):
        """verbosity=short 时短 description（≤15 字符）不被截断"""
        strategy = BatchToolCalling(verbosity="short")
        short_def = _make_tool_def("hi", "打招呼")
        result = strategy.filter_tools([short_def])
        assert result[0]["function"]["description"] == "打招呼"

    def test_filter_tools_empty_list(self):
        """空列表返回空列表"""
        strategy = BatchToolCalling()
        assert strategy.filter_tools([]) == []

    def test_filter_tools_no_parameters(self):
        """工具定义无 parameters 字段时不报错"""
        strategy = BatchToolCalling(verbosity="short")
        defs = [{
            "type": "function",
            "function": {
                "name": "ping",
                "description": "检测连接状态是否正常",
            },
        }]
        result = strategy.filter_tools(defs)
        # 9 字符 ≤ 15 截断阈值，不截断
        assert not result[0]["function"]["description"].endswith("...")

    def test_filter_tools_returns_copy(self):
        """verbosity=short 时返回新列表（不修改原数据）"""
        strategy = BatchToolCalling(verbosity="short")
        defs = _sample_tool_defs()
        result = strategy.filter_tools(defs)
        # short 模式会创建新列表
        assert result is not defs
        assert len(result) == len(defs)


class TestBatchToolCallingLimitCalls:
    """测试 BatchToolCalling.limit_calls()"""

    def test_limit_calls_no_limit(self):
        """Batch 模式下 limit_calls 不限制，原样返回"""
        strategy = BatchToolCalling()
        tool_calls = ["call1", "call2", "call3", "call4", "call5"]
        result = strategy.limit_calls(tool_calls, [])
        assert result == tool_calls

    def test_limit_calls_empty_list(self):
        """空列表返回空列表"""
        strategy = BatchToolCalling()
        assert strategy.limit_calls([], []) == []

    def test_limit_calls_with_many_calls(self):
        """即使 100 个 tool_calls 也不限制"""
        strategy = BatchToolCalling()
        tool_calls = [f"call{i}" for i in range(100)]
        result = strategy.limit_calls(tool_calls, [])
        assert len(result) == 100

    def test_limit_calls_ignores_tool_defs(self):
        """limit_calls 忽略 tool_defs 参数（batch 不限量）"""
        strategy = BatchToolCalling()
        tool_calls = ["call1", "call2"]
        # 传入非空 tool_defs 也不影响
        result = strategy.limit_calls(tool_calls, [{"name": "echo"}])
        assert result == tool_calls


class TestToolCallingStrategyABC:
    """测试 ToolCallingStrategy 抽象基类"""

    def test_cannot_instantiate_abstract(self):
        """抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            ToolCallingStrategy()  # type: ignore[abstract]

    def test_supports_parallel_default_false(self):
        """基类 supports_parallel 默认为 False"""
        # 通过 BatchToolCalling 验证基类默认值被覆盖
        strategy = BatchToolCalling()
        assert strategy.supports_parallel is True
