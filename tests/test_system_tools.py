"""测试系统工具 — collabroom/core/system_tools.py"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from collabroom.core.system_tools import (
    # 安全
    _is_blocked_device,
    _check_sensitive_path,
    _safety_limit,
    MAX_OUTPUT_CHARS,
    MAX_TERMINAL_TIMEOUT,
    BLOCKED_DEVICE_PATHS,
    SENSITIVE_PATH_PREFIXES,
    SENSITIVE_EXACT_PATHS,
    # 跟踪
    _track_read,
    _track_search,
    _check_staleness,
    _update_mtime,
    _read_tracker,
    # 工具函数
    _read_file,
    _write_file,
    _patch,
    _search_files,
    _terminal,
    _get_current_time,
    _execute_code,
    # 注册
    SYSTEM_TOOLS,
    register_all,
    register_defaults,
)
from collabroom.core.tool import Registry, tool_result, tool_error


# ═══════════════════════════════════════════════════════════════
# _is_blocked_device — 阻塞设备路径检测
# ═══════════════════════════════════════════════════════════════

class TestIsBlockedDevice:
    """_is_blocked_device() 阻塞设备路径检测"""

    def test_dev_zero_被阻止(self):
        """/dev/zero 应被识别为阻塞设备"""
        assert _is_blocked_device("/dev/zero") is True

    def test_dev_random_被阻止(self):
        """/dev/random 应被识别为阻塞设备"""
        assert _is_blocked_device("/dev/random") is True

    def test_dev_urandom_被阻止(self):
        """/dev/urandom 应被识别为阻塞设备"""
        assert _is_blocked_device("/dev/urandom") is True

    def test_dev_stdin_被阻止(self):
        """/dev/stdin 应被识别为阻塞设备"""
        assert _is_blocked_device("/dev/stdin") is True

    def test_dev_tty_被阻止(self):
        """/dev/tty 应被识别为阻塞设备"""
        assert _is_blocked_device("/dev/tty") is True

    def test_普通文件不被阻止(self):
        """普通文件路径不应被阻止"""
        assert _is_blocked_device("/tmp/test.txt") is False

    def test_普通目录不被阻止(self):
        """目录路径不应被阻止"""
        assert _is_blocked_device("/home/user") is False

    def test_空字符串不被阻止(self):
        """空字符串不应被阻止"""
        assert _is_blocked_device("") is False

    def test_相对路径不被阻止(self):
        """相对路径不应被阻止"""
        assert _is_blocked_device("test.py") is False

    def test_proc_fd路径被阻止(self):
        """/proc/self/fd/0 等路径应被阻止"""
        assert _is_blocked_device("/proc/self/fd/0") is True
        assert _is_blocked_device("/proc/1234/fd/1") is True
        assert _is_blocked_device("/proc/1/fd/2") is True

    def test_proc普通文件不被阻止(self):
        """/proc/cpuinfo 等普通 proc 文件不应被阻止"""
        assert _is_blocked_device("/proc/cpuinfo") is False

    def test_符号链接解析到设备(self, tmp_workspace):
        """符号链接解析后指向阻塞设备 → True"""
        # 创建指向 /dev/zero 的符号链接
        os.symlink("/dev/zero", os.path.join(tmp_workspace, "zero_link"))
        assert _is_blocked_device(os.path.join(tmp_workspace, "zero_link")) is True


# ═══════════════════════════════════════════════════════════════
# _check_sensitive_path — 敏感路径检测
# ═══════════════════════════════════════════════════════════════

class TestCheckSensitivePath:
    """_check_sensitive_path() 敏感路径检测"""

    def test_etc路径被拒绝(self):
        """/etc/ 下的路径被拒绝写入"""
        result = _check_sensitive_path("/etc/hosts")
        assert result is not None
        assert "拒绝写入" in result

    def test_boot路径被拒绝(self):
        """/boot/ 下的路径被拒绝写入"""
        result = _check_sensitive_path("/boot/config.txt")
        assert result is not None

    def test_dev路径被拒绝(self):
        """/dev/ 下的路径被拒绝写入"""
        result = _check_sensitive_path("/dev/sda")
        assert result is not None

    def test_proc路径被拒绝(self):
        """/proc/ 下的路径被拒绝写入"""
        result = _check_sensitive_path("/proc/cmdline")
        assert result is not None

    def test_sys路径被拒绝(self):
        """/sys/ 下的路径被拒绝写入"""
        result = _check_sensitive_path("/sys/class/foo")
        assert result is not None

    def test_docker_sock精确匹配被拒绝(self):
        """/var/run/docker.sock 精确匹配被拒绝"""
        result = _check_sensitive_path("/var/run/docker.sock")
        assert result is not None

    def test_run_docker_sock精确匹配被拒绝(self):
        """/run/docker.sock 精确匹配被拒绝"""
        result = _check_sensitive_path("/run/docker.sock")
        assert result is not None

    def test_普通路径通过(self):
        """普通用户路径不被拒绝"""
        result = _check_sensitive_path("/home/user/test.txt")
        assert result is None

    def test_tmp路径通过(self):
        """/tmp/ 路径不被拒绝"""
        result = _check_sensitive_path("/tmp/work/test.py")
        assert result is None

    def test_sensitive_path_prefixes常量(self):
        """验证 SENSITIVE_PATH_PREFIXES 包含关键路径"""
        assert "/etc/" in SENSITIVE_PATH_PREFIXES
        assert "/boot/" in SENSITIVE_PATH_PREFIXES
        assert "/dev/" in SENSITIVE_PATH_PREFIXES
        assert "/proc/" in SENSITIVE_PATH_PREFIXES
        assert "/sys/" in SENSITIVE_PATH_PREFIXES

    def test_sensitive_exact_paths常量(self):
        """验证 SENSITIVE_EXACT_PATHS 包含关键路径"""
        assert "/var/run/docker.sock" in SENSITIVE_EXACT_PATHS
        assert "/run/docker.sock" in SENSITIVE_EXACT_PATHS


# ═══════════════════════════════════════════════════════════════
# _safety_limit — 输出截断
# ═══════════════════════════════════════════════════════════════

class TestSafetyLimit:
    """_safety_limit() 输出截断测试"""

    def test_短文本不截断(self):
        """小于 MAX_OUTPUT_CHARS 的文本原样返回"""
        text = "短文本"
        result = _safety_limit(text)
        assert result == text

    def test_超长文本截断(self):
        """超过上限时截断并附加标记"""
        text = "x" * (MAX_OUTPUT_CHARS + 100)
        result = _safety_limit(text)
        assert len(result) < len(text)
        assert "输出截断" in result
        assert str(MAX_OUTPUT_CHARS) in result
        assert "x" * MAX_OUTPUT_CHARS in result

    def test_正好等于上限不截断(self):
        """正好等于 MAX_OUTPUT_CHARS 的文本不截断"""
        text = "y" * MAX_OUTPUT_CHARS
        result = _safety_limit(text)
        assert result == text

    def test_自定义max_chars(self):
        """自定义截断上限"""
        text = "abcdefghij"
        result = _safety_limit(text, max_chars=5)
        assert len(result) > 5  # 有附加标记
        assert "abcde" in result
        assert "截断" in result

    def test_empty文本(self):
        """空文本不报错"""
        result = _safety_limit("")
        assert result == ""


# ═══════════════════════════════════════════════════════════════
# _track_read/_track_search — 重复检测
# ═══════════════════════════════════════════════════════════════

class TestReadTracker:
    """_track_read() 和 _track_search() 重复检测"""

    @pytest.fixture(autouse=True)
    def _reset_tracker(self):
        """每个测试前重置全局读取跟踪器"""
        _read_tracker.clear()

    def test_第一次读取_允许(self):
        """首次读取不阻止"""
        ok, msg = _track_read("/tmp/test.txt", 1, 500)
        assert ok is True
        assert msg == ""

    def test_连续4次相同读取_阻止(self):
        """连续 4 次相同 (path, offset, limit) → 阻止"""
        # 前 3 次允许
        for _ in range(3):
            ok, _ = _track_read("/tmp/test.txt", 1, 500)
            assert ok is True
        # 第 4 次阻止
        ok, msg = _track_read("/tmp/test.txt", 1, 500)
        assert ok is False
        assert "BLOCKED" in msg

    def test_不同offset不算重复(self):
        """不同 offset 不算重复读取"""
        # 先读 offset=1
        _track_read("/tmp/test.txt", 1, 500)
        _track_read("/tmp/test.txt", 1, 500)
        _track_read("/tmp/test.txt", 1, 500)
        # 改变 offset
        ok, msg = _track_read("/tmp/test.txt", 501, 500)
        assert ok is True

    def test_改变参数重置计数(self):
        """改变 (path, offset, limit) 任意一个 → 重置计数"""
        for _ in range(3):
            _track_read("/tmp/a.txt", 1, 500)
        # 改变 offset
        ok, _ = _track_read("/tmp/a.txt", 2, 500)
        assert ok is True

    def test_track_search_连续4次阻止(self):
        """连续 4 次相同搜索 → 阻止"""
        for _ in range(3):
            ok, _ = _track_search("hello", ".", "content")
            assert ok is True
        ok, msg = _track_search("hello", ".", "content")
        assert ok is False
        assert "BLOCKED" in msg

    def test_track_search_第3次警告(self):
        """连续 3 次相同搜索 → 警告但允许"""
        _track_search("hello", ".", "content")
        _track_search("hello", ".", "content")
        ok, msg = _track_search("hello", ".", "content")
        assert ok is True
        assert "警告" in msg

    def test_track_search_不同pattern重置(self):
        """改变 pattern 重置计数"""
        for _ in range(3):
            _track_search("hello", ".", "content")
        # 新搜索
        ok, _ = _track_search("world", ".", "content")
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# _check_staleness / _update_mtime
# ═══════════════════════════════════════════════════════════════

class TestStaleness:
    """_check_staleness() 和 _update_mtime() 陈旧检测"""

    def test_未读取过的文件_无警告(self, tmp_workspace):
        """未在 read_mtimes 中记录的文件 → None"""
        path = os.path.join(tmp_workspace, "new.txt")
        Path(path).write_text("hello")
        assert _check_staleness(path) is None

    def test_读取后未修改_无警告(self, tmp_workspace):
        """记录 mtime 后文件未变化 → None"""
        path = os.path.join(tmp_workspace, "stable.txt")
        Path(path).write_text("stable")
        _track_read(path, 1, 500)
        assert _check_staleness(path) is None

    def test_读取后被外部修改_警告(self, tmp_workspace):
        """记录 mtime 后文件被外部修改 → 返回警告"""
        path = os.path.join(tmp_workspace, "changing.txt")
        Path(path).write_text("v1")
        _track_read(path, 1, 500)
        # 外部修改
        time.sleep(0.01)
        Path(path).write_text("v2")
        warn = _check_staleness(path)
        assert warn is not None
        assert "警告" in warn or "被修改" in warn

    def test_update_mtime后无警告(self, tmp_workspace):
        """_update_mtime 更新记录后不会报陈旧警告"""
        path = os.path.join(tmp_workspace, "tracked.txt")
        Path(path).write_text("v1")
        _track_read(path, 1, 500)
        # 修改后立即 update
        Path(path).write_text("v2")
        _update_mtime(path)
        assert _check_staleness(path) is None

    def test_文件不存在时不报错(self):
        """文件不存在时 _check_staleness 返回 None 不抛异常"""
        assert _check_staleness("/nonexistent/path.txt") is None


# ═══════════════════════════════════════════════════════════════
# _read_file — 文件读取
# ═══════════════════════════════════════════════════════════════

class TestReadFile:
    """_read_file() 文件读取测试"""

    def test_正常读取文件(self, sample_text_file):
        """正常读取文本文件，返回带行号内容"""
        result = _read_file(sample_text_file)
        # _read_file 返回纯文本（带行号），不是 JSON
        assert "hello world" in result
        assert "第1行" in result or "第2行" in result

    def test_正常读取返回行号格式(self, sample_text_file):
        """返回带行号的文件内容"""
        result = _read_file(sample_text_file)
        assert "hello world" in result

    def test_分页读取_offset参数(self, sample_text_file):
        """offset=2 从第二行开始显示"""
        result = _read_file(sample_text_file, offset=2)
        assert "第2行" in result or "foo" in result

    def test_分页读取_limit参数(self, sample_text_file):
        """limit=1 只显示一行"""
        result = _read_file(sample_text_file, limit=1)
        lines = result.split("\n")
        # 第一行是文件头信息，后面是具体内容
        content_lines = [l for l in lines if l.strip() and "文件:" not in l and "..." not in l]
        assert len(content_lines) <= 1

    def test_文件不存在_返回错误(self, tmp_workspace):
        """文件不存在时返回 tool_error 格式"""
        path = os.path.join(tmp_workspace, "no_such_file.txt")
        result = _read_file(path)
        assert "error" in result
        assert "不存在" in result or "no_such" in result

    def test_设备路径被拒绝(self):
        """读取 /dev/zero 被拒绝"""
        result = _read_file("/dev/zero")
        assert "error" in result
        assert "拒绝" in result or "设备" in result

    def test_不是文件的路径被拒绝(self, tmp_workspace):
        """读取目录路径返回错误"""
        result = _read_file(tmp_workspace)
        assert "error" in result

    def test_offset自动钳制到至少1(self, sample_text_file):
        """offset 为 0 时自动调整到 1"""
        result = _read_file(sample_text_file, offset=0)
        # 不应报错
        assert "error" not in result

    def test_limit自动钳制(self, sample_text_file):
        """limit 超过 2000 时自动调整"""
        result = _read_file(sample_text_file, limit=9999)
        assert "error" not in result


# ═══════════════════════════════════════════════════════════════
# _write_file — 文件写入
# ═══════════════════════════════════════════════════════════════

class TestWriteFile:
    """_write_file() 文件写入测试"""

    def test_正常写入文件(self, tmp_workspace):
        """正常路径下写入文件成功"""
        path = os.path.join(tmp_workspace, "output.txt")
        result = _write_file(path, "hello world")
        data = json.loads(result)
        assert data.get("ok") is True
        assert os.path.exists(path)
        assert Path(path).read_text() == "hello world"

    def test_自动创建父目录(self, tmp_workspace):
        """写入路径的父目录不存在时自动创建"""
        path = os.path.join(tmp_workspace, "deep/nested/dir/output.txt")
        result = _write_file(path, "deep content")
        data = json.loads(result)
        assert data.get("ok") is True
        assert os.path.exists(path)

    def test_敏感路径被拒绝(self):
        """写入 /etc/ 路径被拒绝"""
        result = _write_file("/etc/test.conf", "data")
        data = json.loads(result)
        assert "error" in data
        assert "拒绝" in data["error"] or "敏感" in data["error"]

    def test_覆盖已有文件(self, tmp_workspace):
        """写入已存在的文件会覆盖内容"""
        path = os.path.join(tmp_workspace, "overwrite.txt")
        _write_file(path, "old content")
        _write_file(path, "new content")
        assert Path(path).read_text() == "new content"

    def test_写入大内容(self, tmp_workspace):
        """正常写入较大内容"""
        path = os.path.join(tmp_workspace, "large.txt")
        content = "line\n" * 999 + "line"  # 1000 行，不带尾部换行
        result = _write_file(path, content)
        data = json.loads(result)
        assert data.get("ok") is True
        assert data.get("lines") == 1000

    def test_写入空内容(self, tmp_workspace):
        """写入空字符串也成功"""
        path = os.path.join(tmp_workspace, "empty.txt")
        result = _write_file(path, "")
        data = json.loads(result)
        assert data.get("ok") is True
        assert Path(path).read_text() == ""

    def test_写入后更新mtime(self, tmp_workspace):
        """写入后 _update_mtime 被调用，后续 _check_staleness 无警告"""
        path = os.path.join(tmp_workspace, "mt_test.txt")
        # 先读取建立记录
        Path(path).write_text("v1")
        _track_read(path, 1, 500)
        # 写入
        _write_file(path, "v2")
        # 应该无陈旧警告
        warn = _check_staleness(path)
        assert warn is None


# ═══════════════════════════════════════════════════════════════
# _patch — 文本替换
# ═══════════════════════════════════════════════════════════════

class TestPatch:
    """_patch() 文本替换测试"""

    def test_精确替换成功(self, tmp_workspace):
        """精确匹配后替换成功"""
        path = os.path.join(tmp_workspace, "patch.txt")
        Path(path).write_text("hello world")
        result = _patch(path, "hello", "你好")
        data = json.loads(result)
        assert data.get("ok") is True
        assert data.get("replacements") == 1
        assert Path(path).read_text() == "你好 world"

    def test_replace_all替换所有匹配(self, tmp_workspace):
        """replace_all=True 替换所有匹配项"""
        path = os.path.join(tmp_workspace, "patch_all.txt")
        Path(path).write_text("foo bar foo baz foo")
        result = _patch(path, "foo", "qux", replace_all=True)
        data = json.loads(result)
        assert data.get("ok") is True
        assert data.get("replacements") == 3
        assert Path(path).read_text() == "qux bar qux baz qux"

    def test_未找到匹配_返回错误(self, tmp_workspace):
        """old_string 在文件中不存在 → 返回错误"""
        path = os.path.join(tmp_workspace, "patch_no.txt")
        Path(path).write_text("hello world")
        result = _patch(path, "goodbye", "再见")
        data = json.loads(result)
        assert "error" in data
        assert "未找到" in data["error"]

    def test_空文件未找到匹配_返回错误(self, tmp_workspace):
        """空文件中查找 → 返回错误"""
        path = os.path.join(tmp_workspace, "empty_patch.txt")
        Path(path).write_text("")
        result = _patch(path, "anything", "else")
        data = json.loads(result)
        assert "error" in data

    def test_敏感路径被拒绝(self):
        """patch /etc/hosts 被拒绝"""
        result = _patch("/etc/hosts", "old", "new")
        data = json.loads(result)
        assert "error" in data

    def test_文件不存在_返回错误(self, tmp_workspace):
        """目标文件不存在 → 返回错误"""
        path = os.path.join(tmp_workspace, "no_file.txt")
        result = _patch(path, "old", "new")
        data = json.loads(result)
        assert "error" in data

    def test_返回diff_preview(self, tmp_workspace):
        """返回值包含 diff_preview"""
        path = os.path.join(tmp_workspace, "diff_test.txt")
        Path(path).write_text("original")
        result = _patch(path, "original", "modified")
        data = json.loads(result)
        assert "diff_preview" in data


# ═══════════════════════════════════════════════════════════════
# _search_files — 文件搜索
# ═══════════════════════════════════════════════════════════════

class TestSearchFiles:
    """_search_files() 文件搜索测试"""

    def test_内容搜索_找到匹配(self, tmp_workspace):
        """在文件内容中搜索匹配的字符串"""
        # 创建测试文件
        path = os.path.join(tmp_workspace, "search_test.py")
        Path(path).write_text('print("hello world")\n# comment line\n')
        result = _search_files("hello world", target="content", path=tmp_workspace)
        assert "hello" in result

    def test_内容搜索_无匹配(self, tmp_workspace):
        """搜索不存在的字符串 → 无匹配"""
        Path(os.path.join(tmp_workspace, "empty.py")).write_text("# nothing here\n")
        result = _search_files("zzzz_not_found_zzzz", target="content", path=tmp_workspace)
        assert "无匹配" in result

    def test_文件搜索_按文件名glob(self, tmp_workspace):
        """target='files' 按文件名匹配"""
        path = os.path.join(tmp_workspace, "my_module.py")
        Path(path).write_text("# module\n")
        result = _search_files("*.py", target="files", path=tmp_workspace)
        assert "my_module.py" in result

    def test_路径不存在_返回错误(self, tmp_workspace):
        """搜索路径不存在 → 错误"""
        result = _search_files("hello", target="content", path="/no/such/path")
        assert "error" in result
        assert "不存在" in result

    def test_file_glob过滤(self, tmp_workspace):
        """使用 file_glob 只搜索特定类型文件"""
        Path(os.path.join(tmp_workspace, "test.py")).write_text("hello")
        Path(os.path.join(tmp_workspace, "test.txt")).write_text("hello")
        # 只搜 .py
        result = _search_files("hello", target="content", path=tmp_workspace, file_glob="*.py")
        assert "test.py" in result
        # .txt 不应出现在结果中（除非 glob 匹配它）
        # 注意：file_glob 是用 Path.match，有通配符语义

    def test_limit限制结果数(self, tmp_workspace):
        """limit 参数限制最大返回结果数"""
        for i in range(5):
            Path(os.path.join(tmp_workspace, f"file_{i}.py")).write_text(f"# content {i}\n")
        result = _search_files("content", target="content", path=tmp_workspace, limit=2)
        # 检查结果数量（应显示前 2 个）
        assert "file_" in result


# ═══════════════════════════════════════════════════════════════
# _terminal — 命令执行
# ═══════════════════════════════════════════════════════════════

class TestTerminal:
    """_terminal() 命令执行测试"""

    def test_简单命令执行(self):
        """执行 echo 命令返回输出"""
        result = _terminal("echo hello")
        data = json.loads(result)
        assert data.get("exit_code") == 0
        assert "hello" in data.get("output", "")

    def test_失败命令返回非零exit_code(self):
        """执行失败命令返回非零 exit_code"""
        result = _terminal("exit 1")
        data = json.loads(result)
        assert data.get("exit_code") == 1

    def test_stderr捕获(self):
        """stderr 输出被捕获"""
        result = _terminal("echo error >&2")
        data = json.loads(result)
        output = data.get("output", "")
        assert "error" in output or "stderr" in output

    def test_workdir参数(self, tmp_workspace):
        """指定 workdir 在该目录执行"""
        subdir = os.path.join(tmp_workspace, "subdir")
        os.makedirs(subdir, exist_ok=True)
        result = _terminal("pwd", workdir=subdir)
        data = json.loads(result)
        assert subdir in data.get("output", "")

    def test_timeout较小值被限制(self):
        """传入超时值大于 MAX_TERMINAL_TIMEOUT 时被限制"""
        assert MAX_TERMINAL_TIMEOUT == 120
        # 即使传 999，实际 timeout 也是 min(999, 120)=120
        result = _terminal("echo ok", timeout=999)
        data = json.loads(result)
        assert data.get("exit_code") == 0

    def test_超时命令_返回timeout标记(self):
        """执行超时的命令返回 timeout 标记"""
        result = _terminal("sleep 200", timeout=1)
        data = json.loads(result)
        assert data.get("exit_code") == -1
        assert "timeout" in data.get("output", "")

    def test_空命令可能出错(self):
        """空命令执行（取决于 shell 行为）"""
        result = _terminal("")
        data = json.loads(result)
        assert "exit_code" in data


# ═══════════════════════════════════════════════════════════════
# _execute_code — Python 代码执行
# ═══════════════════════════════════════════════════════════════

class TestExecuteCode:
    """_execute_code() Python 代码执行测试"""

    def test_执行print语句(self):
        """执行 print 输出正常"""
        result = _execute_code("print('hello from code')")
        data = json.loads(result)
        assert data.get("exit_code") == 0
        assert "hello from code" in data.get("output", "")

    def test_执行计算(self):
        """执行计算并打印结果"""
        result = _execute_code("print(1 + 2)")
        data = json.loads(result)
        assert data.get("exit_code") == 0
        assert "3" in data.get("output", "")

    def test_语法错误返回非零exit_code(self):
        """语法错误被捕获，返回非零 exit_code"""
        result = _execute_code("this is invalid python !!!")
        data = json.loads(result)
        assert data.get("exit_code") != 0

    def test_运行时错误(self):
        """运行时异常被捕获"""
        result = _execute_code("raise ValueError('test error')")
        data = json.loads(result)
        assert data.get("exit_code") != 0
        assert "ValueError" in data.get("output", "")

    def test_超时代码(self):
        """执行超时代码返回 timeout 标记"""
        result = _execute_code("import time\ntime.sleep(60)")
        data = json.loads(result)
        assert data.get("exit_code") == -1
        assert "timeout" in data.get("output", "")

    def test_empty代码(self):
        """空代码不报错"""
        result = _execute_code("")
        data = json.loads(result)
        assert data.get("exit_code") == 0


# ═══════════════════════════════════════════════════════════════
# 工具注册
# ═══════════════════════════════════════════════════════════════

class TestSystemToolsRegistration:
    """系统工具注册测试"""

    def test_SYSTEM_TOOLS包含所有工具(self):
        """SYSTEM_TOOLS 包含所有定义的系统工具"""
        tool_names = {t.name for t in SYSTEM_TOOLS}
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "patch" in tool_names
        assert "search_files" in tool_names
        assert "terminal" in tool_names
        assert "execute_code" in tool_names
        assert "get_current_time" in tool_names

    def test_register_all注册所有工具(self):
        """register_all 将所有系统工具注册到 Registry"""
        reg = Registry()
        register_all(reg)
        registered = reg.list_tools()
        assert "read_file" in registered
        assert "terminal" in registered
        assert "execute_code" in registered
        assert len(registered) == len(SYSTEM_TOOLS)

    def test_register_defaults排除指定工具(self):
        """register_defaults 可以排除指定工具"""
        reg = Registry()
        register_defaults(reg, exclude={"terminal", "execute_code"})
        registered = reg.list_tools()
        assert "terminal" not in registered
        assert "execute_code" not in registered
        assert "read_file" in registered

    def test_register_defaults全部注册(self):
        """register_defaults 不带 exclude 注册全部"""
        reg = Registry()
        register_defaults(reg)
        assert len(reg.list_tools()) == len(SYSTEM_TOOLS)


# ═══════════════════════════════════════════════════════════════
# _get_current_time
# ═══════════════════════════════════════════════════════════════

class TestGetCurrentTime:
    """_get_current_time() 时间查询测试"""

    def test_返回合法JSON(self):
        """返回合法的 JSON 字符串"""
        result = _get_current_time()
        data = json.loads(result)
        assert "datetime" in data
        assert "timezone" in data
        assert "timestamp" in data

    def test_timestamp是正浮点数(self):
        """timestamp 是合理的时间戳"""
        result = _get_current_time()
        data = json.loads(result)
        ts = data["timestamp"]
        assert ts > 0
        # 在合理范围内（2024~2030年）
        assert 1700000000 < ts < 2000000000
