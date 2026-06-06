"""系统级真实工具 — Agent 通过它们操作真实文件系统和终端

设计原则：
  1. 每个工具返回字符串（LLM 可消费），成功/失败都结构化
  2. JSON Schema 参数描述，LLM 能理解调用方式
  3. 安全限制：超时、输出大小上限、路径安全

安全加固（参考 Hermes Agent）：
  - 路径安全：阻止写 /etc/ /boot/ /dev/ /proc/ 等敏感系统路径
  - 陈旧检测：写/改文件前对比 mtime，警告文件被外部修改
  - 重复阻断：同一 read_file/search_files 连调 4 次直接拒绝
"""

from __future__ import annotations
import os, sys, json, subprocess, tempfile, time, traceback, threading
from pathlib import Path
from .tool import Tool, Registry, tool_error, tool_result

# ═══════════════════════════════════════════════════════════════
# 安全限制
# ═══════════════════════════════════════════════════════════════

MAX_OUTPUT_CHARS = 50_000          # 单个工具结果最大字符数
MAX_TERMINAL_TIMEOUT = 120         # 终端命令最长超时（秒）

# 写入保护的敏感系统路径（参照 Hermes agent/file_safety.py）
SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",
    "/dev/", "/proc/", "/sys/",
)
SENSITIVE_EXACT_PATHS = {
    "/var/run/docker.sock", "/run/docker.sock",
}

# 读取会挂死的设备路径
BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


# ═══════════════════════════════════════════════════════════════
# 陈旧检测 & 重复检测（线程安全）
# ═══════════════════════════════════════════════════════════════

_tracker_lock = threading.Lock()
_read_tracker: dict[str, dict] = {}   # {task_id: {data}}


def _track_read(filepath: str, offset: int, limit: int) -> tuple[bool, str]:
    """记录读操作，检测重复读取。返回 (允许继续?, 警告或拒绝消息)"""
    try:
        p = Path(filepath).resolve()
        mtime = p.stat().st_mtime if p.exists() else 0
    except OSError:
        return True, ""

    key = (str(p), offset, limit)

    with _tracker_lock:
        td = _read_tracker.setdefault("default", {})
        td.setdefault("last_read", None)
        td.setdefault("consecutive_reads", 0)
        td.setdefault("read_mtimes", {})

        # Staleness: 记录 mtime
        td["read_mtimes"][str(p)] = mtime

        # Loop detection
        if td["last_read"] == key:
            td["consecutive_reads"] += 1
        else:
            td["last_read"] = key
            td["consecutive_reads"] = 1

        if td["consecutive_reads"] >= 4:
            return False, (
                f"BLOCKED: 你已连续 {td['consecutive_reads']} 次读取同一个文件 ({filepath})。"
                f"内容没有变化，请基于已有的信息继续工作。"
            )
    return True, ""


def _track_search(pattern: str, path: str, target: str) -> tuple[bool, str]:
    """记录搜索操作，检测重复搜索。返回 (允许继续?, 警告消息)"""
    key = (pattern, path, target)
    with _tracker_lock:
        td = _read_tracker.setdefault("default", {})
        td.setdefault("last_search", None)
        td.setdefault("consecutive_searches", 0)

        last_s = td.get("last_search")
        if last_s == key:
            td["consecutive_searches"] = td.get("consecutive_searches", 0) + 1
        else:
            td["last_search"] = key
            td["consecutive_searches"] = 1

        count = td["consecutive_searches"]
        if count >= 4:
            return False, (
                f"BLOCKED: 你已连续 {count} 次执行相同的搜索。"
                f"结果没有变化，请使用已有的信息。"
            )
        if count >= 3:
            return True, (
                f"警告：这是连续第 {count} 次相同搜索，结果未变化。"
                f"如需要更多结果，修改搜索条件。"
            )
    return True, ""


def _check_staleness(filepath: str) -> str | None:
    """检查文件自上次读取后是否被修改。返回警告消息或 None"""
    try:
        p = Path(filepath).resolve()
        if not p.exists():
            return None
        current_mtime = p.stat().st_mtime
    except OSError:
        return None

    with _tracker_lock:
        td = _read_tracker.setdefault("default", {})
        td.setdefault("read_mtimes", {})
        last_mtime = td["read_mtimes"].get(str(p))

    if last_mtime is not None and current_mtime != last_mtime:
        # 更新记录，避免连续警告
        with _tracker_lock:
            td = _read_tracker.setdefault("default", {})
            td.setdefault("read_mtimes", {})[str(p)] = current_mtime
        return (
            f"警告: {filepath} 自上次读取后被修改（或已被其他操作更新），"
            f"请重新读取以确认当前内容。"
        )
    return None


def _update_mtime(filepath: str):
    """写/改文件后更新 mtime 记录，避免自身操作触发陈旧警告"""
    try:
        p = Path(filepath).resolve()
        mtime = p.stat().st_mtime if p.exists() else time.time()
    except OSError:
        return
    with _tracker_lock:
        td = _read_tracker.setdefault("default", {})
        td.setdefault("read_mtimes", {})
        td["read_mtimes"][str(p)] = mtime


# ═══════════════════════════════════════════════════════════════
# 路径安全检查
# ═══════════════════════════════════════════════════════════════

def _is_blocked_device(filepath: str) -> bool:
    """检查路径是否是会挂死的设备文件"""
    norm = os.path.normpath(os.path.expanduser(filepath))
    if norm in BLOCKED_DEVICE_PATHS:
        return True
    if norm.startswith("/proc/") and norm.endswith(("/fd/0", "/fd/1", "/fd/2")):
        return True
    try:
        resolved = os.path.realpath(norm)
        if resolved != norm and resolved in BLOCKED_DEVICE_PATHS:
            return True
    except OSError:
        pass
    return False


def _check_sensitive_path(filepath: str) -> str | None:
    """检查是否试图写敏感系统路径。返回错误消息或 None"""
    try:
        resolved = str(Path(filepath).resolve())
    except (OSError, ValueError):
        resolved = filepath
    normalized = os.path.normpath(os.path.expanduser(filepath))

    err_msg = (
        f"拒绝写入敏感系统路径: {filepath}\n"
        f"如需修改系统文件，请手动使用终端工具。"
    )
    for prefix in SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix) or normalized.startswith(prefix):
            return err_msg
    if resolved in SENSITIVE_EXACT_PATHS or normalized in SENSITIVE_EXACT_PATHS:
        return err_msg
    return None


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _safety_limit(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """截断过长输出，末尾加标记"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... (输出截断，共 {len(text)} 字符，仅显示前 {max_chars})"

def _build(name: str, description: str, parameters: dict, fn) -> Tool:
    """Builder：统一创建 Tool 实例"""
    return Tool(name=name, description=description, parameters=parameters, fn=fn)


# ═══════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════

def _read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """读取文件内容，带行号和分页"""
    # 设备路径检查
    if _is_blocked_device(path):
        return tool_error(f"拒绝读取设备文件: {path}，该路径会导致进程挂死")

    # 重复读取检测
    ok, msg = _track_read(path, offset, limit)
    if not ok:
        return tool_error(msg)

    try:
        p = Path(path).resolve()
        if not p.exists():
            return tool_error(f"文件不存在: {path}")
        if not p.is_file():
            return tool_error(f"不是文件: {path}")

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        offset = max(1, offset)
        limit = max(1, min(limit, 2000))
        end = min(offset + limit - 1, total)

        selected = lines[offset - 1:end]
        content = "".join(
            f"{i + offset:>6}|{line}"
            for i, line in enumerate(selected)
        )
        result = f"文件: {p} ({total} 行, 显示 {offset}-{end})\n{content}"
        if end < total:
            result += f"\n... (还有 {total - end} 行未显示, 调整 offset/limit 查看)"
        return result
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


def _write_file(path: str, content: str) -> str:
    """写入/覆盖文件，自动创建目录"""
    # 敏感路径检查
    err = _check_sensitive_path(path)
    if err:
        return tool_error(err)

    # 陈旧检测
    stale_warn = _check_staleness(path)

    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        size = len(content)

        _update_mtime(path)

        result = tool_result(
            ok=True, path=str(p), chars=size,
            lines=content.count("\n") + 1,
        )
        if stale_warn:
            result = json.dumps({
                "ok": True, "path": str(p), "chars": size,
                "lines": content.count("\n") + 1,
                "_warning": stale_warn,
            }, ensure_ascii=False)
        return result
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


def _patch(path: str, old_string: str, new_string: str,
           replace_all: bool = False) -> str:
    """精确文本替换（find-and-replace），返回差异或错误"""
    # 敏感路径检查
    err = _check_sensitive_path(path)
    if err:
        return tool_error(err)

    # 陈旧检测
    stale_warn = _check_staleness(path)

    try:
        p = Path(path).resolve()
        if not p.exists():
            return tool_error(f"文件不存在: {path}")

        text = p.read_text(encoding="utf-8")

        if replace_all:
            if old_string not in text:
                return tool_error(f"未找到匹配: {old_string[:50]!r}")
            count = text.count(old_string)
            new_text = text.replace(old_string, new_string)
        else:
            idx = text.find(old_string)
            if idx == -1:
                return tool_error(f"未找到匹配: {old_string[:50]!r}")
            count = 1
            new_text = text[:idx] + new_string + text[idx + len(old_string):]

        # 生成 diff 预览
        old_lines = old_string.split("\n")
        new_lines = new_string.split("\n")
        diff = f"- {old_lines[0]}" if old_lines else "- (空)"
        if len(old_lines) > 1 or len(new_lines) > 1:
            diff += f"\n  ... ({len(old_lines)} 行 → {len(new_lines)} 行)"
        diff += f"\n+ {new_lines[0]}" if new_lines else "+ (空)"

        p.write_text(new_text, encoding="utf-8")
        _update_mtime(path)

        result = tool_result(
            ok=True, path=str(p), replacements=count, diff_preview=diff,
        )
        if stale_warn:
            result = json.dumps({
                "ok": True, "path": str(p), "replacements": count,
                "diff_preview": diff, "_warning": stale_warn,
            }, ensure_ascii=False)
        return result
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


def _search_files(pattern: str, target: str = "content",
                  path: str = ".", file_glob: str | None = None,
                  limit: int = 30) -> str:
    """搜索文件内容或文件名

    target="content": 在文件内容中搜索（类似 grep）
    target="files":   按文件名 glob 搜索（类似 find）
    """
    # 重复搜索检测
    ok, msg = _track_search(pattern, path, target)
    if not ok:
        return tool_error(msg)

    try:
        root = Path(path).resolve()
        if not root.exists():
            return tool_error(f"路径不存在: {path}")

        results = []
        if target == "files":
            for f in root.rglob(pattern):
                if f.is_file():
                    results.append(str(f.relative_to(root)))
                    if len(results) >= limit:
                        break
            summary = f"找到 {len(results)} 个文件 (显示前 {limit} 个)"
        else:
            count = 0
            extensions = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml",
                          ".cfg", ".ini", ".conf", ".html", ".css", ".js", ".ts",
                          ".jsx", ".tsx", ".vue", ".rs", ".go", ".java", ".c", ".h",
                          ".cpp", ".hpp", ".sh", ".bash", ".zsh", ".csv", ".xml",
                          ".sql", ".rb", ".php", ".swift", ".kt", ".scala", ".lua"}
            for f in root.rglob("*"):
                if not f.is_file() or f.stat().st_size > 1024 * 1024:
                    continue
                if file_glob and not f.match(file_glob):
                    continue
                if f.suffix not in extensions:
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    if pattern in text:
                        count += 1
                        for lineno, line in enumerate(text.split("\n"), 1):
                            if pattern in line:
                                rel = f.relative_to(root)
                                context = line.strip()[:120]
                                results.append(f"{rel}:L{lineno}  {context}")
                                break
                        if len(results) >= limit:
                            break
                except (OSError, UnicodeDecodeError):
                    continue
            summary = f"搜索 '{pattern}' 在 {path} 中 → {count} 个文件匹配 (显示前 {limit})"

        output = summary + "\n" + "\n".join(results) if results else summary + "\n(无匹配)"

        # 拼接警告
        if "警告" in msg:
            output += f"\n\n{msg}"

        return _safety_limit(output, 8000)
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


def _terminal(command: str, timeout: int = 30,
              workdir: str | None = None) -> str:
    """执行 shell 命令，捕获输出

    安全限制：
      - 超时: max {MAX_TERMINAL_TIMEOUT}s
      - 输出: max {MAX_OUTPUT_CHARS} 字符
      - 非交互式（不支持 stdin 输入）
    """
    try:
        timeout = min(timeout, MAX_TERMINAL_TIMEOUT)
        cwd = os.path.abspath(workdir) if workdir else os.getcwd()

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            executable="/bin/bash",
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout.rstrip())
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        output = "\n".join(output_parts)
        output = _safety_limit(output)

        return tool_result(exit_code=result.returncode, output=output)
    except subprocess.TimeoutExpired:
        return tool_result(
            exit_code=-1,
            output=f"[timeout] 命令执行超过 {timeout}s，已终止",
        )
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


def _get_current_time() -> str:
    """返回当前系统时间（含时区信息）"""
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc).astimezone()
    tz_name = now.strftime("%Z")
    tz_offset = now.strftime("%z")
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    return tool_result(
        datetime=formatted,
        timezone=f"{tz_name} (UTC{tz_offset})",
        timestamp=now.timestamp(),
    )


def _execute_code(code: str) -> str:
    """在隔离的子进程中执行 Python 代码

    适合：数据处理、计算、快速验证逻辑
    限制：无第三方库（stdlib only）、30 秒超时、无网络
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        os.unlink(tmp_path)

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout.rstrip())
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        output = "\n".join(output_parts)
        output = _safety_limit(output)

        return tool_result(exit_code=result.returncode, output=output)
    except subprocess.TimeoutExpired:
        return tool_result(
            exit_code=-1,
            output="[timeout] 代码执行超过 30s，已终止",
        )
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# 工具注册表
# ═══════════════════════════════════════════════════════════════

SYSTEM_TOOLS = [
    _build(
        name="read_file",
        description="读取文件内容，支持行号和分页。返回带行号的文件内容。",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（绝对或相对路径）",
                },
                "offset": {
                    "type": "integer",
                    "description": "起始行号（从 1 开始，默认 1）",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "最多读取行数（默认 500，最大 2000）",
                    "default": 500,
                },
            },
            "required": ["path"],
        },
        fn=_read_file,
    ),
    _build(
        name="write_file",
        description="写入/覆盖文件。自动创建父目录。返回写入的文件路径和字符数。",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "文件内容（覆盖写入）",
                },
            },
            "required": ["path", "content"],
        },
        fn=_write_file,
    ),
    _build(
        name="patch",
        description="在文件中精确查找并替换文本（find-and-replace）。返回替换次数和 diff 预览。",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "old_string": {
                    "type": "string",
                    "description": "要查找的原文（必须精确匹配）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "是否替换所有匹配项（默认 false，只替换第一个）",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        fn=_patch,
    ),
    _build(
        name="search_files",
        description="搜索文件内容或文件名。target='content' 类似 grep，target='files' 类似 find。自动跳过二进制/大文件。",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索关键词（文件内容搜索时为精确匹配）",
                },
                "target": {
                    "type": "string",
                    "enum": ["content", "files"],
                    "description": "搜索目标：'content'（内容搜索）或 'files'（文件名 glob 搜索）",
                    "default": "content",
                },
                "path": {
                    "type": "string",
                    "description": "搜索根目录（默认当前目录）",
                    "default": ".",
                },
                "file_glob": {
                    "type": "string",
                    "description": "文件过滤模式，如 '*.py' 只搜 Python 文件（仅 content 模式生效）",
                },
                "limit": {
                    "type": "integer",
                    "description": "最大返回结果数（默认 30）",
                    "default": 30,
                },
            },
            "required": ["pattern"],
        },
        fn=_search_files,
    ),
    _build(
        name="terminal",
        description=f"在 shell 中执行命令。支持管道、重定向。超时 {MAX_TERMINAL_TIMEOUT}s，输出上限 {MAX_OUTPUT_CHARS//1000}K 字符。",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"超时秒数（默认 30，最大 {MAX_TERMINAL_TIMEOUT}）",
                    "default": 30,
                },
                "workdir": {
                    "type": "string",
                    "description": "工作目录（可选，默认当前目录）",
                },
            },
            "required": ["command"],
        },
        fn=_terminal,
    ),
    _build(
        name="execute_code",
        description="在子进程中执行 Python 代码。适合数据处理、计算验证。stdlib only，30s 超时，无网络。",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码字符串",
                },
            },
            "required": ["code"],
        },
        fn=_execute_code,
    ),
    _build(
        name="get_current_time",
        description="获取当前系统时间（含时区信息）。返回格式化的日期时间、时区和 Unix 时间戳。",
        parameters={
            "type": "object",
            "properties": {},
        },
        fn=_get_current_time,
    ),
]


def register_all(registry: Registry):
    """一键注册所有系统工具到指定 Registry"""
    for tool in SYSTEM_TOOLS:
        registry.register(tool)


def register_defaults(registry: Registry, exclude: set[str] | None = None):
    """注册工具，可选择排除某些工具（按 name）"""
    for tool in SYSTEM_TOOLS:
        if exclude and tool.name in exclude:
            continue
        registry.register(tool)
