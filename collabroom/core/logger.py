"""结构化日志 — JSON 格式 + trace_id 追踪

用法:
  from collabroom.core.logger import get_logger

  log = get_logger("room")
  log.info("消息处理", sender="user", message_count=5)
  log.warning("举手异常", member="Alice", error=str(e))

输出格式:
  {"t": "2026-06-06T12:00:00", "l": "INFO", "c": "room",
   "trace": "abc123", "msg": "消息处理", "sender": "user", "message_count": 5}

设计原则:
  - 零依赖（只用 stdlib logging + json）
  - 每个 log 调用都是 key=value 结构，不拼接字符串
  - trace_id 由 contextvars 自动传播（无需显式传参）
  - trace_id 在 Agent.run() 入口生成，贯穿整个 planning 循环
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

# ── trace_id 上下文变量 ─────────────────────────────
# 在 Agent.run() 入口设置，自动传播到所有子调用

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前线程/协程的 trace_id，返回设置的 ID

    Args:
        trace_id: 指定 ID，不传则自动生成 UUID

    Returns:
        设置的 trace_id
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前 trace_id"""
    return _trace_id.get()


# ── JSON Formatter ──────────────────────────────────

class JSONFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON"""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "l": record.levelname,
            "c": record.name,
            "trace": _trace_id.get(),
            "msg": record.getMessage(),
        }
        # 如果消息本身是格式化的，保持原样
        # 额外字段（从 extra 传入）
        if hasattr(record, "extra_fields"):
            data.update(record.extra_fields)
        return json.dumps(data, ensure_ascii=False)


# ── StructuredLogger ────────────────────────────────

class StructuredLogger:
    """包装 stdlib Logger，支持 key=value 结构化参数"""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        # 如果没有 handler，加一个默认的
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self._logger.addHandler(handler)
        self.name = name

    def _log(self, level: int, msg: str, **kwargs: Any):
        """结构化日志：msg 是事件名称，kwargs 是键值对字段"""
        # 防止 msg 作为关键字参数传入（与位置参数冲突）
        kwargs.pop("msg", None)
        extra = {"extra_fields": kwargs}
        self._logger.log(level, msg, extra=extra)

    def info(self, msg: str, **kwargs: Any):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any):
        self._log(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any):
        self._log(logging.DEBUG, msg, **kwargs)


# ── 快速获取 Logger ────────────────────────────────

_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """获取或创建结构化 Logger"""
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def set_level(level: int):
    """全局设置日志级别"""
    logging.getLogger().setLevel(level)
    for name in _loggers:
        logging.getLogger(name).setLevel(level)
