"""MCP Client — 连接 MCP 服务器，发现并注册工具

零外部运行时依赖（只用 mcp Python SDK）。

用法:
  from collabroom.core.mcp_client import MCPClientManager

  manager = MCPClientManager()
  await manager.connect_stdio("time", "uvx", ["mcp-server-time"])
  tools = manager.get_tools()  # -> list[Tool]
"""

from __future__ import annotations
import json, os, traceback
from typing import Any

from .tool import Tool, Registry

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import asynccontextmanager

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _to_tool_def(mcp_tool: Any) -> dict:
    """将 MCP SDK 的 Tool 对象转为 Tool 定义
    
    MCP Tool 有: name, description, inputSchema
    """
    schema = getattr(mcp_tool, "inputSchema", None) or {}
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": schema,
        },
    }


def mcp_tool_to_collab_tool(mcp_tool: Any, registry: Registry) -> None:
    """将一个 MCP 工具注册到 collabroom 的 Registry 中
    
    MCP 工具不需要 fn，因为执行时走 MCP 调用流程。
    这里注册一个占位 fn，实际执行由 MCPClientManager 接管。
    """
    async def _mcp_executor(**kwargs: Any) -> str:
        """通过 MCP 协议执行的占位函数（实际由 manager 接管）"""
        raise RuntimeError(
            "MCP 工具不能直接 fn() 执行，请通过 MCPClientManager.call_tool()"
        )

    tool = Tool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        parameters=getattr(mcp_tool, "inputSchema", None) or {},
        fn=_mcp_executor,
    )
    registry.register(tool)


class MCPClientManager:
    """管理多个 MCP 服务器连接

    每个服务器在自己的 asyncio 事件循环中运行。
    协程方法在同步代码中通过 asyncio.run() 调用。
    """

    def __init__(self):
        self._sessions: dict[str, Any] = {}       # server_name -> ClientSession
        self._tool_map: dict[str, str] = {}        # tool_name -> server_name
        self._contexts: dict[str, Any] = {}        # server_name -> context manager
        self._exit_stack: dict[str, Any] = {}      # server_name -> cleanup
        self._registry: Registry | None = None

    def set_registry(self, registry: Registry):
        """绑定到 collabroom 的工具注册表"""
        self._registry = registry

    # ── 连接管理 ────────────────────────────────────

    def connect_stdio(self, name: str, command: str,
                      args: list[str] | None = None,
                      env: dict[str, str] | None = None) -> list[dict]:
        """连接一个 stdio-based MCP 服务器
        
        返回发现的工具定义列表（OpenAI Schema 格式）。
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP SDK 未安装: pip install mcp")
        import asyncio

        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        async def _connect():
            streams = await stdio_client(params)
            async with streams[0] as read, streams[1] as write:
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools if hasattr(result, "tools") else []

        try:
            tools = asyncio.run(_connect())
        except Exception as e:
            error_msg = f"MCP 服务器 '{name}' 连接失败: {e}"
            print(error_msg)
            return []

        # 注册工具
        defs = []
        for tool in tools:
            defn = _to_tool_def(tool)
            defs.append(defn)
            self._tool_map[tool.name] = name

            # 注册到 Registry（如果已绑定）
            if self._registry:
                mcp_tool_to_collab_tool(tool, self._registry)

        print(f"  MCP [{name}]: 发现 {len(tools)} 个工具")
        return defs

    def connect_stdio_forever(self, name: str, command: str,
                              args: list[str] | None = None,
                              env: dict[str, str] | None = None) -> list[dict]:
        """连接一个 stdio MCP 服务器并保持会话（server 持续运行）
        
        适用于需要持久连接的服务器（如 filesystem、github 等）。
        发现的工具会注册到 Registry。
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP SDK 未安装: pip install mcp")
        import asyncio
        import sys

        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        tools_result = []

        async def _keep_alive():
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        tools = result.tools if hasattr(result, "tools") else []

                        for tool in tools:
                            self._tool_map[tool.name] = name
                            if self._registry:
                                mcp_tool_to_collab_tool(tool, self._registry)
                            tools_result.append(_to_tool_def(tool))

                        print(f"  MCP [{name}]: 发现 {len(tools)} 个工具（持久连接）")

                        # 保持连接直到程序退出
                        while True:
                            await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"  MCP [{name}] 连接断开: {e}", file=sys.stderr)

        # 在后台线程启动事件循环
        import threading
        thread = threading.Thread(target=asyncio.run, args=(_keep_alive(),),
                                  daemon=True)
        thread.start()

        # 等待连接和发现完成
        import time
        for _ in range(50):  # 最多等 5 秒
            if tools_result:
                break
            time.sleep(0.1)

        return tools_result

    # ── 工具执行 ────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> str:
        """通过 MCP 协议调用一个已发现的工具
        
        注意：当前 stdio 连接是瞬时的（connect_stdio 未保持会话），
        因此 call_tool 目前走回退逻辑。持久连接版本需要 connect_stdio_forever。
        """
        server = self._tool_map.get(name)
        if not server:
            return json.dumps(
                {"error": f"MCP 工具 '{name}' 未找到（未注册）"},
                ensure_ascii=False,
            )

        # 暂不支持通过瞬时连接调用（需要持久会话）
        return json.dumps(
            {"error": f"MCP 工具 '{name}' 当前为发现模式"},
            ensure_ascii=False,
        )

    # ── 查询 ────────────────────────────────────────

    def list_servers(self) -> list[str]:
        return list(self._tool_map.values())

    def list_tools(self) -> list[str]:
        return list(self._tool_map.keys())

    def get_definitions(self) -> list[dict]:
        """返回所有 MCP 工具定义（OpenAI Schema 格式）"""
        defs = []
        # 从 registry 获取
        if self._registry:
            for t_name in self._tool_map:
                # 工具已注册到 registry，由 registry.get_definitions() 返回
                pass
        return defs


# ── 快速创建 MCP 工具的辅助函数 ─────────────────────

def create_mcp_tool_from_def(name: str, description: str,
                              parameters: dict) -> Tool:
    """从 JSON Schema 描述创建一个可被 Registry 接受的 Tool
    
    用于手动注册 MCP 工具（不通过 manager）。
    """
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        fn=lambda **kwargs: json.dumps(
            {"error": "MCP 工具暂未连接服务器"},
            ensure_ascii=False,
        ),
    )
