"""CLI Gateway — 终端交互界面

重构为 BaseGateway 子类。
"""

from __future__ import annotations
import sys, shutil, os

from . import BaseGateway


def _term_width() -> int:
    return shutil.get_terminal_size().columns


def _divider(char: str = "─", title: str = "") -> str:
    w = _term_width()
    if title:
        left = (w - len(title) - 2) // 2
        right = w - left - len(title) - 2
        return f"{char * left} {title} {char * right}"
    return char * w


class CLIGateway(BaseGateway):
    """终端交互式 Gateway"""

    def __init__(self, room, name: str = "cli"):
        super().__init__(room, name)
        self._running = False

    def run(self):
        """启动 CLI 交互循环"""
        self._running = True
        self._header()
        while self._running:
            raw = self._prompt()
            if not raw:
                continue
            if raw == "/exit":
                print("\n再见 👋\n")
                break
            if raw == "/agents":
                members = self.list_members()
                print(f"\n当前成员: {', '.join(m['name'] for m in members)}")
                for m in members:
                    print(f"  {m['name']}: {m['role']}")
                continue
            if raw == "/help":
                self._show_help()
                continue
            if raw == "/history":
                history = self.get_history(tail=10)
                print(f"\n对话历史（最近 10 条）:")
                for m in history:
                    tag = f"@{m['sender']}" if m['kind'] == 'dm' else m['sender']
                    print(f"  {tag}: {m['content'][:200]}")
                continue
            if raw == "/clear":
                os.system("clear" if sys.platform != "win32" else "cls")
                continue

            # 私信: @名字 内容
            if raw.startswith("@"):
                parts = raw[1:].split(" ", 1)
                if len(parts) == 2:
                    target, content = parts
                    if target in self.room.members:
                        reply = self.room.dm("user", target, content)
                        self._show_message(target, reply or "(无回应)", "dm")
                    else:
                        print(f"\n  没有叫 {target} 的成员。输入 /agents 查看。")
                else:
                    print(f"\n  格式: @名字 内容")
                continue

            # 普通发言
            self._handle_public(raw)

        self._running = False

    def stop(self):
        self._running = False

    # ── 内部 ──

    def _header(self):
        print()
        print(_divider("╌", self.room.name))
        print(f"  成员: {', '.join(self.room.list_members())}")
        print(f"  输入 /help 看命令  |  @名字 内容 私信  |  ctrl+c 退出")
        print(_divider("╌"))
        print()

    def _show_help(self):
        print(f"\n{_divider('─', '帮助')}")
        print(f"  直接输入         普通发言，所有 Agent 都会回应")
        print(f"  @名字 内容       私信给某个 Agent")
        print(f"  /agents          列出所有成员")
        print(f"  /history         看对话历史")
        print(f"  /clear           清屏")
        print(f"  /help            这个菜单")
        print(f"  ctrl+c           退出")
        print(f"{_divider('─')}")
        print()

    def _show_message(self, sender: str, content: str, kind: str = "public"):
        label = f"@{sender}" if kind == "dm" else f"【{sender}】"
        print(f"\n{label}")
        print(f"{content}")
        print()

    def _prompt(self) -> str:
        try:
            return input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    def _handle_public(self, text: str):
        print(f"\n  (用户发言: {text[:50]}…)")
        responses = self.handle_message("user", text)
        if not responses:
            print(f"\n  (没有 Agent 回应)")
        else:
            for r in responses:
                self._show_message(r["sender"], r["content"])


# ── 向后兼容：保持 run_gateway(room) 可用 ──

def run_gateway(room):
    """保持原有的函数式接口"""
    gw = CLIGateway(room)
    gw.run()
