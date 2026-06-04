"""CLI Gateway — 终端交互界面"""

import sys, shutil

def _term_width() -> int:
    return shutil.get_terminal_size().columns

def _divider(char: str = "─", title: str = "") -> str:
    w = _term_width()
    if title:
        left = (w - len(title) - 2) // 2
        right = w - left - len(title) - 2
        return f"{char * left} {title} {char * right}"
    return char * w

def header(title: str, members: list[str]):
    print()
    print(_divider("╌", title))
    print(f"  成员: {', '.join(members)}")
    print(f"  输入 /help 看命令  |  @名字 内容 私信  |  ctrl+c 退出")
    print(_divider("╌"))
    print()

def show_message(sender: str, content: str, kind: str = "public"):
    label = f"@{sender}" if kind == "dm" else f"【{sender}】"
    print(f"\n{label}")
    print(f"{content}")
    print()

def show_help():
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

def prompt() -> str:
    try:
        return input("\n>> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"

def run_gateway(room):
    """启动 CLI Gateway 交互循环"""
    header(room.name, room.list_members())

    while True:
        raw = prompt()

        if not raw:
            continue
        if raw == "/exit":
            print("\n再见 👋\n")
            break
        if raw == "/agents":
            print(f"\n当前成员: {', '.join(room.list_members())}")
            for name in room.list_members():
                member = room.members[name]
                desc = member.role_desc[:60]
                print(f"  {name}: {desc}")
            continue
        if raw == "/help":
            show_help()
            continue
        if raw == "/history":
            print(f"\n对话历史（最近 10 条）:")
            print(room.format_history(tail=10))
            continue
        if raw == "/clear":
            import os
            os.system("clear" if sys.platform != "win32" else "cls")
            continue

        # 私信: @名字 内容
        if raw.startswith("@"):
            parts = raw[1:].split(" ", 1)
            if len(parts) == 2:
                target, content = parts
                if target in room.members:
                    reply = room.dm("user", target, content)
                    if reply:
                        show_message(target, reply, "dm")
                    else:
                        print(f"\n  ({target} 没有回应)")
                else:
                    print(f"\n  没有叫 {target} 的成员。输入 /agents 查看。")
            else:
                print(f"\n  格式: @名字 内容")
            continue

        # 普通发言 → 一轮对话
        print(f"\n  (用户发言: {raw[:50]}…)")
        responses = room.round(raw)
        if not responses:
            print(f"\n  (没有 Agent 回应)")
        else:
            for name, reply in responses:
                show_message(name, reply)
