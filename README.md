# CollabRoom v0.1.0

让多个 AI Agent 在一个房间里协作交流。每个 Agent 是独立的个体——有不同的角色、性格、工具、记忆——在一个共享的"房间"里交流讨论。

## 快速开始

```bash
git clone git@github.com:NaUyLL/CollabRoom.git
cd CollabRoom

# 需要 Python 3.10+，依赖 DeepSeek API Key
export DEEPSEEK_API_KEY=your_key_here

# 启动团队讨论
python examples/team_demo.py
```

## 架构

```
┌──────────────────────────────────────────┐
│  Gateway  （传输层）                      │
│  CLI / 飞书 / HTTP 接入                  │
├──────────────────────────────────────────┤
│  Room     （会话层）                      │
│  管理多个 Agent 实例                     │
│  协调发言、@mention、多轮交互             │
│  防循环、停止词检测                      │
├──────────────────────────────────────────┤
│  Core     （逻辑层）                      │
│  Agent(memory, planning, tool_calling)   │
│  每个 Agent 是完整独立个体               │
└──────────────────────────────────────────┘
```

## 核心交互机制

用户发言后 Room 自动执行：

1. **举手阶段** — 每个 Agent 轻量决策（YES/NO，约 50 token）是否要发言
2. **发言阶段** — 举手的 Agent 轮流发言，后发言者能看到前面所有人的内容
3. **@mention 链式回应** — Agent 发言中 @其他Agent 自动触发回应
4. **防循环** — 同一对 Agent 来回 @ 最多 2 次，单轮总深度最多 5 次
5. **停止词** — 用户输入「够了」「停止」「结束」截断本轮

## 使用方式

```python
from collabroom import Room, AgentMember, CoreAgent, LLM

# 构造一个 Agent
agent = CoreAgent(
    llm=LLM(),
    registry=...,
    system_prompt="你是架构师",
)

# 注册到房间
room = Room("设计讨论室")
room.register(AgentMember("架构师", "系统架构师", agent))

# 发起讨论
responses = room.round("帮我设计订单系统")
```

## 文件结构

```
collabroom/
├── __init__.py      # 公共 API 导出
├── room.py          # Room + AgentMember（核心）
├── core/
│   ├── llm.py       # LLM 客户端
│   ├── loop.py      # Agent 引擎
│   ├── tool.py      # 工具注册表
│   ├── types.py     # 数据结构
│   ├── memory/      # 记忆策略（可插拔）
│   ├── planning/    # 规划策略（可插拔）
│   └── tool_calling/# 工具调用策略（可插拔）
├── gateway/
│   └── cli.py       # CLI 交互入口
└── examples/
    ├── team_demo.py      # 团队讨论演示
    └── scenario_test.py  # 角色场景测试
```

## 运行演示

```bash
python examples/team_demo.py
```

房间里有架构师、开发者、测试工程师三个角色，输入需求后他们会自动举手发言。

## License

MIT
