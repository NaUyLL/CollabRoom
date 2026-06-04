# CollabRoom

> 让多个 AI Agent 在一个房间里协作交流。

CollabRoom 是一个轻量级的多 Agent 协作框架。每个 Agent 是独立的个体——有不同的角色、性格、工具、记忆——在一个共享的"房间"里交流讨论。

## 快速开始

```bash
git clone ...
cd collabroom
pip install -e .

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

## 核心概念

- **Room**: Agent 协作的会议室，管理发言顺序和历史
- **AgentMember**: 房间里的一个 Agent 成员，有自己的角色和记忆
- **CoreAgent**: Agent 的底层引擎（memory/planning/tool 全部可插拔）

## 交互方式

用户发言后，Room 会自动：
1. 所有 Agent **举手**决定是否要发言（轻量 LLM 调用）
2. 举手的 Agent **轮流发言**
3. 发言中 **@其他Agent** 会触发链式回复
4. **停止词** 可以截断本轮讨论

## License

MIT
