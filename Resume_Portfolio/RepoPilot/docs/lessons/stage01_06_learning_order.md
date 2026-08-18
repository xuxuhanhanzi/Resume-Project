# RepoPilot Stage 1–6 唯一学习入口

> 适用对象：第一次系统学习 LLM Agent、Python 工程经验较少的学习者  
> 学习方式：先运行离线 Demo，再读图，再读少量源码，最后完成练习  
> 当前证据：Ruff、strict mypy 通过，pytest 33/33 通过；真实本地 Qwen 和 Docker daemon 实测尚未完成

## 1. 先理解我们到底在学什么

RepoPilot 不是一个聊天机器人，也不是一个完整商业产品。它是一个 **Agent Runtime Lab**：
用一个很小的 Python Bug Fix 任务，观察大语言模型如何获得工具、改变环境、读取结果、
保存状态、接受权限限制，并由测试程序判断最终结果。

如果只记住一句话，请记住：

```text
LLM 负责“建议下一步做什么”；
Runtime 负责“这一步能不能做、怎样做、做完发生了什么”；
Verifier 负责“任务到底有没有完成”。
```

完整数据流是：

```text
用户任务
  ↓
Task Spec：固定目标、路径、测试和预算
  ↓
Context Builder：挑选本轮模型真正需要的信息
  ↓
ModelProvider：请求本地模型或离线 ScriptedProvider
  ↓
ToolCall：模型提出结构化动作
  ↓
Policy：允许 / 拒绝 / 请求人工批准
  ↓
Tool / Sandbox：读取、搜索、修改、测试
  ↓
ToolResult：把环境变化变成结构化观察
  ↓
Checkpoint + Trace：保存状态与证据
  ↓
再次调用模型，或者交给 Deterministic Verifier
```

## 2. 学习前的最低 Python 知识

不要求你熟练掌握 Python，但需要认识以下写法。遇到不懂的词，先查
[`agent_glossary.md`](agent_glossary.md)。

### 2.1 变量和函数

```python
name = "RepoPilot"

def add(left: int, right: int) -> int:
    return left + right
```

- `name` 是变量；
- `def` 定义函数；
- `left: int` 是类型提示，表示希望它是整数；
- `-> int` 表示函数预期返回整数；
- 类型提示通常由 mypy 检查，不会自动替你修正运行时数据。

### 2.2 class 与对象

```python
class Counter:
    def __init__(self) -> None:
        self.value = 0
```

`class` 是对象的模板，`Counter()` 才会创建一个实际对象。RepoPilot 中的
`AgentRuntime`、`ToolRegistry`、`ContextBuilder` 都是类。

### 2.3 dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int
```

`dataclass` 适合表示“有固定字段的数据”。RepoPilot 用它表示 Message、ToolCall、
ToolResult、Task Spec 和 EvaluationRecord。`frozen=True` 表达“构造后不应随意修改”。

### 2.4 async / await

```python
async def request_model() -> str:
    response = await some_network_call()
    return response
```

模型请求和工具执行可能等待网络或子进程。`async` 允许程序等待时处理别的工作；
`await` 表示“暂停当前协程，等这个操作完成”。初学时先把它理解为一种可组合的等待机制。

## 3. 环境准备

以下命令以 Windows PowerShell 为准：

```powershell
cd D:\Users\27475\Desktop\Resume_Project\RepoPilot
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\dev.py check
```

每条命令的意思：

1. `cd`：切换到项目目录；
2. `python -m venv .venv`：创建项目独立 Python 环境；
3. `pip install -r`：安装冻结的开发依赖；
4. `pip install -e .`：以 editable 模式安装 RepoPilot，修改源码后无需重复安装；
5. `scripts/dev.py check`：依次执行格式、Lint、类型和测试门禁。

如果最后不是 `33 passed`，不要继续学习。先保存完整错误输出，再判断是依赖、路径、
Python 版本还是代码问题。

## 4. 第一次运行：完全不需要真实模型

```powershell
.\.venv\Scripts\python.exe -m repopilot demo scripted
```

它会创建一个新的临时工作区，复制 `examples/bugfix_demo`，然后按固定脚本依次：

```text
list_files
→ read_file
→ apply_patch
→ run_tests
→ git_diff
→ finish
```

这里的 `ScriptedProvider` 不会“思考”，只是返回提前写好的 ModelResponse。为什么还值得运行？
因为 Agent 工程的大部分错误不在模型：状态丢失、权限绕过、路径错误、重复执行副作用、
测试结果误判，都可以在没有真实模型时独立验证。

运行后观察 `artifacts/<run_id>/`：

```text
baseline.json       任务开始前的文本快照
checkpoint.json     可恢复的 AgentState
tool_journal.json   已完成工具调用及结果
events.jsonl        按时间追加的运行事件
```

## 5. Stage 1–6 学习顺序

| 阶段 | 核心问题 | 讲义 | 通过标准 |
|---|---|---|---|
| Stage 1 | Agent 为什么不是一次模型调用？ | [`stage01_minimal_agent_runtime.md`](stage01_minimal_agent_runtime.md) | 能画出 Loop，并追踪一次 ToolCall |
| Stage 2 | 本地模型怎样安全使用代码工具？ | [`stage02_local_model_and_tools.md`](stage02_local_model_and_tools.md) | 能解释 Provider、Schema、8 个 Coding Tools |
| Stage 3 | 模型下一轮到底应该看到什么？ | [`stage03_context_rag_memory_skills.md`](stage03_context_rag_memory_skills.md) | 能区分 Context/RAG/Memory/Skill |
| Stage 4 | 工具和辅助 Agent 怎样编排？ | [`stage04_mcp_planning_parallel_reviewer.md`](stage04_mcp_planning_parallel_reviewer.md) | 能解释 MCP、Planner、并行和 Agent-as-Tool |
| Stage 5 | 崩溃、越权和恶意代码怎么办？ | [`stage05_recovery_permission_sandbox.md`](stage05_recovery_permission_sandbox.md) | 能推演恢复、审批和沙箱边界 |
| Stage 6 | 怎样证明 Agent 真的有用？ | [`stage06_evaluation_and_benchmark.md`](stage06_evaluation_and_benchmark.md) | 能计算 Resolve Rate、Recall@k、MRR 并做消融 |

不要跳过 Stage 1 直接研究 Multi-Agent。后面的每个能力最终都要回到同一个问题：它如何改变
`AgentState → Context → Action → Observation → AgentState`？

## 6. 推荐的 12 天节奏

### 第 1–2 天：Stage 1

- 跑 Scripted Demo；
- 阅读 contracts、budgets 和 loop；
- 手画状态机；
- 修改一个 ScriptedResponse，观察失败轨迹。

### 第 3–4 天：Stage 2

- 学习 HTTP、JSON 和 Tool Schema；
- 分别运行文件读取、搜索、Patch 和测试工具；
- 理解为什么不提供任意 Shell；
- 暂时不要下载大模型。

### 第 5–6 天：Stage 3

- 对两个文件手算一次 BM25 直觉分数；
- 比较 Context 和 Memory；
- 阅读 `python-bugfix/SKILL.md`；
- 观察历史消息裁剪。

### 第 7–8 天：Stage 4

- 跑 MCP Demo；
- 画出 Client、Transport、Server、ToolRegistry；
- 比较 Planner 与 Agent；
- 解释为什么并行仅允许只读工具。

### 第 9–10 天：Stage 5

- 跑 Permission 和 Recovery Demo；
- 手工推演“Patch 完成后进程崩溃”；
- 阅读路径穿越测试；
- 查看 Docker 命令，但不要在 daemon 未验证前执行不可信代码。

### 第 11–12 天：Stage 6

- 检查 10 个微任务；
- 手算 Recall@k 和 MRR；
- 设计一张固定模型、固定预算的消融表；
- 写一段不夸大结论的实验总结。

## 7. 学习时必须遵守的证据纪律

当前已经证明：

- Runtime 的离线闭环可运行；
- 33 个自动测试通过；
- 10 个微任务初始都 fail-to-pass；
- 权限过滤后的 BM25 在这些微任务上 10/10 Top-1 定位 `module.py`；
- MCP、Permission、Recovery 的教学 Demo 可运行。

当前没有证明：

- 任意 Qwen 模型能解决 10 个任务；
- Docker 沙箱已经在本机 daemon 中实际执行敌对代码；
- 教学 MCP 子集等于完整 MCP 标准实现；
- 微任务成绩可以代表真实仓库或 SWE-bench；
- Prompt Injection 已经被“彻底解决”。

学习 Agent 最危险的错误之一，是把“代码接口存在”说成“能力已经验证”。讲义会反复区分：

```text
Implemented：代码路径存在并有机制测试
Qualified：在登记环境中通过约定门禁
Evaluated：在冻结任务上形成可复现指标
Production-ready：经过真实负载、故障、安全和运维验证
```

RepoPilot 当前属于第一个层次，并在离线微型范围内达到部分第二层次。

## 8. 最终学习者验收

完成六册后，你应当能不看源码回答：

1. LLM、Agent、Workflow、Runtime 分别是什么；
2. 一次 ToolCall 从模型输出到 ToolResult 经历哪些边界；
3. 为什么模型不能自己把状态设成 Completed；
4. Context、Session Memory 和 Episodic Memory 有何不同；
5. RAG 为什么在 Agent 中适合做成 Tool；
6. Tool 与 Skill 有何不同；
7. MCP Client/Server 解决什么问题，不解决什么问题；
8. 为什么只读搜索可以并行，Patch 默认不能并行；
9. `pending_calls` 与 `tool_journal` 怎样共同避免重复副作用；
10. HITL 为什么必须位于模型之外；
11. Docker 为什么不是完整安全证明；
12. 为什么 Outcome 比最终自然语言更适合评测代码 Agent。

如果只能背定义、不能沿真实文件追踪数据流，则还没有通过学习门。
