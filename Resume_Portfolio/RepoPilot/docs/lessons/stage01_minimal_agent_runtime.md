# Stage 1 完整讲义：从一次 LLM 调用到最小 Agent Runtime

> 本阶段不需要真实大模型，不需要 GPU，也不执行不可信代码。  
> 建议学习时间：8–12 小时。  
> 对应实现：`src/repopilot/core/`、`providers/scripted.py`、`observability/trace.py`。

## 0. 本阶段要解决的问题

普通模型调用大致是：

```text
输入一段文字 → 模型 → 输出一段文字
```

但代码 Agent 必须多次改变环境：

```text
收到 Bug 描述
→ 查看文件
→ 搜索函数
→ 修改代码
→ 跑测试
→ 根据失败继续修改
→ 验证完成
```

因此我们需要的不只是“模型包装器”，而是一个有状态的 Runtime。本阶段完成后，你应该能回答：

1. LLM、Agent、Workflow、Runtime 有何区别；
2. 为什么 Agent Loop 必须有状态和预算；
3. Message、ToolCall、ToolResult、AgentState 分别保存什么；
4. ScriptedProvider 为什么是核心测试工具；
5. 模型为什么不能直接决定 `COMPLETED`；
6. 一轮 Loop 的数据究竟怎样移动。

## 1. 四个容易混淆的对象

### 1.1 LLM：只负责根据输入预测输出

大语言模型接收一组 Token，预测后续 Token。即使它能写出很聪明的答案，也不自动拥有：

- 文件读取权限；
- 操作系统命令；
- 跨进程状态；
- 测试执行环境；
- “这个补丁确实正确”的证明。

当模型说“我已经修改了文件”，如果 Runtime 没有真正执行工具，环境中什么也没发生。

### 1.2 Workflow：程序提前确定步骤

例如：

```python
prepare_workspace()
run_tests()
build_report()
```

执行顺序由程序员写死，适合确定性步骤。创建工作区、保存 Checkpoint、执行 Policy 和最终验证都
应该尽量是 Workflow。

### 1.3 Agent：模型根据当前状态动态选动作

模型可能先搜索 A，也可能先读取 B；测试失败后可能换假设。动作顺序无法全部提前写死，所以需要
模型做策略选择。

### 1.4 Runtime：把建议变成受控执行

Runtime 负责：

- 构造 ModelRequest；
- 校验 ModelResponse；
- 查找 Tool；
- 调用 Policy；
- 执行并记录 ToolResult；
- 更新 AgentState；
- 保存 Checkpoint；
- 检查预算；
- 调用 Verifier。

可以用一个不完美但直观的类比：模型像司机，Runtime 像汽车的方向盘、刹车、传感器、道路规则和
行车记录仪。司机决定方向，但不能直接让汽车“瞬移”。

## 2. Agent Loop 的第一性结构

RepoPilot 的核心循环位于 [`core/loop.py`](../../src/repopilot/core/loop.py)。先不要逐行读，先看图：

```text
┌──────────────────────────────────────────┐
│ AgentState：消息、预算、计划、pending    │
└─────────────────┬────────────────────────┘
                  ↓
        ContextBuilder.build(...)
                  ↓
        ModelProvider.complete(...)
                  ↓
           ModelResponse
         ┌────────┴────────┐
         ↓                 ↓
     ToolCalls          Final Text
         ↓                 ↓
   Policy + Tools       Verifier
         ↓              ┌──┴──┐
   ToolResults          Pass  Fail
         ↓               ↓     ↓
  写回 Message       Complete  Repair/Fail
         ↓
     保存 Checkpoint
         └────────────→ 下一轮
```

注意：模型没有一条叫做 `set_status_completed` 的工具。它只能“提出结束”，然后 Runtime 强制运行
Verifier。

## 3. 数据契约：为什么不直接传任意字典

核心契约在 [`core/contracts.py`](../../src/repopilot/core/contracts.py)。

### 3.1 Message

```python
@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
```

`role` 常见值：

- `system`：Runtime 提供的最高层规则；
- `user`：用户目标或投影后的任务信息；
- `assistant`：模型文本；
- `tool`：工具观察。

为什么 ToolResult 还要转成 Message？因为下一轮模型只能通过 Context 看到环境变化。文件已经改了，
不代表模型自动知道；Runtime 必须把结果写入历史。

### 3.2 ToolCall

```python
ToolCall(
    call_id="c2",
    name="read_file",
    arguments={"path": "calculator.py"},
)
```

- `call_id`：这次动作的稳定身份，恢复和幂等需要它；
- `name`：工具名；
- `arguments`：结构化参数。

如果只让模型输出 `请读取 calculator.py`，程序还要猜动词、路径和参数，容易命令注入，也无法稳定
校验。结构化调用把“语言理解”和“程序执行”隔开。

### 3.3 ToolResult

重要字段：

```text
ok            是否成功
data          成功数据或部分结果
error         可读错误
error_type    validation/timeout/permission/conflict...
recoverable   是否值得修正后再试
side_effect   环境是否可能已经改变
cached        是否从 Journal 回放
```

`ok=False` 不足以指导 Agent。路径拼错可以改后再试，权限拒绝不能靠重复调用解决，超时和网络限流
可能稍后恢复，Patch 部分写入则必须先确认副作用。

### 3.4 AgentState

`AgentState` 是长于单次模型调用的状态：

```text
run_id / task_id
status
iteration / tool_calls
input_tokens / output_tokens
messages
pending_calls
plan / completed_steps
final_answer / failure_reason
```

模型自己的 Context 可能裁剪，但 AgentState 必须足够恢复任务。不要把“模型能看到的历史”与“系统
持久保存的状态”混为一谈。

## 4. 状态机：为什么比几个布尔值更清楚

`RunStatus` 包含：

```text
CREATED
PREPARING
RUNNING
VERIFYING
APPROVAL_REQUIRED
COMPLETED
FAILED
CANCELLED
```

如果只使用 `done=True/False`，无法区分：

- 正在等待模型；
- 正在等待人工审批；
- 已提出结束但正在验证；
- 因预算失败；
- 用户主动取消。

显式状态使日志、恢复和 UI 都能知道系统此刻在做什么。

### 合法状态路径示例

```text
CREATED → PREPARING → RUNNING
RUNNING → APPROVAL_REQUIRED → RUNNING
RUNNING → VERIFYING → COMPLETED
RUNNING → FAILED
```

当前代码没有实现一个独立的状态转换表，而是在 Runtime 中显式赋值并立即 Checkpoint。这足以教学，
未来生产化可以再加入严格 transition validator。

## 5. Budget：为什么不能写无限循环

预算实现位于 [`core/budgets.py`](../../src/repopilot/core/budgets.py)：

```text
max_iterations
max_tool_calls
max_total_tokens
max_wall_seconds
```

### 四种预算分别防什么

- Iteration：模型重复“想一轮”的次数；
- Tool calls：模型疯狂搜索或测试；
- Tokens：上下文和生成成本；
- Wall seconds：外部工具卡住或整体过慢。

假设每轮模型调用 3000 Token、最多 12 轮，最坏可能达到约 36000 Token。即使单轮便宜，无界循环
仍可能产生高成本。

`BudgetGuard.exceeded_reason(state)` 返回具体原因，而不是一个模糊的 `False`。失败原因会进入
Checkpoint 和 Trace。

## 6. ModelProvider：依赖倒置的第一个例子

Runtime 不直接写：

```python
ollama.chat(...)
```

而依赖 [`providers/base.py`](../../src/repopilot/providers/base.py) 中的 Protocol：

```python
class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...
```

这叫“依赖抽象，而不是依赖具体实现”。好处：

- 测试使用 ScriptedProvider；
- 本地运行使用 LocalOpenAICompatibleProvider；
- 以后可增加别的 Provider；
- Agent Loop 不需要知道厂商响应 JSON 长什么样。

### ScriptedProvider 做了什么

[`providers/scripted.py`](../../src/repopilot/providers/scripted.py) 内部保存一个队列：

```text
第 1 次 complete → 返回预先准备的 Response 1
第 2 次 complete → 返回 Response 2
...
```

它不模拟模型能力，只让 Runtime 的控制流完全可复现。若工具链连 ScriptedProvider 都跑不通，换
更强模型只会让错误更难定位。

## 7. 追踪一次真实的离线集成测试

打开 [`test_agent_runtime.py`](../../tests/integration/test_agent_runtime.py) 中
`test_scripted_agent_completes_coding_loop`。

### 第 1 轮：搜索

ScriptedProvider 返回：

```python
ToolCall("c1", "search_text", {"pattern": "subtract"})
```

Runtime：

1. 把调用写入 `state.pending_calls`；
2. 先保存 Checkpoint；
3. 下一次循环发现 pending；
4. 经过 Policy；
5. 调用 SearchTextTool；
6. 把 ToolResult 写入 Message；
7. 清空 pending，再保存 Checkpoint。

为什么不在拿到 ModelResponse 后立刻执行？保存 pending 后，即使进程在工具前崩溃，恢复时也知道
还有哪个动作没有处理。

### 第 2 轮：Patch

```python
ToolCall(
    "c2",
    "apply_patch",
    {
        "path": "calculator.py",
        "old_text": "return left + right",
        "new_text": "return left - right",
    },
)
```

模型不提供任意 Python 脚本，只提出一次精确文本替换。Runtime 决定它是否在允许路径、是否需要审批、
是否超过修改文件数量。

### 第 3 轮：测试

模型只调用 `run_tests`，不能自己指定任意命令。真实命令来自冻结 Task Spec。

### 第 4 轮：提出结束

模型返回 finish 文本。Runtime 进入 VERIFYING，再次执行确定性验证；通过后才变成 COMPLETED。

## 8. Trace：为什么日志不是随便 print

[`observability/trace.py`](../../src/repopilot/observability/trace.py) 写入 JSONL：

```json
{"event":"model_call_started","iteration":1,"timestamp":"..."}
{"event":"policy_decision","outcome":"allow","tool":"search_text","timestamp":"..."}
{"event":"tool_call_finished","ok":true,"tool":"search_text","timestamp":"..."}
```

与普通文本相比，JSONL 可以稳定查询：

- 哪个工具失败最多；
- 每个任务用了多少轮；
- 哪种 Policy 阻止了动作；
- 恢复时是否发生缓存回放。

Trace 会对包含 `api_key`、`authorization`、`password`、`secret`、`token` 的字段名脱敏。但脱敏规则
不是万能秘密扫描器，生产系统还需要更强的数据分类。

## 9. 动手实验

### 实验 A：运行最小 Demo

```powershell
$env:PYTHONPATH = "src"
python -m repopilot demo scripted --artifacts artifacts\student_stage01
```

检查最新目录的 `checkpoint.json`，回答：

- 最终 status 是什么？
- iteration 和 tool_calls 为什么不同？
- pending_calls 最终为什么为空？

### 实验 B：让 Agent 超预算

阅读测试：

```powershell
python -m pytest -q tests\integration\test_agent_runtime.py::test_budget_stops_provider_that_never_finishes -vv
```

这个 Provider 连续读取文件，却从不 Patch 或 finish。预算达到 2 轮后，状态进入 FAILED。

### 实验 C：人为制造未知工具

在你自己的临时测试中构造：

```python
ToolCall("bad", "delete_everything", {})
```

预测后再运行：ToolRegistry 找不到工具，Runtime 应返回 Validation 类型的失败 Observation，而不是
尝试解释工具名。

## 10. 常见误解

### “用了 while 就是 Agent”

错误。没有状态、结构化动作、环境反馈、预算和验证的循环，只是重复调用模型。

### “最终答案写得很自信就算成功”

错误。代码任务以测试 Outcome 为准。自然语言只用于总结。

### “ScriptedProvider 太假，没有价值”

错误。它隔离模型随机性，是验证 Runtime 控制流最有价值的基线之一。

### “async 会让所有工具自动并行”

错误。`async` 只是允许等待和组合；只有显式 `gather` 等操作才并行调度，而且副作用工具通常不应
并行。

## 11. 思考题与答案

<details>
<summary>问题 1：为什么 ToolResult 需要 error_type，而不是只保存错误字符串？</summary>

程序不能可靠地从任意字符串推断重试策略。Timeout 可能重试，Permission 需要审批，Validation
需要修改参数，Conflict 需要重新读取环境。结构化错误让 Runtime 和 Agent 能做不同决策。
</details>

<details>
<summary>问题 2：模型为什么不能直接修改 AgentState？</summary>

AgentState 是系统控制面。若模型能写 status、预算或权限，它可以把 FAILED 改成 COMPLETED、清零
Token、删除 pending action。模型只能通过受控 Action 影响环境，Runtime 才能更新状态。
</details>

<details>
<summary>问题 3：iteration=4、tool_calls=3 是否矛盾？</summary>

不矛盾。Iteration 统计模型决策轮次；最后一轮可能只提出 finish，不调用工具。一轮也可能提出多个
并行只读工具，因此两个计数本来就不应相等。
</details>

<details>
<summary>问题 4：为什么最终 Verifier 还要重跑测试？Agent 已经调用过 run_tests。</summary>

模型可能在测试后又修改文件，也可能误读 ToolResult，甚至从未调用测试。最终验证属于系统的强制
Workflow，不能依赖模型自觉。
</details>

## 12. Stage 1 验收清单

你需要做到：

- [ ] 不看讲义画出 Agent Loop；
- [ ] 解释四类 Message role；
- [ ] 解释 ToolCall/ToolResult 的字段；
- [ ] 说明 State 与 Context 的区别；
- [ ] 指出四种预算分别防什么；
- [ ] 追踪集成测试的 4 次模型调用；
- [ ] 运行 Scripted Demo 并读懂四种 Artifact；
- [ ] 明确说出“模型提出结束，Verifier 决定完成”。

通过后再进入 Stage 2。下一阶段才会学习真实模型服务和代码工具。
