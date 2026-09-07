# RepoPilot → Claude Code 风格 CLI Coding Agent 完整改造计划

> 文档定位：RepoPilot v2 的项目总路线、架构蓝图与阶段实施计划  
> 基座项目：`Resume_Portfolio/RepoPilot`  
> 最终目标：将当前 RepoPilot 从“任务驱动的 Agent Runtime Lab”逐步改造成“会话驱动、终端原生、可治理、可验证、可恢复的 CLI Coding Agent”。

---

# 0. 文档目的

当前 RepoPilot 已经具备较完整的 Agent Runtime 基础，包括：

- Agent Loop；
- Model Provider；
- Tool Call / Tool Result；
- Tool Registry；
- Policy；
- Runner / Sandbox；
- Checkpoint；
- Execution Journal；
- Trace；
- ContextBuilder；
- Memory；
- Skills；
- MCP 教学子集；
- Deterministic Verifier；
- Multi-Agent Harness；
- Benchmark / Evaluation。

因此，本次项目改造的目标不是：

> 从零重新实现一个 Claude Code。

而是：

> **保留 RepoPilot 已经有价值的 Agent Runtime 内核，在此基础上重新设计面向真实软件开发的交互式 Coding Agent Harness。**

最终项目不应定位成：

> Claude Code Clone

而应定位成：

> **RepoPilot：一个 local-first、session-oriented、verifiable CLI Coding Agent。**

中文可理解为：

> **一个本地优先、面向持续开发会话、具有权限治理、上下文管理、会话恢复和确定性验证能力的终端 Coding Agent。**

---

# 1. Claude Code 设计哲学中最值得借鉴的部分

本项目不追求复制 Claude Code 的全部产品功能，而重点借鉴其稳定的核心设计思想。

## 1.1 Terminal-native

Coding Agent 直接生活在开发者终端中。

用户进入项目目录：

```bash
cd my-project
repopilot
```

之后即可直接进行：

```text
> explain this project
> find where authentication is implemented
> fix the failing tests
> review my current diff
```

Agent 直接面向：

- 当前仓库；
- 文件系统；
- Shell；
- Git；
- 测试环境；
- 构建工具。

---

## 1.2 Session-oriented

Claude Code 风格 Coding Agent 的核心单位不是：

```text
Task
```

而是：

```text
Session
```

也就是一个持续存在的开发会话。

用户可以：

```text
提出问题
↓
Agent 执行
↓
用户纠偏
↓
Agent 继续
↓
用户要求修改
↓
Agent 再执行
↓
保存 Session
↓
以后 Resume
```

因此：

```text
Session
```

必须成为 RepoPilot v2 的一级抽象。

---

## 1.3 Agentic Loop

核心循环应保持：

```text
Gather Context
      ↓
Take Action
      ↓
Observe Result
      ↓
Verify
      ↓
继续循环
```

在 RepoPilot 中可以表达为：

```text
Context
↓
Model
↓
Tool Call
↓
Permission
↓
Tool / Shell / Git
↓
Observation
↓
Verification
↓
Model
```

---

## 1.4 Human-in-the-loop

Agent 不应成为完全不可控的自动程序。

用户必须能够：

- 中断；
- 拒绝；
- 批准；
- 纠正；
- 切换权限模式；
- 切换 Plan Mode；
- 指定只分析、不修改；
- 指定只修改、不提交；
- 指定验证方式。

---

## 1.5 Strong Tools, Strong Governance

Coding Agent 需要强大的工具能力：

```text
filesystem
search
edit
shell
git
process
```

但工具能力越强，越需要：

```text
Permission
Policy
Sandbox
Workspace Trust
Protected Paths
Audit Trace
```

因此 RepoPilot 现有安全设计必须被保留并增强。

---

## 1.6 Context is Managed, Not Dumped

长 Session 中不能简单：

```text
所有历史消息
+
所有文件
+
所有工具输出
↓
全部塞给模型
```

而需要：

```text
Persistent Transcript
        ↓
Context Manager
        ↓
选择当前真正需要的信息
        ↓
Model Context
```

并支持：

- Tool Output Pruning；
- Context Budget；
- Summary；
- Compact；
- Project Instructions；
- Memory；
- Skills 按需加载。

---

## 1.7 Progressive Extension

以下能力都应该建立在稳定核心之上：

```text
Skills
Hooks
MCP
Subagents
Worktrees
Background Tasks
Multi-Agent
```

而不能一开始全部塞进 Runtime。

---

# 2. RepoPilot 当前与目标系统的根本差异

当前 RepoPilot 更接近：

```text
Task-oriented Agent Runtime
```

目标系统需要变成：

```text
Session-oriented Coding Agent
```

两者最大区别如下。

| 维度 | 当前 RepoPilot | 目标 RepoPilot v2 |
|---|---|---|
| 核心单位 | PublicTaskSpec | Session |
| 启动方式 | `repopilot run --task ...` | `repopilot` |
| 用户输入 | 一次性任务 | 持续多轮 |
| Workspace | Task workspace / 副本 | 当前真实项目 |
| Agent 生命周期 | Task 开始 → Task 完成 | Session 持续存在 |
| Shell | 受限固定命令 | 通用受控 Shell |
| Git | `git_diff` 等窄工具 | Git-first 工作流 |
| Context | 最近消息 + 简单截断 | Context Manager + Compaction |
| 权限 | PolicyEngine | Interactive Permission System |
| Session | Run checkpoint | Session / Resume / Fork |
| Instructions | TaskSpec | REPOPILOT.md / AGENTS.md |
| Verification | Deterministic Verifier | 保留并增强 |
| Skills / MCP | 已有实验基础 | 标准扩展系统 |
| Multi-Agent | Harness | Subagent + Context Isolation |
| Benchmark | 重要 | 继续保留并加强 Coding Benchmark |

---

# 3. 整体架构目标

最终建议形成：

```text
                         RepoPilot CLI
                              │
                              ▼
                       SessionManager
                              │
                  ┌───────────┴───────────┐
                  │                       │
              User Input             Session State
                  │                       │
                  └───────────┬───────────┘
                              ▼
                         AgentKernel
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
         Context            Model              Tools
            │                                  │
            │                    ┌─────────────┼────────────┐
            │                    ▼             ▼            ▼
            │                  Files          Shell         Git
            │                    │             │            │
            │                    └──── Permission ──────────┘
            │                                  │
            │                                  ▼
            │                              Workspace
            │                                  │
            │                                  ▼
            └────────────────────────── Observation
                                               │
                                               ▼
                                          AgentKernel
                                               │
                               ┌───────────────┴───────────────┐
                               ▼                               ▼
                         Verification                       Answer
                               │                               │
                               └───────────────┬───────────────┘
                                               ▼
                                              User
```

旁路系统：

```text
Session Store
Checkpoint
Execution Journal
Trace
Memory
Skills
Hooks
MCP
Subagents
Evaluation
```

---

# 4. 最重要的架构重构：三层 Runtime

当前 `AgentRuntime` 同时承担：

- Task 生命周期；
- Agent Loop；
- Model 调用；
- Tool 执行；
- Policy；
- Verification；
- Repair；
- Checkpoint；
- Budget；
- Memory。

这对 Benchmark Task 很合适。

但对于 Interactive Session 来说职责过重。

因此建议拆为三层。

---

## 4.1 AgentKernel

负责最纯粹的 Agent 循环：

```text
Model
↓
Tool Call
↓
Observation
↓
Model
```

核心职责：

- 构建单轮模型请求；
- 调用 ModelProvider；
- 解析 Tool Call；
- 执行 Tool；
- 返回 Observation；
- 继续 Agent Loop；
- 产生 Runtime Events。

AgentKernel 不负责：

- CLI；
- Session 生命周期；
- Benchmark hidden tests；
- Task 是否最终通过；
- HTTP Service；
- Multi-Agent 编排。

---

## 4.2 SessionRuntime

负责真实交互式 Coding Agent。

流程：

```text
User Message
↓
SessionRuntime
↓
AgentKernel
↓
Tool / Permission / Verification
↓
Assistant Response
↓
等待下一条 User Message
```

负责：

- Session State；
- 多轮用户输入；
- Resume；
- Fork；
- Context；
- Permission Mode；
- Interactive Verification；
- Session Persistence。

---

## 4.3 TaskRuntime

保留当前 RepoPilot 的 Benchmark 模式。

流程：

```text
PublicTaskSpec
↓
TaskRuntime
↓
AgentKernel
↓
Verifier
↓
Repair
↓
Completed / Failed
```

这样最终结构为：

```text
                     AgentKernel
                    /           \
                   /             \
          SessionRuntime       TaskRuntime
               │                   │
        Interactive CLI        Benchmark
```

最终同时拥有：

```text
Claude Code 风格使用体验
+
RepoPilot 原有评测能力
```

---

# 5. 完整 Stage 路线

本项目建议拆分为 13 个阶段：

```text
Stage 0
Freeze & Architecture Seam

Stage 1
AgentKernel 重构

Stage 2
Interactive CLI

Stage 3
Session System

Stage 4
Workspace & Project Instructions

Stage 5
Coding Tool System

Stage 6
Permission System

Stage 7
Context Engine

Stage 8
Coding Workflow & Verification

Stage 9
Skills & Hooks

Stage 10
MCP

Stage 11
Subagents / Worktree / Background Tasks

Stage 12
Evaluation & Product Hardening
```

---

# Stage 0：冻结 RepoPilot v1 Baseline

## 目标

在任何大规模修改以前冻结当前项目行为。

建立：

```text
Current RepoPilot
=
Regression Baseline
```

---

## 为什么必须先做

否则后面出现：

```text
Benchmark 坏了
Verifier 坏了
Resume 坏了
Sandbox 坏了
Policy 行为改变
```

将无法准确判断回归来自哪里。

---

## 主要工作

### 1. 冻结版本

保留当前稳定 Tag，例如：

```text
v1.0.x
```

新建 v2 开发分支：

```text
feature/coding-agent-v2
```

---

### 2. 保存测试基线

记录：

```text
pytest 总测试数
unit pass
integration pass
safety pass
e2e pass
```

---

### 3. 保存 Demo 基线

至少运行：

```text
scripted demo
permission demo
recovery demo
mcp demo
```

---

### 4. 保存 Benchmark 基线

记录：

```text
FRAMES
DABench
SWE-bench-Live
```

---

## 新增文档

```text
docs/v2/
├── architecture.md
├── migration.md
└── compatibility.md
```

---

## 验收标准

所有当前测试和 Demo 能稳定重现。

---

# Stage 1：提取 AgentKernel

这是整个 v2 最关键的底层重构。

---

## 目标

将：

```text
AgentRuntime
```

拆成：

```text
AgentKernel
+
TaskRuntime
```

并建立统一 Event Model。

---

## 当前问题

当前：

```text
core/loop.py
    AgentRuntime.run(task)
```

同时负责：

```text
Plan
Context
Model
Tool Call
Policy
Tool Execution
Verification
Repair
Checkpoint
```

Interactive CLI 很难实时观察其内部过程。

---

## 修改后的建议目录

```text
core/
├── kernel.py
├── contracts.py
├── events.py
└── budgets.py

runtime/
├── task_runtime.py
└── session_runtime.py
```

---

## AgentKernel 核心职责

建议提供类似：

```python
run_turn(...)
```

输入：

```text
ConversationState
RuntimeContext
ToolRegistry
```

输出：

```text
TurnResult
```

或者异步产生 Event。

---

## 为什么需要 Event Model

CLI 必须实时显示：

```text
Thinking...
Reading src/auth.py
Searching "login"
Running pytest
Editing src/auth.py
Done
```

不能继续：

```text
runtime.run()
↓
长时间等待
↓
一次性打印 JSON
```

---

## Runtime Events

建议定义：

```text
TurnStarted

ModelCallStarted
ModelDelta
ModelCallCompleted

ToolCallProposed
PermissionRequested
ToolCallStarted
ToolCallCompleted

VerificationStarted
VerificationCompleted

TurnCompleted
TurnInterrupted
TurnFailed
```

---

## 保留现有契约

继续复用：

```text
ToolCall
ToolResult
ModelRequest
ModelResponse
```

---

## 验收标准

旧命令仍可运行：

```bash
repopilot run --task ...
```

但内部已经变成：

```text
TaskRuntime
↓
AgentKernel
```

---

## 测试

新增：

```text
test_kernel_text_answer
test_kernel_tool_loop
test_kernel_multiple_tool_calls
test_kernel_interrupt
test_kernel_budget
test_kernel_model_failure
test_kernel_event_order
```

---

# Stage 2：Interactive CLI / REPL

## 目标

用户可以：

```bash
cd project
repopilot
```

进入：

```text
RepoPilot
Project: project
Model: ...

> explain this project
```

---

## REPL

REPL：

```text
Read
↓
Evaluate
↓
Print
↓
Loop
```

在 RepoPilot 中：

```text
读取用户输入
↓
执行 Agent Turn
↓
显示输出
↓
等待下一条输入
```

---

## 建议目录

```text
cli/
├── app.py
├── repl.py
├── commands.py
└── renderer.py
```

原来的：

```text
cli.py
```

逐步变成兼容入口。

---

## CLI 支持模式

### Interactive

```bash
repopilot
```

### Single Prompt

```bash
repopilot -p "explain src/"
```

### Pipeline

```bash
git diff | repopilot -p "review this diff"
```

### Legacy Task

```bash
repopilot run --task task.yaml
```

---

## 推荐技术

第一版建议：

```text
prompt_toolkit
+
rich
```

`prompt_toolkit`：

- multiline input；
- history；
- autocomplete；
- shortcuts。

`rich`：

- Markdown；
- syntax highlight；
- diff；
- status；
- panel；
- table。

---

## 第一批 Slash Commands

```text
/help
/exit
/clear
/status
/model
/permissions
/context
```

以后扩展：

```text
/resume
/compact
/init
/memory
/mcp
/agents
```

---

## 验收标准

可以连续执行：

```text
> what does this repository do?

> explain core/loop.py

> where is permission handled?
```

三轮属于同一 Session。

---

# Stage 3：Session System

这是 v2 第二个核心阶段。

---

## 目标

让：

```text
Session
≠
Task
```

Session 成为一级持久化对象。

---

## 新目录

```text
session/
├── models.py
├── manager.py
├── store.py
├── transcript.py
└── checkpoint.py
```

---

## SessionState

建议至少包含：

```text
session_id
project_root

created_at
updated_at

model
permission_mode

messages
tool_history

git_branch
workspace

token_usage
compact_state
```

---

## 本地存储

建议：

```text
~/.repopilot/

projects/
└── <project_hash>/
    └── sessions/
        └── <session_id>/
            ├── transcript.jsonl
            ├── metadata.json
            ├── checkpoint.json
            ├── tool_journal.json
            └── snapshots/
```

---

## Transcript

采用 append-only JSONL。

事件例如：

```text
user_message
assistant_message
tool_call
tool_result
permission
verification
system_event
```

优点：

```text
崩溃恢复
容易调试
容易重放
容易迁移
容易生成 Trace
```

---

## Resume

支持：

```bash
repopilot --continue
```

恢复当前项目最近 Session。

支持：

```bash
repopilot --resume
```

显示 Session picker。

支持：

```bash
repopilot --resume <session_id>
```

---

## Fork

支持：

```bash
repopilot --resume <id> --fork
```

逻辑：

```text
Session A

turn1
turn2
turn3
   │
   ├── Session A
   └── Session B
```

---

## 文件 Checkpoint 与 Transcript 分开

必须区分：

```text
Session Transcript
=
对话历史
```

和：

```text
File Checkpoint
=
Agent 修改的文件快照
```

未来可以支持：

```text
/revert
```

恢复 Agent 的文件改动。

---

## 验收标准

1. Session 可保存；
2. 重启程序可 Resume；
3. Tool Journal 不重复执行旧操作；
4. Fork 不污染原 Session；
5. Session 与 Project 绑定。

---

# Stage 4：Workspace & Project Instructions

## 目标

让 Agent 直接理解：

```text
当前项目
```

而不是每次都要求：

```text
PublicTaskSpec
```

---

## 新模块

```text
workspace/
├── project.py
├── discovery.py
└── trust.py
```

---

## ProjectWorkspace

至少描述：

```text
root
git_root
current_branch

language hints
package manager
build system
test framework

project instructions
```

---

## 自动发现项目

检测：

```text
pyproject.toml
requirements.txt
package.json
Cargo.toml
go.mod
pom.xml
build.gradle
CMakeLists.txt
Makefile
```

---

## 解除 Python-only

当前 `PublicTaskSpec` 可以继续保持 Benchmark 限制。

但：

```text
ProjectWorkspace
```

必须语言无关。

---

## REPOPILOT.md

建立 RepoPilot 自己的项目指令文件：

```text
REPOPILOT.md
```

同时可以考虑兼容：

```text
AGENTS.md
```

---

## REPOPILOT.md 内容

例如：

```text
项目架构
开发规范
常用测试命令
不要修改的目录
代码风格
提交规范
验证要求
```

---

## Project Rules

以后可支持：

```text
.repopilot/
└── rules/
    ├── python.md
    ├── frontend.md
    └── testing.md
```

---

## Workspace Trust

第一次打开未知项目：

```text
Trust this workspace?

This repository may contain instructions,
hooks and executable scripts.
```

用户明确批准以后：

```text
REPOPILOT.md
Hooks
MCP
Project Commands
```

才允许生效。

---

# Stage 5：Coding Tool System

这一阶段让 RepoPilot 真正拥有日常软件工程能力。

---

# 5.1 File Tools

现有：

```text
ListFilesTool
ReadFileTool
SearchTextTool
FindSymbolTool
ApplyPatchTool
```

继续保留。

拆分为：

```text
tools/
├── filesystem.py
├── search.py
└── edit.py
```

---

## 新增文件能力

```text
write_file
create_file
move_file
delete_file
mkdir
```

其中：

```text
delete
move
overwrite
```

属于较高风险操作。

---

# 5.2 Search Tools

支持：

```text
glob
grep
regex search
symbol search
```

优先调用：

```text
ripgrep
```

不可用时使用 Python fallback。

---

# 5.3 ShellTool

这是最重要的新增能力之一。

目标支持：

```text
pytest
npm test
ruff
mypy
uv run
pip
docker
cmake
cargo
go test
```

---

## 正确执行路径

必须：

```text
LLM
↓
ShellTool
↓
PermissionEngine
↓
CommandRunner
↓
Local / Sandbox
```

绝不能：

```text
LLM
↓
subprocess.run()
```

---

# 5.4 Process Manager

支持长时间命令：

```text
npm run dev
python server.py
```

新增：

```text
process/
└── manager.py
```

能力：

```text
start
poll
read stdout
read stderr
terminate
```

---

# 5.5 Git Tools

建议保留结构化 Git Tool：

```text
git_status
git_diff
git_log
git_branch
git_show
```

后续写操作：

```text
git_add
git_commit
git_checkout
git_branch_create
```

需要权限。

---

## 验收标准

Agent 可以完成完整小型 Coding Loop：

```text
Read
↓
Search
↓
Edit
↓
Shell Test
↓
Git Diff
```

---

# Stage 6：Permission System

这是 RepoPilot v2 的核心差异化之一。

---

## 目标

从：

```text
PolicyEngine
```

升级成：

```text
PermissionEngine
```

保留：

```text
ALLOW
DENY
REQUIRE_APPROVAL
```

---

## Permission Modes

第一版建议三个。

### Manual

```text
read
→ 自动

edit
→ ask

shell
→ ask
```

---

### Accept Edits

```text
read
→ 自动

edit
→ 自动

safe shell
→ 自动

dangerous shell
→ ask
```

---

### Plan

```text
read
search
git status
git diff

允许

edit
write
side-effect shell

禁止
```

---

## Permission Scope

支持：

```text
User
Project
Session
```

配置：

```text
~/.repopilot/settings.json
```

项目：

```text
.repopilot/settings.json
```

本机项目：

```text
.repopilot/settings.local.json
```

---

## Approval UI

示例：

```text
RepoPilot wants to run:

    pytest tests/test_auth.py

Reason:
Verify the authentication fix.

1. Yes
2. Yes, allow for this session
3. Always allow in this project
4. No
```

---

## Shell 风险等级

### Read-only

```text
git status
git diff
ls
pwd
python --version
```

### Normal

```text
pytest
npm test
ruff check
```

### Side-effect

```text
pip install
npm install
mv
git commit
```

### High-risk

```text
rm -rf
git reset --hard
git clean -fd
curl ... | sh
deployment
credential commands
```

---

## Protected Paths

重点保护：

```text
.git/
.repopilot/
.env
*.pem
credentials
deployment/
.github/workflows/
```

---

## 安全测试

必须加入：

```text
path traversal
symlink escape
secret access
rm command
git reset
.env access
malicious project instruction
shell injection
```

---

# Stage 7：Context Engine

这是 Coding Agent 能否处理大型项目和长 Session 的核心。

---

## 目标

将当前简单 ContextBuilder 升级为：

```text
ContextManager
```

---

## 新结构

```text
context/
├── manager.py
├── budget.py
├── instructions.py
├── projector.py
├── compactor.py
└── cache.py
```

---

## 最重要的概念

严格区分：

```text
Persistent Transcript
```

和：

```text
Active Model Context
```

---

## Context 结构

```text
完整 Transcript
        │
        ▼
ContextManager
        │
        ├── System
        ├── Project Instructions
        ├── Memory
        ├── Recent Conversation
        ├── Relevant Tool Results
        ├── Relevant Files
        └── Skills
        │
        ▼
Model Context
```

---

## Context Budget

可以动态分配，例如：

```text
System                  5%
Project Instructions   10%
Memory                   5%
Conversation            30%
Code / Tool Results     40%
Reserve                 10%
```

不要硬编码绝对比例。

---

## Tool Output Pruning

优先清理：

```text
旧 pytest 输出
旧 grep 输出
旧目录列表
重复 shell stdout
```

优先保留：

```text
用户需求
关键决策
修改文件
测试结论
错误根因
尚未完成的问题
```

---

## Compaction

支持：

```text
/compact
```

生成结构化摘要：

```text
Goal
Completed Work
Files Changed
Important Findings
Current Failures
User Decisions
Next Steps
```

之后用：

```text
Summary
+
Recent Turns
```

代替早期原始内容。

---

## `/context`

显示：

```text
Context Usage

System          4,500
Instructions    2,300
Messages       21,000
Tool Outputs   13,000
Files          18,000

Total          58,800
```

---

## 验收标准

进行 50+ Turn Session：

- 不发生 Context Overflow；
- 关键用户要求不丢失；
- Compaction 后仍可继续任务；
- Tool 输出不会无限膨胀。

---

# Stage 8：Coding Workflow & Verification

这一阶段让 Agent 从：

```text
能操作
```

升级成：

```text
会进行完整软件工程任务
```

---

## 核心行为模式

```text
Explore
↓
Plan
↓
Implement
↓
Verify
↓
Review
↓
Answer
```

注意：

这不是硬编码固定状态机，而是默认行为模式。

---

## Plan Mode

用户：

```text
/plan
```

之后：

```text
Read       ✓
Search     ✓
Git Diff   ✓

Edit       ✗
Write      ✗
Shell Side Effect ✗
```

Agent 只能分析和制定计划。

用户：

```text
implement it
```

再进入 Edit 模式。

---

## VerificationEngine

将当前：

```text
DeterministicVerifier
```

进一步抽象为：

```text
VerificationEngine
```

内部支持：

```text
VerificationStrategy
```

---

## Benchmark Strategy

```text
Hidden deterministic tests
```

---

## Interactive Coding Strategy

组合：

```text
user-specified tests
project test command
lint
type check
build
```

---

## Read-only Question

无需测试验证。

---

## 最终结果格式

不要只输出：

```text
Done.
```

应该输出：

```text
Changed:
- src/auth.py
- tests/test_auth.py

Verification:
✓ pytest tests/test_auth.py
✓ ruff check src/auth.py

Not verified:
- full integration suite
```

---

## Repair Loop

保留 RepoPilot 原有优势：

```text
Verification Fail
↓
写入 Observation
↓
Agent Replan
↓
继续修改
↓
重新验证
```

---

# Stage 9：Skills & Hooks

这些能力必须建立在稳定 Runtime 之后。

---

# 9.1 Skills

目录：

```text
.repopilot/
└── skills/
    ├── review-pr/
    │   └── SKILL.md
    └── debug-tests/
        └── SKILL.md
```

---

## SKILL.md

Frontmatter：

```text
name
description
allowed_tools
```

正文：

```text
workflow
rules
verification
```

---

## Skill 加载原则

不应把所有 Skill 内容每轮都塞给模型。

正确：

```text
Skill Metadata
↓
按任务匹配
↓
加载必要 Skill
↓
加入 Context
```

---

# 9.2 Hooks

新增：

```text
extensions/hooks/
```

支持事件：

```text
SessionStart
UserPromptSubmit

PreToolUse
PostToolUse
ToolFailure

PreVerify
PostVerify

SessionStop
```

---

## 例子

```text
PostEdit
↓
ruff format changed_file.py
```

或者：

```text
PreCommit
↓
pytest
```

---

## 为什么 Stage 1 就需要 Event Model

因为 Hooks 最终就是：

```text
Runtime Event
↓
Hook Dispatcher
↓
User-defined action
```

---

# Stage 10：MCP

当前 RepoPilot 的最小 MCP 教学实现可以保留。

但正式 v2 应使用标准 MCP Client。

---

## 建议目录

```text
extensions/mcp/
├── client.py
├── registry.py
└── config.py
```

---

## Transport

按官方 SDK 能力逐步支持：

```text
stdio
HTTP
Streamable HTTP / SSE
```

---

## 配置

```text
.repopilot/mcp.json
```

例如：

```text
github
filesystem
database
jira
```

---

## MCP 权限

MCP Tool 也必须经过：

```text
MCP Tool Call
↓
PermissionEngine
↓
Execute
```

绝不能绕过 Policy。

---

## Deferred Tool Loading

如果 MCP Server 暴露大量工具：

不要：

```text
100 个 Tool Schema
↓
每轮全部放进 Context
```

而应：

```text
Tool Summary
↓
需要时搜索
↓
加载具体 Schema
```

---

# Stage 11：Subagents / Worktree / Background Tasks

最后再进入高级 Agent 能力。

---

# 11.1 AgentProfile

建议：

```text
agents/
├── profile.py
├── registry.py
└── coordinator.py
```

定义：

```text
name
description
system_prompt
model
tools
permission
skills
max_turns
```

---

# 11.2 第一批 Subagents

只实现三个即可。

## ExploreAgent

Read-only。

负责：

```text
代码搜索
架构理解
关联文件分析
```

---

## ReviewerAgent

负责：

```text
diff review
bug risk
regression risk
code quality
```

---

## TestAgent

负责：

```text
测试策略
失败分析
覆盖不足
```

---

# 11.3 Context Isolation

主 Agent：

```text
Task
+
Conversation
+
Project Context
```

Subagent：

```text
Specific Subtask
+
Selected Files
```

最后返回：

```text
Summary
Evidence
Recommendations
```

防止主 Context 被污染。

---

# 11.4 Git Worktree

并行写操作时：

```text
Agent A
→ worktree A

Agent B
→ worktree B
```

避免多个 Agent 同时修改同一目录。

---

# 11.5 Background Tasks

支持：

```text
/background
```

例如：

```text
run full test suite
```

用户可以继续：

```text
> meanwhile inspect auth module
```

后台完成后：

```text
Background task finished
```

---

# Stage 12：Evaluation & Product Hardening

这是 RepoPilot 和大量简单 Coding Agent 项目拉开差距的部分。

---

# 12.1 Runtime Correctness

使用 ScriptedProvider 完全确定性测试：

```text
tool loop
permission
resume
interrupt
checkpoint
fork
compact
event order
journal replay
```

---

# 12.2 Coding Capability

测试：

```text
SWE-bench
local bug fixtures
repo-level refactor fixtures
```

指标：

```text
Resolved Rate
Verifier Pass Rate
```

---

# 12.3 Safety Benchmark

包括：

```text
path traversal
prompt injection
secret access
rm command
git reset
.env access
symlink escape
malicious REPOPILOT.md
malicious MCP
malicious hook
```

---

# 12.4 Session Reliability

测试：

```text
50-turn session
interrupt
restart
resume
compact
continue
fork
```

要求：

```text
状态一致
文件一致
Transcript 一致
Journal 不重复执行
```

---

# 12.5 Context Efficiency

指标：

```text
tokens / task
tokens / successful task
compaction ratio
tool-output pruning ratio
context overflow rate
```

---

# 12.6 CLI UX

测试：

```text
startup
interactive input
streaming
interrupt
approval
resume
slash commands
terminal resize
invalid command
EOF
Ctrl+C
```

---

# 6. 最终目录规划

完成 v2 后，推荐大致形成：

```text
src/repopilot/

├── cli/
│   ├── app.py
│   ├── repl.py
│   ├── commands.py
│   └── renderer.py
│
├── core/
│   ├── kernel.py
│   ├── contracts.py
│   ├── events.py
│   └── budgets.py
│
├── runtime/
│   ├── session_runtime.py
│   ├── task_runtime.py
│   ├── runner.py
│   └── journal.py
│
├── session/
│   ├── models.py
│   ├── manager.py
│   ├── store.py
│   ├── transcript.py
│   └── checkpoint.py
│
├── workspace/
│   ├── project.py
│   ├── discovery.py
│   └── trust.py
│
├── context/
│   ├── manager.py
│   ├── projector.py
│   ├── budget.py
│   ├── compactor.py
│   └── instructions.py
│
├── providers/
│   ├── base.py
│   ├── capabilities.py
│   └── openai_compatible.py
│
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── filesystem.py
│   ├── search.py
│   ├── edit.py
│   ├── shell.py
│   ├── git.py
│   └── process.py
│
├── security/
│   ├── paths.py
│   ├── permissions.py
│   ├── command_policy.py
│   ├── secrets.py
│   └── sandbox.py
│
├── verification/
│   ├── engine.py
│   ├── strategies.py
│   └── patch_review.py
│
├── memory/
│   ├── project_memory.py
│   └── episodic.py
│
├── extensions/
│   ├── skills/
│   ├── hooks/
│   └── mcp/
│
├── agents/
│   ├── profile.py
│   ├── registry.py
│   └── coordinator.py
│
├── observability/
├── benchmarks/
├── evaluation/
│
├── task.py
└── cli.py
```

注意：

> 这是最终目录目标，不应该在 Stage 1 一次性全部创建。

---

# 7. 当前文件迁移映射

## `cli.py`

当前：

```text
argparse
+
Task command
```

未来：

```text
thin compatibility wrapper
```

主要逻辑迁移到：

```text
cli/app.py
cli/repl.py
```

---

## `task.py`

保留。

继续服务：

```text
Benchmark
Batch Task
Evaluation
```

Interactive Session 不再依赖 `PublicTaskSpec`。

---

## `core/loop.py`

最大重构点。

当前：

```text
AgentRuntime
```

拆成：

```text
AgentKernel
TaskRuntime
```

旧 `loop.py` 可暂时保留兼容层。

---

## `context/builder.py`

逐步被：

```text
ContextManager
```

替代。

但第一阶段不能直接删除。

---

## `tools/coding.py`

逐步拆成：

```text
filesystem.py
search.py
edit.py
git.py
```

新增：

```text
shell.py
process.py
```

---

## `runtime/policy.py`

保留设计思想。

逐步升级为：

```text
security/permissions.py
```

继续沿用：

```text
ALLOW
DENY
REQUIRE_APPROVAL
```

---

## `runtime/checkpoint.py`

拆分职责：

```text
Session Checkpoint
Tool Journal
File Snapshot
```

不要继续全部混在一起。

---

## `mcp/`

旧教学实现保留。

生产路径逐步改成：

```text
标准 MCP SDK
```

---

## `orchestration/`

前期不动。

Stage 11 再接入：

```text
Subagents
```

---

## `benchmarks/`

保留。

---

## `evaluation/`

保留。

这是 RepoPilot 的重要差异化能力。

---

# 8. Model Provider 升级

当前 OpenAI-compatible Provider 设计应保留。

但 Provider 接口需要增加：

```text
Capabilities
```

例如：

```text
supports_tools
supports_streaming
supports_parallel_tools
supports_reasoning

context_window
max_output_tokens
```

---

## Streaming

当前：

```text
complete()
↓
ModelResponse
```

未来建议：

```text
stream()
↓
ModelEvent
```

CLI 实时渲染。

最终仍然聚合：

```text
ModelResponse
```

这样：

```text
Interactive Session
```

可以流式显示，

而：

```text
TaskRuntime
```

仍可以等待最终结果。

---

# 9. Memory 重新定义

未来必须严格区分：

```text
Session Transcript
Project Instructions
Project Memory
Episodic Memory
Context
```

---

## Session Transcript

回答：

> 当前这次 Session 说过什么？

---

## REPOPILOT.md

回答：

> 这个项目长期要求 Agent 遵守什么？

---

## Project Memory

回答：

> Agent 以前从这个项目中学到了什么？

例如：

```text
测试命令是 uv run pytest
auth 代码位于 src/security/
修改 schema 后需要运行 migration test
```

---

## Episodic Memory

回答：

> 以前某类任务是怎么解决的？

现有 EpisodicMemoryStore 可以继续作为研究能力保留。

---

# 10. 最终 CLI UX

完成核心 Stage 后，目标体验：

```text
PS D:\Projects\demo> repopilot

╭─────────────────────────────────╮
│ RepoPilot                       │
│ Model: qwen-coder               │
│ Mode: Manual                    │
│ Project: demo                   │
│ Branch: feature/login           │
╰─────────────────────────────────╯

> 为什么登录测试失败？

● Running
  pytest tests/test_login.py

● Read
  src/auth/login.py

● Search
  "verify_password"

RepoPilot:
问题来自 verify_password() 对 None token
没有进行检查……

> 修复它

● Edit
  src/auth/login.py

Permission required

Modify src/auth/login.py?

> yes

● Running
  pytest tests/test_login.py

✓ 12 passed

Changed:
  src/auth/login.py

Verification:
  ✓ pytest tests/test_login.py

> /exit

Session saved:
  7c3fa...
```

重新进入：

```bash
repopilot --continue
```

继续：

```text
> 我们继续刚才的登录问题
```

此时项目才真正完成：

```text
Agent Runtime Lab
↓
CLI Coding Agent
```

的核心转型。

---

# 11. 三个里程碑

12 个 Stage 不应该一次性实现。

建议拆成三个里程碑。

---

# Milestone A：RepoPilot v2 Alpha

完成：

```text
Stage 0
Stage 1
Stage 2
Stage 3
Stage 4
```

达到：

```text
repopilot
↓
Interactive Session
↓
读取当前项目
↓
多轮对话
↓
Session Persistence
↓
Resume / Fork
↓
Project Instructions
```

这是第一个 MVP。

---

# Milestone B：RepoPilot v2 Beta

完成：

```text
Stage 5
Stage 6
Stage 7
Stage 8
```

达到：

```text
Filesystem
Search
Shell
Git
Permission
Workspace Trust
Context Compaction
Plan Mode
Verification
Repair Loop
```

到这里：

> RepoPilot 已经是一款真正可用的 Claude Code 风格 Coding Agent。

这是最重要的项目节点。

---

# Milestone C：RepoPilot v2.0

完成：

```text
Stage 9
Stage 10
Stage 11
Stage 12
```

加入：

```text
Skills
Hooks
MCP
Subagents
Worktree
Background Tasks
Advanced Evaluation
```

形成完整版。

---

# 12. 校招简历优先级

并不是所有功能的简历价值都一样。

最高优先级：

```text
1. Session-oriented Agent Runtime

2. Context Management / Compaction

3. Shell + Permission + Sandbox

4. Deterministic Verification / Repair Loop

5. Evaluation
```

第二优先级：

```text
MCP
Skills
Hooks
Subagents
Worktree
Background Tasks
```

---

## 不应该陷入的误区

不要追求：

> Claude Code 有 50 个功能，所以 RepoPilot 也必须做 50 个功能。

而应该做到：

```text
核心 Harness
做深

Context
做深

Safety
做深

Verification
做深

Evaluation
做真
```

---

# 13. 最终项目技术故事

完成改造以后，项目可以形成完整技术演进：

```text
RepoPilot v1
=
Task-oriented Agent Runtime Lab

        ↓

发现问题

Benchmark Runtime
不适合长期真实软件开发

        ↓

架构重构

AgentKernel
+
SessionRuntime
+
TaskRuntime

        ↓

Interactive Harness

CLI
Session
Streaming
Context

        ↓

Coding Environment

Filesystem
Search
Shell
Git

        ↓

Governance

Permissions
Sandbox
Workspace Trust

        ↓

Reliability

Checkpoint
Journal
Verification
Repair

        ↓

Extensibility

Skills
Hooks
MCP
Subagents

        ↓

Evaluation

SWE Benchmark
Safety Benchmark
Session Benchmark
Context Benchmark
```

最终项目不应被描述为：

> 我仿照 Claude Code 做了一个终端 Chatbot。

而应描述为：

> **基于已有可评测 Agent Runtime，重新设计 session-oriented coding-agent harness，引入持久化会话、动态上下文管理、受治理 Shell/Git 执行、确定性验证与恢复机制，并使用软件工程 Benchmark、安全测试与 Session Reliability Test 对系统进行评估。**

---

# 14. 正式开发顺序

最终建议固定为：

```text
RepoPilot v1 Baseline
        │
        ▼
Stage 0
冻结现有测试和 Benchmark
        │
        ▼
Stage 1
AgentKernel / TaskRuntime 解耦
        │
        ▼
Stage 2
Interactive CLI
        │
        ▼
Stage 3
Session / Resume / Fork
        │
        ▼
Stage 4
Workspace / REPOPILOT.md
        │
        ▼
======== v2 Alpha ========
        │
        ▼
Stage 5
Filesystem / Search / Shell / Git
        │
        ▼
Stage 6
Permission / Trust / Sandbox
        │
        ▼
Stage 7
Context Manager / Compact
        │
        ▼
Stage 8
Plan / Verify / Repair
        │
        ▼
======== v2 Beta ========
        │
        ▼
Stage 9
Skills / Hooks
        │
        ▼
Stage 10
MCP
        │
        ▼
Stage 11
Subagents / Worktree / Background
        │
        ▼
Stage 12
Benchmark / Safety / Hardening
        │
        ▼
========= v2.0 =========
```

---

# 15. 每个 Stage 的固定开发模板

为了保证后续不会陷入无序开发，每一个 Stage 都必须按照以下结构执行。

---

## A. Stage 目标

明确：

> 这一阶段最终让 RepoPilot 获得什么能力？

---

## B. 为什么现在做

说明：

```text
依赖哪些前置能力
为什么不能提前
为什么不能推迟
```

---

## C. 阅读旧代码

只阅读：

```text
本 Stage 真正需要修改的模块
```

不再逐文件学习整个 RepoPilot。

---

## D. 设计

明确：

```text
新增文件
修改文件
保留文件
废弃文件
```

---

## E. 核心契约

优先确定：

```text
Class
Protocol
Dataclass
Event
Input
Output
```

再写实现。

---

## F. 实现顺序

遵循：

```text
Contract
↓
Minimal Implementation
↓
Unit Test
↓
Integration
↓
CLI
↓
Regression
```

---

## G. 验收标准

每阶段必须有能够直接运行的结果。

例如 Stage 2：

```bash
repopilot
```

能够进入交互模式。

---

## H. 测试

至少考虑：

```text
Unit Test
Integration Test
CLI Test
Safety Test
Regression Test
```

---

## I. 文档

每 Stage 更新：

```text
docs/v2/
```

并记录：

```text
architecture decision
implemented feature
known limitation
test result
```

---

# 16. 项目推进原则

整个 v2 改造遵循以下原则。

## 原则一

> **保留 RepoPilot 已经有价值的设计，不为了模仿 Claude Code 而盲目推倒重写。**

---

## 原则二

> **Session 是 v2 的一级抽象。**

---

## 原则三

> **AgentKernel 与 Session / Benchmark Harness 解耦。**

---

## 原则四

> **强 Tool 能力必须与强 Permission / Sandbox 同步建设。**

---

## 原则五

> **Context 不等于完整 Transcript。**

---

## 原则六

> **模型说完成，不等于任务真的完成。**

Verifier / Verification Engine 继续作为 RepoPilot 的核心特点。

---

## 原则七

> **先单 Agent，再 Subagent；先核心 Runtime，再扩展协议。**

---

## 原则八

> **每完成一个 Stage，都必须保留旧功能 Regression Test。**

---

## 原则九

> **真实 Benchmark 和安全测试比“功能数量”更重要。**

---

# 17. 当前真正的下一步

正式实施时，不再继续旧版“逐模块完整学习”的路线。

下一步应进入：

```text
Stage 0
RepoPilot v1 Baseline 冻结

↓

Stage 1
AgentKernel / TaskRuntime 架构拆分
```

Stage 1 开始前，重点只审查：

```text
core/loop.py
core/contracts.py
runtime/checkpoint.py
runtime/policy.py
tools/base.py
context/builder.py
task.py
cli.py
```

目标不是逐行学习，而是回答：

```text
哪些逻辑属于 AgentKernel？
哪些属于 TaskRuntime？
哪些属于 SessionRuntime？
哪些状态需要重新定义？
哪些接口必须保持兼容？
```

完成这个架构拆分以后，再进入真正的代码改造。

---

# 18. 最终目标

完成 RepoPilot v2 后，用户应该能够：

```bash
cd project
repopilot
```

然后：

```text
> explain this repository

> find why login tests fail

> only analyze, do not edit

> implement the fix

> run tests

> review the diff

> commit this change

> /compact

> /exit
```

以后：

```bash
repopilot --continue
```

继续同一个开发任务。

同时系统依旧具备：

```text
Policy
Sandbox
Checkpoint
Journal
Trace
Verifier
Benchmark
Safety Evaluation
```

最终形成：

> **Claude Code 风格的交互体验 + RepoPilot 自己的可治理、可恢复、可验证、可评测 Agent Runtime。**

这就是 RepoPilot v2 的完整改造目标。
