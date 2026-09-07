# RepoPilot v1.3.0

本版实施 Claude Code 风格后续路线的首项：将交互 coding turn 的完成过程投影为可审计的
`Explore → Plan → Implement → Verify → Review → Answer` 工作流信号，并提供运行时生成的最终证据报告。

## 工作流控制器

- `AgentState` 新增可持久化的 `workflow_stage`，不会改变既有权限模式，也不会把流程阶段当作模型必须
  遵循的固定状态机。
- 每个 turn 从 `explore` 开始；成功记录待办、执行写入动作、运行不可变 `run_tests`、准备最终回答时，
  Runtime 会依次记录对应阶段事件。中断、恢复与 fork 会保留最后的阶段状态。
- `workflow_stage_changed` 与 `workflow_report_created` 被写入 session JSONL trace，因此 `/trace` 可以
  显示阶段与报告事实而不显示模型隐藏推理。

## 最终验证报告

- 运行时在 turn 前后对受允许路径做有界文本快照，最终报告只列出实际变化的文件。
- 仅 `run_tests` 的真实 ToolResult 可进入报告的 `Verification` 部分；没有执行测试时，改动任务会明确
  标为 `Not run in this turn`。
- 正常 text 输出在模型回答后附加 `Workflow evidence`；JSONL 的最终 `result` 事件新增结构化
  `workflow` 字段，方便 IDE 与脚本消费。
- 快照超过既有文件/字节上限或读取失败时，报告显示 inventory 不可用，而不是伪造 changed files。

## 可靠性修复

- 会话锁检测若发现锁属于当前 Python 进程，会直接认定它仍有效，不再执行冗余的
  `os.kill(current_pid, 0)` liveness probe。这避免 Windows/Codex 监督环境把无信号探测误判为进程终止。
- 修复交互 `/model` 切换时审批对象未传入 session UI 的闭包问题；模型切换后审批提示会使用新的
  Provider/模型标签。
