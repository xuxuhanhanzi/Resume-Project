# 实验记录：P11 最小可执行 Multi-Agent 闭环

## 1. 目标

把原有只包含状态图、消息 envelope 和 trace 摘要的 `MultiAgentHarness` 接入真实
`AgentRuntime`，至少完成一个可验证的软件修复任务，并证明协调层可持久化、取消、
超时和在进程中断后恢复，而不会重复已经完成的 Executor 周期。

## 2. 环境

- 平台：Windows 本地工作区
- Python：3.12.3
- 代码路径：`D:\Users\27475\Desktop\Resume_Project\RepoPilot`
- 模型：测试使用确定性的 `ScriptedProvider`
- 执行环境：`LocalTrustedRunner(trusted=True)`，仅用于项目自带可信 fixture
- 数据：`examples/bugfix_demo`

## 3. 实验变量

- 主变量：协调层从数据结构骨架升级为可执行状态机
- 固定项：原 `AgentRuntime`、工具策略、确定性 Verifier、ExecutionJournal 和任务契约
- 对比对象：原 `MultiAgentHarness` 仅有 `send/detect_drift/trace_summary`

## 4. 实现与命令

实现顺序：

1. Planner 以无工具只读模型调用生成计划；
2. Executor 调用现有 `AgentRuntime`，保持唯一写者；
3. `AgentRuntime` 只有在确定性 Verifier 通过后才返回 Completed；
4. Reviewer 接收冻结任务、Executor 答案和统一 diff，只能返回 pass/replan JSON；
5. 协调层保存节点、周期、消息和反馈；Executor 独立保存 checkpoint/journal；
6. Reviewer 拒绝时以反馈启动新的有界 Executor 周期。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests\unit\test_harness.py tests\integration\test_multi_agent_runtime.py
.\.venv\Scripts\python.exe scripts\dev.py check
```

## 5. 输出路径

- 实现：`src/repopilot/orchestration/harness.py`
- 集成测试：`tests/integration/test_multi_agent_runtime.py`
- 运行时协调 checkpoint：`<artifacts_root>/<run_id>/checkpoint.json`
- 冻结基线：`<artifacts_root>/<run_id>/baseline.json`
- Executor checkpoint/journal：沿用 `AgentRuntime` 自身 artifact 目录

## 6. 结果

| 指标 | 数值 |
|---|---:|
| Harness 专项测试 | 8 passed |
| 全仓测试 | 74 passed |
| Ruff | passed |
| strict mypy | 101 source files, 0 errors |
| 真实 fixture 修复 | passed |
| 进程取消后 Reviewer 阶段恢复 | passed |
| 显式取消 | passed |
| 节点超时持久化为失败 | passed |

## 7. 失败与异常

- 初次 Lint 触发 Ruff `SIM105`，原因是手写 `try/except CancelledError: pass`；改为
  `contextlib.suppress(asyncio.CancelledError)` 后通过。
- 本轮没有运行不可信外部仓库，也没有扩大模型或数据评测范围。

## 8. 结论

P11 已不再只是消息数据结构：最小状态图能够调用真实单 Agent Runtime，状态转移由
确定性验证和只读 Reviewer 结果驱动，并具备协调层恢复、取消和超时证据。

该结果不证明生产级 Multi-Agent。尚未实现 rule/model/hybrid 路由、级联降级、任务 API、
1/4/8 并发、故障注入和相对最佳单 Agent 的准确率/成本/延迟对照。

## 9. 下一步

1. 增加显式路由决策与降级理由记录；
2. 增加提交/状态/取消/事件 API；
3. 做 1/4/8 并发和故障注入；
4. 在同一冻结任务集上与单 Agent 做成对对照。
