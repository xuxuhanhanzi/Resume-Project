# Agent Card

## Runtime

- 版本：`1.0.3`
- 正式评测模型：本地 `qwen2.5:7b`，digest `845dbda0ea48...`
- 微调消融模型：Qwen2.5-1.5B base 与匹配的 SFT+DPO Adapter
- 离线测试模型：`ScriptedProvider`
- Prompt：`src/repopilot/context/builder.py`
- 状态：Created/Preparing/Running/Verifying/Approval Required/Completed/Failed/Cancelled
- 默认预算：12 iterations、32 tool calls、32k tokens、600 seconds

## Tools

`list_files`、`read_file`、`search_text`、`find_symbol`、`retrieve_code`、`apply_patch`、
`run_tests`、`git_diff`、`inspect_failure`；Reviewer 以只读 Agent-as-Tool 形式提供。

## Safety and evaluation

- 路径规范化、allow/deny paths、精确替换 Patch、高风险文件审批；
- 不可信执行只允许 DockerSandboxRunner，默认断网、只读 RootFS、非 Root、资源限制；
- Benchmark：FRAMES 60 题、DABench 35 题、SWE-bench-Live 5 个干净任务，以及
  `repopilot_python_micro_v1` 10 个开发夹具；
- 结果优先级：测试 Outcome > Tool/State Trajectory > Final Text。

## Known limitations

- MCP 是三个核心方法的教学子集，不是完整 SDK/Transport 实现；
- Multi-Agent Harness 已接入 Planner、AgentRuntime/确定性 Verifier 和只读 Reviewer，
  支持持久化节点状态、进程中断恢复、取消、超时与有界返工；fixed/rule/model/hybrid
  路由支持级联降级、运行时 binding 和决策审计；ASGI API 支持 Bearer 鉴权的提交、
  状态、取消、可续传 SSE、限流、超时和并发限制；真实 Runtime/Verifier 的 1/4/8 负载
  已运行，但使用 ScriptedProvider，尚无本地 Qwen 服务容量证据；
- 1.5B SFT+DPO Adapter 在当前单次 DABench 消融中由 base 9/35 退化到 0/35；
- SWE-bench-Live 干净 holdout 当前为 0/5，代码修复能力门尚未通过；
- 无 gVisor/微虚拟机证据；
- 无多进程持久队列、多租户、真实 Qwen 并发压测或大规模 SWE-bench 结果。
