# 实验记录：P11 结构化服务与 1/4/8 并发基线

- 日期：2026-08-11
- 状态：核心实现与合成负载完成；真实任务负载待运行
- 目标：为 Multi-Agent coordinator 增加 transport-neutral 的提交、状态、事件、取消、
  超时与并发限制，并用确定性模拟 runner 验证 1/4/8 调度行为。
- 单一主变量：新增 service orchestration 层；不改变 AgentRuntime、路由、模型或 evaluator。

## 验收边界

1. run ID 唯一，状态和事件序号单调且可查询；
2. `max_concurrency` 是硬上限；
3. cancel 通过共享 event 传递给 harness，timeout/exception 形成可定位终态；
4. 事件流在终态后自然结束，不泄漏后台任务；
5. 输出 1/4/8 concurrency 的 P50/P95、吞吐、成功/失败数；
6. 模拟负载只证明调度层，不证明真实模型性能或生产容量。

## Evidence

- 实现：`src/repopilot/orchestration/service.py`；
- 测试：`tests/integration/test_orchestration_service.py`；
- 产物：`artifacts/runs/p11_service_load/simulated_1_4_8.json`；
- 统一门禁：135 files formatted；Ruff 通过；strict mypy 106 source files / 0 errors；
  pytest 84 passed。

24 个任务、每个 runner 固定等待 20ms 的结果：

| concurrency | completed | failed | wall (s) | throughput (task/s) | P50 (s) | P95 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0 | 0.746 | 32.15 | 0.387 | 0.715 |
| 4 | 24 | 0 | 0.186 | 128.72 | 0.109 | 0.187 |
| 8 | 24 | 0 | 0.093 | 257.21 | 0.062 | 0.093 |

结论仅限调度层：硬并发限制、状态/事件、合作式取消、超时和异常归因均通过。因为工作负载
没有模型推理、仓库 I/O、Verifier 或 GPU，这些数字不能写成 RepoPilot 的真实吞吐或生产容量。
