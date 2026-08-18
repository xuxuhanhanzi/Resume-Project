# 实验记录：P11 HTTP/SSE 传输与故障归因

- 日期：2026-08-14
- 状态：HTTP/SSE transport 完成；真实模型 1/4/8 负载仍待运行
- 主变量：在既有 `TaskService` 上新增 ASGI 边界，不改变 AgentRuntime、路由或 evaluator。

## 验收目标

1. JSON submit/status/cancel 走真实 HTTP；
2. SSE 事件有单调 ID，支持 `Last-Event-ID` 断线续传；
3. Bearer 鉴权、固定窗口限流、请求体上限和 task path allowlist fail-closed；
4. 客户端断开不隐式取消任务；模型/runner 故障形成可查询终态；
5. 不把 transport 测试描述为真实模型容量测试。

## 自动化结果

- 新增 5 个集成测试：公开 health/鉴权、提交与状态、SSE 全量和续传、断开/显式取消、
  409/429/413，以及 task path 逃逸；
- `ruff check`：通过；
- strict mypy：110 source files，0 errors；
- integration：39 passed；统一 `dev.py check`：89 passed。

## localhost 黑盒结果

Uvicorn 0.52.3 在 `127.0.0.1:8765` 启动后，通过 PowerShell HTTP 客户端验证：

| 项目 | 结果 |
|---|---|
| `GET /health` | 200 |
| 未携带 token 查询 run | 401 |
| 提交合法 task path | 202 |
| 全量 SSE | event ID 0, 1, 2 |
| `Last-Event-ID: 0` 重连 | 仅返回 1, 2；不重复 submitted |
| 关闭的本地模型端口 | run=`failed`，3 events，原因归因为 `ModelProviderError` |

本次故障注入使用“127.0.0.1:11434 主动拒绝连接”，证明 HTTP 层不会让后台异常变成永久
RUNNING 或无响应。它不是模型成功率或吞吐证据。

## 异常记录

- 默认清华 PyPI 镜像在下载 Uvicorn 时发生 `SSL: UNEXPECTED_EOF_WHILE_READING`；有限重试
  失败后改用官方 PyPI，成功安装 0.52.3。
- Uvicorn 在 Windows PTY 收到 Ctrl+C 后未及时退出；验收结束后仅终止明确记录的测试
  PID 28192。没有批量终止进程。

## 结论与下一步

HTTP/SSE 传输门关闭。下一门必须使用真实 AgentRuntime + Verifier 在固定任务副本上运行
1/4/8 并发，并记录 P50/P95、吞吐、CPU/RAM、模型失败、工具超时、取消与恢复；现有
`simulated_1_4_8.json` 继续只作为调度层证据。
