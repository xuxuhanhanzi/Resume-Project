# HTTP / SSE API

RepoPilot 提供依赖最小的 ASGI 传输层。任务只可引用 `--task-root` 下的受信任 YAML；服务
默认监听 `127.0.0.1`，所有 `/v1/*` 路由均要求 Bearer token，`/health` 除外。

## 安装与启动

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
$env:REPOPILOT_API_TOKEN = '<至少 16 字符的随机令牌>'
repopilot-api --task-root evaluation\cases --model qwen2.5:7b `
  --artifacts artifacts\service --max-concurrency 4
```

不可信任务还必须传入 `--sandbox-image <固定镜像引用>`；否则服务 fail-closed。服务状态
保存在当前进程内，Multi-Agent 与 executor checkpoint 则写入 `--artifacts`，用于恢复。

## 提交与查询

```powershell
$headers = @{ Authorization = "Bearer $env:REPOPILOT_API_TOKEN" }
$body = @{ run_id = 'demo-001'; task_path = 'subtract/task.yaml'; timeout_seconds = 300 } |
  ConvertTo-Json -Compress
Invoke-RestMethod -Method Post http://127.0.0.1:8765/v1/runs `
  -Headers $headers -ContentType application/json -Body $body
Invoke-RestMethod http://127.0.0.1:8765/v1/runs/demo-001 -Headers $headers
```

取消使用 `DELETE /v1/runs/{run_id}`。重复 run ID 返回 409；过大请求返回 413；超过固定窗口
限流返回 429。run ID 仅允许 1-64 个字母、数字、点、下划线或连字符。

## SSE 与断线续传

事件端点为 `GET /v1/runs/{run_id}/events`，事件 ID 单调递增。客户端重连时传入最后已经
处理的 ID：

```text
Last-Event-ID: 12
```

服务将从事件 13 开始发送。也可用 `?after=12`。SSE 客户端断开不会隐式取消任务；任务
取消必须显式调用 DELETE，避免网络抖动造成副作用不完整。

## 安全边界

- token 使用常量时间比较，日志不输出凭据；
- task path 解析后必须仍位于 allowlisted task root；
- 请求体、并发度、任务超时与请求速率均有上限；
- HTTP 传输不会把 hidden tests 或 evaluator-only 字段注入 PublicTaskSpec；
- 单进程内存状态不等于分布式队列或多租户控制面，不作生产级声明。
