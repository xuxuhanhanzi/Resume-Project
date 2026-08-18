# 实验记录：P11 真实 Runtime / Verifier 1-4-8 负载

- 日期：2026-08-14
- 正式 run：`p11_real_runtime_1_4_8_120c`
- 主变量：`TaskService.max_concurrency` = 1 / 4 / 8
- 固定项：每档 120 个独立 bugfix 工作区、同一 ScriptedProvider 轨迹、同一真实
  AgentRuntime/MultiAgentHarness、文件 patch、Python 子进程 verifier。

## Claim 边界

本实验不是 sleep 模拟：每个任务都会复制夹具、执行 planner → AgentRuntime → verifier →
reviewer，修改 `calculator.py`，启动 `verify.py` 子进程并保存双层 checkpoint。模型响应来自
确定性 `ScriptedProvider`，因此结果证明调度、文件 I/O、checkpoint 和 verifier 的容量，
**不证明 Qwen/LLM 推理吞吐**。

## 命令

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe scripts\benchmark_real_orchestration_service.py `
  --tasks 120 `
  --run-root artifacts\runs\p11_real_runtime_1_4_8_120c\work `
  --output artifacts\runs\p11_real_runtime_1_4_8_120c\real_runtime_1_4_8_120c.json
```

## 正式结果

| 并发 | 成功 | 吞吐 task/s | wall | P50 | P95 | P99 | Working Set | Peak WS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120/120 | 10.08 | 11.90s | 5.97s | 11.35s | 11.81s | 30.79 MiB | 30.79 MiB |
| 4 | 120/120 | 33.26 | 3.61s | 1.86s | 3.45s | 3.58s | 31.32 MiB | 31.47 MiB |
| 8 | 120/120 | 34.21 | 3.51s | 1.82s | 3.42s | 3.47s | 32.07 MiB | 32.23 MiB |

- 总计：360 completed，0 failed，0 cancelled；360 个 coordinator checkpoints 与 360 个
  独立工作区均生成；
- P50/P95/P99 从 submit 起算，包含信号量排队时间；
- `process_cpu_seconds` 只覆盖父 Python 进程，不覆盖 verifier 子进程，因此不作为总 CPU
  使用率结论；Working Set/Private Bytes 使用 Windows `GetProcessMemoryInfo`；
- 4→8 并发吞吐仅提升约 2.9%，在本工作负载与机器上已接近 I/O/子进程启动饱和点。

原始 JSON SHA-256：
`534A25E86337FD39A742BB1447A7EB106F31C2D584005F9A4E049FD197E64188`。
提交到 `docs/evidence/system/p11_real_runtime_1_4_8_120c.json` 的跨平台 LF 副本 SHA-256：
`CF0D6302F702E6A5E8D7547C3D4C0880637CF0327A25FA01FF2817BE0029DF21`；两者 JSON 内容相同，
哈希差异来自 Windows 原始文件的 CRLF。

## Pilot 与失败 run

- `p11_real_runtime_1_4_8`：12 task/档的 36/36 pilot；不足 3 秒，未作为正式数字；
- `p11_real_runtime_1_4_8_120`：首个 120 task/档成功 run，缺少内建内存字段；
- `p11_real_runtime_1_4_8_120b`：并发 1 完成后，64 位 Windows pseudo-handle 因 ctypes
  默认返回类型被截断，`GetProcessMemoryInfo` errno 6；失败产物保留；
- `120c` 显式声明 `GetCurrentProcess.restype=c_void_p` 与 API 参数类型后通过。

## 故障与恢复覆盖

- 模型不可达：真实 HTTP 提交形成 `failed` + `ModelProviderError`，事件流正常结束；
- 模型异常、service timeout、合作式 cancel：`test_orchestration_service.py`；
- coordinator 进程中断后 reviewer-stage resume、显式取消、node timeout：
  `test_multi_agent_runtime.py`；
- SSE 断开不取消任务、Last-Event-ID 恢复：`test_http_api.py`。

## 结论

P11 的真实 Runtime/Verifier 1-4-8 调度与主要故障归因门已关闭。由于本机 Ollama 端口未
运行，真实 Qwen 推理容量仍未测；该缺口不影响 runtime 负载结论，但必须继续作为发布限制。
