# RepoPilot v1.0.0 Release Notes

## 已验证能力

- 本地模型兼容的 AgentRuntime、结构化工具、预算、权限、trace、checkpoint 和幂等 journal；
- 由确定性 verifier 决定完成状态的单 Agent 与 Multi-Agent 执行链；
- fixed/rule/model/hybrid 路由，模型失败时记录可归因级联；
- Bearer 鉴权的 JSON submit/status/cancel 与可续传 SSE；
- 固定 digest Docker fail-closed 沙箱；
- 三领域独立 benchmark contract 与哈希证据。

## 冻结结果

| 领域/系统 | 冻结结果 | 边界 |
|---|---|---|
| FRAMES | 11/60 = 18.3%，3 次一致 | oracle-document，qwen2.5:7b |
| DABench | 25/35、26/35、26/35，mean 73.3% | qwen2.5:7b，三次重复 |
| SWE-bench-Live | 0/5 resolved | fresh holdout，正式负结果 |
| 1.5B Adapter | base 9/35 → SFT+DPO 0/35 | 单次诊断负结果，不影响 7B 主指标 |
| Runtime load | 360/360；1/4/8 = 10.08/33.26/34.21 task/s | ScriptedProvider + 真实 I/O/verifier，非 LLM 吞吐 |

## 已知限制

- 不提供 Web UI、数据库、多租户或分布式持久队列；
- HTTP service 的 run registry 是单进程内存状态，任务 checkpoint 才持久化；
- 本机发布时 Ollama 未运行，因此没有真实 Qwen 1/4/8 服务容量数据；
- SWE 代码修复为 0/5，已按无新干预不追加 validation 的停止规则冻结；
- Docker profile 不是 gVisor/Firecracker，也未运行破坏性 fork bomb/磁盘填满实验；
- 大型 benchmark/artifact 不全部提交，权威路径与哈希见 evidence index 和实验文档。

## 最小复现

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\dev.py check
.\.venv\Scripts\python.exe -m repopilot demo scripted
```

Docker 现场安全门见 `docs/experiments/2026-08-14_r7_live_sandbox_security.md`，API 用法见
`docs/api.md`。
