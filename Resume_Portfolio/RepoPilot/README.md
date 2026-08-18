# RepoPilot

RepoPilot 是一个本地优先、反馈驱动、可治理的 Agent Runtime Lab + 评测骨架。它覆盖
知识研究、数据分析、软件工程三个领域，用最小实现验证 Agent 的规划、检索、工具、
执行反馈、验证、记忆和治理机制在不同任务形态下的效果。

它不是生产平台：没有 Web 前端、数据库、多租户或完整 MCP 兼容。
项目最新状态和对外指标以 `docs/STATUS.md` 为唯一权威口径。

当前冻结版本：`v1.0.4`；功能与声明边界见
[`docs/RELEASE_NOTES_v1.0.0.md`](docs/RELEASE_NOTES_v1.0.0.md)。1.0.1-1.0.2 修复 Windows
干净克隆时三类历史冻结数据使用不同换行口径的问题，1.0.3 固化成功复现记录；1.0.4
关闭真实本地 Qwen 服务容量实验门。

## 三领域评测

| 场景 | 数据集 | 主指标 | 当前结果 |
|---|---|---|---|
| 知识研究 | Google FRAMES | 答案准确率 | **18.3% (11/60)**，3 次确定性重复，✅ 可引用 |
| 数据分析 | InfiAgent-DABench | 问题准确率 | **73.3%**，3 次为 25/35、26/35、26/35，✅ 可引用 |
| 软件工程 | SWE-bench-Live | 官方 resolved rate | **0.0% (0/5)**，干净 holdout，✅ 可引用负结果 |

三领域分开报告，禁止合成跨领域总分。

## 已实现闭环

```text
Task Spec → Context → Local Model → Structured Tool Call
         → Policy → Tool/Sandbox → Observation → Checkpoint
         → Deterministic Verifier → Complete/Repair/Fail
```

核心边界：

- `ModelProvider` 首选本地 Qwen 的 OpenAI-compatible 服务；
- `ScriptedProvider` 让 Runtime、恢复和安全测试完全离线；
- 模型只建议动作，只有确定性 Verifier 可以完成任务；
- 所有路径先规范化，任意 Shell 不是默认工具；
- 不可信执行必须使用 Docker 沙箱，缺少 Docker 时 fail-closed；
- 隐藏测试不属于 Public Task Spec，也不能挂载进 Agent 工作区；
- 每次状态转换、工具调用和验证都有 Checkpoint 与 JSONL Trace。

## 检索改进 (P8A，历史 10 题开发消融)

| 配置 | 平均准确率 | 说明 |
|---|---|---|
| B1 BM25 | 16.7% | 词法匹配 baseline |
| R1 Dense | 0.0% | 通用 embedding 在百科 chunk 上失效 |
| **R2 Hybrid (BM25+Dense RRF)** | **20.0%** | **最佳，+3.3pp** |
| R3 Hybrid+Rerank | 20.0% | embedding reranker 无额外收益 |
| A2 No-retrieval | 20.0% | 首段 vs 检索，差异不显著 |

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q --basetemp D:/Temp/rp_pytest
.\.venv\Scripts\python.exe -m repopilot demo scripted
```

HTTP/SSE 服务使用可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
$env:REPOPILOT_API_TOKEN = '<至少 16 字符的随机令牌>'
repopilot-api --task-root evaluation\cases --model qwen2.5:7b
```

端点、SSE 续传和安全边界见 [`docs/api.md`](docs/api.md)。

## 目录导航

- `src/repopilot/core/`：Agent Loop、状态、预算和共享契约；
- `providers/`、`tools/`、`context/`：决策和环境接口；
- `retrieval/`：BM25、Dense、Hybrid (RRF)、Reranker、FirstChunk 消融；
- `memory/`、`skills/`、`mcp/`、`orchestration/`：Agent 扩展机制 + Multi-Agent harness；
- `runtime/`、`security/`、`verification/`：可靠性、安全边界 + patch reviewer；
- `benchmarks/`：三领域 contracts、adapters、executors、runner；
- `evaluation/`：指标、报告、manifest、实验矩阵；
- `docs/lessons/`：Stage 1–6 教学讲义与术语表；
- `docs/experiments/`：实验报告（P5–P12）；
- `artifacts/`：本地 Trace、Checkpoint、实验结果，不提交版本库。

## 声明边界

- 最小 MCP 实现只覆盖教学子集，不宣称完整 MCP 兼容；
- 10 题检索消融只用于方法诊断，正式 FRAMES 指标使用 60 题、3 次重复；
- 旧 SWE 三题已被开发污染；正式负结果使用 5 个未污染任务，当前 resolved 为 0/5；
- P10 已完成 QLoRA SFT + DPO，并在匹配的 Qwen2.5-1.5B 上完成一次下游消融：base
  为 9/35，SFT+DPO adapter 为 0/35。该结果表明当前微调退化，不改变 7B 主指标；
- `orchestration/MultiAgentHarness` 已能以 Planner → AgentRuntime/Verifier → Reviewer
  执行真实任务，并支持持久化状态、进程中断恢复、取消、超时和有界返工；新增的
  fixed/rule/model/hybrid 路由支持只读模型决策、级联降级、运行时 binding 与 checkpoint
  审计；ASGI 服务层支持 Bearer 鉴权的 JSON 提交/状态/取消、SSE 事件流与断线续传、
  限流、请求上限、超时和硬并发限制；真实 Runtime/Verifier 1/4/8 已完成 360/360，
  其模型为 ScriptedProvider，因此只代表调度容量；另一个固定 digest 的真实
  Qwen2.5-7B/Q4_K_M 服务实验完成 1/4/8 并发各 24 请求（72/72 非空），吞吐为
  5.52/16.80/18.62 req/s，P95 为 4.16/1.38/1.24s。该短请求结果仍不代表端到端 coding-agent 吞吐；
- 三领域分开报告，不给综合总分。

## 复现实验

```powershell
# FRAMES B1 BM25
python scripts/run_frames_oracle_smoke.py --run-id NEW_ID --count 10 `
  --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e `
  --planned --reasoned --reviewed --reasoning-output-tokens 512

# FRAMES R2 Hybrid
python scripts/run_frames_oracle_smoke.py --run-id NEW_ID --count 10 `
  --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e `
  --planned --reasoned --reviewed --reasoning-output-tokens 512 --retriever-type hybrid
```

以上命令复现的是 10 题历史开发消融。正式 FRAMES 60 题证据与全部 run 哈希见
`docs/evidence/artifact_index.json`；固定条件为 temperature=0、seed=7、被测 Agent
network=deny。
