# RepoPilot 项目总计划与接手手册

> 状态日期：2026-08-07  
> 项目路径：`D:\Users\27475\Desktop\Resume_Project\RepoPilot`  
> 当前主线：面向知识研究、数据分析、软件工程三类任务的本地优先、反馈驱动、可治理 Agent 系统  
> 本文件是当前交接状态的第一入口；旧文档中的环境状态可能已经过期。

## 0. 接手者先读

### 0.1 用户已经确认的方向

1. 项目不能只聚焦软件工程，否则无法体现 Agent 的通用系统能力。
2. 固定使用三个权威公开数据集，不自行构造主评测集：
   - 知识研究：Google FRAMES。
   - 数据分析：InfiAgent-DABench。
   - 软件工程：SWE-bench-Live。
3. 三个领域分别报告各自官方主指标，禁止制造含义模糊的跨领域综合分数。
4. 项目最终要覆盖卡码大模型路线中的 Prompt、RAG、Agent、微调、部署、Transformer、AI 编程等技术，但要区分“代码存在”“真实运行”“正式评测证明”。
5. 用户偏好简洁、聚焦的阶段汇报；详细信息放入文件，不在对话中堆砌。

### 0.2 当前最重要的事实

- Stage 1–6 的教学型 Agent Runtime 已实现。
- 三个官方数据集的接入协议和烟雾链路均已跑通。
- FRAMES 和 DABench 的核心基线/消融已完成三次重复。
- SWE-bench-Live 三任务 Agent 官方结果为 `0/3`，尚未通过能力门槛。
- 当前完成的是系统实现和本地模型推理评测，**没有进行 SFT、LoRA、QLoRA、DPO、RLHF 或其他模型权重训练**。
- 根仓库有大量未提交改动；不得用 `git reset --hard`、`git checkout --` 或清理命令破坏现状。
- 当前机器的 Docker daemon、Ollama 和三个 SWE 官方镜像均可用。

### 0.3 接手后的第一条原则

不要立即扩大样本或开始微调。先完成“接手验证与状态冻结”，然后按本文第 9 节的顺序推进。

---

## 1. 项目研究问题与边界

### 1.1 核心研究问题

研究一个本地优先的 Agent 系统，如何通过规划、检索、工具、执行反馈、验证、记忆和治理机制，在不同任务形态下提高成功率，同时控制 tokens、耗时、安全风险和失败恢复成本。

三个场景回答不同问题：

| 场景 | 数据集 | 能力问题 | 主指标 |
|---|---|---|---|
| 知识研究 | Google FRAMES | 能否检索并组合多来源证据，输出正确答案 | 答案准确率 |
| 数据分析 | InfiAgent-DABench | 能否理解表格任务、生成和执行代码、按格式返回答案 | 官方问题准确率 |
| 软件工程 | SWE-bench-Live | 能否定位真实仓库问题、修改代码并保持回归测试 | 官方 resolved rate |

共享工程指标包括：迭代数、工具调用数、输入/输出 tokens、耗时、恢复次数、无效调用、预算耗尽和安全拦截。

### 1.2 不应宣称的内容

- 不宣称当前系统是生产平台。
- 不把 10 题 smoke 结果外推为完整数据集性能。
- 不把内部 verifier 结果当成 SWE-bench-Live 官方 resolved。
- 不把教学型 MCP 子集称为完整 MCP 兼容实现。
- 不把 Reviewer Agent-as-Tool 称为完整 A2A 或通用多 Agent 框架。
- 不把当前 Qwen 推理实验称为“训练完成”。
- 不把三个领域合成一个总分。

---

## 2. 仓库与环境现状

### 2.1 Git 状态

- 根仓库分支：`main`
- 根仓库基线 commit：`e9dd59cfeb3579256e9be30313aaaed245f06a4e`
- 根仓库：大量 modified/untracked 文件，当前工作尚未形成提交。
- 官方 evaluator 子仓库：`external/SWE-bench-Live-python-only`
- evaluator 分支：`python-only`
- evaluator commit：`ad79b850f15e33992e96f03f6e97f05ddf9aa0be`
- evaluator 有一项本地 Windows 兼容补丁：写 `patch.diff` 与 `eval.sh` 时强制 `newline="\n"`。

接手者禁止直接清理工作树。先运行：

```powershell
git status --short
git -C external/SWE-bench-Live-python-only status --short
```

确认所有文件后，再请求用户是否允许创建一次有意图的 checkpoint commit。

### 2.2 本机运行环境

- 操作系统：Windows / PowerShell。
- 项目 Python 要求：`>=3.11,<3.13`。
- 当前主 Python：Anaconda Python；可运行 pytest 和 mypy，但没有安装 ruff。
- 根项目锁文件：`requirements-dev.lock`。
- Docker：client/server `29.6.1`，daemon 已验证。
- Ollama endpoint：`http://127.0.0.1:11434`。

本地模型：

| 模型 | digest | 量化 | 状态 |
|---|---|---|---|
| `qwen2.5:7b` | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` | Q4_K_M | 正式重复实验模型 |
| `qwen2.5:3b` | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | Q4_K_M | 早期 smoke |
| `repopilot-qwen3:4b` | `01016d3e9a64528db06d687a2078c5856072f57401eef837bd3f8066102b88d9` | Q4_K_M | 非 thinking 派生模型 |
| `qwen3:4b` | `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` | Q4_K_M | 早期路由实验 |

Docker 镜像：

| 用途 | 镜像 / digest |
|---|---|
| DABench | `repopilot-dabench:py311-v1` / `sha256:fc82eedb08c10ba2471b62781d34edf8a5e70bc788bf5defb38fc6546ee302dd` |
| cfn-lint | `sha256:d05f79ab3d900cacdca27cbc7c109241bd3869846e864012002b35e5a521ff13` |
| Babel | `sha256:b9847102c2e7d0f82738cd1a73fe11b242ca2152dc9c28091430ec5a12a7bd70` |
| Mesa | `sha256:dde2624bdba6be282761e0376c8e8726f76dd50b951f5dc1d6444db225fe1c4b` |

### 2.3 推荐重新建立根项目虚拟环境

不要删除现有文件。新建根项目 `.venv`：

```powershell
cd D:\Users\27475\Desktop\Resume_Project\RepoPilot
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy src tests scripts
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

最后一次已知门禁：`66 passed`；mypy 对 61 个源文件通过。ruff 因当前 Anaconda 环境缺少模块而未执行，不代表代码存在 ruff 错误。

### 2.4 每次工作开始前的检查

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
docker version
ollama list
python scripts/dev.py check
pytest -q
```

---

## 3. 代码结构与已实现能力

### 3.1 主要模块

| 目录 | 作用 |
|---|---|
| `src/repopilot/core/` | ReAct 风格循环、状态、预算、共享契约 |
| `src/repopilot/providers/` | 本地 OpenAI-compatible 与 scripted provider |
| `src/repopilot/tools/` | 结构化代码读取、搜索、AST、编辑、diff 工具 |
| `src/repopilot/context/` | 上下文裁剪、工具输出压缩、信任边界 |
| `src/repopilot/retrieval/` | 可检查 BM25 基线 |
| `src/repopilot/memory/` | session 与 episodic memory |
| `src/repopilot/skills/` | SKILL.md 发现、选择与渐进加载 |
| `src/repopilot/mcp/` | 教学型 MCP JSON-RPC 子集 |
| `src/repopilot/orchestration/` | Planner、只读并行、Reviewer 模式 |
| `src/repopilot/runtime/` | checkpoint、幂等 journal、policy、sandbox runner |
| `src/repopilot/verification/` | 确定性验证与失败反馈 |
| `src/repopilot/benchmarks/` | 三领域 contracts、adapter、executor、runner、manifest |
| `src/repopilot/evaluation/` | 成功率、成本、延迟、安全指标和报告 |

### 3.2 Stage 1–6 已完成部分

| Stage | 已完成内容 | 证据 |
|---|---|---|
| 1 | Provider-neutral 消息、结构化工具调用、状态、预算、ReAct loop、trace、checkpoint | `tests/unit/test_core_contracts.py`、`tests/integration/test_agent_runtime.py` |
| 2 | 本地模型 provider、代码工具、路径限制、确定性 verifier | `tests/unit/test_task_and_provider.py`、`tests/unit/test_tools.py` |
| 3 | Context、BM25 RAG、Memory、Skills | `tests/unit/test_context_retrieval_memory_skills.py` |
| 4 | MCP 子集、Planner、只读并行、Reviewer Agent-as-Tool | `tests/unit/test_mcp_orchestration.py` |
| 5 | Retry、checkpoint resume、HITL、policy、Docker fail-closed、安全边界 | `tests/safety/test_security_boundaries.py` |
| 6 | 统一评测、10 个微型修复任务、三领域 benchmark contracts | `tests/integration/test_benchmark_runner.py` 与三个 adapter 测试 |

完整实现说明：`docs/implementation/stage_01_06_summary.md`。

---

## 4. 数据集与协议冻结状态

### 4.1 Google FRAMES

- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 数据 SHA-256：`4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff`
- 完整 `test.tsv` 已下载。
- 当前只对前 10 题物化了 39 个离线 Wikipedia 页面。
- 语料 manifest SHA-256：`3200861c8dd642730b5464ce48a2c9fcb77a412df0fdef7d72e6688eecd97fa3`
- 当前 executor：oracle pages 内 BM25，每页取 2 个 chunk；chunk 2000 chars，overlap 250 chars。
- 当前 10 题是开发/机制实验集，不足以支持完整性能结论。

关键文件：

- `evaluation/benchmarks/data/frames/`
- `evaluation/benchmarks/corpora/frames/`
- `src/repopilot/benchmarks/adapters/frames.py`
- `src/repopilot/benchmarks/executors/research.py`
- `scripts/run_frames_direct_smoke.py`
- `scripts/run_frames_oracle_smoke.py`

### 4.2 InfiAgent-DABench

- 数据 revision：`b455d578e30fee513abd79936cbdf7a6de026cb5`
- questions/labels 已下载。
- 当前只下载了 3 个 CSV，覆盖冻结的 10 题：3 easy、5 medium、2 hard。
- 生成代码只在断网 Docker 中执行。
- B1 与 A1 的唯一主变量是 `max_attempts=3` 与 `max_attempts=1`。

关键文件：

- `evaluation/benchmarks/data/dabench/`
- `evaluation/benchmarks/manifests/dabench_agent_smoke10_v1.json`
- `src/repopilot/benchmarks/adapters/dabench.py`
- `src/repopilot/benchmarks/executors/data_analysis.py`
- `scripts/run_dabench_agent_smoke.py`

### 4.3 SWE-bench-Live

- 数据 revision：`a637bd46829f3132e12938c8a0ca93173a977b8e`
- lite parquet SHA-256：`7ee0a75c41bfc954fd441b67ce738fc5c1cbae00721c4e30e7db4d893057c9ab`
- 官方 evaluator commit：`ad79b850f15e33992e96f03f6e97f05ddf9aa0be`
- 当前三任务：cfn-lint-3767、babel-1141、mesa-2394。
- public task 与 evaluator-only 数据已拆分；Agent 脚本只读取 `smoke_public.jsonl`。
- 三个官方镜像和源码 workspace 均已物化。

关键文件：

- `evaluation/benchmarks/data/swebench_live/`
- `evaluation/benchmarks/workspaces/swebench_live/`
- `evaluation/benchmarks/manifests/swebench_live_agent_smoke3_v1.json`
- `scripts/prepare_swebench_live_smoke.py`
- `scripts/run_swebench_live_agent_smoke.py`
- `scripts/normalize_swebench_predictions.py`
- `external/SWE-bench-Live-python-only/`

数据保密边界：gold patch、test patch、FAIL_TO_PASS、PASS_TO_PASS 不能进入 Agent prompt、workspace 或工具输出。

---

## 5. 已完成实验与结果

### 5.1 FRAMES 三次重复

| 配置 | 三次准确率 | 平均准确率 | 平均 F1 | 平均耗时 |
|---|---|---:|---:|---:|
| B0 直接模型 | 1/10, 1/10, 1/10 | 10.0% | 0.1333 | 10.45s |
| B1 完整 Agent | 2/10, 2/10, 1/10 | 16.7% | 0.1920 | 169.55s |
| A4 无 Planner | 1/10, 1/10, 0/10 | 6.7% | 0.1271 | 120.74s |
| A5 无 Reviewer | 2/10, 1/10, 1/10 | 13.3% | 0.1595 | 109.26s |

当前决策：

- Planner 暂时保留。
- Reviewer 改为可选升级路径；无 Reviewer 平均节省约 48.0% 输入 tokens 和 35.6% 耗时。
- B1 相比直接模型的绝对提升有限，且成本约为 16.2 倍耗时、59 倍输入 tokens。
- 下一轮优先改检索和证据合成，不继续堆叠控制模块。

### 5.2 DABench 三次重复

| 配置 | 三次准确率 | 平均耗时 | 平均恢复次数 |
|---|---|---:|---:|
| B1 有反馈 | 9/10, 9/10, 9/10 | 99.66s | 2.33 |
| A1 无反馈 | 8/10, 8/10, 8/10 | 73.12s | 0 |

当前决策：

- 环境反馈保留；三次均稳定增加 1 个正确任务。
- 代价是平均耗时增加约 36%。
- `dabench-dev-0007` 被反馈机制恢复。
- `dabench-dev-0006` 两组均失败，应专项审计答案格式与分组边界。

### 5.3 SWE-bench-Live 官方结果

| 任务 | 生成结果 | 官方结果 | 失败原因 |
|---|---|---|---|
| cfn-lint-3767 | 迭代耗尽、空补丁 | empty patch | 未形成有效编辑 |
| babel-1141 | 非空补丁、可见测试通过 | unresolved | 缺失秒位被设为 `None` 后与整数排序，目标测试 `TypeError` |
| mesa-2394 | 非空补丁、可见验证失败 | unresolved | 在注册 Agent 时错误清空集合，目标测试失败并破坏 17 个回归测试 |

官方结果：`resolved=0/3`。

重要研究边界：这三个任务的官方详细失败报告已经被人工查看，因此从现在起应标记为 **development tasks**。可以用于通用失败机制调试，但不能再作为无偏最终测试。后续确认必须冻结新的、未查看 evaluator 结果的 holdout 任务。

另外，cfn-lint 的 gold patch 官方 smoke 出现 4 个 PASS_TO_PASS 失败，说明该任务存在数据/镜像负例风险，不应单独用于证明模型能力。

### 5.4 聚合证据位置

- 最新矩阵：`evaluation/benchmarks/experiment_matrix.yaml`
- 三次重复聚合：`artifacts/benchmarks/20260807_p5_p6_formal_repeats/results.json`
- 聚合脚本：`scripts/summarize_p5_p6_repeats.py`
- FRAMES/DABench 报告：`docs/experiments/2026-08-07_p5_p6_formal_repeat_results.md`
- SWE 报告：`docs/experiments/2026-08-07_p5_swebench_live_agent_smoke3.md`
- SWE 官方原始证据副本：`artifacts/benchmarks/20260807_swebench_live_agent_official_smoke3/`

重新生成聚合表：

```powershell
python scripts/summarize_p5_p6_repeats.py
```

---

## 6. 已解决的工程问题与不能回退的修复

1. Docker daemon 已恢复并验证断网运行，不要再沿用旧文档中“daemon 不可用”的结论。
2. Windows 会把官方 evaluator 的 `patch.diff` 与 `eval.sh` 写成 CRLF；子仓库本地补丁强制 LF。不要删除该补丁，除非迁移到 Linux 并重新验证。
3. `qwen2.5:7b` 中的冒号会成为官方 evaluator 日志目录名，在 Windows 非法。新脚本把 submission name 规范为 `qwen2.5_7b`。
4. SWE workspace 的 baseline 曾扫描 Babel 大型 CLDR 数据导致严重性能问题。`capture_text_snapshot(..., include_paths=task.allowed_paths)` 已限制扫描范围。
5. cfn-lint 镜像复制到 Windows 后会因行尾和超长路径显示大量无关工作树变化。Agent 脚本的 clean check、diff、changed files 已严格限定到 `allowed_paths`。
6. 所有失败 run 目录均故意保留，不能覆盖或批量删除。

---

## 7. 卡码技术覆盖审计

需求来源：

- [卡码大模型面经汇总](https://notes.kamacoder.com/interview/llm/)
- [卡码大模型学习路线](https://notes.kamacoder.com/llm/)

网页当前覆盖 Prompt、结构化输出、Context、Function Calling、RAG、Embedding、向量数据库、切片、Query 改写、Context 压缩、评估、Agent、Reflection、规划、MCP、Memory、微调、部署、多模态、Transformer、AI 编程与 Harness 等方向。

### 7.1 覆盖状态表

| 技术 | 当前状态 | 已有证据 | 后续动作 |
|---|---|---|---|
| 结构化 Prompt / JSON 输出 | 已实现并测试 | provider、action fallback、schema tests | 增加 prompt 版本注册与 A/B 记录 |
| Few-shot / CoT | 部分运行 | FRAMES reasoned answer | 冻结 few-shot 示例并做单变量实验 |
| Reflection / 自检 | Reviewer 形式已运行 | FRAMES A5 | 改为失败触发的可选升级，不默认全量调用 |
| Context Engineering | 已实现 | trimming、压缩、checkpoint | 增加 context attribution 与污染检测 |
| Function Calling / Tool Use | 已实现并测试 | structured tools、policy、journal | 增加 tool selection、invalid-call 指标 |
| ReAct / Loop Engineering | 已实现 | bounded loop、预算、终止、验证 | 加强失败反馈摘要和停止条件 |
| BM25 RAG | 已实现并真实运行 | FRAMES | 保留为 lexical baseline |
| Chunking | 已实现固定窗口 | 2000/250 配置 | 增加段落/语义/递归切片对照 |
| Embedding / 向量检索 | 未实现 | 无 | 实现本地 embedding 与可冻结索引 |
| 向量数据库 | 未实现 | 无 | 选 FAISS/SQLite 向量层；记录版本与索引 hash |
| 混合检索 | 未实现 | 无 | BM25+dense，先用 RRF |
| Rerank | 未实现 | 无 | 增加本地 cross-encoder reranker |
| Query 改写 | 原型 | FRAMES Planner 每来源 query | 独立为 retriever 模块并做消融 |
| Context 压缩 | 只有工具输出压缩 | ContextBuilder | 增加证据压缩且验证事实保真 |
| RAG 评估 | 部分实现 | F1、evidence 指标 | 加 Recall@k、MRR、nDCG、faithfulness |
| GraphRAG / LightRAG | 未实现 | 无 | 作为扩展实验，不阻塞主线 |
| Memory | 代码已实现，正式三域未验证 | unit tests | 先接入实际 benchmark，再做 A3 消融 |
| Skills | 教学实现 | registry tests | 增加版本/hash、真实任务激活评估 |
| MCP | 教学子集 | MCP tests | 接真实只读工具 server，再决定是否扩协议 |
| Planner | 已真实验证 | FRAMES A4 | 保留，优化 query 计划格式 |
| Reviewer / Agent-as-Tool | 已真实验证 | FRAMES A5 | 改为置信度/失败触发 |
| 只读并行工具 | 已实现，未做领域效果实验 | unit tests | 做延迟/正确率消融 |
| HITL / 审批 | 已实现 | safety tests | 在高风险写工具 demo 中验证 |
| Checkpoint / Resume | 已实现 | recovery tests | 增加崩溃注入实验 |
| Trace / Observability | 已实现 JSONL | run artifacts | 增加统一 trace 汇总和失败面板 |
| 权限 / Sandbox | 已实现并真实运行 | Docker、security tests | 增加恶意 prompt/文件/网络红队集 |
| 规则/模型/混合路由 | 只有早期原型 | Qwen3 smoke | 先定义 routing baseline，再做 A6 |
| 多 Agent 通信/图编排 | 仅模式级实现 | Planner/Reviewer/parallel | 实现显式消息 envelope、状态图和成本治理 |
| SFT | 未实现 | 无 | 在公开 trace 语料上建立训练/验证协议 |
| LoRA / QLoRA | 未实现 | 无 | 在 AutoDL 训练，记录基模、adapter、数据 hash |
| DPO | 未实现 | 无 | 从公开成功/失败轨迹构造 preference pairs |
| RLHF / RLAIF | 未实现 | 无 | 先做选型与小规模验证，不作为首个训练方案 |
| 蒸馏 | 未实现 | 无 | 完成教师许可、数据来源和污染审计后再做 |
| 自部署推理 | 部分实现 | Ollama Q4_K_M | 增加 vLLM/llama.cpp 或其他后端对照 |
| 流式/异步/并发 | 异步内部调用，未压测 | async runtime | 增加 API、并发和吞吐压测 |
| 成本/延迟 | 已记录 | shared metrics | 增加 P50/P95、吞吐、峰值内存 |
| 多模态 | 未实现 | 无 | 独立扩展轨，不强行混入三个文本主基准 |
| Transformer 原理/手写 | 只有知识要求 | 无项目证据 | 建独立 educational lab，不作为 Agent 主指标 |
| Spec-driven AI 编程 | 部分实现 | task spec、plan、verifier | 完善规格版本、变更审查和回滚演示 |

原则：所谓“运用所有技术”应表现为有边界的模块、对照实验或教学实验，不应把不相关技术强行放进同一推理链。

---

## 8. 尚未完成部分

### 8.1 立即未完成

- 根仓库尚未形成可交付 commit，旧文档状态未全部更新。
- ruff 尚未在锁定虚拟环境运行。
- FRAMES 绝对准确率低，尚无 dense/hybrid/rerank 检索。
- DABench 只覆盖 3 个表和 10 题，未扩到更有代表性的验证集。
- SWE-bench-Live 0/3，且现有三题已被开发过程污染。
- P6 的 A2 无检索、A3 无记忆、A6 无路由尚未完成。

### 8.2 实验设计上的无效或退化项

- A3 当前不能直接跑：FRAMES/DABench 正式 executor 尚未真正使用 episodic memory；在未接入前做“无 memory”消融没有意义。
- A6 当前不能直接跑：B1 本身使用固定 `qwen2.5:7b`，不存在可移除的 adaptive routing。必须先建立合法 routing baseline。
- A2 必须先定义“无检索”含义：不能同时改变文档可见性和检索算法。建议比较“同一 oracle 文档、固定长度首段”与“BM25 选段”，并单独报告 full-context 上限。

### 8.3 模型训练与生产能力

- 没有训练数据治理、去重、污染审计或 train/valid/test split。
- 没有 PEFT/TRL/Transformers 训练环境。
- 没有模型 checkpoint、adapter、训练日志或训练评测。
- 没有 API 服务、并发压测、P95、吞吐、峰值内存或故障注入报告。
- 没有真正通用的 multi-agent graph/harness。

---

## 9. 后续详细执行计划

### P7：接手验证与可复现状态冻结（最高优先级，0.5–1 天）

目标：把当前可运行但未提交的状态变成可安全继续的基线。

任务：

1. 建立根 `.venv`，执行 pytest、mypy、ruff。
2. 校验所有 JSON/YAML manifest，重新运行 `scripts/summarize_p5_p6_repeats.py`。
3. 检查根仓库和 evaluator 子仓库 diff；不要清理用户改动。
4. 更新 `README.md`、`docs/project_context_summary.md` 中 Docker、正式实验和训练状态的过期描述。
5. 将 evaluator LF 补丁写入明确的 patch 文件或 ADR，避免子仓库改动丢失。
6. 经用户确认后，创建一个有意图的 checkpoint commit；不要把超大镜像、workspace 或无必要日志提交进 Git。

通过门槛：

- `pytest=66 passed` 或更多。
- mypy、ruff 全通过。
- manifest 可加载。
- 当前 results 聚合可复现。
- Git 中每类文件的保留/忽略策略明确。

产物：`docs/experiments/<date>_p7_takeover_freeze.md`、更新后的 README/上下文摘要、可审查 commit。

### P8A：FRAMES 检索与证据合成改进（2–4 天）

假设：当前主要瓶颈是检索片段未覆盖多跳事实，以及生成阶段未严格绑定证据。

先做诊断：

1. 对 10 题逐题标注失败类型：召回缺失、关系链错误、时间/数值错误、答案抽取错误、Reviewer 改坏。
2. 在不调用模型的情况下计算 oracle page 内 Recall@k/MRR/nDCG。
3. 记录正确答案所需字符串是否存在于检索 context，区分 retrieval failure 与 reasoning failure。

按单变量顺序实现：

1. B1-L：当前 BM25，作为冻结 baseline。
2. R1：只替换为 dense retrieval。
3. R2：BM25+dense，使用 RRF 混合。
4. R3：在 R2 上只增加 reranker。
5. R4：在最佳检索器上只增加 query rewrite。
6. R5：在最佳检索器上只增加 context compression。
7. G1：固定检索 context，只改证据链 JSON 输出与 citation verifier。

建议接口：新增通用 `Retriever`、`Reranker`、`ContextCompressor` protocol，避免继续把逻辑写死在 `OracleDocumentModelExecutor`。

门槛：

- 10 题开发集检索 Recall@k 明显提高。
- 答案准确率至少稳定超过当前 B1 的 16.7%，且三次重复。
- tokens/耗时增幅有记录。
- 通过门槛后再冻结 50 个未调参 FRAMES validation tasks。

### P8B：SWE 失败驱动可靠性改进（2–4 天）

假设：当前失败来自编辑前定位不足、缺少针对性测试、验证反馈过长或不结构化，以及错误 patch 未被自审阻止。

通用改进，不针对隐藏答案硬编码：

1. 将 verifier 输出压缩成结构化字段：失败测试、异常类型、最短 traceback、changed files、回归计数。
2. 在编辑前要求模型明确：根因假设、目标文件、预期测试、回归风险。
3. 增加“先写/选择针对性可见测试，再修改代码”的两阶段策略。
4. 每次 patch 后依次运行：syntax/import → targeted tests → affected tests → full configured tests。
5. 增加 patch reviewer：检查无关修改、集合清空、类型混用、边界条件和测试缺失。
6. 对空补丁/重复无效工具调用增加早停与重新定位。
7. 保持 max files、路径白名单、网络 deny 和官方 evaluator 边界。

实验协议：

- 已查看报告的 3 题只作为 dev smoke。
- 改进后先在 dev smoke 检查机制是否工作。
- 再从 lite 中冻结至少 3 个全新 Python holdout，不查看 gold/test patch。
- holdout 官方结果生成前不得读取 evaluator-only 字段。

门槛：dev smoke 至少不再出现空补丁和明显回归破坏；新 holdout 至少有 1/3 resolved 后才扩到 10 题。

### P8C：DABench 扩展与失败审计（1–2 天）

1. 审计 `dabench-dev-0006` 的输出格式/分组失败，不能从标签反向硬编码。
2. 按 revision 下载更多官方 CSV，记录每个文件 SHA-256。
3. 冻结 30–50 题 stratified validation set，覆盖 easy/medium/hard 和不同表。
4. 重跑 B1 feedback 与 A1 no-feedback，各 3 次。
5. 增加代码执行成功率、格式错误率、超时率和每难度结果。

门槛：扩展集上反馈收益仍为正，且 Docker/格式失败率可解释。

### P9：补齐机制消融（2–4 天）

严格每次一个主变量：

1. A2 无检索：先冻结有效定义，再运行 FRAMES。
2. A3 无记忆：先把 memory 接入真实 benchmark；确认检索到的 episode 会进入 context 后再消融。
3. A6 无路由：先实现 rule/model/hybrid routing baseline，再与 fixed model 对照。
4. P9-P：只读并行 on/off，观察耗时和错误率。
5. P9-S：Skills on/off，必须记录 skill 版本与激活准确率。
6. P9-M：MCP 本地工具与直接 tool adapter 对照，比较协议开销和可靠性。
7. Reviewer 使用“全量”与“失败/低置信度触发”对照。

门槛：相同任务、模型 revision、预算、evaluator；每组 3 次；报告均值、波动和 paired task 变化。

### P10：微调与蒸馏实验（1–2 周，建议 AutoDL）

前置条件：P8/P9 找到稳定的系统瓶颈后再训练，避免用训练掩盖错误协议。

数据治理：

1. 只使用公开任务描述、公开源码、Agent 轨迹和人工可发布标注。
2. evaluator-only 测试、gold patch、隐藏答案不得进入训练数据。
3. 记录数据来源、许可、hash、去重方法、污染审计和 split。
4. 训练集、验证集、最终 benchmark holdout 必须隔离。

推荐顺序：

1. SFT：训练规划、工具选择、错误修复摘要等结构化能力。
2. LoRA/QLoRA：优先低成本 adapter，冻结基模 revision。
3. DPO：使用同任务的成功/失败或低成本/高成本轨迹构造 preference pairs。
4. 蒸馏：只有教师模型许可、生成数据许可和质量审核通过后实施。
5. RLHF/RLAIF：作为后续研究，不作为第一个训练方案。

每项比较：base、Prompt/RAG-only、SFT、LoRA/QLoRA、DPO；同时报告三领域主指标、tokens、延迟和安全。

训练产物必须包含：环境、GPU、CUDA、Transformers/PEFT/TRL 版本、命令、数据 hash、checkpoint hash、adapter、日志和失败记录。

### P11：Multi-Agent Harness 与生产工程（1–2 周）

1. 实现显式状态图：planner → retriever/executor → verifier → reviewer → finish/replan。
2. 定义 Agent 间消息 envelope、任务 ID、权限、预算和 trace correlation ID。
3. 并行只允许无副作用工具；写操作保持串行和审批。
4. 实现规则路由、模型路由、混合路由和级联降级。
5. 增加漂移检测：目标摘要、计划偏差、证据缺失、重复工具调用。
6. 建立 trace 聚合：成功、失败类型、tokens、P50/P95、工具错误、恢复路径。
7. 增加服务层：结构化 API、流式输出、并发限制、取消、超时。
8. 做 1/4/8 并发压测、故障注入、checkpoint resume、安全红队。

门槛：正确率不低于单 Agent 最佳配置；成本与延迟收益明确；失败可定位、可恢复。

### P12：最终正式评测与交付（1–2 周）

分三级控制资源：

| 层级 | FRAMES | DABench | SWE-bench-Live | 用途 |
|---|---:|---:|---:|---|
| Dev | 10 | 10 | 已污染 3 题 | 快速调试 |
| Validation | 50 | 30–50 | 新 holdout 10 | 配置选择 |
| Final | 尽量完整 824 | 尽量完整 257 | Lite 300 或公开声明的冻结子集 | 最终报告 |

最终协议：

- 固定数据 revision、任务 ID、模型/adapter hash、prompt/tool/skill 版本、镜像 digest。
- 随机过程至少 3 次；确定性过程也保留重复确认。
- 任务级 paired comparison，报告 bootstrap CI；适合时使用 McNemar。
- 三领域分开报告，不给综合总分。
- 每项 claim 链接 manifest、命令、trace、结果表和失败审计。

最终交付物：README、架构说明、复现实验脚本、模型卡、Agent 卡、安全威胁模型、数据/许可清单、结果表、演示和论文/简历版项目描述。

---

## 10. 接手者第一天的具体动作

按顺序执行，不要跳步：

1. 阅读本文件、`evaluation/benchmarks/experiment_matrix.yaml` 和两个最新实验报告。
2. 运行环境预检与 66 个测试。
3. 建立根 `.venv`，补跑 ruff。
4. 检查所有 dirty/untracked 文件，禁止清理。
5. 创建 `docs/experiments/<date>_p7_takeover_freeze.md`，记录环境和检查结果。
6. 修正 README 和 project context 中的过期状态。
7. 为 FRAMES 写逐题失败审计脚本；先做只读分析，不改算法。
8. 输出 retrieval failure / reasoning failure / extraction failure 分类表。
9. 根据分类结果冻结 P8A 的第一个单变量实验。

第一天不要做：

- 不扩大 SWE 样本。
- 不开始 LoRA。
- 不根据隐藏测试名称硬编码补丁。
- 不同时替换检索器、prompt、模型和预算。
- 不删除失败 artifacts。

---

## 11. 常用复现实验命令

新运行必须使用新的 `run-id`，现有目录不可覆盖。

### FRAMES 直接模型

```powershell
python scripts/run_frames_direct_smoke.py --run-id NEW_RUN_ID --count 10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-output-tokens 512
```

### FRAMES 完整 Agent

```powershell
python scripts/run_frames_oracle_smoke.py --run-id NEW_RUN_ID --count 10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --planned --reasoned --reviewed --reasoning-output-tokens 512
```

### DABench 有反馈

```powershell
python scripts/run_dabench_agent_smoke.py --run-id NEW_RUN_ID --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-attempts 3
```

### SWE Agent patch 生成

```powershell
python scripts/run_swebench_live_agent_smoke.py --run-id NEW_RUN_ID --task-id NEW_TASK_ID --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-iterations 12 --max-tool-calls 32 --max-total-tokens 32000 --max-wall-seconds 600 --max-output-tokens 1024
```

### SWE 官方 evaluator

从 `external/SWE-bench-Live-python-only` 运行其独立 `.venv`：

```powershell
cd D:\Users\27475\Desktop\Resume_Project\RepoPilot\external\SWE-bench-Live-python-only
.\.venv\Scripts\python.exe -m swebench.harness.run_evaluation --dataset_name SWE-bench-Live/SWE-bench-Live --split lite --instance_ids INSTANCE_ID --namespace starryzhang --predictions_path ABSOLUTE_PREDICTIONS_JSONL --max_workers 1 --run_id NEW_EVAL_RUN_ID
```

---

## 12. 安全、数据和文件操作规则

1. 禁止递归或批量删除文件/目录。
2. 禁止 `rm -rf`、`Remove-Item -Recurse`、`git reset --hard`。
3. 不覆盖已有 run directory；失败结果也是研究证据。
4. 不执行模型生成的任意宿主机 shell；不可信代码只进断网 Docker。
5. gold/hidden/evaluator-only 数据必须与 Agent workspace 分离。
6. 不把 API key、token、个人信息写入 trace。
7. 任何新依赖都要锁版本；任何新模型/镜像都要记录 digest。
8. Windows 超长路径和 CRLF 可能制造假 diff；SWE 补丁只从允许路径导出。

---

## 13. 状态更新规范

后续工作人员每完成一个阶段，应在本文件顶部状态日期后追加一条变更记录，并同步更新：

1. `evaluation/benchmarks/experiment_matrix.yaml`
2. 对应 `docs/experiments/<date>_<stage>_<run>.md`
3. manifest / config revision
4. artifact 路径
5. 通过与失败门槛
6. 下一步决策

不要覆盖旧结论；新证据与旧结论冲突时，追加“更新结论”和证据链接。

---

## 14. 当前一句话交接结论

RepoPilot 已经从“软件修复 Agent”扩展为三领域通用 Agent 评测骨架，DABench 的反馈机制已有稳定正收益，FRAMES 的 Planner 有初步收益但检索/证据合成仍弱，SWE 官方能力尚未过门槛；下一位工作人员应先冻结可复现状态，再优先完成 FRAMES 检索诊断和 SWE 通用失败反馈改进，之后才进入剩余消融、微调、多 Agent Harness 和最终规模评测。
