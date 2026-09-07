# 高可信 Agent 项目案例与筛选标准

> 建立日期：2026-08-23  
> 用途：为 RepoPilot 的工程研究、技术对标与简历项目表达收集**可公开核验的项目级证据**。  
> 边界：这不是个人候选人库，也不是对个人能力的排名。开源仓库不能自动证明某位贡献者的具体职责；
> 任何简历只能陈述自己可以用提交记录、设计文档、测试、发布物或实验记录证明的贡献。

## 1. 为什么 A-01 不能做高水平基准

原始 A-01 是 `fszale/resume` 的个人简历仓库。它有公开的关联实现，因而不能简单说成“没有技术内容”；
但它的简历仓库仅有 1 Star / 0 Fork，关联 `agent-kernel` 的公开采用也很有限。更关键的是，它缺少本
语料库要求的多源工程证据：持续发布、独立使用、规模化测试或可复现评测。

因此，A-01 的正确定位是：

- 可以研究它如何把 Agent、HITL、评估与治理写进项目介绍；
- 不可作为 RepoPilot 的质量、影响力、安全性或架构成熟度标杆；
- 不可将其自述的角色、指标或组织化结果转写为自己的项目事实。

Star 不是质量的充分条件：新项目、细分项目或企业内部项目都可能 Star 很低。但在本次目标是寻找
“公开、高水平、可对标”的样本时，低 Star 且缺乏其他独立证据，足以使它不通过高门槛准入。

## 2. 新的多信号准入标准

### 2.1 必须条件

候选项目必须同时满足：

1. **直接工程证据**：链接直接指向可阅读、可运行的 Agent 代码库，而非只指向简历、文章或模板；
2. **工程可检查性**：能看到源码、文档、测试/评测、变更或发布信号中的至少两类；
3. **任务边界明确**：能说清 Agent 的任务、工具、状态、审批/安全或评测中的至少两项；
4. **来源可追溯**：优先官方组织或项目维护者仓库；对厂商宣传不把其单独当作证据。

### 2.2 分层门槛

| 层级 | 额外要求 | 允许用途 |
| --- | --- | --- |
| H1：核心对标 | ≥10k Star；活跃 issue/PR 或大量 Fork；公开测试/评测与文档；有安全、状态/恢复、可观测性或发布治理证据 | RepoPilot 的架构与质量对标 |
| H2：强参考 | ≥1k Star，或拥有等价的机构/论文/正式发布证据；代码、测试和文档齐全 | 单项机制与工程做法对标 |
| H3：候选灵感 | 不满足 H1/H2，但实现、测试、复现实验足够完整 | 仅作为灵感，不能引用其影响力或指标 |
| 排除 | 仅简历页、教程、提示词集合、无可运行代码、无工程证据，或指标无法追溯 | 不进入高可信案例库 |

**使用规则**：Star/Fork 是公开采用的代理信号，不是技术质量评分；它只在缺少外部证据时充当筛选门槛。
每次引用数字时都要标注“复核日期”，因为数字会变化。

## 3. H1 核心项目案例（复核于 2026-08-23）

| 编号 | 项目 | 公开采用信号 | 可核验工程证据 | 对 RepoPilot / 简历的可迁移点 |
| --- | --- | ---: | --- | --- |
| H1-01 | OpenHands | 84.8k Star、11.1k Fork | 源码、`__tests__`、`tests/e2e`、Docker、规范、变更日志与 Windows 指引 | Coding Agent 的工作区边界、Docker 沙箱、前后端分离、长任务运行 |
| H1-02 | Cline | 66.7k Star、7.2k Fork | `evals`、SDK、CLI、测试/发布目录、规范与安全策略 | 交互 CLI、headless JSON、工作树、多 Agent 团队、持久任务 |
| H1-03 | AutoGen（历史参照） | 60.6k Star、9.1k Fork | 多语言源码、测试/基准、MCP 示例、文档与安全文件 | 事件驱动多 Agent、可流式模型客户端、工具/Agent 组合；**新项目不应基于它扩展** |
| H1-04 | CrewAI | 57.5k Star、8.2k Fork | 源码、测试配置、文档、贡献/安全文件与示例 | 角色型协作（Crew）与可控事件流（Flow）的分层、状态和人工审阅 |
| H1-05 | Aider | 48.4k Star、4.9k Fork | `benchmark`、`tests`、Git 集成、变更历史、Docker | CLI Coding Agent 的 repo map、差异审阅、提交、lint/test 闭环 |
| H1-06 | LangGraph | 40.3k Star、6.8k Fork | 多包源码、示例、文档、贡献/安全文件、7k+ commits | 持久状态、恢复、人工介入、执行图和调试可观测性 |
| H1-07 | OpenAI Agents SDK（Python） | 28.9k Star、4.6k Fork | `integration_tests`、`tests`、安全文件、文档/示例、类型检查与 coverage 工具 | handoff、guardrail、工具调用、Sandbox Agent 与跨 provider 适配 |
| H1-08 | Microsoft Agent Framework | 13.1k Star、2.2k Fork | Python/.NET/Go、设计文档、ADR、样例、贡献/安全文件 | 可恢复编排、治理、HITL、A2A/MCP、多语言运行时 |

### H1-01：OpenHands — Coding Agent 的环境隔离

- **任务范围**：面向软件开发的 Agent 运行栈与 Agent Canvas。
- **可验证点**：仓库公开列出单元测试、端到端测试、Docker、架构与自托管文档；README 明确区分“直接运行时
  Agent 可获得主机文件系统权限”与 Docker 沙箱运行，说明它把运行边界当作一等设计问题。
- **可学而不照抄的简历表达**：如果自己实现了等价功能，可以写“设计 workspace 级权限与容器化执行边界，
  将文件操作和命令执行纳入可追溯会话”；不能写成“实现了 OpenHands 级别系统”。
- **来源**：[仓库](https://github.com/OpenHands/OpenHands)

### H1-02：Cline — 多形态 Coding Agent 与可自动化接口

- **任务范围**：IDE、CLI、SDK 和看板的同一 Agent 引擎。
- **可验证点**：仓库同时可见 CLI、SDK、`evals`、安全策略、发布相关文件；README 明确描述交互式与
  headless CLI、JSON 输出、工作树隔离、会话持久与计划任务。
- **可学而不照抄的简历表达**：把交互体验写成“实现交互模式与可脚本化模式的同一事件协议”，并附上真实的
  回归测试、会话恢复或 JSONL 样例。
- **来源**：[仓库](https://github.com/cline/cline)

### H1-03：AutoGen — 有价值但已退役的对照组

- **任务范围**：多 Agent 消息传递、AgentChat 与扩展层。
- **可验证点**：仓库有 3k+ commits、多语言目录、MCP 示例、评测工具和安全文件。
- **关键限制**：维护方已明确标为 maintenance mode，并建议新项目迁移至 Microsoft Agent Framework；
  所以它适合研究经典编排模式，不适合当作 RepoPilot 的新增依赖选择。
- **来源**：[仓库](https://github.com/microsoft/autogen)

### H1-04：CrewAI — 自主协作与确定性工作流分层

- **任务范围**：用角色、任务、工具构成协作 Agent（Crews），并用事件驱动 Flows 施加确定性控制。
- **可验证点**：有源码、文档、测试配置、贡献与安全文件，且项目给出 Crews/Flows 的显式职责边界。
- **可学而不照抄的简历表达**：不要只写“多 Agent 协作”；应写出“哪些步骤由确定性状态机控制，哪些步骤
  交给模型/角色自主决策，以及失败时如何分支或人工接管”。
- **来源**：[仓库](https://github.com/crewAIInc/crewAI)

### H1-05：Aider — 小而深的终端编码闭环

- **任务范围**：在现有代码库中结对编程的终端 Agent。
- **可验证点**：仓库公开 `benchmark`、`tests`、Git 集成、代码库地图和自动 lint/test 能力。
- **可学而不照抄的简历表达**：以可复现闭环为中心，例如“构建代码索引 → 生成补丁 → 运行 lint/test → 记录
  diff/恢复点”，而非以调用模型名称为中心。
- **来源**：[仓库](https://github.com/Aider-AI/aider)

### H1-06：LangGraph — 可恢复、可检查的状态机

- **任务范围**：长运行、有状态 Agent 与工作流的低层编排。
- **可验证点**：仓库公开显示 durable execution、HITL、短/长期记忆和运行路径调试能力，并含多包源码、
  文档、示例、贡献与安全文件。
- **可学而不照抄的简历表达**：将“多轮对话”落实为 checkpoint、状态 schema、失败恢复、人工修改状态和
  execution trace，而不是笼统宣称“支持 memory”。
- **来源**：[仓库](https://github.com/langchain-ai/langgraph)

### H1-07：OpenAI Agents SDK — Agent 原语与沙箱接口

- **任务范围**：以 agent、tool、guardrail、handoff 和 sandbox 为核心的轻量多 Agent SDK。
- **可验证点**：有 `tests`、`integration_tests`、示例、文档、类型检查与 coverage 工具；README 还明确
  Windows 环境应使用 Docker 或托管沙箱，而非默认 Unix 本地沙箱。
- **可学而不照抄的简历表达**：用“模型无关的工具接口、审批、handoff 与受限执行”描述工程边界，并说明
  Windows/Docker 的实际兼容性测试。
- **来源**：[仓库](https://github.com/openai/openai-agents-python)

### H1-08：Microsoft Agent Framework — 面向治理的可演进编排

- **任务范围**：Python、.NET 和 Go 中可部署的单/多 Agent 与工作流。
- **可验证点**：仓库公开 Python/.NET/Go 实现、设计文档、ADR、样例、贡献与安全文件；README 将耐久性、
  可恢复、可观测、治理、HITL 和 provider 灵活性列为适用边界。
- **可学而不照抄的简历表达**：把“支持多模型”写成可证明的 provider interface、契约测试、失败降级与
  运行记录，而不是仅罗列 API 名称。
- **来源**：[仓库](https://github.com/microsoft/agent-framework)

## 4. 对 RepoPilot 的直接使用方式

1. **Coding Agent 核心对标**：优先比较 Aider（终端闭环）、Cline（交互/自动化协议）和 OpenHands（隔离运行）。
2. **状态与恢复对标**：优先研究 LangGraph 与 Microsoft Agent Framework 的 checkpoint、恢复和人工介入。
3. **多 Agent 不作为默认复杂度**：参考 CrewAI/AutoGen 的分工思想，但只有在单 Agent 的工具链和评测已
   证明瓶颈时才引入 coordinator/subagent。
4. **简历写作证据链**：每一句项目表述必须可映射到“代码路径 + 测试/实验 + 会话/运行记录 + 限制”。
   对开源项目学的是证据组织方式，不是借用其 Star、用户量、指标或作者经历。

## 5. 复核清单

新条目进入本文件前，逐项填写：

- [ ] 直接代码仓库，不是仅简历/文章；
- [ ] 复核日期与 Star/Fork（如采用）；
- [ ] 至少两类工程证据：测试、评测、CI、发布、设计文档、安全策略、示例；
- [ ] 明确 Agent 任务、工具、状态、审批/安全、评测中的至少两项；
- [ ] 写出不应借用的结论或指标；
- [ ] 只有在自己能证明贡献边界时，才转换为个人简历 bullet。
