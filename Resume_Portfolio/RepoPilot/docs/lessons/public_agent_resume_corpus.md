# 公开 Agent 项目简历语料库（去标识化索引）

> **状态（2026-08-23 更新）**：这是宽松准入的“公开表达发现池”，不再作为高水平项目的对标基准。
> 其中的 A 类页面主要用于研究简历表述，不能以此推定代码质量、实际个人贡献或项目影响力。
> 需要研究可复核的高水平 Agent 工程时，请优先使用
> [高可信 Agent 项目案例与筛选标准](high_confidence_agent_project_cases.md)。

> 建库日期：2026-08-23  
> 目的：收集公开、可访问的 Agent 项目简历/作品集表达，作为 RepoPilot 后续整理、项目表述和面试准备的
> 研究材料。  
> 收集原则：不复制完整简历；不保存电话、邮箱、住址、薪资、出生日期或其他联系信息；不抓取登录后的
> 求职平台页面；仅保存技术摘要、来源链接与可信度标签。

## 1. 语料边界与使用说明

本语料库不是人才库，也不用于筛选或评估真实个人。每条记录仅描述公开页面中自行披露的**项目或技术
表达方式**，以便研究“Agent 项目怎样写得具体、可验证、可追问”。

来源分为三层：

| 层级 | 含义 | 可如何使用 |
| --- | --- | --- |
| A：公开简历/作品集 | 页面明确是公开简历、CV 或作者的项目型 Portfolio，且出现 Agent 项目或 Agent 工程经验 | 用于观察真实表达结构；不照抄、不假设其指标可信 |
| B：简历项目表述参考 | 公开项目页提供简历 bullet、项目经验模板或专门的项目表述素材 | 用于归纳写法与关键词，不作为候选人经历 |
| C：工具/标准参考 | 公开的简历 Agent、结构化简历标准或真值约束工具 | 用于研究如何组织材料、校验事实与输出格式 |

**重要限制**：所有项目能力、角色、规模与量化结果均为页面作者的自述；本仓库只验证“该公开页面存在且
包含该类描述”，不验证个人身份、个人贡献或指标真实性。用于自己的简历时，只有自己能够拿出代码、测试、
设计文档或实际记录证明的内容才能写成事实。

求职网站的公开页面常带有登录、访问频控、平台条款和更敏感的个人资料，因此本轮不批量采集、存储或
镜像这些页面。GitHub 公开简历/作品集与开源项目页已能覆盖本阶段需要的技术表达样本。

---

## 2. 样本总览

当前索引共 **22** 条：11 条 A 类公开简历/作品集、7 条 B 类项目表述参考、4 条 C 类工具/标准参考。
其中 A 类用于观察现实表述，B/C 类用于补足“如何把机制写成可追问项目成果”的方法。

| 编号 | 层级 | Agent 项目方向 | 主要标签 | 来源 |
| --- | --- | --- | --- | --- |
| A-01 | A（已降级） | Agentic 组织转型与 Agent Factory | digital twin、HITL、guardrail、evaluation | [来源](https://github.com/fszale/resume) |
| A-02 | A | 科研文献分析 Harness | multi-agent、provenance、reflection、provider abstraction | [来源](https://github.com/eliaswestonfarber/resume) |
| A-03 | A | 企业内部 briefing / RAG 工具 | PDF、web research、SSE、MCP、skills | [来源](https://github.com/dr5hn/resume) |
| A-04 | A | 安全受限环境的 Agent 研究 | sandbox、hallucination、安全、on-prem inference | [来源](https://github.com/CodyKochmann/resume) |
| A-05 | A | 云端生产 Agent | Google ADK、MCP、RAG、OpenTelemetry、Terraform | [来源](https://gist.github.com/doughayden/ff355296263ebe1e97cf012064588be7) |
| A-06 | A | 本地优先 Agent 平台 | MCP、RAG、semantic memory、模型路由、health check | [来源](https://github.com/shreejitverma/shreejitverma) |
| A-07 | A | SQL Agent / Resume Chatbot / RAG | LangChain、multi-agent、FAISS、Pydantic、async | [来源](https://github.com/Rishi-Kora) |
| A-08 | A | 多步研究与营销 Assistant | LangGraph、并行、持久状态、time travel | [来源](https://github.com/ayush-dhanker) |
| A-09 | A | 企业 Agent 编排与 HR Agent | OpenAI Agents SDK、multi-agent、RCA、structured extraction | [来源](https://github.com/Enesjashari) |
| A-10 | A | AI 辅助 DevOps / Agentic 工作流 | Claude Code、MCP、prompt/context engineering | [来源](https://github.com/miroadamy/resume/blob/main/miro-adamy-cv-side-projects.md) |
| A-11 | A | Agent 系统与 CV 公开入口 | LangGraph、MCP、CrewAI、RAG、向量库 | [来源](https://github.com/KaushikML/KaushikML) |
| B-01 | B | LLM/Agent 工程岗位简历模板 | Harness、AutoResearch、评测、业务归因 | [来源](https://github.com/zxt-wakeup/llm-agent-engineer-resume) |
| B-02 | B | 金融研究 Agent 的简历项目素材 | LangGraph、RAG、rerank、citations、parallel | [来源](https://github.com/LeelaissakAttota/agentic-financial-intelligence-platform/blob/main/RESUME_CONTENT.md) |
| B-03 | B | 多 Agent 辩论系统 bullet | LangGraph、审计报告、置信度、SQLite | [来源](https://github.com/xyma2003/multi-agent-debate) |
| B-04 | B | Agent 工程项目三维写法 | architecture、business、result、evaluation | [来源](https://github.com/adongwanai/AgentGuide) |
| B-05 | B | Agent + RAG 项目表述建议 | hybrid retrieval、RRF、rerank、MCP、Ragas | [来源](https://github.com/YifanJiang5/Academy_paper_agent/blob/main/README.md) |
| B-06 | B | 证据驱动的中文项目经历生成 | contribution boundary、metrics、risk review | [来源](https://github.com/oxygen914/project-resume-writer) |
| B-07 | B | Coding Agent 风格简历经验 | Google ADK、Cloud Run、工具 API、OTel | [来源](https://gist.github.com/doughayden/ff355296263ebe1e97cf012064588be7) |
| C-01 | C | Agent-readable resume schema | JSON Schema、agent instructions、human verification | [来源](https://github.com/danielrosehill/AI-Resume) |
| C-02 | C | 公开人才资料标准 | machine-first profile、schema、validator | [来源](https://github.com/neogene-ai/open-talent-protocol) |
| C-03 | C | 事实约束的简历改写 MCP | ATS、truthfulness guardrail、structured JSON | [来源](https://github.com/mutamiri-sudo/ats-resume-writer-mcp) |
| C-04 | C | Claude Code 多 Agent 简历生成插件 | agent、skill、hook、PDF rendering、ATS | [来源](https://github.com/andywxy1/rescume) |

---

## 3. A 类：公开简历 / Portfolio 中的 Agent 项目表达

### A-01：组织级 Agentic 平台与治理

> **高水平基准结论：不通过。** 该链接是个人简历仓库而非主要工程仓库；截至复核日为 1 Star、0 Fork。
> 其关联 `agent-kernel` 虽包含文档和代码，但公开采用与独立验证信号仍不足。因此本条仅可用于
> 观察“表达结构”，不得作为 RepoPilot 的架构、质量或影响力对标样本。

- **项目表达焦点**：不只写“做了 Agent”，而把 Agent 放入数字孪生、组织流程、模型路由、持久化、
  人工在环、治理和评估的组合中。
- **值得学习的写法**：把技术组件和交付边界一起说清楚：Agent 的角色、作用域、审批/guardrail、
  运行环境和可持续改进机制。
- **可迁移机制**：Agent Factory、角色化技能、模型路由、持久状态、HITL、评估循环。
- **风险提示**：该类“组织转型”措辞容易泛化；自己简历中必须落回可验证的模块、实验或文档。
- **原始页面**：[A-01 公开简历](https://github.com/fszale/resume)

### A-02：科研文献 Agent Harness

- **项目表达焦点**：以明确领域任务（系统综述与结构化数据提取）开场，再说明 orchestrator—specialist
  协作、反思/验证、跨模型 provider abstraction 和字段级来源追溯。
- **值得学习的写法**：把“结果可信”写为独立工程能力，而不是只写模型名称；例如将数据字段与原文位置/
  来源关系绑定。
- **可迁移机制**：多 Agent 分工、结构化提取、Reviewer、可追溯 evidence、跨模型对照。
- **风险提示**：页面含有量化复现实验自述；不能在自己的项目中借用该数值或结论。
- **原始页面**：[A-02 公开简历](https://github.com/eliaswestonfarber/resume)

### A-03：内部 Briefing Agent 与 AI 工程化

- **项目表达焦点**：将 PDF 分析、网页研究、交互式流程、流式输出和多格式导出组合为一个面向工作流的
  Agent，而不是孤立的聊天功能。
- **值得学习的写法**：同时列出 RAG ingest、工具标准化、MCP、skill 分层与团队使用规范，体现“系统
  能力 + 落地治理”。
- **可迁移机制**：文档管线、web research、SSE streaming、项目/全局 skills、MCP 集成。
- **原始页面**：[A-03 公开简历](https://github.com/dr5hn/resume)

### A-04：安全环境中的 Agent 研究

- **项目表达焦点**：把 Agent 放在敏感网络/受限环境的约束中描述，包括 sandbox、幻觉风险、离线推理和
  成本/基础设施考量。
- **值得学习的写法**：安全和系统边界是可写入 Agent 项目经历的工程内容，不是只附在“技术栈”末尾的
  关键词。
- **可迁移机制**：安全沙箱、风险建模、受限网络、模型部署优化。
- **原始页面**：[A-04 公开简历](https://github.com/CodyKochmann/resume)

### A-05：云端生产 Agent 的全栈表达

- **项目表达焦点**：用 Agent SDK / 框架之外的系统能力证明工程深度：外部系统 API、Agent Engine、
  云运行环境、数据库、可观测性和 IaC。
- **值得学习的写法**：简历项目的技术栈应说明“Agent 怎样进入真实系统”，例如工具 API、追踪、部署、
  权限和基础设施，而非只写 LangChain/LangGraph。
- **可迁移机制**：工具集成、云端运行、OpenTelemetry、CI/CD、Terraform。
- **原始页面**：[A-05 公开简历 Gist](https://gist.github.com/doughayden/ff355296263ebe1e97cf012064588be7)

### A-06：本地优先的自主 Agent 平台

- **项目表达焦点**：强调统一 provider、MCP 工具调用、本地数据边界、RAG/语义记忆、量化模型与
  健康检查。
- **值得学习的写法**：一个 Agent 项目可围绕“数据是否离开本机、如何切换模型、如何恢复/检测运行状态”
  展开，而不仅围绕对话效果。
- **可迁移机制**：local-first、MCP、ChromaDB、持久记忆、模型路由、环境健康检查。
- **原始页面**：[A-06 公开 Portfolio](https://github.com/shreejitverma/shreejitverma)

### A-07：多个小型 Agent 项目的 Portfolio 表达

- **项目表达焦点**：将 SQL Agent、RAG 知识助手和基于简历的对话 Agent 分开呈现，每个项目保留一个
  明确任务和一组技术标签。
- **值得学习的写法**：初级/学习型项目不必硬写“生产级”；可诚实地通过任务边界、输入输出、并发/校验
  等实现细节展示深度。
- **可迁移机制**：自然语言到 SQL、多 Agent 分派、FAISS RAG、Pydantic 结构化输出、异步处理。
- **原始页面**：[A-07 公开 Portfolio](https://github.com/Rishi-Kora)

### A-08：状态化 LangGraph Agent

- **项目表达焦点**：将“多步研究 Agent”和“领域助手”描述为不同用例，并强调并行信息源、迭代改善、
  持久状态和 state rewind。
- **值得学习的写法**：相比“用了 LangGraph”，写清图的运行语义：哪些节点并行、状态怎样保存、何时
  回退或继续。
- **可迁移机制**：规划、并行、条件路由、持久状态、time travel、Streamlit UI。
- **原始页面**：[A-08 公开 Portfolio](https://github.com/ayush-dhanker)

### A-09：企业业务场景下的多 Agent

- **项目表达焦点**：将根因分析、指标分析、结构化信息提取、候选人与岗位匹配等业务任务，分别与 Agent
  编排关联。
- **值得学习的写法**：业务类 Agent 应说明输入对象、行动/分析边界、人工决策位置和实际数据的处理方式。
- **可迁移机制**：OpenAI Agents SDK、多 Agent 编排、RCA、KPI 分析、结构化抽取。
- **风险提示**：页面中含有多项效率/数量自述，因未独立验证，本语料库不将其当作可复用事实。
- **原始页面**：[A-09 公开 Portfolio](https://github.com/Enesjashari)

### A-10：AI 辅助 DevOps 与 Agentic 工作流

- **项目表达焦点**：将 Coding Agent、MCP、prompt/context engineering 放进 DevOps、IaC、CI/CD 的
  实际工作流，而不是仅列工具名称。
- **值得学习的写法**：对 RepoPilot 这类 Coding Agent，强调“研发任务如何被 Agent 改变”比强调聊天
  能力更贴近岗位价值。
- **可迁移机制**：Claude Code、MCP、IaC、GitOps、AI 辅助日常开发。
- **原始页面**：[A-10 公开 CV](https://github.com/miroadamy/resume/blob/main/miro-adamy-cv-side-projects.md)

### A-11：Agent 系统技能与项目入口

- **项目表达焦点**：公开 profile 同时提供简历入口、Agent 相关技术栈和项目导航，便于招聘者从一页
  检查“简历—代码—项目”的一致性。
- **值得学习的写法**：简历若声明 Agent 能力，应有可访问的 demo、仓库或实验报告作为旁证；对研究项目
  可链接到 benchmark、trace 或技术文档。
- **可迁移机制**：LangGraph、MCP、CrewAI、RAG、向量检索、测试/CI。
- **原始页面**：[A-11 公开 Portfolio](https://github.com/KaushikML/KaushikML)

---

## 4. B 类：项目表述与 bullet 的公开参考

### B-01：LLM / Agent 工程岗位模板

该模板把 Agent 项目放在“LLM Agent、AutoResearch Harness、实验优化、复杂业务归因、Harness Engineering”
等语境中。它说明技术项目的表述可以同时包含：任务领域、系统架构、可靠性/评测机制和业务/研究结果。

- **可借鉴**：把“模型调用”升级为“运行时 + 工具 + 状态 + 评测”的系统表达。
- **不可照搬**：模板中的任何经历、指标、职位和项目名。
- **来源**：[B-01 模板](https://github.com/zxt-wakeup/llm-agent-engineer-resume)

### B-02：金融研究 Multi-Agent 的简历素材

公开页面给出一组 resume-ready 项目素材：Agent 角色划分、并行数据/检索、文档智能、混合检索、
cross-encoder 重排、内联引用和评测。它的价值在于展示如何将复杂架构压缩成有限 bullet。

- **可借鉴**：每个 bullet 只突出一个主轴：编排/并行、RAG 质量、评测或证据链。
- **风险**：页面量化指标和“生产就绪”都是作者自述；不得移植到自身项目。
- **来源**：[B-02 项目素材](https://github.com/LeelaissakAttota/agentic-financial-intelligence-platform/blob/main/RESUME_CONTENT.md)

### B-03：多 Agent 辩论项目的单条 bullet

该项目将“多个带不同角色/偏置的模型独立分析 → 结构化论证 → 分歧检测 → 让步追踪 → 可审计共识报告”
浓缩为一条简历 bullet，并补充技术栈。

- **可借鉴**：用“任务机制 + 验证/可解释产物 + 技术栈”而非“搭建了多 Agent”。
- **来源**：[B-03 项目与 bullet](https://github.com/xyma2003/multi-agent-debate)

### B-04：Agent 项目的三维表达框架

公开指导把项目表达分成三层：

| 层 | 应写什么 | RepoPilot 对应例子 |
| --- | --- | --- |
| 架构 | loop、工具、状态、context、权限、恢复、观测 | 本地 CLI runtime、工具策略、checkpoint、trace |
| 业务/任务 | 为何需要 Agent、工具和数据源怎样围绕任务工作 | 代码诊断、测试失败排查、patch 生成与验证 |
| 结果 | 评测集、失败类型、成本/效率、消融结论 | benchmark、正确失败、安全拒绝、预算消融 |

这套框架最有价值之处是：它避免把“使用了某 SDK”误写成项目成果。

- **来源**：[B-04 AgentGuide](https://github.com/adongwanai/AgentGuide)

### B-05：Agent + RAG 的表述深度

该材料建议从检索模块化、稀疏/稠密混合召回、融合排序、重排、多模态文档处理、MCP 工具接入和 RAG
评估等维度描述 Agent 中的知识能力。

- **可借鉴**：把 RAG 写成 Agent 的一项可验证工具能力，而不是笼统的“接入知识库”。
- **风险**：检索指标需有真实数据集、标注与运行记录；不能由模型估算。
- **来源**：[B-05 Agent + RAG 材料](https://github.com/YifanJiang5/Academy_paper_agent/blob/main/README.md)

### B-06：证据优先的项目经历生成

该 Skill 的核心观点非常适合本项目：仓库 README 只能证明“项目有某能力”，不能自动证明“某个人完成了
该能力”；所谓“主导”“从零搭建”“服务规模”“效果提升”等陈述需要独立证据。

- **可借鉴**：将代码、配置、测试、设计文档和本人工作记录分开；指标区分“已证实”“代码可证范围”与
  “待验证假设”。
- **来源**：[B-06 Project Resume Writer](https://github.com/oxygen914/project-resume-writer)

### B-07：生产 Agent 的系统层表达

与 A-05 同源，但作为表达素材可单列：它把 Agent SDK 与工具 API、云平台、数据库、可观测性、IaC 一起
出现，避免形成“只会 Prompt/框架”的印象。

- **来源**：[B-07 公开简历 Gist](https://gist.github.com/doughayden/ff355296263ebe1e97cf012064588be7)

---

## 5. C 类：简历资料的结构化与真实性工具参考

### C-01：面向 Agent 的结构化简历

该公开标准把简历拆为 JSON schema，并区分基本经历、技能、作品集、偏好和对 Agent 的操作说明。对本
项目的启发是：若未来构建“Agent 项目案例库”，应让每条样本都有规范字段，而不是只存一段散文。

- **来源**：[C-01 Agent Resume Standard](https://github.com/danielrosehill/AI-Resume)

### C-02：机器优先的专业资料协议

该项目提供专业资料 schema 与 validator。它提醒我们：数据规范解决的是结构与互操作性，不会自动证明
资料事实正确，也不能替代隐私和使用授权。

- **来源**：[C-02 Open Talent Protocol](https://github.com/neogene-ai/open-talent-protocol)

### C-03：带真值护栏的 ATS 简历改写 MCP

该 MCP 示例明确禁止凭空编造日期、指标、职位或资历，并可返回纯文本或结构化结果。这与 RepoPilot 的
研究原则一致：模型可提出改写候选项，但事实边界必须由输入证据与用户确认控制。

- **来源**：[C-03 ATS Resume Writer MCP](https://github.com/mutamiri-sudo/ats-resume-writer-mcp)

### C-04：基于 Claude Code 的多 Agent 简历插件

该插件将简历处理组织为解析、JD 分析、匹配、生成、PDF 渲染等能力，并展示 Agent、Skill、Hook 与
模板渲染的组合。它更适合作为“工作流产品”参考，而不是个人简历样本。

- **来源**：[C-04 Rescume](https://github.com/andywxy1/rescume)

---

## 6. 跨样本观察：Agent 项目简历写法的共性

### 6.1 强表达通常包含四个层次

```text
业务/研究问题
  + Agent 决策与执行结构
  + 可靠性、证据或安全机制
  + 可验证产物或真实评测
```

仅写“基于 LangChain/LangGraph 开发 AI Agent”只覆盖技术名词；加入以上四层后，读者才能判断：为何要用
Agent、它如何工作、有什么边界、是否经过验证。

### 6.2 高频技术主题（按语料出现方式归纳）

| 主题 | 常见的有效表述 | 不充分的表述 |
| --- | --- | --- |
| 多 Agent | 角色、输入/输出契约、协调/汇总、冲突处理、独立上下文 | “用了多个 Agent 协作” |
| RAG | 数据源、chunk/检索/重排、证据引用、评测集 | “接入向量数据库” |
| 工具调用 | 具体工具类别、schema、权限、失败恢复 | “支持 Function Calling” |
| 状态与长任务 | checkpoint、持久状态、回退、预算、恢复 | “支持多轮对话” |
| 安全 | sandbox、HITL、最小权限、审计、真值约束 | “安全可靠” |
| 评测 | 任务集、指标、对照、失败类型、回归 | “效果很好/准确率高” |
| 生产化 | API、可观测性、部署、IaC、CI/CD、成本 | “可部署上线” |

### 6.3 量化指标的正确位置

公开样本中常见“准确率、耗时、成本、自动化比例”等数字。对 RepoPilot，只有当指标拥有以下任一证据时，
才适合写入简历：

- 可重复 benchmark 的运行结果、任务清单、固定模型/预算和评分脚本；
- CI 测试、trace、artifact 或 release 文档；
- 明确的手工统计口径、时间范围和原始记录。

如果没有证据，应改写为实现事实，例如“建立 XX 任务集与评测脚本”“记录 Token、延迟、工具调用和正确
失败”，不要写成未经证明的改善百分比。

---

## 7. 对 RepoPilot 简历材料的可用提炼方向（暂不作为最终简历）

从本语料库看，RepoPilot 可以优先准备以下四类**事实型材料**，供下一步再由本人确认贡献范围和指标：

1. **本地 Coding Agent Runtime**：会话、流式输出、工具调用、权限模式、取消与 checkpoint 恢复；
2. **可靠性 Harness**：循环预算、结构化状态、上下文压缩、错误恢复、sandbox/策略与审计；
3. **扩展与协作**：MCP、Skills/Plugins、规划、只读并行探索、reviewer/验证器；
4. **研究评测**：任务协议、trace、benchmark、正确失败、安全拒绝、消融或模型对照。

建议的事实核验顺序：

```text
源码与测试 → 可运行命令/trace → 设计文档 → 实验结果 → 本人贡献确认 → 简历 bullet
```

这样可避免将“仓库具备的功能”误写成“个人已经证明的业务影响”。

---

## 8. 后续维护格式

新增样本时请使用以下字段，避免语料库退化为链接收藏夹：

```yaml
id: A-12
tier: A | B | C
source_url: https://...
source_type: public_resume | public_portfolio | resume_bullet_reference | tool_or_schema
accessed_on: YYYY-MM-DD
agent_task: 一句话描述任务
mechanisms: [tools, state, rag, evaluation]
evidence_on_page: 页面明确可见的模块/项目描述
claim_status: self_reported | template | independently_reproduced
reusable_writing_pattern: 可学的表达结构
do_not_copy: 不可迁移的身份、公司、指标或贡献声明
```

每次新增后都应先检查：是否确为公开页面、是否包含不必要个人信息、是否把作者自述误标为已验证事实，
以及是否真的能为 RepoPilot 的下一步整理提供新信息。
