# RepoPilot：可写入简历的证据驱动 Agent 实验计划

> 制定日期：2026-08-23  
> 状态：研究设计，尚未产生新的实验结果。  
> 核心目标：将 RepoPilot 收敛为一个**本地优先、可控的 Coding Agent 研究项目**；针对可替换的
> Agent 模块提出假设，用公开数据集和冻结评测流程验证“方法 A 是否优于方法 B”。

## 0. 结论先行：项目应该怎么聚焦

不要把 RepoPilot 继续写成“实现了 RAG、记忆、MCP、多 Agent、微调和安全”的功能清单。这样的项目难以
证明每个模块真正带来收益，也难以在面试中解释因果关系。

建议的项目主线是：

> **Evidence-Bounded Coding Agent：以自适应代码检索、验证驱动控制环与受限工具契约，提升真实仓库修复
> 任务中的证据质量、修复有效率和失败可解释性。**

这条主线把已有的 RepoPilot 模块（代码检索、ToolSpec、验证器、trace、checkpoint、reviewer、Docker
边界）变成可测研究对象。最小可写入简历的成果由四项正式实验构成：

1. **R1：代码 RAG / 上下文选择**：新检索方法相对 BM25 基线的检索质量与端到端修复效果；
2. **R2：验证驱动控制环**：Plan–Execute–Verify 和风险触发 reviewer 是否改善有效修复/错误接受率；
3. **R3：工具契约**：严格 schema 和阶段化工具暴露是否提高有效调用并降低不相关/无效调用。
4. **R4：长期记忆**：带来源和时效边界的 typed episodic memory 是否在固定成本下优于窗口和 BM25 记忆。

AgentDojo 对抗安全可在这四项完成后作为第二阶段；它不阻塞主项目交付。

## 1. 现状审计：什么能保留，什么不能再当作结论

RepoPilot 已有如下可直接利用的实现基础：

- `src/repopilot/retrieval/bm25.py`：可解释的代码 BM25 基线；
- `src/repopilot/context/builder.py`：上下文组装边界；
- `src/repopilot/tools/`：结构化工具定义、权限与执行路径；
- `src/repopilot/verification/`、`src/repopilot/orchestration/`：验证、reviewer、并行只读执行与状态图；
- `src/repopilot/observability/trace.py`、session/checkpoint：运行记录与恢复证据；
- `docs/evaluation/coding_development_protocol.md`：受限开发集与 receipt 验证框架。

历史的 FRAMES smoke10 检索结果只能用作探索日志，**不能**写成“Hybrid RAG 已显著优于 BM25”：样本只有
10 题、问题处于 oracle document 设置，且历史记录本身已指出波动很大。旧的 `docs/project_resume_description.md`
亦只能视为历史草案；其中每个数字都必须在本计划的冻结评测完成后重新审计。

## 2. 高质量项目和论文如何影响本计划

| 参考来源 | 可学习的工程/研究方法 | 本计划中的对应决策 | 不可照抄的内容 |
| --- | --- | --- | --- |
| [SWE-agent, NeurIPS 2024](https://arxiv.org/abs/2405.15793) 与 [代码库](https://github.com/SWE-agent/SWE-agent) | 将 Agent–Computer Interface、工具和轨迹作为主要研究对象；以真实 issue 修复评估 | R2 用真实 GitHub issue 修复任务，记录工具轨迹和 verifier 结果 | 不借用其基准分数或“state of the art”表述 |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | Docker 化、可复现的真实仓库 issue–patch 验证；Verified 由工程师确认可解 | R1/R2 的端到端冻结验证集 | 不能把非官方/小样本结果称为 SWE-bench 官方成绩 |
| [CodeRAG-Bench, Findings of NAACL 2025](https://aclanthology.org/2025.findings-naacl.176/) 与 [代码](https://github.com/code-rag-bench/code-rag-bench) | 将检索指标与执行型代码生成指标分开；提供 repo-level 检索和执行评估 | R1 同时报告 NDCG/Recall、Pass@1、token 与延迟 | 不把论文中的任一检索器分数写为 RepoPilot 结果 |
| [Aider](https://github.com/Aider-AI/aider) | 代码库地图、Git diff 与 lint/test 组成闭环 | R1 以代码相关上下文为中心；R2 以验证证据为终止条件 | 不宣称兼容全部模型或全部语言 |
| [Cline](https://github.com/cline/cline) | 同一 Agent 引擎提供交互和 headless JSON 输出；工作树/自动化是独立能力 | 结果统一落为 JSONL/receipt，保证可比较 | 不把 SDK、IDE 和多 Agent 看板当作本阶段必需功能 |
| [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard) | 将工具调用正确性拆为单调用、并行、多轮、相关性与格式敏感性 | R3 使用其非在线数据的 AST/可执行调用指标 | 不将 Agent 的端到端能力简化成纯 function-call 分数 |
| [AgentDojo, NeurIPS 2024 D&B](https://github.com/ethz-spylab/agentdojo) | 在有状态工具环境中同时测效用与 prompt injection 防御 | R3 的可选安全扩展 | 不用私有/攻击性环境做未经隔离的运行 |

## 3. 候选模块优先级

| 优先级 | 模块 | 要回答的问题 | 新方法 | 比较基线 | 数据集 | 可行性 | 决策 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| P0 | 评测与证据基础 | 差异是否来自模块而不是模型、提示词或测试变化？ | 冻结 manifest、receipt、配对运行、置信区间 | 无 | 所有正式评测 | 5/5 | 必须先做 |
| P1 | 代码 RAG / context | 能否更准地定位跨文件证据，并在固定 token 预算内提高修复？ | 自适应 hybrid + 符号/测试邻接扩展 + 预算选择 | 当前 BM25 top-k | CodeRAG-Bench + SWE-bench Verified | 5/5 | 主成果 |
| P2 | 验证驱动控制环 | 规划、测试与 reviewer 在什么条件下有净收益？ | PEV + 风险触发 reviewer + 一次受限 replan | 直接 ReAct、无 reviewer、全量 reviewer | SWE-bench Verified | 4/5 | 主成果 |
| P3 | Tool contract | 更严格的 schema/阶段化工具集合是否减少误调用？ | typed schema + precondition + stage-scoped allowlist + recovery code | 宽松命令字符串、平铺工具表 | BFCL v4 非在线子集 | 4/5 | 主成果 |
| P4 | 长期记忆 | 历史经验检索能否在固定成本下提高跨会话回答？ | provenance-aware typed episodic memory | 滑动窗口、BM25 memory | LongMemEval | 3/5 | 正式成果 |
| 暂缓 | 并行子 Agent | 并行是否真正降低墙钟时间且不降低正确性？ | 只读子 Agent task packet | 单 Agent | SWE-bench 诊断子集 | 2/5 | 有足够主结果后再做 |
| 暂缓 | 模型路由 / 微调 | 路由或训练是否优于同成本单模型？ | 路由器或 LoRA | 单模型 | 多数据集 | 1/5 | 当前不做 |

**为什么不把通用 FRAMES RAG 当主实验？** RepoPilot 的目标是 Coding Agent；CodeRAG-Bench 与 SWE-bench
能够分别测“代码证据是否找对”和“补丁是否通过真实测试”，比百科式 QA 更贴合项目叙事。FRAMES 可保留为
探索性附录，但不能承担核心简历结论。

## 4. P0：所有正式实验共享的证据协议

### 4.1 冻结项

每个实验变体必须记录并固定：

- RepoPilot commit、Python/依赖 lock、Docker 镜像 digest；
- provider、模型精确名称、base URL、请求日期、temperature、top_p、最大输出/工具/时间预算；
- 系统提示词、工具集、权限模式、运行平台和 CPU/RAM/GPU；
- 数据集版本、下载校验和、许可、实例 ID 列表和拆分 manifest；
- 原始输出位置、去敏后的 trace JSONL、测试日志、结果 receipt 与汇总脚本版本。

配置 A 和 B 必须在**同一实例、相同模型与预算**上配对执行。检索实验若完全确定性则无需重复；Agent 端到端
实验即使 temperature=0 也应对同一配置至少运行 3 次，防止云端模型非确定性掩盖差异。

### 4.2 数据拆分和统计规则

1. 下载后按 `sha256(dataset_version + instance_id)` 固定排序并写入版本库；不得按看过的结果重新挑题。
2. 每个正式数据集分为 **development 20% / validation 40% / final holdout 40%**；只在 development 调参。
3. `validation` 只用于一次候选方案选择；最终简历数字只能来自从未用于方案选择的 `final holdout`。
4. 二值成功率报告点估计、分母、95% Wilson 区间和配对 bootstrap 差值区间；不要只报告百分比。
5. 除主指标外，强制报告成本（input/output token、API 费用，如可得）、p50/p95 墙钟时间、工具次数、
   中断/超时和错误接受率。

### 4.3 简历结论准入

只有同时满足以下条件时才能写“优于”：

- final holdout 上主指标的配对 95% CI 下界大于 0，或预注册的非劣界成立；
- 没有通过扩大 token/时间/工具预算获得不可比优势；
- 原始 trace、patch、Docker 验证结果和汇总脚本可在本地复查；
- 文案指出数据集、样本数、模型和约束，不外推为普适结论。

如果结果为负或不显著，仍可保留为“实验发现”，例如“在固定 4k token 预算下，Dense-only 检索未超过
BM25”；这比只展示成功案例更可信。

## 5. R1：自适应代码 RAG 与上下文预算实验

### 5.1 研究问题与假设

**问题**：当前运行时的 BM25 文件检索无法处理语义描述、测试—实现关联与固定上下文预算之间的冲突。

**主假设 R1-H**：在相同检索/生成 token 预算下，`Adaptive Code Hybrid` 比当前 `BM25 top-k` 有更高的
代码证据 Recall@10 / NDCG@10，并且不会降低 RepoEval 的 Pass@1。

### 5.2 方法与隔离消融

`Adaptive Code Hybrid (ACH)` 由四个可独立启用的组件构成：

1. **Hybrid retrieval**：BM25 与 code embedding 的 Reciprocal Rank Fusion；
2. **查询类型路由**：若输入含 stack trace、文件路径、符号或精确标识符，提高 BM25 权重；若是自然语言
   功能描述，使用平衡权重；路由规则须是确定性的、可记录的；
3. **符号/测试邻接扩展**：从命中函数扩展到定义、调用者/被调用者、同名测试和 import 相邻文件；
4. **证据预算选择**：在固定 `context_token_budget` 下做去重与多样性选择，避免多个高度重复 chunk 挤掉测试。

不能把四项一起与 BM25 比完就声称每项有效。依次执行：

| 运行 | 唯一主变量 | 固定项 | 比较 |
| --- | --- | --- | --- |
| R1-A | Hybrid RRF | chunker、top-k、budget、模型 | BM25 vs Hybrid |
| R1-B | 查询类型路由 | 使用同一 Hybrid、同一 budget | fixed-weight Hybrid vs routed Hybrid |
| R1-C | 符号/测试邻接 | 使用 R1-B、同一 budget | no expansion vs expansion |
| R1-D | 预算选择 | 使用 R1-C、总 token 相同 | naïve top-k concatenation vs diversified budget selector |

`R1-D` 胜出且其他消融不出现严重回退时，才将 ACH 作为候选默认实现。若某组件无效，应关闭它而不是
为了“复杂”保留。

### 5.3 数据集、指标和成功条件

| 阶段 | 数据 | 用途 | 主要指标 | 备注 |
| --- | --- | --- | --- | --- |
| 检索开发/验证 | CodeRAG-Bench 的 `repoeval_repo` | 有 qrels 的 repo-level 代码检索 | NDCG@10、MRR@10、Recall@5/10、检索延迟 | CodeRAG-Bench 已提供 BM25/dense 与 BEIR 格式管线 |
| 端到端开发/验证 | CodeRAG-Bench `repoeval-function` | 固定上下文下的执行型生成 | Pass@1、编译/测试失败率、输入 token | 先验证原始 ground truth 在环境中可通过 |
| 最终泛化 | SWE-bench Verified 的冻结 holdout | issue → patch → Docker 测试 | resolved rate、patch apply rate、错误接受率、token/时间 | 仅最终一次使用，不用于选择权重 |

**成功判据**：final holdout 上 `Δ NDCG@10` 的 paired bootstrap 95% CI 下界 > 0，且 `Pass@1/resolved`
的下界不为负；否则只能写“改善检索代理指标”而不能写“改善修复能力”。

### 5.4 建议实现边界

- 先复用 `BM25CodeIndex`，新增独立的 `CodeChunk` / `RetrievalEvidence` schema；不能直接将原始全文塞入
  prompt；
- 代码 symbol 提取先支持 Python `ast` 与测试文件命名规则，其他语言只退化为路径/词法，不伪装为通用 AST；
- embedding/reranker 的模型、维度、设备和量化方式必须记录；8GB 显存不足时可 CPU rerank，但要报告延迟；
- 提供离线 qrel 评估，使 R1 的核心结论不依赖任何 API 模型。

## 6. R2：验证驱动 Agent 控制环实验

### 6.1 研究问题与假设

**问题**：单一 ReAct loop 容易把“模型回答完成”当作“补丁已被验证”；全量 reviewer 则可能提高成本却不提高
修复率。

**主假设 R2-H1**：在相同模型、工具与总预算下，显式 `Plan → Execute → Verify`（PEV）相较于直接 ReAct
降低错误接受率，并且不降低 resolved rate。

**主假设 R2-H2**：风险触发的只读 reviewer 相较于“每次都 reviewer”，在 resolved rate 非劣的前提下
降低 reviewer 调用与端到端 token 成本。

### 6.2 方法与隔离消融

| 运行 | 唯一主变量 | 对照 | 指标重点 |
| --- | --- | --- | --- |
| R2-A | 控制环 | direct ReAct vs PEV（均不启用 reviewer） | resolved、错误接受、工具步数 |
| R2-B | reviewer 策略 | PEV + no reviewer vs always reviewer vs risk-gated reviewer | resolved、reviewer token、p95 时间 |
| R2-C | replan 上限 | risk-gated + 0/1/2 次 replan | 修复率、循环率、预算耗尽率 |

风险门必须为可解释的确定性规则，初版可由以下信号组成：测试失败、改动安全敏感目录、diff 超过阈值、
新增依赖、静态 patch anti-pattern、模型低置信度或关键证据缺失。它不是“让另一个模型猜是否有风险”。

### 6.3 数据集与指标

| 阶段 | 数据 | 用途 | 样本要求 |
| --- | --- | --- | --- |
| smoke | `repopilot-synthetic-dev-v1` | 验证工具、patch、receipt、Docker 路径正确，不产出能力结论 | 全部 fixture |
| development | SWE-bench Verified development manifest | 确定阈值和 replan 上限 | 按仓库分层的 20% |
| validation | SWE-bench Verified validation manifest | 一次方案选择 | 按仓库分层的 40% |
| final holdout | SWE-bench Verified final manifest | 产生简历可用数字 | 按仓库分层的 40%，只运行一次 |

核心指标：

- **resolved rate**：官方 Docker evaluator 的通过率；
- **错误接受率**：Agent 宣称/状态记录为完成，但官方 evaluator 不通过的比例；
- **patch apply rate、verification pass rate、replan rate、budget exhaustion rate**；
- 每实例 input/output tokens、工具次数、reviewer 调用、p50/p95 墙钟时间。

**实践前置**：官方 SWE-bench 文档建议约 120GB 可用空间、16GB RAM、8 CPU 核；若当前 Windows Docker
环境达不到，应改用隔离的 Linux/云端评测机并保持项目代码、数据和 Docker digest 不变。不能以 3 个
smoke task 的结果替代正式集。

## 7. R3：工具契约与阶段化能力暴露实验

### 7.1 研究问题与假设

**问题**：将宽泛自然语言命令交给模型会混淆工具选择、参数生成、权限判断与失败恢复；只改提示词不能
说明这种失败是否被真正约束。

**主假设 R3-H**：类型化参数 + precondition validation + 结构化 recovery code 相较于宽松文本参数，能
提高有效调用率和不相关工具拒绝率。阶段化 allowlist 能在不降低任务完成率的条件下减少工具选择错误。

### 7.2 对照设计

| 运行 | 工具集合 | 参数契约 | 恢复信息 | 用途 |
| --- | --- | --- | --- | --- |
| R3-B0 | 平铺全量工具 | 宽松 `command/query` 字符串 | 自由文本错误 | 弱工程基线 |
| R3-B1 | 平铺全量工具 | JSON Schema：required/enum/path scope | 结构化错误码 | 隔离 schema 价值 |
| R3-A | 任务阶段 allowlist | 与 B1 相同 | 与 B1 相同 | 隔离阶段化暴露价值 |

`R3-A` 的阶段仅允许在 runtime 内由确定性状态转换：例如探索阶段可 read/search，修改阶段才能 apply patch，
验证阶段只允许 test/diff。权限确认仍由 policy 层执行，allowlist 不是权限绕过。

### 7.3 数据集与指标

- **主评测**：[BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard) 可离线执行的 `simple`、`parallel`、
  `multi-turn` 和 `relevance` 子集。冻结所用 release、任务 ID 与官方 evaluator commit；不运行需要实时
  外部服务的 live 任务。
- **项目契约回归集**：从 RepoPilot 的 `read_file/search/retrieve_code/apply_patch/run_tests` schema 生成
  版本化的正例、缺参例、越界路径例、无关请求例和可恢复失败例。它仅验证 RepoPilot 契约，不能替代 BFCL。
- **可选安全扩展**：[AgentDojo](https://github.com/ethz-spylab/agentdojo) `workspace` suite，比较无过滤与
  tool filter/阶段化策略的 utility 与 attack success。此扩展要求先写 DeepSeek 的兼容 adapter 和完整隔离，
  不纳入 R1–R4 四项正式实验的交付门槛。

指标：AST exact/executable call accuracy、relevance rejection、schema rejection、参数错误、非法路径拒绝、
失败后恢复成功率、每任务工具数与延迟。对于 AgentDojo 同时报告 benign utility 和 attack success，不能只
报告防御成功率。

## 8. R4 / P4：长期记忆正式实验

### 8.1 研究问题与边界

**问题**：简单窗口只能看见最近对话；普通 BM25 情节记忆虽然可解释，却容易错过同义线索、知识更新和
时间条件。Agent 的记忆也不应把不可信的模型总结直接当作事实。

**主假设 R4-H**：在相同 reader model、回答 token 上限和可见记忆预算下，`Typed Provenance Memory (TPM)`
比滑动窗口和单纯 episodic BM25 具有更高的 evidence recall 与 QA accuracy，并且不降低 abstention
正确性。所有索引、抽取和回答成本均纳入总成本。

这个实验验证的是**跨会话记忆模块**，不是 Coding Agent 修复能力。因此 R4 的结果只能写为“Agent memory
在 LongMemEval 上的表现”，不可与 R1/R2 的 SWE-bench resolved rate 合并成单一总分。

### 8.2 方案定义

| 变体 | 记忆内容与取回方式 | 目的 |
| --- | --- | --- |
| R4-B0：window | 仅向 reader 提供最近 `W` 个 turn，token 预算与其他变体一致 | 最低成本、无索引基线 |
| R4-B1：BM25 episode | 以 session 为 unit 的原始文本 BM25 top-k | 现有可解释基线 |
| R4-B2：hybrid episode | 同一 session unit 的 BM25 + embedding RRF | 隔离语义检索价值 |
| R4-B3：time-aware hybrid | B2 加问题日期解析与时间范围 pruning | 隔离时间约束价值 |
| R4-C1：TPM | B3 加类型化、带来源的 memory card 与预算选择 | 候选新方案 |

`TPM` 的 memory card 固定为以下 schema，禁止由模型自行扩展字段：

```text
memory_id, source_session_id, source_turn_ids, observed_at, scope,
kind(fact|preference|decision|event|failure), statement, entities,
valid_from, valid_to, confidence, extraction_model, content_hash
```

- `source_session_id` 和 `source_turn_ids` 是强制字段，任何 card 无来源即丢弃；
- `valid_from/valid_to` 未知时明确置空，不能由模型编造日期；
- 原始 session 永远保留为可回读证据，card 只是索引和压缩层；
- 记忆抽取按内容哈希缓存，一个相同 session 只抽取一次；总 ingestion token 和失败率必须写入结果；
- 回答前的预算选择需至少包含原文片段、来源 ID、时间条件和 card 摘要，防止 summary-only 幻觉。

### 8.3 单变量消融顺序

| 运行 | 唯一主变量 | 固定项 | 比较 |
| --- | --- | --- | --- |
| R4-1 | 是否使用检索 | reader、history budget、top-k、answer prompt | window vs BM25 episode |
| R4-2 | 检索分数 | session granularity、top-k、budget | BM25 vs Hybrid RRF |
| R4-3 | 时间约束 | 使用同一 Hybrid、同一 budget | Hybrid vs time-aware Hybrid |
| R4-4 | card/来源层 | 使用 R4-3 检索池、同一 reader/budget | time-aware Hybrid vs TPM |
| R4-5 | 最终端到端确认 | 冻结 R4-4 配置 | B0、B1、R4-C1 在 final holdout 上配对运行 |

虽然 TPM 有多个工程组成，R4-4 只在 R4-3 已固定的候选集上加入 card、来源约束和预算 selector；R4-1 至
R4-3 的结果分别解释检索、语义与时间条件的贡献。若 R4-4 不优于 R4-3，TPM 不进入默认运行时。

### 8.4 数据、拆分与防泄漏

使用官方清洗版 [LongMemEval（ICLR 2025）](https://github.com/xiaowu0162/LongMemEval)：`longmemeval_s_cleaned.json`
的 500 个实例。它覆盖信息抽取、多会话推理、知识更新、时间推理和 abstention；其长历史可达约 115k token，
适合检验“不能直接塞满上下文”的设计。

拆分不能只按 `question_id` 随机切分。下载后应先将所有共享任一 `haystack_session_id` 的问题连成同一组，
再按组的稳定 SHA-256 hash 和 `question_type` 分层划为：

- development：20%，只用于 `W`、top-k、embedding、时间范围规则和 card prompt 的选择；
- validation：40%，只用于一次候选确认；
- final holdout：40%，从未查看运行结果，只在 R4-5 运行一次。

所有实例均保留在本地不可变 manifest；开发/验证/最终集的 question type 与 abstention 比例须写入 receipt。
检索 recall 的计算遵循官方定义：30 个 abstention instance 不进入 evidence recall 分母，但必须进入端到端
abstention 评估。

**不以 LongMemEval-V2 作为第一轮正式数据集**：V2 虽更接近 Agent trajectory，但可含极长的多模态轨迹，
会把实现与成本风险扩大到无法解释的程度。完成 R4 后才可将 V2 用作单独的外部泛化附录。

### 8.5 指标、判定与评估器边界

| 层次 | 指标 | 是否依赖 LLM judge |
| --- | --- | --- |
| 检索 | session Recall@1/3/5、turn recall、MRR、p50/p95 retrieval latency | 否，使用 `answer_session_ids` / `has_answer` 标签 |
| 回答 | 官方 QA auto-eval accuracy、按 question type 分解 | 是，官方脚本使用指定 judge |
| abstention | abstention precision、recall、F1、错误自信回答率 | 官方 judge 或冻结盲审协议 |
| 成本 | ingestion/retrieval/reader/judge 分项 token、请求数、费用、缓存命中率 | 否 |
| 可信度 | 无来源 card 比例、答案所引用 session 的 evidence coverage | 否 |

官方 LongMemEval `evaluate_qa.py` 的发布示例使用 `gpt-4o` judge。若后续没有获得独立的 OpenAI judge 授权，
可以用冻结版本的 DeepSeek 盲评器完成**内部比较**，但报告必须标为“LongMemEval reimplementation with
DeepSeek judge”，不能称为官方 leaderboard score。无论采用哪一种 judge，都要额外抽取 final holdout 中
按题型分层的 50 题，去掉变体名称后进行人工双盲复核；报告 judge 与人工的一致率和所有分歧。

R4 达标条件：

1. final holdout 上 TPM 相对 B1 的 `Δ session Recall@5` 的 paired bootstrap 95% CI 下界 > 0；
2. QA accuracy 的差值不为负，abstention F1 不下降；
3. 总 token 成本（包括 card 抽取）不高于 B1 的预注册上限；若超出，必须报告 quality/cost frontier，不能
   简化为“更好”；
4. 每个正确答案可回链至少一个 source session；无来源卡片不得贡献结论。

### 8.6 工程交付物

- `src/repopilot/memory/`：memory card schema、content-addressed extraction cache、provenance validator；
- `src/repopilot/retrieval/`：session/turn retriever、time-aware filter、预算 selector；
- `evaluation/memory/longmemeval/`：下载校验、group split、无 LLM retrieval evaluator、QA adapter；
- `configs/evals/r4_*.yaml`：每个变体的冻结预算/模型/索引参数；
- `docs/experiments/*_r4_*.md` 与 `artifacts/evals/r4/`：运行记录、raw output hash、结果与成本表。

## 9. 推荐执行顺序与交付物

| 阶段 | 完成条件 | 主要交付物 |
| --- | --- | --- |
| S0 | 安装数据、核对许可、生成不可变 split manifest、Docker smoke | `configs/evals/*.yaml`、`artifacts/dataset_receipts/` |
| S1 | R1-A 至 R1-D 离线检索、RepoEval 与最终 SWE-bench holdout | qrel report、官方 evaluator 结果、patch/trace 记录 |
| S2 | R2-A 至 R2-C 全流程消融 | trace funnel、错误接受审计、成本表 |
| S3 | R3-B0/B1/A 的 BFCL 与项目契约回归 | AST 结果、拒绝/恢复分析 |
| S4 | R4-1 至 R4-5 的 LongMemEval 检索和端到端评测 | group split、provenance audit、judge/human consistency、成本 frontier |
| S5 | 四项主实验的复现实验、独立复查、最终事实审计 | `docs/results/resume_evidence_report.md`、可复查 artifact index |

每次运行按 `YYYY-MM-DD_<module>_<variant>_<dataset>_<run-id>.md` 在 `docs/experiments/` 记录，至少包含：
假设、主变量、冻结项、确切命令、数据 manifest、结果、失败、统计方法和下一步。原始模型输出和 patch 仅
保存在本地受控 artifact 目录；文档只保存去敏摘要和哈希。

## 10. 结果出来后才能使用的简历表述模板

以下均为占位模板，方括号中必须替换为复查过的 final holdout 数字，不能在实验前使用。

1. **代码 RAG**  
   “设计并实现面向 Python 仓库的自适应混合检索：结合 BM25、语义检索、符号/测试邻接扩展与固定预算
   上下文选择；在 [数据集/holdout N] 上将 [NDCG@10 或 Recall@10] 从 [B] 提升至 [A]（配对 95% CI
   [x, y]），并在相同 [token/时间] 预算下使 [Pass@1/resolved] [变化]。”

2. **控制环与验证**  
   “构建 Plan–Execute–Verify 控制环与风险触发只读 reviewer；在 [SWE-bench Verified holdout N] 的官方 Docker
   评测中，将错误接受率从 [B]% 降至 [A]%，并在 resolved rate 非劣（95% CI [x, y]）的条件下减少
   [token/时间] [Z]%。”

3. **工具契约**  
   “将 Coding Agent 工具重构为阶段化 allowlist、JSON Schema 和可恢复错误契约；在 [BFCL release/subset N]
   上将有效函数调用率从 [B]% 提升至 [A]%，并将不相关工具误调用率降低 [Z]pp。”

4. **长期记忆**  
   “实现带 source-turn、时间范围与内容哈希的 typed episodic memory；在 [LongMemEval_S holdout N] 上将
   session Recall@5 从 [B] 提升至 [A]（配对 95% CI [x, y]），在 [总 token 成本] 下保持 [QA accuracy /
   abstention F1] [变化]；结果经 [官方/DeepSeek 盲评] 与 [N] 题人工双盲复核。”

没有统计支持时，应该写“实现并建立评测协议”，而非“提升”“优于”“显著降低”。

## 11. 清理与归档边界

本计划**不执行删除操作**。用户所说的“删除之前项目”可能指旧的简历语料文档、历史实验日志，或整个
RepoPilot 项目，三者后果完全不同。新方案验收后，可按明确名单选择以下之一：

1. 只归档/删除旧的简历资料：`docs/lessons/public_agent_resume_corpus.md`、
   `docs/lessons/high_confidence_agent_project_cases.md`；
2. 归档旧的、不可用于结论的实验摘要，保留原始日志与哈希；
3. 不删除 RepoPilot，只以本计划作为新的项目叙事。

在得到具体路径清单前，必须保留现有文件，避免误删代码、数据或可复现证据。

## 12. 主要参考

- [卡码笔记 Agent 知识体系整理](../lessons/kamacoder_agent_knowledge_summary.md)
- [SWE-agent paper](https://arxiv.org/abs/2405.15793)；[SWE-agent code](https://github.com/SWE-agent/SWE-agent)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [CodeRAG-Bench paper](https://aclanthology.org/2025.findings-naacl.176/)；[CodeRAG-Bench code](https://github.com/code-rag-bench/code-rag-bench)
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard)；[evaluation data/harness](https://github.com/EnlightenedAI/BFCL)
- [AgentDojo paper/repository](https://github.com/ethz-spylab/agentdojo)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
