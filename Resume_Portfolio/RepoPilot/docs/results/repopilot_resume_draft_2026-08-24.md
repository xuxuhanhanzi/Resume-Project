# RepoPilot 简历项目描述（结果留空版）

## 推荐主版本：LLM / Agent 工程师岗位

**RepoPilot｜可验证 Coding Agent Runtime（个人项目）**　2026.03–至今  
`Python · Docker · Git · LLM Tool Calling · Agent Orchestration · Evaluation`

- **构建**本地优先的 Coding Agent Runtime，将任务执行拆分为 `Task Spec → Model → Policy → Tool/Sandbox → Observation → Checkpoint → Verifier`；将模型输出限制为结构化动作建议，由独立 Policy 和确定性 Verifier 分别负责权限约束与完成判定。
- **实现**可持久化的 `AgentState`、checkpoint 与 execution journal，用于中断恢复和工具副作用去重；在默认拒绝的 Docker sandbox 中约束网络、身份、Linux capabilities、CPU、内存和 PID，降低不可信代码执行的越权与资源失控风险。
- **设计并落地** Planner–Execute–Verify 控制环：Planner 生成有界计划，Executor 仅获得有限写权限，Verifier 独立检查；Reviewer 保持只读，并由测试失败、风险 diff、依赖变化和证据缺失等可解释信号触发，支持受限 replan。
- **搭建**面向 SWE-bench Verified 的证据驱动实验链路：以 SHA-256 绑定公开数据、split manifest、源码快照、trace、patch 与 receipt；隔离 gold patch/隐藏测试等 evaluator-only 信息，并通过预调用 token 预算与 public-base 工作区收据保证配对运行可复查。
- **最终实验结果：**`[待填写：SWE-bench Verified final holdout 的样本数、模型与固定预算、resolved rate、错误接受率、token/时间成本，以及相对 direct ReAct 的 paired 95% CI。]`

## 精简版本：一页中文简历

**RepoPilot｜可验证 Coding Agent Runtime（个人项目）**　2026.03–至今

- 构建本地优先 Coding Agent Runtime，解耦模型决策、策略约束、工具执行与确定性验证；以结构化 Tool Contract、路径作用域与 Docker sandbox 控制不可信代码执行边界。
- 实现可恢复执行机制（`AgentState`、checkpoint、execution journal）和 PEV 多角色编排；通过只读 Reviewer、风险门与受限 replan 将“模型完成”与“补丁验证通过”分离。
- 建立 SWE-bench Verified 的可复核实验协议：冻结数据/拆分/预算/源码快照，隔离 evaluator-only 内容，保存 trace/patch/receipt 后再恢复 public-base 工作区。
- **最终实验结果：**`[待最终评测完成后填写。]`

## 使用规则

- 应聘 LLM Agent、AI Infra、后端平台或 Developer Tool 岗位时，优先使用“推荐主版本”；篇幅紧张时使用“精简版本”。
- 将项目链接放在标题行或个人主页中；不要在项目 bullet 中重复堆叠框架名。
- 最后一条只能替换为 final holdout 的官方 Docker evaluator 结果；development、validation 和 smoke 结果不得写成项目性能结论。
- 根据具体 JD 调整技术栈和首条关键词，但保留“受限执行、可恢复、可验证、可复核评测”四个可证实的差异点。
