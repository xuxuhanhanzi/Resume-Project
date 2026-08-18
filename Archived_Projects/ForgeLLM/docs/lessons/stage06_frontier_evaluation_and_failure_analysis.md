# Stage 6 讲义五：前沿评测实践与失败分析

## 1. “最新模型分数”为什么特别难比较

前沿模型报告常同时变化：模型版本、thinking budget、system prompt、采样参数、工具环境、答案提取器和重复运行次数。两个表格即使用了同名 benchmark，也可能不是同一实验。

因此 Stage 6 学习的重点不是背榜单，而是审查 protocol equivalence：

- 数据版本相同吗？
- prompt/chat template 相同吗？
- reasoning effort/token budget 相同吗？
- pass@1 还是多次采样？
- 工具调用和执行环境相同吗？
- 是否有污染或题目公开后的训练风险？

<details>
<summary>思考题：为什么不能把不同 reasoning budget 的分数直接当成模型能力差？</summary>

更多生成 token、并行样本或工具调用本身就是额外推理资源。比较时必须固定预算，或画出质量—成本曲线；只比较最高分会把 compute scaling 混进模型差异。
</details>

## 2. Kimi K3 报告给 Stage 6 的启示

Kimi K3 官方材料明确绑定 reasoning effort、采样设置、重复次数和具体 harness。其价值不只是某个分数，而是提醒我们：agentic/reasoning 模型的“模型”已经包含推理协议。

本项目不加载 K3 权重，也不复现大规模 benchmark；只把其评测披露方式映射为本地 Manifest：generation settings、模型 revision、数据哈希和重复运行不可省略。

<details>
<summary>思考题：引用 K3 的评测协议是否意味着 ForgeLLM 与 K3 可比较？</summary>

不意味着。方法论可以借鉴，模型规模、任务、工具和预算完全不同。Stage 6 只声称实现了可追踪的本地协议，不声称达到前沿模型评测覆盖度。
</details>

## 3. LiveBench 与动态题的意义

静态公开 benchmark 容易随着时间进入训练语料。LiveBench 的核心动机是持续加入新题、用客观答案评分并标注时间，从而降低长期污染风险。

动态并非万能：新题仍需质量控制，版本变化也使跨时间分数更难直接比较。正确做法是同时保存 benchmark release/date，而不是只写名称。

<details>
<summary>思考题：动态 benchmark 为什么仍需保存题目哈希？</summary>

“LiveBench”是不断变化的集合。没有 release 或哈希，两次同名运行可能用不同题，差异无法归因给模型。
</details>

## 4. lm-evaluation-harness 的可复现思想

主流 harness 强调任务配置、模型参数、chat template、Tokenizer、few-shot、generation kwargs 和 decontamination 记录。Harness 的价值是统一接口与复现元数据，不是自动保证任务正确。

自定义任务仍可能有答案提取 bug、错误 stop token、数据版本漂移或模板泄漏。Stage 6 用更小的自研实现逐项理解这些边界，未来接入 harness 时也必须做小样例人工对照。

<details>
<summary>思考题：使用知名 harness 后为何仍要保存原始输出？</summary>

指标实现、答案提取器或依赖版本可能有 bug。原始输出让我们能在不重新推理的情况下重算评分、定位截断和审计错误；只有汇总分无法完成这些工作。
</details>

## 5. DeepSeek V4 的“待核验”是一种能力

截至本阶段 F0 审计，没有找到可归因训练技术的 DeepSeek 官方 V4 技术报告。第三方推理框架出现模型别名，只能说明生态可能预留接口，不能证明模型架构或后训练方法。

研究中的成熟做法是登记“待核验”，而不是用传闻补齐技术路线。官方材料发布后，再创建新的 registry revision；不要回写成“我们早已知道”。

<details>
<summary>思考题：为什么“不知道”也应写入实验记录？</summary>

它明确证据边界，防止后续文档把猜测当事实，也让未来更新知道需要寻找什么来源。可审计的不确定性优于无法追踪的自信断言。
</details>

## 6. 失败运行为什么不能删除

Stage 6 v1 在多轮案例上触发消息交替错误。失败 Artifact 保存了：异常类型、消息、完整 traceback，以及已经写出的 Manifest。v2 修复后使用新数据目录和输出目录。

保留失败有三项价值：

- 证明改动是由可复现问题驱动；
- 防止只呈现成功路径的幸存者偏差；
- 为多轮消息 Schema 增加回归测试。

<details>
<summary>思考题：失败发生在第 8 题，前 7 题的输出能否和 v2 拼接？</summary>

不能。v1 与 v2 的案例文件哈希不同，属于不同 run identity。即使前几题文本碰巧相同，拼接也破坏 Manifest 绑定。可把 v1 当失败诊断证据，但正式汇总必须来自完整 v2。
</details>

## 7. 结论分级

推荐用四级语言：

- **证明实现正确性**：单元测试、手算、张量对齐；
- **证明本地流程可运行**：真实权重、真实输出、完整 Artifact；
- **在冻结样本上观察到差异**：点估计与区间支持；
- **支持广泛能力结论**：需要代表性、多数据集、外部复现和更强统计证据。

ForgeLLM Stage 6 主要达到前三级的一部分；不宣称第四级。

<details>
<summary>思考题：Q3 完成 16 条 rollout 和一步更新属于哪一级？</summary>

属于“本地流程可运行”，再加上部分实现正确性测试。一次更新没有足够训练预算、对照和重复，不能进入“能力改善”级别。
</details>

## 8. 你应该怎样写最终摘要

使用以下模板：

```text
在 [数据哈希/案例数] 与 [生成协议] 下，
[模型身份] 在 [指标及区间] 上呈现 [观察]。
[另一指标] 显示 [冲突/限制]。
证据位于 [路径]。
该结论仅覆盖 [边界]，不支持 [过度外推]。
```

避免“全面提升”“达到主流水平”“已对齐”等没有覆盖范围的词。

<details>
<summary>思考题：如果所有自动门通过，能否写“模型已通过最终验收”？</summary>

只能写“Stage 6 自动化门通过”，并单列哪些模型行为门通过。学习者盲评与 G6-L 尚未完成时，完整 Stage 6 仍是 pending；本地门通过也不等于外部通用能力验收。
</details>

## 9. 本讲义使用的主要公开材料

- [Kimi K3 官方仓库与技术报告入口](https://github.com/MoonshotAI/Kimi-K3)：用于核验 reasoning effort、sampling/repeat 与 harness 披露；
- [LiveBench 论文](https://arxiv.org/abs/2406.19314)：用于理解动态题、客观评分与污染时间边界；
- [lm-evaluation-harness Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)：用于任务配置、指标和去污染记录；
- [lm-evaluation-harness Model Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)：用于模型参数、Tokenizer 与 Chat Template 记录。

这些来源支持评测方法论，不支持把 ForgeLLM 的本地结果与对应前沿模型横向排名。
