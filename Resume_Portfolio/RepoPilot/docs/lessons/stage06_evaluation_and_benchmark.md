# Stage 6 完整讲义：Agent Evaluation、Benchmark、指标与消融

> 本阶段不再增加 Agent 功能，而是学习“现有证据究竟允许我们说什么”。  
> 建议学习时间：12–18 小时。  
> 当前边界：10 个可信微任务和离线 Runtime 已验证；正式本地 Qwen、外部 hidden tests、Docker 实跑尚未形成结果。

## 0. 为什么“看起来不错”不是评测

模型最终可能写：

```text
I fixed the issue, all tests pass, and the code is production-ready.
```

这段话不能证明：

- 文件真的修改了；
- 修改的是允许文件；
- 测试真的运行了；
- 测试退出码是 0；
- 没有删测试；
- 没有破坏原有功能；
- 过程没有越权；
- 结果能在新工作区复现。

因此 Agent Eval 的证据优先级是：

```text
真实 Outcome
>
Tool / State Trajectory
>
Final Natural-language Text
```

## 1. Task → Trial → Trace → Outcome → Grader

### 1.1 Task

固定目标和环境契约，例如“修复 subtract 返回加法的问题”。Task 不是一句随意 Prompt，还包含工作区、
路径、测试和预算。

### 1.2 Trial

某个模型、Prompt、工具、预算在某个 Task 上的一次独立运行。即使 Task 相同，随机模型的多次 Trial
也可能结果不同。

### 1.3 Trace

模型调用、ToolCall、Policy、ToolResult、Checkpoint 和状态转换的完整轨迹。

### 1.4 Outcome

环境最终发生的事实：测试是否通过、哪些文件改变、是否超预算、是否被安全策略阻止。

### 1.5 Grader

把 Outcome 转为评分。优先级：

```text
Deterministic Grader > LLM Judge > Human
```

不是说人最差，而是能用确定规则回答的问题，不应先交给更昂贵、更不稳定的主观判断。

## 2. PublicTaskSpec 与 EvaluatorTaskSpec

实现位于 [`task.py`](../../src/repopilot/task.py)。

### 2.1 PublicTaskSpec

模型可见或 Runtime 可投影的字段：

```text
task_id
workspace
problem_statement
language
allowed_paths / forbidden_paths
visible_tests
setup_command / test_command
network_policy
max_changed_files
budget
trusted_fixture
```

### 2.2 EvaluatorTaskSpec

只给独立评测器：

```text
public
hidden_tests
gold_patch_sha256
```

### 2.3 为什么 Gold Patch 不给 Agent

Gold Patch 是参考答案或参考修改。Agent 能读取它就不是解决任务，而是复制答案。

### 2.4 当前项目的真实边界

当前 10 个微任务提交了 `verify.py`，它们属于可见开发检查，不是 hidden tests。代码已实现
HiddenTestGrader 和数据契约隔离，但正式独立 hidden suite 尚未建设。因此现在不能报告“隐藏测试通过率”。

## 3. DeterministicVerifier 怎样验收

实现位于 [`verification/verifier.py`](../../src/repopilot/verification/verifier.py)。

流程：

```text
读取当前文本快照
→ 与 baseline 比较 changed files
→ 没有修改：失败，可恢复
→ 超过 max_changed_files：失败
→ 没有固定 test_command：失败
→ 在 Runner 中执行测试
→ exit_code == 0 且未 timeout：Passed
```

### 为什么“没有修改”会失败

当前任务类型是 Bug Fix。模型只解释答案但不改变仓库，不算完成。其他任务类型如 Documentation Query
可能允许零修改，需要不同 Verifier；不要把当前规则错误推广到所有 Agent。

### 为什么 exit code 很重要

命令约定：

```text
0      成功
非 0   失败或异常
```

只检查 stdout 是否包含 `passed` 很危险，恶意脚本可以打印 passed 后退出 1。

## 4. FAIL_TO_PASS 与 PASS_TO_PASS

### FAIL_TO_PASS

修复前失败、修复后通过。它证明目标 Bug 得到修复。

### PASS_TO_PASS

修复前通过、修复后仍通过。它帮助发现回归。

一个 Agent 可能让目标测试通过，但破坏 20 个原有测试。因此真实代码 Benchmark 要同时报告两者。
当前微任务极小，主要展示 fail-to-pass 机制，尚不能代表完整回归能力。

## 5. EvaluationRecord 字段

[`evaluation/metrics.py`](../../src/repopilot/evaluation/metrics.py) 的记录包含：

```text
task_id
status
hidden_tests_passed
iterations
tool_calls
input_tokens
output_tokens
wall_seconds
changed_files
security_blocks
```

一条 Record 对应一个 Task Trial。不要在 Record 中只保存一个总分，否则以后无法分析成本、失败模式和
安全性。

## 6. Resolve Rate

假设 10 个任务中 6 个状态为 COMPLETED：

```text
Resolve Rate = resolved / tasks = 6 / 10 = 0.6 = 60%
```

RepoPilot 的 `summarize()` 只把 RunStatus.COMPLETED 计为 resolved。

### 60% 并不完整

还需要说明：

- 哪 10 个任务；
- 是否看过答案；
- 模型和量化；
- 每题是否只跑一次；
- 工具、Prompt、预算；
- 是否 hidden；
- 是否有安全违规；
- 置信区间。

小样本中 6/10 的不确定性很大，不能写成“模型真实成功率就是 60%”。

## 7. 平均成本和效率指标

若两个任务 Token 分别为 100 和 300：

```text
average_tokens = (100 + 300) / 2 = 200
```

类似指标：

- average_iterations；
- average_tool_calls；
- average_wall_seconds；
- input/output tokens；
- first-attempt success；
- timeout rate；
- recovery rate。

平均值可能被极端任务影响，正式报告可同时给中位数、分位数和逐题原始记录。

## 8. Retrieval Metrics：定位要单独评测

端到端失败可能来自：

```text
检索没找到文件
模型理解错
Patch 失败
测试环境坏
预算太小
```

若不单独评测检索，就无法知道该优化哪一层。

### 8.1 File Recall@k

定义：前 k 个结果找回了多少相关文件。

例子：相关文件为 `{a.py, b.py}`，Top-3 为 `[x.py, a.py, y.py]`：

```text
Recall@3 = 找到的相关文件数 / 相关文件总数
         = 1 / 2
         = 0.5
```

### 8.2 MRR

Mean Reciprocal Rank，平均倒数排名。每个任务只看第一个相关结果：

```text
相关文件第 1 名 → reciprocal rank = 1/1 = 1
相关文件第 2 名 → 1/2 = 0.5
相关文件第 5 名 → 1/5 = 0.2
没有找到       → 0
```

两个任务分别是第 2 名和第 1 名：

```text
MRR = (0.5 + 1.0) / 2 = 0.75
```

### Recall@k 与 MRR 关注不同问题

- Recall@k：在有限 Context 内找回多少相关文件；
- MRR：第一个相关文件出现得多早。

## 9. 当前 10 个微任务

Manifest：[`evaluation/micro_benchmark.yaml`](../../evaluation/micro_benchmark.yaml)。任务包括：

```text
subtract
divide
even
clamp
slugify
normalize
safe_get
average
starts_with
absolute
```

每个任务：

- 只有一个允许修改的 `module.py`；
- `verify.py` 被禁止修改；
- 固定 test_command；
- 最多改一个文件；
- 固定迭代、工具、Token、时间预算；
- `trusted_fixture: true`。

### 为什么保留这么简单的题

它们适合验证：

- Task Spec 加载；
- fail-to-pass；
- 路径限制；
- 模型格式遵循；
- Tool Loop；
- Artifact；
- 检索指标代码。

它们不适合证明：

- 多文件架构理解；
- 复杂依赖；
- 大仓库检索；
- 长期规划；
- 通用代码 Agent 能力。

## 10. 10/10 BM25 Top-1 应怎样表述

合格表述：

> 在 `repopilot_python_micro_v1` 的 10 个单文件允许路径任务中，权限过滤后的 BM25 对任务描述检索
> 10/10 将 `module.py` 排在 Top-1。

不合格表述：

> RepoPilot 检索准确率 100%，已经解决代码定位。

前者说明数据集、范围和指标；后者把极小结果推广到所有代码库。

## 11. Ablation：每次只改变一个主变量

Ablation（消融）用于回答某个组件是否真的有贡献。

示例：

| Variant | Model | Tools | Budget | 唯一变化 |
|---|---|---|---|---|
| A | 固定 Qwen | 基础 Tools | 固定 | 无 BM25 |
| B | 同一个 | 同一组 + retrieve_code | 同一个 | 加 BM25 |

必须固定：

- Task IDs；
- 模型 checkpoint/量化；
- Prompt 版本；
- 除主变量外的 Tools；
- Token/iteration/time 预算；
- 推理参数；
- Trial 数；
- Grader。

[`evaluation/report.py`](../../src/repopilot/evaluation/report.py) 会拒绝不同 Variant 使用不同 task_id 集合。

### 不公平对比示例

```text
基础 Agent：3B 模型、8k Token、5 轮
RAG Agent：7B 模型、32k Token、20 轮
```

即使 RAG Agent 更好，也无法知道收益来自 RAG、模型大小还是预算。

## 12. Baseline 设计

建议逐步比较：

```text
B0：模型直接输出答案，不允许工具
B1：基础 ReAct + read/search/patch/test
B2：B1 + BM25 RAG
B3：B2 + Skill
B4：B3 + Planner
B5：B4 + Reviewer
```

每一步只增加一个关键组件。若 Resolve Rate 相近但 Token 增加很多，新组件可能不值得默认开启。

## 13. 安全指标

Agent 不应只按成功率排名。至少记录：

- Policy Block Rate；
- forbidden-path attempt；
- secret leakage；
- network violation；
- arbitrary command attempt；
- test tampering；
- sandbox escape test；
- approval frequency。

一个 Agent 成功率高但经常尝试读取秘密，不能直接投入使用。

### Policy Block Rate 的两面性

过低可能代表 Policy 没检测；过高可能代表工具设计或 Prompt 让正常动作频繁被阻止。指标需要结合任务
和人工审计解释，不能简单认为越高越安全。

## 14. Trajectory Evaluation

即使最终通过，也应检查关键轨迹：

- 是否读取 forbidden 文件；
- 是否重复相同失败；
- 是否修改无关文件；
- 是否在测试前后偷偷改变测试；
- 是否从 Journal 正确回放；
- 是否发生 Tool 报失败但留下副作用；
- 是否超预算后仍继续。

本项目曾出现“Patch 实际成功、ToolResult 却失败、Verifier 仍通过”。这正说明 Outcome 与 Trajectory 都
重要，优先级不是“只看 Outcome、完全忽略过程”。

## 15. LLM Judge 什么时候使用

适合：

- 代码可读性；
- 解释是否清晰；
- 需求语义是否大致满足；
- 两个合理实现的偏好比较。

不适合替代：

- 测试是否退出 0；
- 文件是否越界；
- JSON 是否符合 Schema；
- 是否存在指定函数；
- 资源是否超限。

若使用 Judge，应记录 Judge 模型、Prompt、位置交换、一致性和人工抽查。

## 16. 多次 Trial 与随机性

本地模型即使 temperature=0，也可能因后端、量化或硬件出现差异；temperature>0 时更明显。

更可靠的报告：

```text
每任务跑 N 次
→ 报 pass@1 / 成功比例
→ 保存每次随机种子与 Trace
→ 使用配对统计比较 Variant
```

当前微型资格测试主要是确定性 ScriptedProvider，不是模型随机性实验。

## 17. 本地 Qwen 正式评测计划

在讲义加入真实模型结论前，应冻结：

### 模型身份

```text
model_id
revision / file SHA-256
Base/Instruct/Coder
parameter size
quantization
chat template
context limit
license/source
```

### 服务身份

```text
backend and version
GPU/driver/CUDA
tool-call mode
generation parameters
```

### Agent 身份

```text
Git commit
Prompt hash
Tool schemas
Skill version
budgets
sandbox policy
```

### 运行顺序

1. 在新复制的工作区运行每题；
2. 不修改原始 micro benchmark；
3. 先注册 Prompt，不边看结果边改；
4. 保存 raw Trace；
5. 运行独立 hidden grader；
6. 汇总结果和失败类别；
7. 失败案例进入 regression suite；
8. 只有完成以上步骤后写教学结论。

## 18. 动手实验

### 实验 A：验证 10 任务门禁

```powershell
python -m pytest -q tests\integration\test_micro_benchmark.py -vv
```

测试同时证明：

- 数量恰好 10；
- Task ID 不重复；
- 所有 Task Spec 可加载；
- 每题初始测试失败；
- 权限过滤后的 BM25 Top-1 是 module.py。

### 实验 B：手算 Summary

```powershell
python -m pytest -q tests\unit\test_evaluation.py::test_outcome_summary -vv
```

测试包含一个 Completed、一个 Failed，因此 Resolve Rate = 0.5，平均 Token = 120。

### 实验 C：手算 Retrieval Metrics

```powershell
python -m pytest -q tests\unit\test_evaluation.py::test_retrieval_metrics -vv
```

先手算 Recall@2 和 MRR，再看断言。

### 实验 D：错误消融

```powershell
python -m pytest -q tests\unit\test_evaluation.py::test_ablation_requires_identical_task_sets -vv
```

Base 使用 task a，RAG 使用 task b，系统拒绝比较。

## 19. 怎样写一段诚实结论

建议模板：

```text
在 <任务集版本>、<模型/服务版本>、<固定 Agent scaffold> 和 <预算> 下，
<Variant> 在 <N> 个任务、每题 <Trials> 次运行中达到 <Outcome 指标>。
相对 <Baseline>，<主指标变化>，同时 <Token/延迟/安全指标变化>。
结果只支持 <范围内结论>；不支持 <外推结论>。
主要失败类型为 <列表>，原始 Trace 位于 <路径>。
```

当前可以写：

> Stage 1–6 Runtime 在可信微型 Fixture 上完成离线机制资格验证，33 个自动测试通过，10 个微任务均满足
> 初始 fail-to-pass 和 BM25 Top-1 定位门禁。尚无真实本地 Qwen Resolve Rate、正式 hidden-test 或 Docker
> 敌对执行证据，因此不能声称模型能力或生产安全已经验收。

## 20. 常见误解

### “测试 33/33 通过，所以 10 个任务 Agent 都能解”

错误。33 个测试验证 Runtime 机制；没有真实模型的 10 题 Trial，就没有模型 Resolve Rate。

### “隐藏测试类已经实现，所以隐藏测试已通过”

错误。代码接口存在不等于数据集和正式运行存在。

### “平均 Token 更低，所以 Agent 更好”

错误。若成功率也大幅下降，成本低可能只是更早失败。必须联合看 Outcome 和效率。

### “消融 B 比 A 高 10%，一定是组件有效”

不一定。样本小、随机性、任务不一致或 Prompt 偷改都可能造成差异，需要固定变量和统计不确定性。

### “LLM Judge 说正确，所以不需要运行代码”

错误。能执行确定性测试时，先运行测试。

## 21. 思考题与答案

<details>
<summary>问题 1：为什么 Eval 优先真实 Outcome，但仍记录 Trajectory？</summary>

Outcome 是最终用户价值的强证据；Trajectory 解释为什么成功/失败并检测越权、投机和隐藏副作用。
只看轨迹可能奖励“过程漂亮但没解决”，只看结果可能漏掉危险捷径。
</details>

<details>
<summary>问题 2：10 个任务中 10 个都通过，能否说真实成功率 100%？</summary>

只能说这 10 个登记 Trial 的观察结果是 10/10。任务可能太简单，样本很小，也可能有污染。真实总体
成功率存在不确定性，不能精确等于 100%。
</details>

<details>
<summary>问题 3：为什么不同模型必须使用相同 Agent Scaffold？</summary>

否则结果同时混合模型和工具/Prompt/预算差异，无法回答模型本身的贡献。若目标是比较“最佳系统”，
可以各自优化，但必须明确那是系统比较，不是纯模型比较。
</details>

<details>
<summary>问题 4：为什么失败案例要加入 Regression Suite？</summary>

优化新 Prompt 或 Tool 后，旧问题可能重新出现。把真实失败冻结为回归任务，才能防止“修新问题、复发
旧问题”。加入前仍要审查隐私、许可证和数据污染。
</details>

## 22. Stage 6 验收清单

- [ ] 画出 Task → Trial → Trace → Outcome → Grader；
- [ ] 区分 Public、visible、hidden 和 Gold Patch；
- [ ] 手算 Resolve Rate、average tokens、Recall@k、MRR；
- [ ] 区分 FAIL_TO_PASS 与 PASS_TO_PASS；
- [ ] 设计一个只改变单变量的消融；
- [ ] 说出至少五个安全指标；
- [ ] 运行 10 任务、Summary、Retrieval、Ablation 测试；
- [ ] 写出一段带范围和限制的结论；
- [ ] 明确当前没有真实 Qwen Resolve Rate 和 Docker 敌对执行证据。

## 23. Stage 1–6 最终口述题

不看讲义，用 10 分钟讲清楚：

```text
Task Spec 如何进入 Context
→ 模型如何提出 ToolCall
→ Policy/Tool/Sandbox 如何执行
→ ToolResult 如何更新 State
→ Checkpoint/Journal 如何恢复
→ Verifier 如何完成任务
→ Eval 如何比较不同 Scaffold
```

如果你能沿真实文件指出每个对象，并能说出当前证据边界，就已经完成 Stage 1–6 的第一轮学习。
