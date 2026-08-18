# Stage 0 完整讲义：从工程优先切换到模型优先

> 学习模式：完整讲义（用户明确要求）  
> 生效日期：2026-07-24（Asia/Singapore）  
> 当前主计划：[`model_first_21_week_learning_plan.md`](../model_first_21_week_learning_plan.md)  
> 阶段计划：[`learning_stage_00_model_first_transition.md`](../stages/learning_stage_00_model_first_transition.md)

## 0. 这节课要解决什么

Stage 0 不教授一个新模型算法，而是修正“应该把有限学习时间花在哪里”。

前三天已经学习了项目范围、Python 包/依赖、CLI、配置 Schema 和 fail-fast。继续沿旧路线逐日深入 Git 快照、日志、CI、Manifest、Unicode、过滤、数据流水线和实验文档，会让 Tokenizer、Transformer 和训练闭环被显著推迟。

新的目标不是变成“不讲工程”，而是建立三层知识结构：

```text
第一层：核心算法——系统学习并亲手实现
第二层：模型正确性工程——必须掌握并验证
第三层：一般支撑工程——够用、可查、按问题补学
```

完成本讲义后，你应能回答：

- 为什么旧路线需要修订；
- 哪些工程知识可以后移，哪些不能删除；
- 如何判断一个主题应该用完整讲义还是执行卡；
- 如何用阶段门而不是“看完课程”判断进度；
- 为什么评测必须在训练前设计；
- 下一阶段为什么是 Tokenizer，而不是继续学 Day 4–14。

## 1. 路线优化不是降低标准

### 1.1 旧路线的问题

旧路线隐含的前置链是：

```text
Day 1–14 工程与数据全部深入学习
→ 正式数据审计
→ Tokenizer
→ Transformer
→ 预训练
→ 后训练
```

该路线在工程项目中很稳健，但不完全符合当前目标岗位：LLM 算法/训练。

主要问题有四个：

1. **学习与实现重复。** Day 1–10 的工程/数据代码和测试已经存在，再用 14 天逐行掌握全部基础设施，收益下降。
2. **支撑知识挤占核心知识。** 包、CLI、日志和 CI 有价值，但它们不是 Decoder、预训练或后训练的数学核心。
3. **按天完成会制造虚假前置。** 不理解某个 Git 快照细节，不应该阻止学习 BPE；真正阻塞 BPE 的是文本、词表、merge 和评测知识。
4. **目标反馈过晚。** 如果几周后才开始模型，无法尽早暴露 PyTorch、数学、显存和训练方面的真实薄弱点。

### 1.2 新路线保留了什么

新路线不会取消：

- 独立 Python 环境；
- 可运行的质量命令；
- train/validation/test 隔离；
- 模型数值、梯度和因果性测试；
- Checkpoint/Resume；
- 数据与 Tokenizer 版本；
- 实验配置、指标、硬件和成本记录；
- 失败实验与结论边界。

这些内容直接影响“模型是否真的正确”，属于第二层的模型正确性工程。

### 1.3 新路线后移了什么

以下内容默认改为按需查阅：

- CLI 框架扩展；
- TOML Schema 的全部边界；
- Git diff 哈希实现细节；
- 结构化日志框架设计；
- 完整 CI 平台治理；
- Docker/Kubernetes；
- 多格式 Reader；
- 复杂数据治理和服务监控。

“后移”不等于“永远不学”。当真实模型任务遇到相应问题时，再学习具体需要的部分。

## 2. 三层知识结构

## 2.1 第一层：核心算法

这些知识构成当前主线，默认使用完整讲义：

- Tokenizer 和 BPE；
- PyTorch Tensor/Autograd/Module；
- RMSNorm、RoPE、SwiGLU、Attention、GQA；
- Decoder-only Transformer；
- 语言模型 loss 和生成；
- optimizer、scheduler、混合精度；
- Checkpoint/Resume 的训练状态；
- C++/CUDA 自定义算子；
- SFT、LoRA、QLoRA、DPO；
- Policy Gradient、PPO、GRPO；
- 模型评测与验收。

核心算法的掌握标准：

```text
能解释
→ 能推导关键关系
→ 能亲手实现
→ 能写正确性测试
→ 能运行受控实验
→ 能解释指标和失败
→ 能说明结论边界
```

## 2.2 第二层：模型正确性工程

这部分虽然带有工程性质，但不能降级：

- shape、dtype、device；
- 参考实现数值对齐；
- gradient check；
- causal leakage test；
- Tiny Overfit；
- 固定训练/验证划分；
- checkpoint schema；
- RNG 与 data cursor 恢复；
- NaN/Inf 与 OOM 处理；
- 固定 baseline 和实验主变量；
- 指标分母、硬件和重复次数。

原因很简单：没有这些证据，即使 loss 下降或生成了合理句子，也无法判断代码是否正确。

## 2.3 第三层：一般支撑工程

这部分默认使用执行卡：

- 安装和环境；
- CLI；
- 配置文件；
- Git/CI；
- 日志；
- 普通数据处理；
- 文档结构。

掌握标准不是闭卷解释全部源码，而是：

- 会执行当前需要的命令；
- 能找到权威文件；
- 能读懂明确错误；
- 能按排查顺序定位；
- 需要时能在 30–120 分钟内定向补学。

## 3. 如何决定学习深度

以后遇到新主题，依次问五个问题：

```text
1. 它是否直接决定模型数学或训练行为？
2. 错误是否会让模型结论失真？
3. 求职面试是否要求独立解释或实现？
4. 当前 Stage 是否必须使用它？
5. 是否已有成熟工具，只需正确调用？
```

判断规则：

| 情况 | 教学方式 |
|---|---|
| 直接决定模型数学/训练，且需要实现 | 完整讲义 + 代码 + 测试 + 实验 |
| 会影响模型正确性，但已有实现 | 深入关键不变量和验证，不扩展平台细节 |
| 只是当前任务的工具 | 1–2 页执行卡 |
| 未来可能需要、当前无触发 | 记录到候选队列，不学习 |
| 用户明确要求深入 | 覆盖默认规则，生成完整讲义 |

### 示例一：RMSNorm

RMSNorm 影响模型数学，需要理解公式、shape、epsilon、dtype、梯度，并与参考实现对齐，因此属于核心完整讲义。

### 示例二：GitHub Actions 缓存

缓存只影响 CI 速度，不应影响模型数学。只要质量命令可以运行，当前无需深入缓存 key、跨平台恢复和缓存失效策略。

### 示例三：Checkpoint

Checkpoint 看似工程功能，但 Resume 不正确会重复/跳过训练数据、重置 optimizer 或改变 LR，直接破坏训练结论，因此属于模型正确性工程，必须深入。

## 4. Day 1–3 给后续留下了什么

### Day 1：边界意识

已经建立：

- 项目目标和资源有限；
- P0/P1/P2 不能混在一起；
- 结论必须有证据；
- 8 GB GPU 和 20 USD/月预算决定实验规模。

以后真正需要保留的是“范围和证据边界”，不是继续学习所有项目管理术语。

### Day 2：运行基础

已经建立：

- `.venv` 隔离；
- `src` 包结构；
- 两种 CLI 入口；
- 开发依赖和统一命令。

以后能正确使用环境、运行测试、定位 import 来源即可。

### Day 3：可信输入

已经建立：

- CLI 与核心逻辑分离；
- TOML 语法不等于业务 Schema；
- 配置应在昂贵任务前 fail-fast；
- resolved config 代表程序实际使用的值。

这些原则会直接用于模型和训练配置，但无需继续扩展通用配置平台。

## 5. 新路线的阶段门

新路线不再用“看完第几天”作为主要进度，而使用可验证 Gate。

| Gate | 证明的问题 |
|---|---|
| S0 | 路线切换完成，代码基线有效，旧学习门槛不再冲突 |
| G1 Tokenizer | 分词原理、实现、确定性与指标是否正确 |
| G2 Model/Systems | 模型前向、梯度、生成、系统 reference 与 C++/CUDA 自定义算子是否正确 |
| G3 Pretrain | 训练、验证、保存、恢复和指标是否形成闭环 |
| G4 Supervised Post-train | SFT/LoRA/QLoRA 是否有正确 mask、Adapter 和冻结前后评测 |
| G5 Preference/RL | RM/DPO/PG/PPO/GRPO 是否有可检查 loss、reward 和受控实验 |
| G6 Evaluation | 最终模型结论是否量化、可重复且不过度声明 |

阶段门的含义是：

```text
产物存在
+ 正确性测试通过
+ 实验指标满足预先门槛
+ 失败和局限有记录
= 才能进入下一阶段
```

“读完讲义”“代码文件存在”“跑出一条生成文本”都不能单独通过 Gate。

## 6. 评测为什么必须提前

如果先训练、后决定指标，会产生两个问题：

1. 容易只挑有利样例；
2. 容易在看到结果后修改成功标准。

所以每个 Stage 开始前要先固定：

- baseline；
- 数据和切分；
- 主指标；
- 辅助指标；
- 成功门槛；
- 失败处理；
- 最大时间、显存和费用。

例如 Tokenizer 不能只展示一段分词结果，至少要固定 round-trip、bytes/token、fertility、unknown rate 和特殊 Token 行为。

预训练不能只看 train loss，需要 validation loss、tokens seen、吞吐、显存和 Resume。

SFT/DPO/RL 不能只看“回答感觉更好”，还要检查格式遵循、固定任务、Base 能力回退、长度偏差、KL 和 reward hacking。

## 7. Stage 0 实际执行结果

### 7.1 路线文件

当前唯一主计划：

- [`model_first_21_week_learning_plan.md`](../model_first_21_week_learning_plan.md)

支撑执行卡：

- [`minimum_engineering_support_card.md`](../reference/minimum_engineering_support_card.md)

旧计划曾在路线切换时保留用于审计；当前模型优先路线和新手学习入口稳定后，这些容易误导学习顺序的过时计划已经删除。历史决策仍可由本讲义和 Stage 0 实验记录追溯。

### 7.2 质量门

实际执行：

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
```

结果：

- Ruff format：27 个文件已格式化；
- Ruff lint：通过；
- mypy strict：27 个源文件无问题；
- pytest：33/33 通过；
- 平台：Windows 11；
- Python：3.12.3。

该结果证明当前已有代码在当前环境和测试范围内有效。它不能证明：

- 正式 Tokenizer 已实现；
- Transformer 已实现；
- GPU 训练可运行；
- Linux CI 已通过；
- 模型质量达到任何水平。

## 8. 以后一次核心学习任务如何推进

核心模块采用以下循环：

```text
1. 先固定学习目标和成功门
2. 学习最小数学和输入/输出
3. 手写清晰参考实现
4. 写 shape/数值/梯度/失败测试
5. 与成熟框架或第二实现对照
6. 运行只改变一个主变量的实验
7. 分析指标、失败和资源
8. 口述解释与归档
```

文档和测试不是为了“看起来专业”，而是为了回答：

```text
我的实现和参考是否一致？
如果不一致，差异来自哪里？
结果对哪些输入、设备和精度成立？
一次实验究竟证明了什么？
```

## 9. 自定义 C++/CUDA 的正确位置

“PyTorch 没有现成模块”时，不要直接跳到 C++。

正确决策链：

```text
能否用 PyTorch Tensor 算子组合？
→ 能：先写 Python nn.Module/函数并验证数学
→ Profile 是否证明性能瓶颈，或是否必须接入外部内核？
→ 否：保留 Python 实现
→ 是：实现 C++ CPU 算子
→ 注册 Schema/Dispatcher/Autograd/FakeTensor
→ 实现 CUDA Kernel
→ opcheck/gradcheck/数值对照/benchmark
```

它安排在首个预训练基线之后，因为那时已经具备：

- 正确的 Python 参考；
- 真实输入 shape/dtype；
- 性能 Profile；
- 训练梯度需求；
- 可比较的速度基线。

否则容易优化一个并非瓶颈、甚至数学错误的算子。

## 10. 下一阶段：Tokenizer 的输入与退出条件

Stage 1 开始前只需：

- 项目质量门通过；
- 有可公开或许可明确的小型文本；
- train/validation/test 角色明确；
- 不使用 test 学习词表；
- Unicode/byte 基本术语从零讲解。

Stage 1 不需要提前完成：

- Docker；
- Linux GPU；
- 大规模数据平台；
- Transformer；
- 云端长训练；
- vLLM/FastAPI。

Tokenizer 的退出条件包括：

- 能手算 BPE；
- 手写训练/encode/decode；
- 保存加载一致；
- 输入顺序不影响确定性结果；
- round-trip 和特殊 Token 测试通过；
- 使用固定指标比较基线与工程 Tokenizer。

## 11. 思考题与对应答案

### Q1｜核心判断

为什么取消 Day 4–14 的逐日验收，不等于认为测试、数据和实验记录不重要？

<details>
<summary>查看参考答案与解析</summary>

取消的是“所有支撑知识都必须在进入模型前系统学习”的前置关系，不是取消正确性要求。与模型结论直接相关的测试、数据隔离、Checkpoint/Resume 和评测仍属于必须深入的第二层知识；一般 CLI、CI、日志细节改为按需学习。

</details>

### Q2｜分类

RMSNorm、GitHub Actions 缓存、Checkpoint/Resume 分别属于哪一层？

<details>
<summary>查看参考答案与解析</summary>

- RMSNorm：核心算法；
- GitHub Actions 缓存：一般支撑工程；
- Checkpoint/Resume：模型正确性工程。

Checkpoint 虽然是工程功能，却会直接影响 optimizer、LR、随机状态和数据位置，因此不能降级为普通查阅项。

</details>

### Q3｜证据边界

33/33 测试通过后，最强的合理结论是什么？

<details>
<summary>查看参考答案与解析</summary>

当前 Windows/Python 环境中，现有 33 项测试覆盖的包、CLI、配置、运行元数据和数据夹具行为按预期工作。不能推出 Tokenizer、Transformer、GPU 训练、Linux CI 或模型质量已经完成。

</details>

### Q4｜教学深度

什么时候支撑知识应该升级为详细讲义？

<details>
<summary>查看参考答案与解析</summary>

当真实任务暴露具体缺口且该缺口阻塞当前核心 Stage，或用户明确要求深入学习时。默认只补当前缺口，并设置 30–120 分钟时间盒，不自动恢复整套工程课程。

</details>

### Q5｜评测

为什么必须在训练前冻结 Baseline 和指标？

<details>
<summary>查看参考答案与解析</summary>

否则容易看到结果后修改成功标准、挑选有利样例或更换分母。预先固定 Baseline、数据、指标、门槛和失败条件，才能让实验可证伪并减少选择性报告。

</details>

### Q6｜Tokenizer

为什么 Tokenizer 仍需要最小数据隔离，而不能完全跳过数据知识？

<details>
<summary>查看参考答案与解析</summary>

Tokenizer 的词表是从数据学习出来的。如果 test 数据参与词表学习，就会产生评测泄漏；如果规范化、去重和语言分布不固定，不同 Tokenizer 的指标也不可比较。因此只压缩一般数据治理，保留与 Tokenizer 正确性直接相关的最小边界。

</details>

### Q7｜C++ 决策

PyTorch 没有 `MySpecialBlock` 类，是否应该直接写 C++？

<details>
<summary>查看参考答案与解析</summary>

不应该。先判断它能否由已有 Tensor 算子组合。如果能，应先用 Python `nn.Module` 实现并验证数学。只有必须接入外部 C/C++/CUDA、需要新的算子语义，或 Profile 证明性能瓶颈时，才进入自定义算子。

</details>

### Q8｜阶段门

“完成代码文件”和“通过阶段门”有什么差别？

<details>
<summary>查看参考答案与解析</summary>

代码文件只证明实现存在。阶段门还需要正确性测试、预先指标、实际实验、失败记录和结论边界。例如 Attention 文件存在不能证明没有因果泄漏，必须通过 causal test、数值和梯度对照。

</details>

### Q9｜资源

为什么不应为了学习 PPO/GRPO 立即启动大模型 RL？

<details>
<summary>查看参考答案与解析</summary>

当前只有约 8 GB 本地显存和每月 20 USD 外部预算。大模型 RL 同时需要 policy、reference、可能的 reward/critic 和采样资源。应先在小环境理解 Policy Gradient/PPO，在微型语言模型验证 loss 与梯度，再根据预算决定正式运行，避免将环境失败误当算法学习。

</details>

### Q10｜学习进度

为什么新路线用 Gate 而不是固定“每天学一章”？

<details>
<summary>查看参考答案与解析</summary>

核心模块的难度和调试时间差异很大。固定天数容易造成看完即完成或为了追进度跳过正确性。Gate 用可观察证据判断是否掌握，允许困难模块多花时间，也允许已具备的支撑知识快速通过。

</details>

### Q11｜过度优化

自定义 CUDA 算子输出正确但没有比 PyTorch 组合实现更快，是否算失败？

<details>
<summary>查看参考答案与解析</summary>

不一定。若目标是学习自定义算子闭环，正确性、Autograd、Dispatcher 和测量方法仍然形成有效证据。性能结论应写成“当前输入和硬件下未观察到加速”，并用 Profile 分析原因，不能删除结果或伪造提升。

</details>

### Q12｜下一步

Stage 0 之后为什么直接进入 Tokenizer，而不是正式数据平台或 Transformer？

<details>
<summary>查看参考答案与解析</summary>

Tokenizer 是原始文本与模型 Token ID 之间的输入契约，也是 Transformer `vocab_size`、特殊 Token、序列长度和训练指标的基础。当前已有最小数据夹具和流水线，足以开始原理实现；先冻结 Tokenizer，再构造模型，可以减少模型输入契约反复变化。

</details>

## 12. Stage 0 验收清单

- [x] 用户确认 Day 1–3 学习与测试完成；
- [x] 新模型优先 21 周计划已经建立；
- [x] 旧 Day 4–14 课程退出当前学习入口；路线稳定后已删除过时计划，历史决策证据继续保留；
- [x] Prompt 支持 core/support 两种教学深度；
- [x] 最低工程支撑执行卡已经建立；
- [x] Ruff format、lint、mypy、33 项 pytest 全部通过；
- [x] 下一阶段唯一指向 Tokenizer；
- [x] 未把任何尚未实现的模型能力标为完成。

## 13. 本日结论

Stage 0 的真正成果不是少学一些，而是建立了更严格的学习资源分配：

```text
对模型数学和训练结论负责的知识——深入
只用于支撑当前任务的知识——够用
未来可能需要但当前没有触发的知识——后移
```

从现在开始，ForgeLLM 的学习主线不再由 Day 4–14 阻塞。下一份核心完整讲义应是 Stage 1 的 Tokenizer 与 BPE，而不是继续扩展一般工程基础设施。
