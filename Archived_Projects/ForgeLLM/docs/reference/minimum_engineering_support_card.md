# ForgeLLM 最低工程支撑执行卡

> 用途：模型优先路线中的快速查阅资料。这里的内容只要求“会用、会定位、会排错”，不要求在进入 Tokenizer 前逐项深入学习。  
> 当前主计划：`docs/model_first_21_week_learning_plan.md`

## 1. 开始任何任务前

在仓库根目录确认：

```powershell
git status --short
git branch --show-current
.\.venv\Scripts\python.exe -m forgellm doctor
```

需要知道：

- 当前修改不一定属于本次任务，不能覆盖或删除；
- `.venv` 是当前项目环境；
- `doctor` 只证明包入口和最小运行信息可用，不证明模型正确。

## 2. 权威质量命令

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
```

执行顺序：

```text
Ruff format check
→ Ruff lint
→ mypy strict
→ 全量 pytest
```

首个失败会停止。修复顺序按最早失败项，不要同时猜测多个原因。

只运行相关测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit -q
.\.venv\Scripts\python.exe -m pytest tests\integration -q
.\.venv\Scripts\python.exe -m pytest tests\smoke -q
```

## 3. 新增模型模块时的最低结构

每个核心模块至少包含：

```text
src/forgellm/<area>/<module>.py
tests/unit/test_<module>.py
config（存在可调参数时）
docs/experiments/<date>_<stage>_<run>.md（运行实验时）
```

模型数学模块优先写成不依赖 CLI 的纯 Python/PyTorch 类或函数。CLI 只负责参数解析、选择用例、输出结果和错误码。

## 4. 配置最低规则

- 正式运行参数必须能保存；
- 未知字段默认拒绝；
- 整数配置显式排除 `bool`；
- 在创建输出或启动训练前完成校验；
- 保存程序实际使用的 resolved config；
- 同一实验中途不能偷偷修改参数。

配置错误排查：

```text
文件是否存在
→ TOML 是否能解析
→ 顶层 table 是否正确
→ 是否有 unknown/missing 字段
→ 类型是否正确
→ 范围/跨字段约束是否满足
```

## 5. 模型正确性测试优先级

一般工程覆盖率不是当前主目标。优先测试会改变模型结论的行为：

1. shape；
2. dtype/device；
3. 数值参考对照；
4. 梯度存在且有限；
5. 因果性；
6. Tiny Overfit；
7. 保存/加载；
8. Checkpoint/Resume；
9. 失败输入与 NaN/Inf；
10. 固定输入回归。

测试通过只能证明已编码的行为，不等于模型具备通用能力。

## 6. 数据的最低安全边界

进入 Tokenizer/训练前必须确认：

- 来源和许可证基本可用；
- train/validation/test 分离；
- test 不参与词表学习和调参；
- UTF-8/Unicode 策略固定；
- 精确重复不会跨 split 泄漏；
- 样本数、字节数、输入/配置哈希有记录；
- 未确认的大数据不直接下载；
- “规则未发现风险”不能写成“风险不存在”。

近似去重、复杂 PII 平台和多格式 Reader 不是当前 Tokenizer 的默认前置门槛。

## 7. 每次训练运行最少记录

- 目标和唯一主变量；
- 模型、训练、数据、Tokenizer 配置；
- Git commit 与 dirty 状态；
- Python/PyTorch/CUDA/GPU；
- seed、dtype、batch、sequence length；
- 精确命令；
- train/validation 指标；
- tokens/s、显存、时间和费用；
- checkpoint 和日志路径；
- 失败、判断、处理和下一步；
- 该运行能证明什么、不能证明什么。

## 8. Checkpoint 最低内容

预训练阶段至少保存：

- model；
- optimizer；
- scheduler；
- step/tokens seen；
- Python/PyTorch/CUDA RNG；
- sampler 或 data cursor；
- AMP scaler（使用时）；
- model/train/data/tokenizer 配置哈希；
- schema version。

“权重能加载”不等于“训练能等价恢复”。必须比较连续训练与中断恢复的下一批数据、loss、LR 和参数。

## 9. 支撑知识何时需要补学

只有出现具体触发信号时才深入：

| 触发信号 | 定向补学 |
|---|---|
| 包无法导入/入口不一致 | editable install、`src` layout、console script |
| 配置被忽略/解析失败 | TOML、Schema、fail-fast |
| 测试不稳定 | fixture、seed、临时目录、确定性 |
| 训练结果无法定位 | Run ID、配置/环境/Git 快照 |
| Linux/云端运行失败 | CI、平台差异、依赖构建 |
| 数据指标异常 | Unicode、去重、split、Manifest |
| 日志泄露或不可分析 | 结构化日志、敏感字段规则 |

补学目标是解决当前问题，默认时间盒为 30–120 分钟，不重新执行完整 14 天课程。

## 10. 当前不需要深入的内容

- CLI 框架扩展；
- 复杂配置库；
- Docker/Kubernetes；
- 数据库、消息队列；
- 全量 CI 平台治理；
- 完整数据湖/数据平台；
- 服务监控和自动扩缩容；
- 为“以后可能需要”提前抽象。

## 11. 三个快速自检

### Q1

新增 RMSNorm 时，应先写 CLI 命令还是独立 PyTorch 模块？

<details>
<summary>查看答案</summary>

先写独立 PyTorch 模块和数值/梯度测试。只有确实需要给用户提供运行入口时才增加 CLI；数学正确性不应依赖 CLI。

</details>

### Q2

全量 pytest 通过是否能证明预训练模型正确？

<details>
<summary>查看答案</summary>

不能。它只证明当前测试覆盖的行为通过。模型仍需要因果性、参考数值、梯度、Tiny Overfit、训练/验证、Resume 和正式运行证据。

</details>

### Q3

工程知识什么时候从执行卡升级为详细讲义？

<details>
<summary>查看答案</summary>

当真实任务暴露具体缺口、该缺口阻塞核心阶段，或者用户明确要求深入学习时。补学应针对具体问题，不能自动恢复整套工程课程。

</details>
