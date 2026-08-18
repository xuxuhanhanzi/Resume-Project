# ForgeMM 完整项目实现计划

> 版本：v1.0  
> 冻结日期：2026-08-07  
> 项目定位：校招大模型算法 / 多模态后训练项目  
> 当前状态：独立项目与数据目录已建立；训练和正式实验尚未开始  
> 训练底座：Qwen2.5-VL-3B-Instruct  
> 训练框架：ms-swift 固定版本依赖，不复制或改名上游仓库  
> 数据：ChartQA + ChartQAPro  
> 主要算力：单张 RTX 4090 24 GB；本地仅要求可演示推理

## 1. 项目目标

ForgeMM 研究图表问答模型中的“答案正确但证据或计算过程错误”问题。项目不重新实现
通用训练框架，而是在 ms-swift 的成熟 SFT/GRPO 流程之上实现图表领域的差异化模块：

1. 将 ChartQA 转换为可验证的结构化推理数据；
2. 训练模型输出答案、证据和可执行运算；
3. 使用确定性 Verifier 分别评价答案、证据和运算一致性；
4. 实现受 Faithful GRPO 启发的动态约束式 Chart-FGRPO；
5. 在 ChartQA 冻结测试集和 ChartQAPro 外部测试集上进行严格对照和消融；
6. 输出可在普通本地显卡上进行 4-bit 推理演示的 LoRA Adapter。

一句话定位：

> 基于 Qwen2.5-VL-3B 与 ms-swift 构建图表可信推理后训练系统，通过结构化 QLoRA
> SFT、标准 GRPO 和动态约束式 Chart-FGRPO，提高答案、视觉证据和计算过程的一致性。

## 2. 非目标与主张边界

本项目不做以下主张：

- 不宣称从零实现训练框架；
- 不宣称提出通用 FGRPO；
- 不宣称完整复现原论文；
- 不宣称达到 SOTA；
- 不用训练 reward 代替冻结测试结果；
- 不用 ChartQA test 或 ChartQAPro 调参；
- 不把 ms-swift、Qwen2.5-VL 或公开数据集代码描述为个人实现。

统一表述为：

> FGRPO-inspired Chart-FGRPO / 受 FGRPO 启发的图表约束式 GRPO。

只有在多随机种子、冻结测试集和统计检验满足预登记门槛后，才能在简历中填写数值提升。

## 3. 主要依据

- Qwen2.5-VL：<https://github.com/QwenLM/Qwen2.5-VL>
- ms-swift：<https://github.com/modelscope/ms-swift>
- ms-swift GRPO：<https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html>
- Faithful GRPO，2026：<https://arxiv.org/abs/2604.08476>
- ChartQA：<https://github.com/vis-nlp/ChartQA>
- ChartQAPro：<https://github.com/vis-nlp/ChartQAPro>
- ChartQAPro 论文：<https://aclanthology.org/2025.findings-acl.978/>

方法借鉴边界：原 Faithful GRPO 主要面向视觉空间推理；ForgeMM 将“任务目标 + 约束”
思想迁移到图表问答，并使用表格、annotation 与可执行运算构建确定性 Verifier。

## 4. 研究问题与假设

核心问题：

> 在同一个 Structured SFT 起点上，动态约束式 Chart-FGRPO 是否优于只优化答案的
> 标准 GRPO，以及固定权重的多奖励 GRPO？

研究假设：相比标准 GRPO，Chart-FGRPO 应当：

1. 提高证据引用正确率；
2. 提高可执行运算一致性；
3. 提高答案、证据和运算同时正确的可信正确率 FCR；
4. 不显著损害 ChartQA relaxed accuracy；
5. 对 ChartQAPro 的外部分布表现产生可复核的正向影响。

若训练 reward 上升但冻结测试指标不改善，则假设不成立，必须记录负结果。

## 5. 总体架构

```text
ChartQA / ChartQAPro
        │
        ▼
Data Audit + EvidenceStore
        │
        ├── answer-only records
        ├── evidence-verifiable records
        └── operation-verifiable records
        │
        ▼
Structured SFT JSONL
        │
        ▼
ms-swift QLoRA SFT
        │
        ▼
共同 Structured SFT checkpoint
        │
        ├── E3: answer-only GRPO
        ├── E4: fixed multi-reward GRPO / ms-swift GDPO scaling
        └── E5: dynamic constrained Chart-FGRPO
        │
        ▼
Frozen Evaluation
        ├── ChartQA test
        ├── ChartQAPro
        ├── faithful metrics
        └── efficiency / local demo
```

责任边界：

| 模块 | ms-swift | ForgeMM |
|---|---:|---:|
| 模型加载与模板 | 是 | 配置与兼容测试 |
| QLoRA/SFT | 是 | 数据协议、配置、验收 |
| GRPO rollout 与 loss | 是 | 扩展 advantage 合成 |
| vLLM/权重同步 | 是 | 显存 smoke 与配置冻结 |
| ChartQA/ChartQAPro loader | 否 | 是 |
| EvidenceStore | 否 | 是 |
| Parser/Executor/Verifier | 否 | 是 |
| 三类 reward | 否 | 是 |
| 动态拉格朗日约束 | 否 | 是 |
| 统一评测与统计检验 | 否 | 是 |

## 6. 目标代码结构

```text
ForgeMM/
├── configs/
│   ├── data/
│   ├── sft/
│   ├── grpo/
│   └── evaluation/
├── datasets/
│   ├── ChartQA/
│   └── ChartQAPro/
├── docs/
│   ├── designs/
│   ├── experiments/
│   └── results/
├── scripts/
│   ├── audit_datasets.py
│   ├── build_evidence_store.py
│   ├── build_sft_dataset.py
│   ├── build_grpo_dataset.py
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── evaluate.py
├── src/forgemm/
│   ├── data/
│   │   ├── chartqa.py
│   │   ├── chartqapro.py
│   │   ├── evidence_store.py
│   │   └── schemas.py
│   ├── reasoning/
│   │   ├── parser.py
│   │   ├── executor.py
│   │   └── normalizer.py
│   ├── rewards/
│   │   ├── answer.py
│   │   ├── evidence.py
│   │   └── operation.py
│   ├── trainers/
│   │   ├── swift_adapter.py
│   │   ├── chart_fgrpo.py
│   │   └── dual_state.py
│   └── evaluation/
│       ├── chartqa_metrics.py
│       ├── faithful_metrics.py
│       └── statistics.py
└── tests/
    ├── unit/
    ├── integration/
    └── smoke/
```

## 7. 数据管理

### 7.1 已迁移数据

ChartQA：

```text
datasets/ChartQA/train
datasets/ChartQA/val
datasets/ChartQA/test
```

- 62,652 个文件；
- 1,036,819,201 字节；
- train/val/test 与源目录的文件数和字节数完全一致；
- 关键 annotation 的 SHA-256 已记录于 manifest。

ChartQAPro：

```text
datasets/ChartQAPro/chartqapro_test.parquet
```

- 209,486,545 字节；
- SHA-256：`6209a9a6f7307b761e70ff9e708cb7505e0327d7eb932aa26152b8240f633da5`；
- 1,948 条，六个字段无缺失；
- 前 20 张图片解码通过；
- 未迁移 Arrow 缓存和锁文件。

原始目录暂时保留，正式数据 smoke 通过前不删除。

### 7.2 数据用途冻结

| 数据 | 用途 | 是否训练 |
|---|---|---:|
| ChartQA train human | Structured SFT 与 GRPO | 是 |
| ChartQA train augmented | 审计后补充训练 | 是 |
| ChartQA val human | 开发、阈值和 checkpoint 选择 | 否 |
| ChartQA test human | 最终 ChartQA 测试 | 否 |
| ChartQAPro | 最终外部分布测试 | 否 |

### 7.3 EvidenceStore

EvidenceStore 将图片、CSV、annotation 和官方 QA 统一为带版本的证据记录：

```text
source identifiers
image sha256
question / reference answer
normalized cells or visual evidence
gold operation and operands
answer_type / task_type
evidence_mask / operation_mask
conflict status and exclusion reason
schema_version
```

可信样本必须满足：

1. split 来源明确且无跨分区泄漏；
2. 图像和必要表格存在；
3. operation 属于白名单；
4. executor 可以安全执行；
5. 执行结果能够复现官方答案；
6. CSV、annotation 和图片冲突已冻结处理；
7. 每条处理记录能够回溯源文件与哈希。

### 7.4 数据规模门槛

- Structured SFT：目标 4,000，最低 3,000；
- GRPO：目标 2,000 个可信 prompt，最低 1,500；
- ChartQA val 证据开发集：200；
- ChartQA test 证据测试集：300；
- ChartQA test human 答案评测：全量 1,250；
- ChartQAPro：全量 1,948。

可信样本不足时缩小 operation 范围，不降低验证标准，不使用测试数据补训练集。

## 8. 模型输出协议

统一输出：

```text
<evidence>
e1=cell(row="2015", column="Favorable", value=38)
e2=cell(row="2016", column="Favorable", value=43)
</evidence>
<operation>
subtract(ref=e2, ref=e1)
</operation>
<answer>
5
</answer>
```

初始 operation 白名单：

```text
lookup, equal, sum, difference, average, ratio,
product, median, count, argmax, argmin, compare
```

task type 和 constraint mask 由数据管道产生，模型不得自行声明，以免通过错误分类逃避约束。

## 9. Parser、Executor 与 Verifier

### 9.1 Parser

- 严格识别三个标签且每个只出现一次；
- 拒绝未闭合、重复、嵌套错误和标签外答案；
- 将 evidence 与 operation 转为类型化对象；
- 不使用 `eval`；
- 输出稳定错误码用于分析。

### 9.2 Executor

- 只执行白名单 operation；
- operand 必须引用已解析 evidence；
- 显式处理百分数、货币、负数、千分位和单位；
- 除零、空值、非数值和未定义引用返回失败；
- 数值比较使用与 ChartQA 指标一致的容差规则。

### 9.3 Reward

三个 reward 均限制在 `[0, 1]`：

- `R_task`：最终答案 exact/relaxed correctness；
- `R_evidence`：引用证据的 precision、recall、F1 与数值一致性；
- `R_operation`：operation 可执行且结果与最终答案一致。

格式解析失败时三个 reward 均为 0。对于无法可靠验证 evidence 或 operation 的题目，由
数据元信息 mask 对应约束，不允许模型自己生成 mask。

## 10. Chart-FGRPO

优化目标：

```text
maximize E[R_task]
subject to E[R_evidence] >= tau_evidence
           E[R_operation] >= tau_operation
```

三个 reward 分别进行组内 advantage 计算：

```text
A = A_task
  + lambda_evidence * A_evidence
  + lambda_operation * A_operation
```

乘子更新：

```text
lambda_k = clip(
    lambda_k + dual_lr * (tau_k - batch_mean_reward_k),
    0,
    lambda_max,
)
```

预登记初值：

| 参数 | 值 |
|---|---:|
| `tau_evidence` | 0.90 |
| `tau_operation` | 0.95 |
| `lambda_evidence_init` | 0.0 |
| `lambda_operation_init` | 0.0 |
| `dual_lr` | 0.01 |
| `lambda_max` | 5.0 |

乘子、阈值、reward 均值、constraint violation 和更新步数必须写入 checkpoint。断点恢复
必须恢复相同 dual state，不能只恢复模型权重。

## 11. ms-swift 依赖策略

### 11.1 原则

- 不 vendoring 整个 ms-swift；
- 不直接修改虚拟环境中的 site-packages；
- Stage 1 smoke 后固定稳定 tag、commit 和完整依赖锁；
- 自定义 reward 优先通过 `external_plugins`；
- Chart-FGRPO 优先通过 Trainer 子类或注册入口扩展；
- 若上游没有稳定 hook，只维护针对固定 commit 的最小 patch，并附测试和变更说明。

### 11.2 必须验证的接口

1. Qwen2.5-VL-3B 模板与图片处理；
2. 自定义数据额外字段是否透传到 reward；
3. 三个 reward 的独立日志；
4. ms-swift `scale_rewards=gdpo` 是否适合作为 E4；
5. advantage 合成的最小替换点；
6. dual state 的 checkpoint hook；
7. QLoRA 与 vLLM LoRA-only weight sync；
8. 单卡 4090 下 generation batch、offload 和显存占用。

## 12. 训练配置

### 12.1 Structured QLoRA SFT 初值

| 配置 | 值 |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct |
| Quantization | NF4 4-bit |
| Compute dtype | BF16 |
| LoRA targets | language q/k/v/o projections |
| Rank / alpha / dropout | 8 / 16 / 0 |
| ViT / aligner | frozen / frozen |
| max_pixels | 262144 |
| max_completion_length | 160 |
| learning rate | `5e-5` |
| epochs | 2 |
| warmup ratio | 0.03 |
| per-device batch | 1 |
| gradient accumulation | 16 |
| scheduler | cosine |

进入 RL 的门槛：

- Format Compliance >= 98%；
- Operation Executable Rate >= 90%；
- ChartQA val relaxed accuracy 不低于 answer-only SFT 超过 1 pp；
- Adapter、配置、训练数据和模型 revision 已冻结并记录哈希。

### 12.2 GRPO 公共初值

| 配置 | 值 |
|---|---|
| num_generations | 4；显存 smoke 可统一降为 2 |
| generation_batch_size | 8；依据 smoke 冻结 |
| temperature / top_p | 1.0 / 1.0 |
| max_completion_length | 160 |
| learning rate | `5e-7` |
| max_steps | 500 |
| gradient accumulation | 8 |
| clip epsilon | 0.2 |
| max_grad_norm | 0.5 |
| beta | 0.001；统一显存失败才全部改为 0 |
| loss type | grpo |

单张 4090 优先采用 QLoRA GRPO、vLLM colocate、LoRA-only weight sync、低
`gpu_memory_utilization` 和必要的 offload。显存参数只能依据 smoke 冻结，不得根据质量
结果为不同实验选择不同资源配置。

## 13. 实验矩阵

| 编号 | 方法 | 目的 |
|---|---|---|
| E0 | 原始 Qwen2.5-VL-3B-Instruct | 底座基线 |
| E1 | Answer-only QLoRA | 简单微调基线 |
| E2 | Structured QLoRA SFT | 全部 RL 的共同起点 |
| E3 | Task-only 标准 GRPO | 主要算法基线 |
| E4 | 固定权重多奖励 GRPO | 排除“只增加 reward”的解释 |
| E5 | 动态拉格朗日 Chart-FGRPO | 最终方法 |
| A1 | E5 去掉 evidence constraint | 证据约束消融 |
| A2 | E5 去掉 operation constraint | 运算约束消融 |

E3/E4/E5 必须固定：

- E2 checkpoint；
- 数据、顺序与随机种子；
- QLoRA、像素和 completion 预算；
- rollout 参数与训练步数；
- checkpoint 选择规则；
- 评测 prompt 和解析器版本。

正式种子为 `17 / 42 / 2026`。E3、E4、E5 跑三个种子；A1、A2 用种子 42。

## 14. 评价指标与成功门槛

定义：

```text
FCR = P(answer correct AND evidence pass AND operation pass)
```

| 指标 | 成功门槛 |
|---|---|
| Format Compliance | >= 98% |
| Truncation Rate | <= 1% |
| Evidence F1 | 相比 E3 提高至少 5 pp |
| Operation Consistency | 相比 E3 提高至少 5 pp |
| FCR | 相比 E3 提高至少 5 pp，paired bootstrap 95% CI 不跨 0 |
| Inconsistency Rate | 相对 E3 降低至少 30% |
| ChartQA Relaxed Accuracy | 不低于 E3 超过 1 pp |
| ChartQAPro Overall | 不低于 E3 超过 1 pp；提升至少 1 pp 才宣称 OOD 改善 |

同时报告：

- 三种子均值和标准差；
- paired bootstrap 95% CI；
- exact/relaxed 的逐样本改对与改错；
- McNemar 或 sign test；
- 按 operation、answer type、question type 分组；
- 峰值显存、GPU 时长、吞吐、输出长度和截断率。

## 15. 分阶段实施

### Stage 0：项目与环境基线

任务：

1. 初始化 Git，但不提交数据 payload；
2. 建立 Linux/Python 独立环境；
3.选择并锁定 ms-swift stable tag + commit；
4. 记录 CUDA、PyTorch、Transformers、vLLM、bitsandbytes 版本；
5. 编写环境审计脚本和第一份实验记录。

验收：干净环境可以依据锁文件重建；`git status` 不出现数据和模型大文件。

### Stage 1：ms-swift 最小可行性 smoke

按顺序完成：

1. Qwen2.5-VL-3B 加载和单图推理；
2. ChartQA 20 条基线推理；
3. 4 至 16 条样本的 QLoRA SFT；
4. `4 prompts × 2 rollouts` 标准 GRPO；
5. 额外字段到 reward plugin 的端到端透传；
6. advantage 扩展点与 checkpoint hook 原型。

验收：推理、SFT、GRPO 均成功，能够记录非恒定 reward，且确认 Chart-FGRPO 的最小修改
边界。未通过前不进行数据大规模构建和租卡长训练。

### Stage 2：数据审计与 loader

任务：

1. 实现统一路径解析；
2. 校验 manifest；
3. 实现 ChartQA 与 ChartQAPro loader；
4. 审计 split 泄漏、缺图、损坏图、表格冲突和重复样本；
5. 输出可追溯数据审计报告。

验收：全部样本可枚举，异常都有原因码，ChartQAPro 只作为 test 暴露。

### Stage 3：EvidenceStore 与 Verifier

任务：

1. 定义版本化 schema；
2. 构建受支持 operation 的可信样本；
3. 实现 parser、normalizer、executor；
4. 实现三个 reward；
5. 建立正常、边界和恶意格式单元测试。

验收：executor 不使用 `eval`；gold operation 可复现答案；随机抽查与自动审计一致。

### Stage 4：Structured SFT

任务：

1. 构建 3,000 至 4,000 条 SFT 数据；
2. 运行 32 样本过拟合测试；
3. 运行短训练并检查格式、截断、loss 和 val accuracy；
4. 完成 E1 与 E2；
5. 冻结共同 E2 checkpoint。

验收：满足第 12.1 节门槛；否则先修正数据和协议，不进入 RL。

### Stage 5：GRPO 与 Chart-FGRPO 工程实现

任务：

1. 用外部插件注册三类 reward；
2. 跑通 E3；
3. 以 ms-swift GDPO scaling 或等价实现完成 E4；
4. 实现 dual state 与 Chart-FGRPO advantage；
5. 验证 lambda 更新方向、clip、mask 和断点恢复；
6. 对 E3/E4/E5 使用相同 smoke 输入做数值对照。

验收：reward 不恒定，advantage 数值符合手算测试，E5 在约束不足时正确提高 lambda，恢复
训练后 dual trajectory 连续。

### Stage 6：快速实验与配置冻结

任务：

1. E3/E4/E5 各运行 50 至 100 steps；
2. 只根据稳定性、OOM、reward variance、KL、clip ratio 和截断率调整公共配置；
3. 冻结正式训练 YAML；
4. 预登记 checkpoint 选择和失败处理。

禁止依据 test/ChartQAPro 质量为某一个方法单独调参。

### Stage 7：正式实验

1. E3/E4/E5 × 3 seeds；
2. A1/A2 × 1 seed；
3. 每次运行保存环境、命令、配置、数据 hash、日志和 checkpoint；
4. 失败 run 不覆盖，另建记录解释重跑原因。

### Stage 8：冻结评测与统计分析

顺序：ChartQA val 选择 checkpoint → ChartQA test 一次正式评测 → ChartQAPro 一次外部评测。

输出总表、分组表、置信区间、显著性检验、改对/改错样例和错误类型报告。

### Stage 9：本地演示与交付

任务：

1. 合并或加载最佳 LoRA Adapter；
2. 基座 4-bit 量化，本地单样本推理；
3. 演示图表上传、结构化推理和 Verifier 结果；
4. 记录显存、首 token 延迟和总延迟；
5. 完善 README、架构图、复现命令和简历表述。

本地演示不承担训练，只要求可稳定展示最终模型能力。

## 16. 测试方案

### 单元测试

- schema 序列化与版本检查；
- 数值、百分数、单位和文本答案标准化；
- 标签解析的正常与异常输入；
- 每个 operation 的正确、边界和错误用例；
- answer/evidence/operation reward；
- mask 行为；
- group advantage 手算一致性；
- lambda 更新、clip 和 state serialization。

### 集成测试

- ChartQA → EvidenceStore → SFT JSONL；
- GRPO 额外字段 → reward plugin；
- completion → parser → executor → reward；
- ms-swift checkpoint → dual state 恢复；
- E3/E4/E5 在固定张量上的差异仅来自 advantage 合成。

### Smoke 测试

- CPU 数据构建；
- 单图基线推理；
- 32 样本 SFT；
- `4 × 2` GRPO rollout；
- 本地 4-bit Adapter 推理。

## 17. 实验记录与产物

每个 run 必须记录：

- 日期、目标和唯一主变量；
- GPU、驱动、CUDA、Python 与依赖；
- Git commit、ms-swift commit 和工作区 diff；
- 数据 manifest 与训练集 hash；
- 完整命令和配置；
- 日志、checkpoint 与结果路径；
- 峰值显存、耗时与失败原因；
- 结论和下一步。

运行命名：

```text
4090_<stage>_<method>_seed<seed>_<tag>
```

示例：

```text
4090_rl_e3_grpo_seed42_formal
4090_rl_e5_chart_fgrpo_seed42_formal
```

## 18. 风险与退路

| 风险 | 处理 |
|---|---|
| 4090 上 GRPO OOM | 统一降低 generations/batch/pixels，启用 offload；仍失败再换线上更大显存 |
| vLLM 与 ms-swift 版本冲突 | 固定官方兼容组合，不在正式实验中升级 |
| 额外字段被丢弃 | 在 Stage 1 建立透传测试，必要时用稳定 adapter 层 |
| Chart-FGRPO 无公开 hook | 针对固定 commit 维护最小 patch 和回归测试 |
| reward 大量零方差 | 检查题目难度和 rollout 温度，不临时增加算法分支 |
| evidence 噪声过高 | 缩小 operation 范围，保留 answer-only mask |
| Structured SFT 损害答案准确率 | 调整数据比例和格式，不进入 RL |
| E5 不优于 E3 | 如实报告负结果，保留 Verifier/数据系统的工程价值 |
| ChartQAPro 不提升 | 不宣称 OOD 改善，分析 question type 差异 |

## 19. 里程碑

| 周期 | 目标 | 交付物 |
|---|---|---|
| 第 1 周 | Stage 0–1 | 环境锁、ms-swift smoke、扩展点结论 |
| 第 2 周 | Stage 2 | loader、数据审计、manifest 报告 |
| 第 3 周 | Stage 3 | EvidenceStore、Verifier、完整单测 |
| 第 4 周 | Stage 4 | E1/E2、Structured SFT checkpoint |
| 第 5 周 | Stage 5–6 | E3/E4/E5 smoke、正式配置冻结 |
| 第 6–7 周 | Stage 7 | 三种子正式训练与消融 |
| 第 8 周 | Stage 8–9 | 冻结评测、本地演示、README 与简历材料 |

时间是执行顺序参考，不以赶进度为由跳过 smoke、数据冻结或对照实验。

## 20. 完成定义

只有满足以下条件，项目才算完成：

- 新环境能够依据锁文件重建；
- ms-swift 版本、commit 和扩展边界清晰；
- 两个数据集均有 manifest、loader 和审计报告；
- Structured SFT、E3、E4、E5 均有可复现记录；
- 三个 reward 与 Chart-FGRPO advantage 有自动化测试；
- E3/E4/E5 使用相同起点和公共配置；
- ChartQA test 与 ChartQAPro 保持冻结；
- 正式结果包含三种子统计与资源成本；
- 最佳 Adapter 可在本地 4-bit 演示；
- README 明确区分上游能力、个人实现和实验结论；
- 简历上的每个数值均能定位到配置、日志和结果表。

## 21. 可使用的简历表述

实现完成但正式结果未出时：

> 基于 Qwen2.5-VL-3B 与 ms-swift 构建图表可信推理后训练系统，自主实现 EvidenceStore、
> 可执行 Verifier、三目标 reward 及 FGRPO-inspired 动态约束 Trainer，并建立 ChartQA/
> ChartQAPro 的统一评测与消融流程。

只有成功门槛满足后，才替换为：

> 相比同起点标准 GRPO，Chart-FGRPO 在冻结测试集上将 FCR 提升 X pp、推理不一致率
> 降低 Y%，ChartQA relaxed accuracy 变化为 Z pp；最终 Adapter 支持本地 4-bit 推理。

## 22. 立即执行的下一步

只执行 Stage 0 与 Stage 1：先固定 ms-swift 候选版本，在 Linux/4090 环境依次跑通
Qwen2.5-VL 推理、微型 QLoRA SFT、标准 GRPO 和额外字段透传。完成兼容性报告后，再开始
EvidenceStore 和正式数据构建。

