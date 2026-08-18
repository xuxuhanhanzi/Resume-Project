# ForgeMM Chart-FGRPO 完整实验与实现方案

> 决策冻结日期：2026-08-07  
> 项目状态：方案已冻结，尚未开始 Chart-FGRPO 数据构建、训练或正式评测  
> 主模型：Qwen2.5-VL-3B-Instruct  
> 主数据集：ChartQA  
> 外部测试集：ChartQAPro  
> 训练框架：ms-swift  
> 计划硬件：单张 RTX 4090 24 GB；本地 RTX 4070 8 GB 用于开发、评测与演示

## 1. 项目定位

ForgeMM 的最终方向确定为：

> 基于 Qwen2.5-VL-3B 的图表可信推理后训练系统，通过结构化 QLoRA SFT、标准 GRPO 与 FGRPO-inspired 约束优化，提高模型答案、图表证据和计算过程的一致性。

项目不从零训练视觉语言模型，不修改 Qwen2.5-VL 的基础架构，也不调用闭源商业模型完成在线训练或评分。正式训练可以在单张 RTX 4090 上完成，训练后的 LoRA Adapter 应能下载到本地，并在 RTX 4070 8 GB 上以 4-bit 方式运行和演示。

项目唯一核心研究变量是：

> 在相同的 Structured SFT 起点上，动态约束式 Chart-FGRPO 是否优于只优化答案的标准 GRPO，以及固定权重的多奖励 GRPO。

## 2. 方法来源与声明边界

主要方法来源：

- Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization，2026，<https://arxiv.org/abs/2604.08476>
- ms-swift GRPO 文档：<https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html>
- ms-swift 多模态 GRPO 示例：<https://swift.readthedocs.io/en/latest/BestPractices/GRPO-Multi-Modal-Training.html>
- ChartQA 官方仓库：<https://github.com/vis-nlp/ChartQA>
- ChartQAPro 官方仓库：<https://github.com/vis-nlp/ChartQAPro>
- ChartQAPro 论文：<https://aclanthology.org/2025.findings-acl.978/>

本项目使用 FGRPO 的以下核心思想：

1. 将答案正确性作为主要优化目标；
2. 将推理一致性和视觉证据一致性作为约束；
3. 分别计算不同 reward 的组内 advantage；
4. 使用拉格朗日乘子动态调整约束强度。

本项目与原论文存在明确差异：

- 原论文主要面向视觉空间推理；本项目面向图表问答；
- 原论文使用 LLM/VLM judge 和空间标注；本项目优先使用确定性表格、annotation 和可执行 operation verifier；
- 本项目只实现证据一致性与运算一致性两个约束；
- 本项目使用 QLoRA 和单张消费级 GPU，不复现原论文的大规模训练设置。

因此，在完整对齐论文代码和训练设置之前，项目统一使用以下表述：

> FGRPO-inspired Chart-FGRPO / 受 FGRPO 启发的图表约束式 GRPO

不得表述为“完整复现 FGRPO”“提出通用 FGRPO”或“达到 SOTA”。

## 3. 研究假设

相比相同 Structured SFT 起点上的标准 GRPO，Chart-FGRPO 应当：

1. 提升模型引用图表证据的正确率；
2. 降低最终答案与中间运算不一致的比例；
3. 提高答案、证据、运算同时正确的可信正确率；
4. 保持或提高 ChartQA 答案准确率；
5. 尽可能改善 ChartQAPro 外部分布泛化能力。

如果训练 reward 上升，但冻结测试集上的指标不改善，则研究假设没有得到支持。

## 4. 模型输出协议

Structured SFT、GRPO 与 Chart-FGRPO 统一使用：

```text
<evidence>
e1=cell(row="2015", column="Favorable", value=38)
</evidence>
<operation>
equal(e1,38)=true
</operation>
<answer>
Yes
</answer>
```

计算题示例：

```text
<evidence>
e1=cell(row="2014", column="Favorable", value=51)
e2=cell(row="2015", column="Favorable", value=44)
</evidence>
<operation>
sum(e1,e2)=95
</operation>
<answer>
95
</answer>
```

不训练自由文本长思维链。`operation` 使用有限、可执行的 DSL，首版支持：

```text
lookup / equal / sum / difference / average / ratio
product / median / count / argmax / argmin / compare
```

模型不输出任务类型和 constraint mask。它们由数据处理程序生成，避免模型通过自报题型逃避约束。

## 5. 数据策略

### 5.1 数据用途

| 数据 | 用途 | 是否参与训练 |
|---|---|---|
| ChartQA train human | 自然语言问题与可信 Structured SFT/GRPO 样本 | 是 |
| ChartQA train augmented | 补充可验证、模板化的结构化训练样本 | 是，通过审计后 |
| ChartQA val human | 开发、阈值冻结和 checkpoint 选择 | 否 |
| ChartQA test human | 最终 ChartQA 答案与可信推理测试 | 否 |
| ChartQAPro | 外部分布最终测试 | 否 |

不得使用 ChartQA test 或 ChartQAPro 调学习率、阈值、训练步数、prompt 或 checkpoint。

### 5.2 EvidenceStore

ChartQA 的 CSV、annotation 和图片可能存在噪声或相互冲突，因此需要构建统一 EvidenceStore：

```text
图片 + CSV + annotation + 官方 QA
                ↓
字段标准化、图片哈希检查、证据融合、冲突检测
                ↓
可信样本 / 仅答案样本 / 排除样本
```

可信样本必须满足：

1. 样本只来自对应训练或评测 split；
2. 图片文件存在且没有跨 split 内容泄漏；
3. 问题属于支持的 operation；
4. 必要证据可以从冻结证据记录中恢复；
5. gold operation 可以被安全 executor 执行；
6. 执行结果能够复现官方答案；
7. CSV 与 annotation 冲突时已经完成冻结裁决，否则排除；
8. 数据处理产物带 schema version、源路径和 SHA-256。

### 5.3 数据规模门槛

- Structured SFT：目标 4,000 条，最低 3,000 条；
- GRPO train：目标 2,000 个可信 prompt，最低 1,500 个；
- 开发证据集：ChartQA val 冻结 200 条；
- 正式证据测试集：ChartQA test 冻结 300 条；
- 完整答案测试：ChartQA test human 全部 1,250 条；
- 外部测试：ChartQAPro 全量。

如果可信数据少于门槛，优先缩小 operation 支持范围，不能伪造证据、使用测试数据补训练集或降低验证标准。

### 5.4 ms-swift 数据字段

训练 JSONL 至少包含：

```text
sample_id
messages
images
reference_answer
gold_evidence
gold_operation
answer_type
task_type
evidence_mask
operation_mask
source_split
source_image_sha256
```

GRPO 配置显式设置并测试额外字段能够传入 reward plugin。数据预处理后必须用最小批次完成端到端字段透传测试。

## 6. Verifier 与 reward

### 6.1 格式解析

解析 `<evidence>`、`<operation>`、`<answer>`，拒绝：

- 标签缺失或重复；
- 标签嵌套错误；
- 超长字段；
- 未声明 evidence ID；
- operation 引用不存在的证据；
- 输出标签之外的异常内容。

格式失败时，答案、证据和 operation reward 均为 0。

### 6.2 答案 reward

- 文本、布尔、年份：标准化严格匹配；
- 普通数值：使用 ChartQA relaxed correctness；
- 训练 reward 与最终评测指标分别记录，不能只报告训练 reward。

### 6.3 证据 reward

预测 evidence 与 gold evidence 进行字段级匹配：

- row/category；
- column/series；
- value；
- 必要时的 bbox 或视觉属性。

主要连续指标为 Evidence Precision、Recall、F1。严格 evidence pass 要求没有错误引用，并覆盖所有必要证据。

### 6.4 运算一致性 reward

使用白名单 AST/DSL executor 执行 operation，禁止 Python `eval`。通过条件为：

1. operation 语法合法；
2. 所有参数来自已声明 evidence；
3. 执行无异常；
4. 执行结果与 `<answer>` 一致。

### 6.5 约束适用 mask

- `answer_mask=1`：所有正式 QA 样本；
- `evidence_mask=1`：证据已冻结且可验证；
- `operation_mask=1`：gold operation 已冻结且可执行。

颜色、图例、视觉交点等无法由当前 EvidenceStore 稳定验证的问题，可以只启用答案 reward。mask 只存在于数据元信息中，不由模型生成。

## 7. Chart-FGRPO 优化目标

任务目标：

```text
maximize E[R_task]
subject to E[R_evidence] >= tau_evidence
           E[R_operation] >= tau_operation
```

分别对三个 reward 计算组内 advantage：

```text
A = A_task + lambda_evidence * A_evidence
           + lambda_operation * A_operation
```

拉格朗日乘子更新：

```text
lambda_k = clip(
    lambda_k + dual_lr * (tau_k - batch_mean_reward_k),
    0,
    lambda_max,
)
```

预登记参数：

| 参数 | 值 |
|---|---:|
| `tau_evidence` | 0.90 |
| `tau_operation` | 0.95 |
| `lambda_evidence_init` | 0.0 |
| `lambda_operation_init` | 0.0 |
| `dual_lr` | 0.01 |
| `lambda_max` | 5.0 |

乘子、阈值、reward 均值和 constraint violation 必须进入 checkpoint 与训练日志，断点恢复后轨迹必须连续。

## 8. 实验矩阵

| 编号 | 模型/方法 | 作用 |
|---|---|---|
| E0 | 原始 Qwen2.5-VL-3B-Instruct | 底座基线 |
| E1 | 现有 Answer-only QLoRA | 历史工程基线 |
| E2 | Structured QLoRA SFT | 所有 RL 方法的共同起点 |
| E3 | Task-only 标准 GRPO | 主要算法基线 |
| E4 | 固定权重、解耦归一化的多奖励 GRPO | 固定 reward 基线 |
| E5 | 动态拉格朗日 Chart-FGRPO | 最终方法 |
| A1 | E5 去掉 evidence constraint | 证据约束消融 |
| A2 | E5 去掉 operation constraint | 运算约束消融 |

E3、E4、E5 固定以下内容：

- 同一个 Structured SFT checkpoint；
- 同一训练数据和顺序；
- 同一 QLoRA 配置；
- 同一 image token/pixel 预算；
- 同一 rollout 参数；
- 同一训练步数和 checkpoint 选择规则；
- 同一组随机种子。

它们之间只改变 reward/advantage 组合方式。

## 9. 评价指标与成功门槛

定义可信正确率：

```text
FCR = P(answer correct AND evidence pass AND operation pass)
```

| 指标 | 成功门槛 |
|---|---|
| Format Compliance | >= 98% |
| Truncation Rate | <= 1% |
| Evidence F1 | 相比标准 GRPO 提高至少 5 pp |
| Operation Consistency | 相比标准 GRPO 提高至少 5 pp |
| FCR | 相比标准 GRPO 提高至少 5 pp，95% paired bootstrap CI 不跨 0 |
| Inconsistency Rate | 相对降低至少 30% |
| ChartQA Relaxed Accuracy | 不低于标准 GRPO 超过 1 pp |
| ChartQAPro Overall | 不低于标准 GRPO 超过 1 pp；提高至少 1 pp 才宣称 OOD 改善 |

正式 RL 实验使用随机种子：

```text
17 / 42 / 2026
```

E3、E4、E5 均运行三个种子。要求 E5 相对 E3 的 FCR 差异在三个种子上方向一致。A1、A2 使用一个预登记种子完成机制消融。

评测同时报告：

- 三种子均值与标准差；
- paired bootstrap 95% CI；
- exact/relaxed 的逐样本改对、改错；
- McNemar 或 sign test；
- 按 task、operation、answer type 和 chart type 的分组结果；
- 峰值显存、GPU 时长、吞吐、响应长度与截断率。

## 10. Structured QLoRA SFT 配置

| 配置 | 冻结值 |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct |
| Quantization | NF4 4-bit |
| Compute dtype | BF16 |
| LoRA targets | language `q_proj/k_proj/v_proj/o_proj` |
| Rank / alpha / dropout | 8 / 16 / 0 |
| ViT / aligner | frozen / frozen |
| max_pixels | 262144 |
| max_completion_length | 160 |
| learning_rate | `5e-5` |
| epochs | 2 |
| warmup_ratio | 0.03 |
| per-device batch | 1 |
| gradient accumulation | 16 |
| scheduler | cosine |

SFT 进入 RL 前必须满足：

- Format Compliance >= 98%；
- Operation Executable Rate >= 90%；
- ChartQA val relaxed accuracy 不低于 answer-only SFT 超过 1 pp；
- Adapter、配置、训练数据和模型 revision 均已哈希冻结。

## 11. GRPO 公共配置

| 配置 | 初始冻结值 |
|---|---|
| `num_generations` | 4 |
| `generation_batch_size` | 8 |
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `max_completion_length` | 160 |
| `learning_rate` | `5e-7` |
| `max_steps` | 500 |
| `gradient_accumulation_steps` | 8 |
| clip epsilon | 0.2 |
| `max_grad_norm` | 0.5 |
| `loss_type` | `grpo` |
| `beta` | 首选 0.001；仅在统一显存 smoke 失败后对所有方法改为 0 |

单张 RTX 4090 优先使用：

- QLoRA GRPO；
- vLLM colocate；
- LoRA-only weight sync；
- 约 0.30 的初始 vLLM GPU memory utilization；
- 必要时启用 sleep/offload。

显存相关调整只能依据 OOM/稳定性 smoke，且必须在正式 E3/E4/E5 前一次性冻结，不能依据质量结果为不同方法选择不同配置。

## 12. ms-swift 集成设计

使用独立 Linux 训练环境，不直接升级当前 Windows 项目环境。计划锁定一个经过 smoke 的 ms-swift 稳定 tag 与 commit，并保存：

- Python、PyTorch、CUDA、cuDNN、vLLM、ms-swift 版本；
- `pip freeze`；
- GPU 型号与驱动；
- 训练配置 YAML；
- ms-swift commit；
- 自定义 patch 的 diff 与 SHA-256。

集成方式：

1. 使用外部 reward plugin 注册 `chart_answer`、`chart_evidence`、`chart_operation`；
2. 使用 ms-swift 的多模态数据格式传入图片；
3. 使用独立 reward 归一化作为固定权重 E4 的基础；
4. 自定义/子类化 GRPO Trainer 的 advantage 合成步骤；
5. 为 E5 增加动态 lambda 更新和 checkpoint state；
6. 不复制或重写整个 ms-swift 仓库，只维护最小可审计扩展。

## 13. 分阶段执行计划

### Stage 0：范围与环境冻结

- 更新 ForgeMM 项目范围和 ADR；
- 登记数据路径、许可、hash 与 split；
- 建立独立 ms-swift 环境；
- 完成 Qwen2.5-VL-3B 推理和 4-bit 加载 smoke。

### Stage 1：EvidenceStore 与数据审计

- 读取 CSV、annotation、QA 与图片；
- 建立统一 schema；
- 检测跨 split 图片泄漏；
- 检测 CSV/annotation/答案冲突；
- 生成可信、仅答案和排除清单；
- 冻结 train/val/test manifest。

### Stage 2：Parser、Executor 与 Verifier

- 实现结构化 parser；
- 实现安全 operation executor；
- 实现三类 reward；
- 完成单元、性质和异常测试；
- 完成离线 scorer 与 ms-swift plugin 一致性测试。

### Stage 3：Structured SFT

- 生成并审计 Structured SFT JSONL；
- 完成 32 条 smoke；
- 完成正式两轮 QLoRA SFT；
- 使用冻结 val 选择 checkpoint；
- 通过格式、运算和答案门。

### Stage 4：GRPO/FGRPO 工程 smoke

- 4 prompts x 4 rollouts 的 reward smoke；
- 20-step 训练、显存与恢复 smoke；
- 验证 reward 非恒定、lambda 更新方向和 on-policy 身份；
- 冻结正式 RL 配置。

### Stage 5：单种子快速实验

- E3、E4、E5 各运行 100 steps；
- 只在 ChartQA val 比较；
- 检查 FCR 方向、reward variance、KL、clip ratio、长度与崩溃；
- 不根据 test 或 ChartQAPro 结果修改方案。

### Stage 6：正式三种子训练

- E3、E4、E5 各运行 500 steps x 3 seeds；
- A1、A2 各运行一个预登记 seed；
- 使用统一规则选择 checkpoint；
- 保存所有成功和失败 run，不覆盖历史结果。

### Stage 7：最终评测

- 完整 ChartQA test human；
- 冻结可信 test300；
- ChartQAPro 全量官方评测；
- 三种子统计、逐样本迁移和失败分类；
- 形成最终结果表与 claim audit。

### Stage 8：本地部署和演示

- 下载最终 LoRA Adapter；
- RTX 4070 8 GB 上以 Transformers + bitsandbytes 4-bit 推理；
- 提供 CLI 与简洁 Gradio 页面；
- 展示 evidence、operation、answer 和 verifier 状态；
- 记录 Adapter 大小、加载时间、峰值显存、首 token 与完整响应延迟。

## 14. 计划代码与配置结构

```text
ForgeLLM/
├── configs/forgemm/chart_fgrpo/
│   ├── sft.yaml
│   ├── grpo_task.yaml
│   ├── grpo_fixed.yaml
│   └── grpo_adaptive.yaml
├── src/forgellm/multimodal/chart_fgrpo/
│   ├── schema.py
│   ├── parser.py
│   ├── evidence_store.py
│   ├── operations.py
│   ├── trace_builder.py
│   ├── rewards.py
│   ├── constraints.py
│   ├── swift_plugin.py
│   ├── swift_trainer.py
│   └── evaluation.py
├── scripts/
│   ├── forgemm_build_evidence_store.py
│   ├── forgemm_build_structured_data.py
│   ├── forgemm_validate_traces.py
│   └── forgemm_chart_fgrpo_eval.py
├── tests/unit/
├── tests/integration/
└── docs/experiments/
```

数据集大文件不直接提交 Git。项目内只保存数据说明、目录约定、下载来源、许可、manifest 和 hash。实际数据目录通过配置或环境变量传入。

## 15. 失败回退规则

| 风险 | 回退 |
|---|---|
| 可信样本不足 | 缩小 operation 范围，不生成虚假证据 |
| CSV/annotation 噪声过高 | 使用冻结高置信子集，冲突样本只做 answer evaluation |
| Structured SFT 格式失败 | 修复 trace 与 SFT，不提前进入 RL |
| reward 大量零方差 | 检查数据难度与采样温度，不增加新算法分支 |
| 单卡 4090 OOM | 依次降低 generation batch、像素预算、关闭 KL reference；所有对照统一 |
| 动态 Trainer 接入失败 | 维护固定版本最小 patch；若只能分段更新权重，则改称 chunk-wise constrained GDPO |
| 训练 reward 上升但 test 无改善 | 记录负结果，不声明能力提升 |
| 证据改善但答案持平 | 只声明可信度改善且答案性能保持 |

## 16. 预期产物

- ChartQA EvidenceStore 与数据审计报告；
- Structured SFT 与 GRPO manifests；
- 安全结构化 parser 和 operation executor；
- 三类独立 reward；
- Chart-FGRPO 自定义 Trainer；
- E0-E5、A1-A2 实验记录；
- ChartQA 与 ChartQAPro 统一评测报告；
- 三种子结果、统计检验和失败分析；
- 本地 4-bit 演示；
- 数据卡、模型卡、实验卡和简历 claim audit。

## 17. 简历表述边界

实验前可以写：

> 基于 Qwen2.5-VL-3B 设计面向图表推理的结构化 QLoRA SFT 与 FGRPO-inspired 约束式强化学习方案，通过确定性证据和可执行运算 Verifier 优化推理可信度。

实验后只有在达到成功门槛时，才可以填入真实数值：

> 相比标准 GRPO，Chart-FGRPO 在冻结测试集上将 FCR 提升 X pp、推理不一致率降低 Y%，同时 ChartQA relaxed accuracy 变化为 Z pp；最终 Adapter 可在本地 RTX 4070 8 GB 上以 4-bit 模式运行。

不得使用：

- “从零训练 3B 多模态模型”；
- “完整复现 FGRPO”；
- “达到 SOTA”；
- “显著提升”但没有统计证据；
- 只用训练 reward 代替冻结测试结果；
- 把 ChartQAPro 调参结果称为外部分布泛化。

## 18. 当前下一步

在任何正式训练之前，按顺序完成：

1. 整理项目文件并更新 ForgeMM 范围文档；
2. 将数据集放入统一但不纳入 Git 的项目数据目录；
3. 建立数据路径配置、许可与 hash 清单；
4. 实现 EvidenceStore 数据审计；
5. 实现 parser、executor 和 verifier 单元测试；
6. 通过数据与方法 smoke 后再租用 RTX 4090。
