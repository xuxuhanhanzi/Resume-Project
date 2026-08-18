# ForgeMM ChartQA 数据诊断记录（2026-08-06）

## 1. 目标

解释 B1–B3 为什么只改变少量答案却没有获得净提升，并判断下一项实验应优先修改模型、训练规模还是数据采样。

本阶段不训练模型，只审计数据与已有实验输出。

## 2. 审计范围

- ChartQA human：train、val、test 全量。
- ChartQA augmented：train、val、test 全量，只做规模、字段、图片引用和重复统计，不与 human 混合解释。
- 当前训练候选前 512 条。
- B1–B3 实际使用的 200 条训练样本。
- 当前验证集前 50 条。
- 原始模型、B1、B2、B3 在这 50 条上的逐样本结果。
- 图片文件名与 SHA-256 内容哈希，用于检查跨划分视觉泄漏。

任务类型由确定性规则划分为 boolean、median、average、ratio、product、sum、difference、counting、extremum/comparison、temporal lookup、direct lookup 和 other。它们是项目内部启发式标签，不是 ChartQA 官方标注。

## 3. 最终命令

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_chartqa_data_audit.py `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --b1-dir "artifacts\forgemm\b1_language_attention_qlora_bounded50_20260805" `
  --b2-dir "artifacts\forgemm\b2_language_attention_ffn_qlora_bounded50_20260806" `
  --b3-dir "artifacts\forgemm\b3_attention_qlora_lr5e5_bounded50_20260806" `
  --output-dir "artifacts\forgemm\chartqa_data_audit_v3_20260806"
```

v1、v2 是开发过程中保留的中间审计。v3 加入任务×答案联合分层，并从候选清单排除了跨划分重复图片，是正式结果。

## 4. 数据规模与完整性

| 划分 | Human 问题 | Human 图片 | Augmented 问题 |
|---|---:|---:|---:|
| Train | 7,398 | 3,699 | 20,901 |
| Val | 960 | 480 | 960 |
| Test | 1,250 | 625 | 1,250 |

- 六个 JSON 文件均符合所需字段结构。
- human 与 augmented 的所有图片引用均存在，缺失图片为 0。
- train/val/test 图片目录中的 PNG 均被 human 或 augmented 数据引用。
- 启发式任务分类的 `other` 比例：train human 0.58%，val human 0.52%，test human 0.64%。分类覆盖足够用于分布诊断，但仍不等于官方任务标签。
- Human 内部完全重复记录：train 1 条、val 0 条、test 0 条。
- Augmented 内部完全重复记录：train 160 条、val 8 条、test 10 条。后续若使用 augmented 数据，必须先去重。

## 5. 跨划分泄漏

按文件名检查时，train/val/test 没有共享图片名，也没有完全相同的“图片名+问题+答案”记录。

按图片内容 SHA-256 检查后发现：

| 对比 | 相同图片内容数量 |
|---|---:|
| Train–Val | 2 |
| Train–Test | 8 |
| Val–Test | 1 |

例如 `train/16970.png` 与 `val/16968.png` 内容完全相同，但问题不同；`train/8314.png` 与 `val/8302.png` 也相同。这不属于问题文本完全复制，却仍构成视觉信息跨划分重复。

最终候选清单已排除所有涉及跨划分重复哈希的样本：

- 训练候选排除 10 张图片、20 条 human 记录。
- 验证候选排除 3 张图片、6 条 human 记录。

另外，human 内部存在不同文件名但相同内容的图片哈希：train 38 组、val 1 组、test 0 组。正式扩展训练集前应继续按图片哈希去重或分组采样。

## 6. 当前训练样本偏差

完整 train human 的主要任务分布包括：direct lookup 1,891、extremum/comparison 1,312、counting 875、difference 828、sum 657、boolean 614、average 583 等。

直接取前 512 条时，boolean 比完整训练集高约 6.93 个百分点，同时 counting、difference、average 和 sum 偏少，说明原始文件前缀并非稳定的代表性抽样。

B1–B3 实际 200 条经过随机打乱后，最大任务类型偏差降到 3.70 个百分点，但答案类型仍明显失衡：

- 文本答案少约 8.22 个百分点。
- 年份答案多约 4.01 个百分点。
- 布尔答案多约 3.97 个百分点。

因此当前训练样本不是严重的任务类别失衡，而是“任务与答案表面类型的联合分布”失衡。

## 7. 当前验证集偏差

完整 val human 有 960 条；当前只使用文件前 50 条。

主要差异：

- direct lookup 高约 8.06 个百分点。
- counting 低约 6.60 个百分点。
- difference 低约 4.73 个百分点。
- median 高约 4.65 个百分点。
- year/date 答案低约 6.23 个百分点。
- 数值答案高约 5.33 个百分点。
- 平均问题长度：完整 val 为 11.07 词，前 50 条只有 9.72 词。

任务分布 Jensen–Shannon divergence 为 0.0528，说明前 50 条不是完整验证集的可靠近似。更重要的是，50 条中每条样本对应 2 个百分点，B1/B2/B3 之间 2–4 个百分点的差异容易由单个样本决定。

## 8. B1–B3 分类结果

在当前 50 条上，模型变化集中在数值答案：

| 模型 | 数值严格准确率 | 全部严格准确率 |
|---|---:|---:|
| 原始模型 | 34.62%（9/26） | 46% |
| B1 | 26.92%（7/26） | 42% |
| B2 | 23.08%（6/26） | 40% |
| B3 | 30.77%（8/26） | 44% |

布尔答案 4/4、文本答案 9/12 在四个模型中都没有变化。主要退化不是通用格式崩坏，而是少数数值推理答案被微调改变。

任务类别方面，当前 3 条 median 问题中：

- 原始模型答对 2 条。
- B1/B2 答对 0 条。
- B3 答对 1 条。

但每个类别的验证样本过少：average 只有 2 条、ratio 2 条、median 3 条、sum 4 条。不能据此得出稳定的任务能力排名，只能定位需要在更大验证集中重点复核的类别。

## 9. 分层候选清单

使用“启发式任务类型 × 答案类型”联合分层、固定随机种子和最大余数配额，生成两个候选清单：

### 训练 200 条

| 指标 | 当前实际 200 | 候选分层 200 |
|---|---:|---:|
| 最大任务比例偏差 | 3.70% | 0.62% |
| 最大答案比例偏差 | 8.22% | 0.99% |
| 任务 JS divergence | 0.01513 | 0.00023 |

### 验证 250 条

| 指标 | 当前前 50 | 候选分层 250 |
|---|---:|---:|
| 最大任务比例偏差 | 8.06% | 0.60% |
| 最大答案比例偏差 | 6.23% | 0.66% |
| 任务 JS divergence | 0.05278 | 0.00040 |

候选清单均不包含已发现的跨划分重复图片。

## 10. 结论

数据诊断支持以下判断：

1. B1–B3 的 50 条验证结果只能视为探索性证据，不能用于稳定选择最终方法。
2. 当前训练 200 条的答案类型分布存在实质偏差，尤其缺少文本答案。
3. 模型退化集中在少数数值/median 样本，不是所有任务同时退化。
4. ChartQA 本地副本完整，但官方划分中存在少量“不同文件名、相同图片内容”的视觉泄漏，项目清单应主动排除。
5. Augmented train 含重复记录，不能未经去重直接加入训练。

## 11. 下一步与 B4 决策

下一步先实现“清洁分层 val250 评估”，不立即训练 B4：

1. 让评测脚本读取 `proposed_stratified_val250_manifest.csv`。
2. 在相同 250 条上评估原始模型、B1、B2、B3。
3. 检查 50 条上的模型排序是否在 250 条上保持。

如果 val250 仍显示所有 Adapter 低于原始模型，则 B4 使用：

- B3 的 Attention QLoRA；
- 学习率 `5e-5`；
- 仍处理 200 条样本；
- 唯一变量改为 `proposed_stratified_train200_manifest.csv` 的任务×答案联合分层样本。

这样 B4 可以直接回答“数据组成失衡是否导致微调无净提升”，不会同时混入更多步数、更多参数或不同学习率。

## 12. 产物

- 正式审计：`artifacts/forgemm/chartqa_data_audit_v3_20260806/audit_report.json`
- 当前训练 200 清单：`actual_train200_manifest.csv`
- 候选分层训练 200：`proposed_stratified_train200_manifest.csv`
- 候选清洁分层验证 250：`proposed_stratified_val250_manifest.csv`
- 当前 val50 四模型对照：`val50_model_comparison.csv`
- 审计模块：`src/forgellm/multimodal/data_audit.py`
- 审计命令：`scripts/forgemm_chartqa_data_audit.py`
- 单元测试：`tests/unit/test_forgemm_data_audit.py`
