# ForgeLLM 项目文件整理与数据集接入计划

> 制定日期：2026-08-07  
> 当前阶段：只读盘点与方案冻结  
> 执行边界：本文制定整理方案；尚未删除、移动或复制任何现有文件  
> 对应实验方案：`docs/multimodal/forgemm_chart_fgrpo_implementation_plan.md`

## 1. 整理目标

本次整理服务于校招简历中的“多模态大模型训练项目”，目标不是简单压缩目录，而是让仓库能够清楚回答四个问题：

1. 项目当前研究目标是什么；
2. 哪些代码、配置和实验记录支撑该目标；
3. 数据集从哪里来、如何复现、是否被误提交到 Git；
4. 哪些内容是可复现实验资产，哪些只是可再生成的本地缓存。

整理后应满足：

- ChartQA 与 ChartQAPro 具备统一、稳定的数据根目录；
- 训练、评测代码不再依赖个人桌面的绝对路径；
- 大数据、模型权重、运行产物不进入 Git；
- 历史实验记录仍可追溯，但不会与当前 Chart-FGRPO 主线混淆；
- 删除动作均有明确对象、理由和确认步骤；
- 在删除任何实验产物前，先保留指标、配置、日志摘要与可复现命令。

## 2. 当前盘点结论

### 2.1 Git 工作区状态

当前 ForgeLLM 工作区并非干净状态：

- `requirements-post-training.lock` 已修改；
- ForgeMM 的文档、脚本、源码和测试目前大多还是未跟踪文件；
- `.audit_tmp/` 未被 `.gitignore` 忽略；
- `artifacts/`、`data/raw/`、`data/processed/`、虚拟环境和常见 Python 缓存已被忽略。

因此，整理前必须先建立一个可恢复检查点。不能在未区分用户改动和新项目文件的情况下直接清理。

### 2.2 主要本地目录占用

| 目录 | 文件数 | 大小 | 判断 |
|---|---:|---:|---|
| `.venv/` | 37,530 | 约 5.09 GiB | 可再生成，本地清理候选 |
| `artifacts/` | 487 | 约 2.07 GiB | 混合了当前实验与历史实验，必须分类后处理 |
| `data/` | 30 | 约 244.9 MiB | 旧阶段原始/处理数据，先归档判断 |
| `.mypy_cache/` | 614 | 约 118.1 MiB | 可再生成，本地清理候选 |
| `.venv-repro/` | 3,870 | 约 97.4 MiB | 可再生成，但删除前确认是否仍用于复现 |
| `.audit_tmp/` | 102 | 约 5.8 MiB | ChartQA 数据审计临时副本，清理候选 |
| `.pytest_cache/`、`.ruff_cache/` | 17 | 小于 0.1 MiB | 可再生成，本地清理候选 |
| `tmp/` | 0 | 0 | 空目录候选 |

`artifacts/` 中占用较大的部分：

| 子目录 | 大小 | 建议 |
|---|---:|---|
| `artifacts/forgemm/` | 约 1.25 GiB | 当前多模态主线，保留并进一步提取指标清单 |
| `artifacts/stage04/` | 约 395.8 MiB | 旧文本模型阶段，归档候选 |
| `artifacts/stage05/` | 约 297.1 MiB | 旧偏好优化阶段，归档候选 |
| `artifacts/stage03/` | 约 139.2 MiB | 旧预训练阶段，归档候选 |
| 其他 stage 目录 | 约 3.7 MiB | 依据文档引用逐项判断 |

### 2.3 ChartQA 位置与状态

已确认有效数据根目录为：

```text
D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset
```

该目录直接包含 `train/`、`val/`、`test/`，是现有 ForgeMM 脚本使用的正确 `--data-root`。

| 分区 | 文件数 | 大小 |
|---|---:|---:|
| train | 54,953 | 约 867.8 MiB |
| val | 3,170 | 约 50.6 MiB |
| test | 4,529 | 约 70.4 MiB |
| 合计 | 62,652 | 约 988.8 MiB |

其中还存在一个无内容的冗余嵌套目录：

```text
D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset\ChartQA Dataset
```

它目前为 0 文件、0 字节，可列为后续明确确认的清理对象，但本阶段不删除。

### 2.4 ChartQAPro 位置与状态

源目录为：

```text
D:\Users\27475\Desktop\ChartQAPro
```

已确认的规范数据文件：

```text
D:\Users\27475\Desktop\ChartQAPro\chartqapro_test.parquet
```

该 parquet：

- 1,948 条样本；
- 约 199.8 MiB；
- 字段为 `Question`、`Answer`、`Question Type`、`image`、`Year`、`Paragraph`；
- 足以作为 ChartQAPro 外部测试集的数据入口。

源目录中还存在：

- 一份约 239.9 MiB 的 Hugging Face Arrow 缓存；
- `dataset_info.json`；
- 3 个 0 字节锁文件。

Arrow 文件是 parquet 被数据加载库转换后的本机缓存，并非第二份独立基准数据。项目接入时只采用 parquet 作为规范源，不复制锁文件和 Arrow 缓存。缓存可在需要时自动重建。

## 3. 冻结后的目标目录结构

大数据集应加入整个简历项目工作区，但不放入 ForgeLLM Git 仓库。目标结构如下：

```text
Resume_Project/
├── datasets/
│   ├── ChartQA/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── DATASET_CARD.md
│   │   └── manifest.json
│   └── ChartQAPro/
│       ├── chartqapro_test.parquet
│       ├── DATASET_CARD.md
│       └── manifest.json
├── ForgeLLM/
│   ├── configs/
│   │   └── data/
│   │       ├── forgemm_datasets.example.yaml
│   │       └── forgemm_datasets.local.yaml   # 本地文件，不提交
│   ├── docs/
│   │   ├── multimodal/
│   │   ├── experiments/
│   │   └── archive/
│   ├── scripts/
│   ├── src/forgellm/multimodal/
│   ├── tests/
│   └── artifacts/                            # 运行产物，不提交
└── RepoPilot/
```

选择工作区级 `datasets/` 而非 `ForgeLLM/data/` 的原因：

1. 避免 Git 状态扫描和误提交大文件；
2. 将公开数据集与项目生成的 `data/raw`、`data/processed` 区分；
3. 将来若第二个 Agent 项目需要复用模型或数据，可通过配置引用；
4. 数据集迁移和仓库代码提交可以独立进行。

数据集代码不得硬编码上述绝对路径。默认解析顺序定为：

1. 命令行参数；
2. 环境变量 `FORGELLM_DATASETS_ROOT`；
3. 不提交的本地配置文件；
4. 相对路径 `../datasets`；
5. 均不存在时给出带示例的明确错误。

## 4. 文件分类决策

### 4.1 立即保留

以下内容属于当前主线或复现证据，不进入删除候选：

- `docs/multimodal/forgemm_chart_fgrpo_implementation_plan.md`；
- `docs/multimodal/ForgeMM_论文与开源项目清单.md`；
- `docs/multimodal/ForgeMM_项目总览与实验进展.md`；
- `docs/experiments/2026-08-05_*` 与 `2026-08-06_*` ForgeMM 实验记录；
- `scripts/forgemm_*.py`；
- `src/forgellm/multimodal/`；
- `tests/unit/test_forgemm_*.py`；
- `artifacts/forgemm/` 中能支撑现有实验结论的结果；
- 所有当前被修改但尚未审定的文件，包括 `requirements-post-training.lock`。

### 4.2 保留但标记为历史基线

以下文档虽然不是新 Chart-FGRPO 主方案，但记录了基线选择、数据审计与失败/成功实验，不能当作垃圾文件删除：

- `forgemm_chartqa_baseline_plan.md`；
- `forgemm_b1_language_qlora_plan.md`；
- `forgemm_b2_language_attention_ffn_qlora_plan.md`；
- `forgemm_b3_low_lr_attention_qlora_plan.md`；
- `forgemm_val250_evaluation_plan.md`；
- `ForgeMM_候选方向与推荐.md`。

计划将它们移动至 `docs/archive/forgemm_baselines/`，并在当前总览中保留索引。原因是部分正式实验记录仍直接引用这些计划文件。

### 4.3 暂不删除、先做归档判断

以下部分属于 ForgeLLM 早期“从零训练小语言模型”的旧阶段。它们与当前多模态微调主线不完全一致，但可能包含简历叙事、工程能力或测试基础，不能仅凭目录名称删除：

- `artifacts/stage01` 至 `artifacts/stage06`；
- `data/raw/` 与 `data/processed/`；
- `scripts/stage*.py`；
- `docs/stages/` 和早期月度学习计划；
- tokenizer、data pipeline、DPO、服务化等旧骨架；
- 各类仅含 `.gitkeep` 的规划目录。

对它们执行两项检查后再决定：

1. 是否被 README、测试、实验记录或简历材料引用；
2. 是否仍承载当前 ForgeMM 可复用的代码。

若两项均为否，则按“单个明确文件”生成最终删除清单；若仍有叙事价值，则移动到 `docs/archive/legacy_text_llm/` 或工作区外的冷归档目录。

### 4.4 明确的本地清理候选

以下项目可再生成，不应作为项目资产：

- `.mypy_cache/`；
- `.pytest_cache/`；
- `.ruff_cache/`；
- `.audit_tmp/`；
- 空 `tmp/`；
- ChartQA 的空嵌套 `ChartQA Dataset/ChartQA Dataset/`；
- ChartQAPro 的 0 字节 `.lock` 文件；
- ChartQAPro 的 Arrow 缓存（在 parquet 完整性验证通过后）；
- 不再使用的虚拟环境 `.venv-repro/`；
- `.venv/` 仅在锁文件和环境重建命令验证通过后才清理。

注意：项目规则禁止批量或递归删除目录。因此这些目录不会由自动整理命令批量删除；需要用户手动删除，或将目录内文件展开成逐个明确路径的删除操作。后续自动执行阶段优先只处理少量、明确且已确认的单文件。

## 5. 数据集接入方案

### 5.1 ChartQA

源：

```text
D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset
```

目标：

```text
D:\Users\27475\Desktop\Resume_Project\datasets\ChartQA
```

采用“先复制、验证、切换配置，再决定是否清理源目录”的方式。目标目录直接包含 `train/`、`val/`、`test/`，不保留多余的 `ChartQA Dataset` 嵌套层。

验证门槛：

- 总文件数保持 62,652；
- 总字节数保持 1,036,819,201；
- train/val/test 文件数分别保持 54,953、3,170、4,529；
- 三个分区各随机抽取图像、JSON 和表格文件进行可读性检查；
- 重新运行 ChartQA 数据审计与 20 样本 smoke test；
- 所有配置切换到统一数据根目录后，不再依赖旧绝对路径。

### 5.2 ChartQAPro

源：

```text
D:\Users\27475\Desktop\ChartQAPro\chartqapro_test.parquet
```

目标：

```text
D:\Users\27475\Desktop\Resume_Project\datasets\ChartQAPro\chartqapro_test.parquet
```

仅复制 parquet，不复制 Hugging Face Arrow 缓存和锁文件。

验证门槛：

- 文件大小保持 209,486,545 字节；
- 数据行数保持 1,948；
- 六个字段完整存在；
- 图像列至少抽查 20 条可解码；
- 答案和问题类型字段无异常整体缺失；
- 运行项目侧 ChartQAPro loader smoke test；
- 将 SHA-256、行数、字段、来源和获取日期写入 `manifest.json`。

### 5.3 版本控制与许可边界

- 两个数据集的实际内容均不提交 Git；
- Git 只跟踪数据说明、目录结构示例、manifest 格式和下载/校验脚本；
- 在 `DATASET_CARD.md` 中记录官方来源、论文、许可证/再分发边界和用途；
- README 使用环境变量或相对路径示例，不记录个人用户名路径；
- ChartQAPro 只用于最终外部测试，不参与超参数选择。

## 6. 分阶段执行顺序

### 阶段 0：建立安全检查点

1. 保存当前 `git status` 和关键目录清单；
2. 审查 `requirements-post-training.lock` 的已有改动归属；
3. 将当前未跟踪的 ForgeMM 源码、测试、脚本和文档纳入一次明确的版本检查点；
4. 确认目标磁盘至少有 1.5 GiB 可用空间；建议保留至少 3 GiB 缓冲；
5. 不在工作区未清洁、未建立检查点时删除历史内容。

### 阶段 1：建立数据目录与配置层

1. 创建工作区级 `datasets/ChartQA/` 与 `datasets/ChartQAPro/`；
2. 创建数据集卡片和 manifest 模板；
3. 增加 `forgemm_datasets.example.yaml`；
4. 增加本地配置的忽略规则；
5. 将 loader 改为统一路径解析，不再散落硬编码路径。

### 阶段 2：接入 ChartQAPro

1. 复制唯一规范文件 `chartqapro_test.parquet`；
2. 验证大小、SHA-256、行数、字段和图像解码；
3. 实现 ChartQAPro loader 与 smoke test；
4. 生成 manifest 和审计报告；
5. 保留原文件作为短期备份，直到完整评测成功。

先接入 ChartQAPro，是因为它只有一个规范文件，验证成本低，可先验证数据目录和配置设计。

### 阶段 3：接入 ChartQA

1. 复制 train/val/test 到目标目录；
2. 对照文件数、总字节数和抽样哈希；
3. 更新训练、评测和数据审计命令；
4. 运行现有单元测试、20 样本 smoke 与 val250 回归；
5. 配置和回归全部通过后，旧位置只作为短期备份。

### 阶段 4：文档归档

1. 新建 `docs/archive/forgemm_baselines/`；
2. 移动六份历史计划文档；
3. 修复实验记录和总览中的链接；
4. 将 `forgemm_chart_fgrpo_implementation_plan.md` 设为当前唯一主方案入口；
5. 在 README 增加“当前主线 / 历史实验”导航。

### 阶段 5：清理生成物

1. 先补充 `.audit_tmp/` 与项目级 `tmp/` 的忽略规则；
2. 用户手动处理可再生成的缓存目录；
3. 对 ChartQAPro 锁文件等少量明确单文件，可在再次确认后逐个删除；
4. 不删除 `artifacts/forgemm/`；
5. 不自动删除任何含实验结果、模型权重或用户修改的目录。

### 阶段 6：旧文本 LLM 资产裁决

1. 生成旧 stage 文件到 README/测试/实验记录的引用图；
2. 提取仍需要保留的指标、配置、日志摘要；
3. 给出逐文件的“保留 / 迁移 / 删除”最终表；
4. 用户确认后再逐项执行；
5. 执行后运行完整测试并检查所有文档链接。

## 7. 删除执行规则

后续任何删除都必须满足：

1. 路径是完整、明确的单个文件；
2. 已证明不是当前 Git 未提交改动；
3. 已证明不被代码、文档或实验记录引用，或引用已更新；
4. 若属于实验产物，关键指标和配置已经提取；
5. 数据迁移类文件必须先通过数量、大小、哈希和加载验证；
6. 每批执行前向用户展示最终清单；
7. 不使用递归删除或批量删除命令。

目录级缓存和虚拟环境由用户手动删除，避免违反项目安全规则。

## 8. 完成标准

项目整理只有同时达到以下条件才算完成：

- 两个数据集均从统一工作区路径稳定加载；
- ChartQA 与 ChartQAPro manifest 可复核；
- Git 仓库内没有数据集二进制、缓存或个人绝对路径配置；
- 当前 Chart-FGRPO 方案成为唯一主入口；
- 历史计划和实验记录有清楚的归档索引；
- 当前 ForgeMM 单元测试、smoke test 和基线回归通过；
- `git status` 中每一项变更都能解释；
- 未误删历史实验结果或用户已有修改；
- README 能让招聘面试官快速理解“底座模型—SFT—GRPO—FGRPO—评测”的完整链路。

## 9. 下一步执行建议

下一轮先执行“阶段 0 + 阶段 1 + 阶段 2”：

1. 建立安全检查点；
2. 创建统一数据目录、配置和 manifest 结构；
3. 仅接入并验证 ChartQAPro parquet；
4. 不删除任何文件；
5. 验证通过后，再接入体积更大的 ChartQA。

这样可以用最小的数据迁移验证整套目录设计，并将不可逆清理推迟到证据充分之后。
