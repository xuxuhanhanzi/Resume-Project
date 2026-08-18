# 实验记录：Stage 0 Model-first Learning Route Transition

## 1. 目标

验证在保留现有工程代码、测试和历史文档的前提下，能否取消 Day 4–14 全量学习阻塞，将唯一当前路线切换为 Tokenizer→Transformer→预训练→C++/CUDA→后训练→综合评测，并保持本地工程质量门通过。

## 2. 环境

- 平台：Windows 11 `10.0.22631`；
- GPU：本次未使用；
- Python：3.12.3；
- PyTorch/CUDA：本次不需要；
- 环境：仓库内 `.venv`；
- 代码路径：`D:\Users\27475\Desktop\Resume_Project\ForgeLLM`；
- 分支：`agent/month01-engineering-study`；
- 基线 commit：`983a64ab15ce160e232210d30b63912b082b8a22`；
- Git 状态：dirty；包含既有未提交文档和本次路线重构文件；
- 数据集：只运行仓库测试夹具，不使用正式数据。

## 3. 实验变量

- 主变量：学习门槛从 Day 1–14 全量工程/数据复盘切换为核心算法完整学习、支撑知识按需补充；
- 固定项：源码、33 项既有测试、`.venv`、Windows 环境、无 GPU、无付费任务、无模型功能新增；
- 对比对象：旧 14 天学习计划与旧 Month 01–03 顺序门禁。

## 4. 命令

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
git rev-parse HEAD
git branch --show-current
git status --short
.\.venv\Scripts\python.exe -c "import platform, sys; print(sys.version); print(platform.platform())"
```

## 5. 输出路径

- 当前主计划：`docs/model_first_21_week_learning_plan.md`；
- Stage 计划：`docs/stages/learning_stage_00_model_first_transition.md`；
- 支撑卡：`docs/reference/minimum_engineering_support_card.md`；
- 讲义：`docs/lessons/stage00_model_first_transition.md`；
- 本实验记录：`docs/experiments/2026-07-24_stage00_model_first_transition.md`；
- 权重/训练日志：无。

## 6. 结果

| 指标 | 结果 |
|---|---:|
| Ruff format | 通过；27 files already formatted |
| Ruff lint | 通过 |
| mypy strict | 通过；27 source files |
| pytest | 33/33 通过 |
| 新主计划 | 已建立 |
| 旧计划处置 | 路线切换时已添加历史标记；2026-07-25 当前路线稳定后删除 5 个会干扰学习顺序的过时文档 |
| 支撑知识执行卡 | 已建立 |
| Prompt core/support 分级 | 已建立 |
| 付费/GPU 运行 | 0 |

## 7. 失败与异常

- 本次质量命令无失败；
- 工作区在执行前已经存在未提交 README、进度台账和新增计划文件，本次在其上增量修改，没有覆盖或删除；
- Git dirty 状态意味着本次结果尚未形成干净 commit/Release 证据，但不影响本地 Stage 0 路线验证；
- Linux CI 未在本实验中运行，因此不得声明跨平台质量门已经通过。

## 8. 结论

Stage 0 假设在当前本地基线上成立：既有代码和 33 项测试保持通过，模型优先路线已建立，Day 4–14 不再构成 Tokenizer 前置阻塞，一般工程/数据知识有明确按需查阅入口。

该结论只证明学习路线与本地工程基线已经完成切换，不证明正式数据、Tokenizer、Transformer、预训练、自定义算子或后训练已经完成。

## 9. 下一步

进入 Stage 1：数据最低闭环与 Tokenizer。第一批任务是生成 Tokenizer/BPE 核心完整讲义、冻结最小输入/特殊 Token 契约和手算测试样例；在 Stage 计划通过前不实现正式训练器。
