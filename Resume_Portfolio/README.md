# 简历作品集项目总览

本目录集中管理四个用于简历展示与后续迭代的项目。各项目保留独立 Git 历史、README、实验记录和可复现入口；请从各项目自己的 README 开始了解具体实现。

统一的 100% 完成定义、验收门槛与执行顺序见 [`COMPLETION_ROADMAP.md`](COMPLETION_ROADMAP.md)。

## 实验 PC 自动化入口

RTX 5090 实验机采用 Windows 11 + WSL2 Ubuntu 基线。克隆仓库后，在
`Resume_Portfolio` 目录运行：

```bash
bash scripts/bootstrap_experiment_pc.sh smoke
bash scripts/run_all_experiments.sh smoke
```

正式实验的数据许可确认、两条启动命令、阶段顺序、断点续跑和产物位置见
[`docs/EXPERIMENT_PC_RUNBOOK.md`](docs/EXPERIMENT_PC_RUNBOOK.md)。

| 类别 | 项目 | 简历定位 | 当前证据口径 |
| --- | --- | --- | --- |
| VLA | `DriveVLA-Guard` | 自动驾驶 VLA 推理时风险约束与评测 | 固定 AutoVLA 与 NAVSIM 协议，聚焦风险重排序、路由和逐场景证据。 |
| 多模态 LLM | `ForgeMM` | 图表问答可信推理的多模态大模型后训练 | QLoRA SFT、GRPO 与 Chart-FGRPO 研究；训练结果尚未验证，不做性能宣称。 |
| Agent | `RepoPilot` | 本地优先、可治理的 Agent Runtime Lab | 覆盖研究、数据分析和软件工程三类评测，指标以 `docs/STATUS.md` 为准。 |
| 实习项目（后续重构） | `Hospital_Workforce_Platform` | 医院人力、排班与薪资管理平台 | Spring Boot 模块化单体，覆盖权限、审计、并发安全排班、容器化和测试/监控。实习经历与后续重构成果须分开表述。 |

## 管理约定

- 项目代码、文档、配置和可引用的实验产物保留在对应项目目录。
- Python 虚拟环境、测试/类型检查/格式化缓存、构建目录和临时审计目录均为本机派生产物，不纳入简历交付物；需要时按项目锁定文件重建。
- 新增项目或实习材料时，在本表补充定位、成果口径与项目入口。
- 四个项目在本作品集仓库中统一交付，但仍保持各自的依赖、配置、实验记录和运行入口。
