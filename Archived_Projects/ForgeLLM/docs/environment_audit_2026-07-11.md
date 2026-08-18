# 开发环境盘点：2026-07-11

## 1. 结论

当前机器足以完成第一个月工程、数据与 Tokenizer 工作，也足以承担后续 5M–20M 模型的本地 Smoke 和部分训练。当前不具备 Docker 环境，也没有已安装的 WSL Linux 发行版；因此第一个月采用 Windows 本地开发 + Linux CI 的分阶段策略，容器与 Linux GPU 运行不作为 G0 的阻塞条件。

## 2. 实测环境

| 项目 | 实测值 | 判断 |
|---|---|---|
| 操作系统 | Windows 11，build 22631（由 Python 平台信息获得） | 可作为首月本地开发环境 |
| PowerShell | 5.1.22621.6931 | 可运行项目管理命令 |
| Git | 2.53.0.windows.2 | 可用 |
| Base Python | 3.12.3，`E:\anaconda3\python.exe` | 未安装 PyTorch，不直接作为训练环境 |
| Conda | 24.11.3 | 可用于建立独立项目环境 |
| 既有环境 | `base`、`SCI` | 不复用 `SCI` 作为 ForgeLLM 正式环境 |
| SCI Python | 3.10.20 | 仅作已有能力参考 |
| SCI PyTorch | 2.5.1+cu121 | CUDA 可用 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU | 用户口述的“4070 Ti”以实测型号修正 |
| GPU 显存 | 8188 MiB | 适合小模型和 Smoke；大模型正式运行需云端或节省显存方案 |
| NVIDIA 驱动 | 610.74 | 当前可识别 GPU |
| Compute Capability | 8.9 | 支持现代 NVIDIA GPU 训练能力 |
| CUDA Toolkit | 12.4 / nvcc 12.4.99 | 已安装；项目实际 CUDA runtime 由 PyTorch 包决定 |
| Docker | 未找到命令 | P1，不能阻塞首月 G0 |
| WSL | WSL2 功能存在，但没有已安装的 Linux 发行版 | 后续 Linux 开发环境待建立 |
| 磁盘 | 沙箱内无法可靠读取整盘容量 | 需用户人工确认 |

## 3. 发现的问题

### E-001：没有 ForgeLLM 独立环境

`base` 为 Python 3.12.3 且没有 PyTorch；`SCI` 虽然有 CUDA PyTorch，但属于已有项目环境。复用会混淆依赖和实验证据。

**处理：** Day 2 建立独立的 ForgeLLM 环境；首月只安装工程、数据和 Tokenizer 所需的最小依赖。训练依赖在模型阶段再锁定，避免过早引入重型栈。

### E-002：本地 Linux/容器链路缺失

Docker 未安装，WSL 没有发行版。vLLM 等后续组件不应以 Windows 原生作为正式目标。

**处理：** 首月以跨平台 Python CLI 为权威入口，并使用 Linux CI 做兼容性证明；最迟在预训练系统或服务阶段前建立 WSL2/云端 Linux 环境。

### E-003：8 GB 显存约束

8 GB 显存无法支持计划中所有 0.5B–1.5B 后训练配置，但不会阻塞首月或 5M–20M 模型主线。

**处理：** 所有正式 GPU 任务遵循 CPU → 本地微型 GPU Smoke → 云端短跑 → 云端正式跑；付费运行必须有预算和停止条件。

### E-004：存储容量未知

当前无法从受限环境可靠读取整盘可用容量。

**处理：** 第一个月正式数据下载前人工确认至少 20 GB 可用空间；训练前按数据、Checkpoint 和 Artifact 预算重新评估。

## 4. Day 2 环境目标

- 独立环境，不修改 `SCI`；
- 项目包可 editable install；
- Python CLI 是 Windows、本地 Linux 和 CI 的统一入口；
- 依赖按 core/dev 分组，首月不安装 vLLM、DeepSpeed、FSDP 附加栈；
- 生成可追踪的环境快照；
- 任何新安装都记录版本和精确命令。

## 5. 证据命令摘要

本记录来自 2026-07-11 的只读盘点：`python --version`、`python -m pip --version`、`conda info --envs`、`git --version`、`nvidia-smi`、`nvcc --version`、`docker --version`、`wsl --status`，以及 `SCI` 环境中的 PyTorch CUDA 查询。

盘点只证明命令在当前机器上被检测到，不等于 ForgeLLM 环境已经可复现；后者由 G0 单独验收。

