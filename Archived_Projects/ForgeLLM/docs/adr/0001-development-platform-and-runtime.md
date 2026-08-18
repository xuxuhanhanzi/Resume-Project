# ADR-0001：首月开发平台与运行时策略

- 状态：接受
- 日期：2026-07-11
- 复审：G0 结束、首次 GPU 训练前、服务阶段开始前

## 背景

当前机器是 Windows 11，具有 Conda、CUDA 12.4 和 8 GB RTX 4070 Laptop GPU。现有 `SCI` 环境有 PyTorch，但它属于其他项目；Docker 未安装，WSL2 没有 Linux 发行版。ForgeLLM 后续会使用偏 Linux 的训练与推理工具，但首月只需要工程、数据和 Tokenizer 闭环。

## 候选方案

### A. 立即安装 WSL2 Linux 和 Docker，全部在 Linux/容器中开发

优点：更接近后续训练与服务环境。  
代价：环境安装可能占用首周主线，Docker 对首月正确性不是必要条件。

### B. 直接复用现有 `SCI` Conda 环境

优点：PyTorch 与 CUDA 已可用，启动快。  
代价：依赖来源不清晰，会污染已有项目并削弱复现证据。

### C. 建立 ForgeLLM 独立环境，Windows 本地开发，Linux CI 验证；Linux GPU/容器按阶段引入

优点：立即推进首月 P0，同时保持项目隔离和 Linux 兼容性。  
代价：后续仍需建立 Linux GPU 与容器环境，Windows 无法作为 vLLM 等组件的正式服务基线。

## 决策

选择方案 C：

- 建立独立 ForgeLLM 环境，不修改或复用 `SCI`；
- Python CLI 是所有平台的权威执行入口；
- Windows 用于首月开发和本地验证，Linux CI 用于跨平台证明；
- 本地 GPU 仅用于 Smoke 和资源可承受的小模型任务；
- WSL2/云端 Linux 最迟在预训练系统或推理服务阶段前建立；
- Docker 属于 P1，不能阻塞第一个月 G0/G1。

## 理由

该方案与当前资源相符，能将环境隔离、工程可复现和 Linux 兼容同时纳入证据链，又不会让非必要基础设施占用首月关键路径。

## 代价与风险

- Windows 和 Linux 的路径、进程与文件系统行为可能不同；
- 后续 GPU 包需要重新固定 Linux wheel/runtime；
- Linux CI 只证明 CPU 接口兼容，不能证明云端 GPU 训练正确。

## 控制措施

- 业务代码使用 `pathlib`，不硬编码盘符或 shell；
- 统一通过 Python CLI 调用，平台命令只做代理；
- GPU 训练前新增云端 Linux 环境记录和 Smoke；
- 服务阶段前单独评审 Docker/WSL/云平台决策。

## 复审条件

- G0 期间跨平台差异造成超过 8 小时额外成本；
- 本地 Windows 无法执行核心测试；
- 进入 FSDP、vLLM、容器或多卡阶段；
- 云平台价格或月预算发生变化。

