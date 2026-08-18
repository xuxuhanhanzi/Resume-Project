# 实验记录：Stage 0 环境与数据身份基线

## 1. 目标

建立 ForgeMM 独立仓库的第一条可复现证据链：冻结开发依赖、统一质量入口、记录当前
硬件与包版本，并验证 ChartQA/ChartQAPro 本地副本与既有 manifest 一致。

本阶段不训练模型，也不把 Windows RTX 4070 环境视为计划中的 Linux RTX 4090 正式环境。

## 2. 环境

- 平台：Windows 11
- GPU：NVIDIA GeForce RTX 4070 Laptop GPU，8,188 MiB
- 驱动：610.74
- Python：3.12.3
- PyTorch：2.6.0+cu124
- Transformers：5.14.1
- bitsandbytes：0.49.2
- 当前执行环境：复用 ForgeLLM Python 3.12 环境完成 Stage 0 只读审计
- 代码路径：`D:\Users\27475\Desktop\Resume_Project\ForgeMM`

## 3. 实验变量

- 主变量：无；本阶段只冻结环境与数据身份
- ms-swift 候选：官方 release `v4.2.2`，短 commit `f279713`
- 固定数据：项目内 `datasets/ChartQA` 与 `datasets/ChartQAPro`

候选版本来源：

- <https://github.com/modelscope/ms-swift/releases/tag/v4.2.2>
- <https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html>

`v4.2.2` 只是 Stage 1 候选。完整 CUDA/PyTorch/vLLM 组合必须在 Linux/4090 上完成
Qwen2.5-VL 推理、QLoRA SFT、GRPO 和额外字段透传后再锁定。

## 4. 命令

```powershell
..\ForgeLLM\.venv\Scripts\python.exe scripts\audit_environment.py `
  --project-root . `
  --output artifacts\runs\stage00_environment\environment.json

..\ForgeLLM\.venv\Scripts\python.exe scripts\dev.py check
```

独立环境安装尝试：

```powershell
..\ForgeLLM\.venv\Scripts\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.lock
```

## 5. 输出路径

- 环境报告：`artifacts/runs/stage00_environment/environment.json`
- 开发依赖锁：`requirements-dev.lock`
- 训练栈候选：`requirements-swift.in`
- 质量入口：`scripts/dev.py`
- CI：`.github/workflows/ci.yml`

## 6. 结果

| 指标 | 结果 |
|---|---:|
| ChartQA 关键文件哈希 | 6/6 一致 |
| ChartQAPro 大小与哈希 | 一致 |
| Format | passed |
| Ruff | passed |
| strict mypy | 28 files, 0 errors |
| pytest | 43 passed |
| ms-swift/vLLM | 未安装，Stage 1 待验证 |
| 独立 `.venv` | 从锁文件重建成功，CPU 质量门通过 |

## 7. 失败与异常

- 首次使用清华 PyPI 镜像下载 `pyarrow==25.0.0` 时持续超时，失败过程被保留。
- 2026-08-11 使用官方 PyPI、120 秒单包超时和同一锁文件重试成功；没有修改锁定版本。
- 独立环境复现门已关闭，证据位于
  `artifacts/runs/stage00_environment_v2/environment.json`。

## 8. 结论

本地数据身份已经冻结，独立 CPU 开发环境可从锁文件重建，Stage 0 基线完成。Linux/4090
训练环境版本矩阵仍属于 Stage 1 smoke，不能据此声称 ms-swift 兼容或训练就绪。

## 9. 下一步

1. 准备 Linux/4090 Stage 1 smoke 配置和最小样本；
2. 依次运行单图推理、QLoRA SFT、`4 prompts × 2 rollouts` GRPO 与字段透传；
3. Stage 1 通过后生成完整训练依赖锁。
