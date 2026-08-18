# 实验计划：Stage 0 环境、上游与 baseline 契约

> 状态：计划中，尚未执行  
> 目标平台：本地 Windows 开发环境 + AutoDL Linux 正式推理环境

## 1. 阶段目标

在实现任何个人模块前，证明官方 AutoVLA checkpoint、数据和 NAVSIM 接口可以在固定环境中端到端运行，并冻结 B0 baseline 的全部身份信息。

## 2. 假设

AutoVLA 官方代码、checkpoint 和 NAVSIM 集成能够在受控 Linux/CUDA 环境中完成至少一个场景的推理与评测；若不能，问题可以被定位为依赖、数据、显存、checkpoint 或接口中的一类。

## 3. 唯一主变量

Stage 0 不比较算法变量。唯一目标是从“未验证”变为“官方链路可运行”。

## 4. 固定项

首次 smoke 前冻结：

- AutoVLA Git commit；
- AutoVLA checkpoint revision 和文件 hash；
- NAVSIM commit/branch；
- Python、PyTorch、CUDA；
- 数据版本和最小场景清单；
- 官方 processor、codebook 和默认生成参数；
- GPU 型号。

## 5. 对比对象

- 官方文档给出的运行方式；
- 本项目 B0 adapter 的运行方式。

B0 adapter 必须在 K=1、greedy、fast mode 下与上游输出一致，才能进入 Stage 2。

## 6. 必须完成的检查

### 6.1 许可证和来源

- 仓库 license；
- checkpoint/model card license；
- NAVSIM/OpenScene/nuPlan 数据条款；
- 可以提交 Git 的文件范围；
- 不可重新分发的权重和数据路径。

### 6.2 环境

记录：

```bash
pwd
git rev-parse HEAD
python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

### 6.3 最小 smoke

依次完成：

1. 模型/processor 仅加载；
2. 官方样例或单场景预处理；
3. 单场景 fast inference；
4. Action Token 到轨迹解码；
5. 单场景 NAVSIM metric；
6. 相同配置重复两次；
7. 捕获峰值显存和分项延迟。

## 7. 指标

- 环境加载成功；
- 单场景完成率；
- 输出 token/轨迹确定性；
- metric 生成成功；
- 峰值显存；
- 模型加载、预处理、生成、解码、评分耗时；
- 所有异常数量和类型。

## 8. 成功门槛

Stage 0 只有同时满足以下条件才完成：

1. 一个场景端到端通过；
2. 相同 seed/config 重复运行输出一致，或确定性差异得到解释；
3. checkpoint、commit、数据和配置可追溯；
4. 结果可以写入独立 JSONL/JSON；
5. 显存不 OOM，或已经确定更高显存的最低需求；
6. B0 的定义与官方默认差异已逐项列出。

## 9. 失败处理

按顺序定位：

1. build/import only；
2. model load only；
3. data preprocess only；
4. generation only；
5. trajectory decode only；
6. NAVSIM eval only。

任何失败记录精确命令、traceback、环境版本、判断、修改和结果。不得通过同时升级多个依赖解决问题。

## 10. 预期产物

- `artifacts/stage00/environment.json`；
- `artifacts/stage00/upstream_lock.json`；
- `artifacts/stage00/license_audit.md`；
- `artifacts/stage00/smoke_predictions.jsonl`；
- `artifacts/stage00/smoke_metrics.json`；
- `artifacts/stage00/timing_memory.json`；
- 实验记录 `docs/experiments/<date>_stage00_official_smoke.md`；
- 冻结的 B0 配置快照。

## 11. Stage 0 后的决策

- `Proceed`：全部门槛通过，进入 B0 baseline 实现；
- `Proceed with constraint`：只能在更高显存或特定依赖版本下运行，约束已记录；
- `Switch baseline`：同一阻塞条件连续三次确认且无法在计划预算内解决，此时才评估 Senna 或 UniDriveVLA checkpoint；
- `Stop`：数据/许可证或 checkpoint 条件不允许形成可公开项目。

