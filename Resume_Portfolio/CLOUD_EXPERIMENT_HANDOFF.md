# 云端实验交接单

> 冻结日期：2026-08-14  
> 当前状态：所有可独立在本机完成的实验均已完成。此文件只列云端实验与由云端产物触发的验收。

## 1. DriveVLA-Guard

### 云端前置

- Linux，单卡显存至少 24GB；
- 官方 AutoVLA checkpoint（约 16.3GB）与 Qwen2.5-VL-3B 权重；
- NAVSIM navtest 数据、metric cache 与项目锁定的 AutoVLA 上游 commit；
- 严格 preflight 必须返回 `ready=true`。

### 执行顺序

1. 记录 GPU、驱动、Python/PyTorch/CUDA、代码 diff、模型/数据 SHA-256；
2. 固定同一 navtest manifest，运行官方/工程 B0；
3. 只替换为 K=4 风险重排序，运行 E2；
4. 只有 NC/DAC/TTC 改善且 EP/Comfort 不越过预登记退化门槛时，才运行 E3/E4；
5. 冻结逐场景预测、PDMS 分解、P50/P95 延迟、峰值显存与失败案例。

### 停止规则

- B0/E2 不满足安全子指标门槛：停止 E3/E4，保留负结果；
- 任一资产 hash、upstream commit 或 manifest 不一致：不得启动正式评测；
- 运行手册：[official_runbook.md](DriveVLA-Guard/docs/official_runbook.md)。

## 2. ForgeMM

### 云端前置

- Linux，单卡 24GB+（预登记 RTX 4090）；
- Qwen2.5-VL-3B-Instruct、ms-swift 4.2.2 与兼容 CUDA/torch/vLLM/bitsandbytes；
- 本地已冻结的 v2 Structured SFT/GRPO 数据上传后重新核对 SHA-256；
- ChartQA val/test 与 ChartQAPro 使用边界保持不变。

### 执行顺序

1. Stage 1：单图推理 → ChartQA 20 条基线 → 32 样本 QLoRA → `4 prompts × 2 rollouts`
   标准 GRPO → 额外字段透传 → advantage/checkpoint hook；
2. Stage 4：E1 answer-only QLoRA、E2 structured QLoRA，满足 Format ≥98%、Operation
   Executable ≥90% 且 val accuracy 退化不超过 1pp 后冻结共同 E2 checkpoint；
3. Stage 5–6：E3/E4/E5 各 50–100 steps quick，只依据 OOM、稳定性、reward variance、
   KL、clip ratio 与截断率统一冻结公共配置；
4. Stage 7：E3/E4/E5 × seeds 17/42/2026；A1/A2 × seed 42；
5. Stage 8：依序进行 val checkpoint 选择、ChartQA test 一次正式评测、ChartQAPro 一次
   外部评测，报告 FCR、Evidence F1、Operation Consistency、准确率、bootstrap CI 与 McNemar；
6. 云端生成最佳 Adapter 后，再触发本机 4-bit 演示验收。该验收依赖云端产物，不是当前可执行实验。

### 停止规则

- Stage 1 任一 smoke 未通过：不得启动正式训练；
- E2 未达到进入 RL 门槛：停止 E3–E5，先修数据/协议；
- E5 未优于 E3：如实保留负结果，不宣称提升；
- 实施计划：[implementation_plan.md](ForgeMM/docs/implementation_plan.md)。

## 3. 无云端实验的项目

- Hospital Workforce Platform：实验闭环；只剩演示视频与远程发布；
- RepoPilot：实验闭环；只剩远程发布；
- 以上外部交付动作不计入云端实验矩阵。
