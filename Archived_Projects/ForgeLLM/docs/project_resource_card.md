# Project Resource Card

本卡在每个可能产生显著下载、GPU 时长或外部费用的阶段前复审。

| 项目 | 值 |
|---|---|
| 每周可投入时间 | 60 小时以上；计划按 60 小时/周排期，超出部分作为缓冲 |
| 目标周期 | 模型优先 21 周路线，另有 Buffer |
| GPU 型号与数量 | 1 × NVIDIA GeForce RTX 4070 Laptop GPU（以 `nvidia-smi` 实测为准） |
| 单卡显存 | 8188 MiB（约 8 GB） |
| 本地或云端 | 本地为开发与 Smoke 主环境；达到阶段门禁后按需租用云端 Linux GPU |
| 训练预算 | 与评测/API 共用外部资源总预算，不超过 20 USD/月；每次付费运行单独登记上限 |
| 评测预算 | 与训练共用外部资源总预算，不超过 20 USD/月；默认优先本地离线评测 |
| 可用存储 | 2026-07-27 `cmd /c dir D:\` 实测约 76,863,045,632 bytes（约 71.6 GiB） |
| Stage 3 外部成本 | 0 USD；19.4MB 数据免费下载，本地 GPU 运行 |
| Stage 4 计划预算 | 默认 0 USD；Qwen3-0.6B-Base 本地 BF16 LoRA/QLoRA，单次正式短跑不超过 45 分钟 |
| 目标岗位优先级 | P1：LLM 算法/训练；P2：AI 全栈（推理、服务、Agent、应用，无内部次序） |

## 资源使用策略

- 本地 8 GB GPU 用于数值验证、微型 batch、5M–20M 小模型和有界短训练；
- 云端只在本地 Smoke、配置冻结、评测接口和停止条件都完成后启用；
- 单次云端任务必须记录平台、GPU、单价、最长时长、预计费用和自动停止条件；
- 月度总预算耗尽后不继续付费实验，优先完成分析、测试、文档和 CPU 工作；
- Stage 4 的 0.6B Base 先在本地完成 BF16 LoRA 与 QLoRA；只有真实资源门失败且用户单独批准费用时才考虑云端；
- 1.5B 以上后训练仍采用本地 Smoke + 条件云端，不预先承诺模型规模或训练时长。

## 当前资源结论

- Stage 3 只下载 19.4MB 原始数据；处理 JSONL 和 token cache 远低于工作盘余量；
- 1M-token 正式运行峰值 allocated CUDA memory 约 235.9 MiB，无需云端；
- 20 USD 的精确预算口径仍在首次付费任务前确认，本阶段没有产生付费资源；
- Stage 4 新增磁盘软上限 10 GiB；正式运行上限为 100k assistant tokens、500 steps 或 45 分钟先到即停；
- Qwen3-0.6B-Base 权重、SmolTalk 小子集和 bitsandbytes 尚未下载/安装，实际显存、磁盘与兼容性在 Stage 4 Phase 0 复审。
