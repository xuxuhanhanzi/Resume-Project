# 实验记录：local_data_reward_and_fgrpo_v2

## 1. 目标

完成所有不依赖 GPU 模型训练的本地实验：训练数据路径可迁移性、全量结构/图片/reward 回放，
以及 E3/E4/E5 固定张量算法对照与 dual checkpoint 恢复。

## 2. 环境

- 平台：Windows 11，RTX 4070 Laptop 8GB；
- Python：3.12.3；NumPy：2.2.6；
- 环境：项目 `.venv`，未安装 torch/transformers/ms-swift；
- 代码基线：`9575814` + 本记录对应 changed files；
- 数据：EvidenceStore v7，human 580 + augmented 1,183 条严格唯一记录。

## 3. 实验变量

- 数据实验：迁移前绝对图片路径 vs `datasets/...` 项目相对路径；
- reward 回放：全部 3,526 个 SFT assistant gold completion；
- 算法实验：E3 task-only、E4 固定双约束权重、E5 动态双变量；
- 固定项：seed 17/42/2026，每种子 200 steps、32 groups/step、4 samples/group。

## 4. 命令

```powershell
.\.venv\Scripts\python.exe scripts\build_training_datasets.py `
  --evidence-store artifacts\runs\stage03_evidence_store_v7\train_human.jsonl `
  --evidence-store artifacts\runs\stage03_evidence_store_v7\train_augmented.jsonl `
  --sft-output artifacts\runs\stage04_training_data_v2\structured_sft.jsonl `
  --grpo-output artifacts\runs\stage04_training_data_v2\grpo.jsonl `
  --summary artifacts\runs\stage04_training_data_v2\summary.json `
  --portable-image-root .

.\.venv\Scripts\python.exe scripts\audit_training_datasets.py `
  --sft artifacts\runs\stage04_training_data_v2\structured_sft.jsonl `
  --grpo artifacts\runs\stage04_training_data_v2\grpo.jsonl `
  --output artifacts\runs\stage04_training_data_v2\full_audit.json --project-root .

.\.venv\Scripts\python.exe scripts\benchmark_chart_fgrpo_cpu.py `
  --output artifacts\runs\stage05_cpu_algorithm_v1\summary.json `
  --steps 200 --groups 32 --group-size 4
```

## 5. 输出路径

- v2 数据与原始审计：`artifacts/runs/stage04_training_data_v2/`；
- CPU 算法原始结果：`artifacts/runs/stage05_cpu_algorithm_v1/summary.json`；
- 可提交摘要：`artifacts/evidence/stage04_training_data_full_audit.json`、
  `artifacts/evidence/stage05_cpu_algorithm_summary.json`。

## 6. 结果

| 本地门 | 结果 |
|---|---:|
| 严格唯一 GRPO groups | 1,763/1,763 |
| 双模板 SFT rows | 3,526/3,526 |
| 可定位图片 | 1,763/1,763 |
| gold completion 三通道 reward 全 1 | 3,526/3,526 |
| E3/E4/E5 finite advantages | 3/3 seeds |
| E5 中途 checkpoint 最终状态精确恢复 | 3/3 seeds |

v1 的绝对图片路径在项目迁移后全部失效；v2 改用项目相对路径并通过全量检查。v2 SFT
SHA-256 为 `fa3ac2ebbd179d5efd8d436a726ad9197e37351e5b17b979fb5474eddfd5548f`，
GRPO SHA-256 为 `eab7b4e929a77da29ad98d7d110b88ec2c32df69ee73de892b2eaa966d9d8b6a`。

## 7. 失败与异常

- v1 路径迁移失败保留为历史证据，不删除、不覆盖；
- CPU 固定张量不包含模型 rollout、KL、显存或训练 loss，不能代替 ms-swift smoke。

## 8. 结论

本地数据、Parser、Verifier、reward、advantage、dual update 与断点恢复实验已经闭环。真实
Qwen2.5-VL 推理、QLoRA、GRPO 与 E0–E5 质量比较仍必须在云端训练环境完成。

## 9. 下一步

在 Linux 24GB+ GPU 上按 Stage 1 顺序完成 ms-swift 4.2.2 推理 → 32 样本 QLoRA →
4×2 GRPO → 字段透传，之后才进入 E0–E5/A1–A2。
