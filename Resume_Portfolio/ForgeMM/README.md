# ForgeMM

实验 PC 的自动安装与顺序运行入口为 `scripts/setup_experiment_pc.sh` 和
`scripts/run_experiment_pc.sh`；正式档会下载并核验 ChartQA、ChartQAPro 与
Qwen2.5-VL-3B，然后通过可续跑的 Stage 04 驱动完成 SFT、GRPO、消融、冻结选型和测试。

ForgeMM 是一个面向图表问答可信推理的多模态大模型后训练项目。项目以
Qwen2.5-VL-3B-Instruct 为底座、以 ms-swift 为训练依赖，通过结构化 QLoRA
SFT、标准 GRPO 和 FGRPO-inspired Chart-FGRPO，研究答案正确性、证据一致性与
运算一致性之间的联合优化。

当前状态：Stage 0 的开发质量门、环境审计和两个数据集的冻结身份校验已完成；
Stage 2–5 的 loader、全量数据审计、EvidenceStore、严格 Parser、安全 Executor、
三通道 Reward、ms-swift reward plugin 契约、Chart-FGRPO advantage/dual state 和冻结评测统计已实现。2026-08-15 已在 AutoDL RTX 4090 上完成 Stage 1：单图/20 样本推理、
NF4 QLoRA、三路字段透传、非恒定奖励 GRPO 与 checkpoint 保存全部通过。正式 E0–E5/A1–A2
质量矩阵仍未运行，因此当前不宣称模型性能提升。

## 项目边界

- ms-swift 负责通用 SFT、QLoRA、GRPO、rollout、checkpoint 与训练日志；
- ForgeMM 负责 ChartQA/ChartQAPro 数据适配、EvidenceStore、Verifier、Reward、
  Chart-FGRPO Trainer 扩展和统一评测；
- 数据集、模型和运行产物均保留在本地，不提交 Git；
- 在实验完成前不宣称性能提升或 SOTA。

详细实施步骤见：[`docs/implementation_plan.md`](docs/implementation_plan.md)。

## 当前证据

- `python scripts/dev.py check`：44 个源码/测试/脚本文件通过格式、Lint、strict mypy，
  pytest 56 passed；
- ChartQA 6 个关键 JSON 的 SHA-256 与冻结 manifest 全部一致；
- ChartQAPro parquet 为 209,486,545 字节，SHA-256
  `6209a9a6f7307b761e70ff9e708cb7505e0327d7eb932aa26152b8240f633da5`，与 manifest 一致；
- 当前 Windows RTX 4070 环境没有安装 ms-swift/vLLM，不视为 Stage 1 训练环境；
- 独立 `.venv` 已从 `requirements-dev.lock` 成功重建，并用自身 Python 3.12.3
  通过环境审计和完整质量门；它是 CPU 开发环境，不是 ms-swift 训练环境。
- ChartQA 全量审计发现 5 张损坏 train 图、16 个跨 split 重复图像哈希；这些训练记录
  已按原因码关闭 evidence/operation mask；
- ChartQAPro 为 1,948 行、2,709 个问题，全部图片可解码，且 loader 只允许 `test`；
- EvidenceStore v7（schema 1.3.0 / builder `chartqa-store-1.1.0` / labeler
  `rules-1.3.0`）在去除 161 条精确重复训练记录后含 1,763 条问题级严格标签：
  human 580 + augmented 1,183；全部 gold operation 均能按参考答案显示精度复现答案。
  该数量达到 GRPO 的 1,500 下限；在不降低标签标准的前提下，每条严格记录生成两个可追踪 SFT 模板视图，形成 3,526 条训练序列和 1,763 个唯一 group key。模板视图不被表述为新增独立 QA。
- Stage 4 v1 暴露出项目迁移后的绝对图片路径失效；该失败版本保留。v2 改用可迁移的
  `datasets/...` 相对路径，1,763/1,763 图片存在，3,526/3,526 gold completion 均可解析且
  task/evidence/operation reward 全为 1。v2 Structured SFT SHA-256 为
  `fa3ac2ebbd179d5efd8d436a726ad9197e37351e5b17b979fb5474eddfd5548f`；GRPO SHA-256 为
  `eab7b4e929a77da29ad98d7d110b88ec2c32df69ee73de892b2eaa966d9d8b6a`。
- Chart-FGRPO CPU 数值门覆盖组内标准化、mask、双变量更新/裁剪、checkpoint 恢复；评测覆盖 FCR、inconsistency、paired bootstrap 95% CI 和 exact McNemar。
- 本地 E3/E4/E5 固定张量实验完成 200 steps × 3 seeds；所有 advantage 有限，E5 从中途
  checkpoint 恢复后的最终 dual state 在三个种子上均精确一致。该结果是算法诊断，不是模型训练结果。
- AutoDL RTX 4090 Stage 1 已完成：20 样本底座推理为 0.899 samples/s；50-step
  Structured QLoRA 最终 token accuracy 0.9889；4×2 GRPO 的 reward/std 分别出现
  0.5/0.7071 与 1.5/2.1213，三路 reward 均非零，checkpoint-4 完整保存。详见
  [`docs/experiments/2026-08-15_stage01_autodl_gpu_smoke.md`](docs/experiments/2026-08-15_stage01_autodl_gpu_smoke.md)。
- `v0.2.1` 已在独立干净克隆中按锁文件标准隔离构建，并通过 Ruff、strict mypy、56 tests 和无 payload 环境审计；`v0.2.2` 固化该记录，`v0.2.3` 关闭本地数据/reward/CPU 算法实验门。

环境审计命令：

```powershell
python scripts/audit_environment.py --project-root . `
  --require-datasets `
  --output artifacts/runs/stage00_environment/environment.json

python scripts/audit_datasets.py `
  --output artifacts/runs/stage02_data_audit/report.json

python scripts/build_evidence_store.py `
  --audit-report artifacts/runs/stage02_data_audit/report.json `
  --output artifacts/runs/stage03_evidence_store_v7/train_human.jsonl `
  --summary artifacts/runs/stage03_evidence_store_v7/summary_human.json

python scripts/build_training_datasets.py `
  --evidence-store artifacts/runs/stage03_evidence_store_v7/train_human.jsonl `
  --evidence-store artifacts/runs/stage03_evidence_store_v7/train_augmented.jsonl `
  --sft-output artifacts/runs/stage04_training_data_v2/structured_sft.jsonl `
  --grpo-output artifacts/runs/stage04_training_data_v2/grpo.jsonl `
  --summary artifacts/runs/stage04_training_data_v2/summary.json `
  --portable-image-root .
```

## 目录

```text
ForgeMM/
├── configs/          # 可提交的训练与评测配置
├── datasets/         # 本地数据；payload 被 Git 忽略
├── docs/             # 设计、实验记录与结果报告
├── scripts/          # 数据、训练、评测入口
├── src/forgemm/      # 项目核心代码
├── tests/            # 单元、集成和 smoke 测试
└── artifacts/        # 本地运行产物
```
