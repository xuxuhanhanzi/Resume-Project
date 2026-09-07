# ForgeMM-Controlled v1-Lite 数据门禁 — 2026-08-24

## 结论

`advance`：受控视觉证据数据已经满足本实验注册的正式最小样本量，并通过自动可审计性检查。
这不是对旧 ChartQA 严格标签门禁的修改；ChartQA 的 val 106 / test 123 结论保持不变。

## 冻结数据

| Split | 记录数 | 正式最小值 | 结果 |
|---|---:|---:|---|
| train | 5,000 | 1,500 | pass |
| val | 600 | 200 | pass |
| test | 1,000 | 500 | pass |

- 数据版本：`forgemm-controlled-v1-lite`
- 生成器：`forgemm-controlled-pillow-1.0.0`
- root seed：`20260824`
- 图像数：6,600；跨 split 重复图像 SHA-256：0。
- oracle：每个 mark 的 source id、row、column、value、bbox，及固定白名单操作和答案。

## 自动审计

审计工件：
`artifacts/runs/2026-08-24_controlled_v1_lite/dataset_audit_with_views.json`。

- 6,600/6,600 图像 SHA-256 与从冻结字段重渲染出的 PNG 一致；
- 6,600/6,600 canonical gold completion 达成 visual full-pass；
- train、val、test 均拒绝 wrong-bbox、wrong-answer、missing-evidence 三种攻击；
- val/test prompt view 没有 answer、gold operation、evidence 或 oracle 字段；
- 三个 train view（answer SFT、structured SFT、GRPO）各有 5,000 条，并按 sample id 一一对齐。

关键 view 哈希：

| 文件 | SHA-256 |
|---|---|
| `views/answer_sft_train.jsonl` | `701b4b8a1f332f509cf3d0cd0740337e91db2f133355daf0905a2557218376e3` |
| `views/sft_train.jsonl` | `f72ccba85615187abad1f806b29d03c85c0e31029f514fcc3866656c1522f230` |
| `views/grpo_train.jsonl` | `276a30ad84535d16cdf8cd1cf028147ef8e965667ee9b571950a935972ad040d` |

## 当前执行状态

新的 visual oracle 与本地 Lite 运行脚本已完成；Ruff、strict mypy 和完整 pytest 均通过
（89 passed）。Ubuntu WSL 已安装，但首次启动需要交互式创建 Linux 用户。完成该一步后，依次运行：

```bash
cd /mnt/d/Users/27475/Desktop/Resume_Project/Resume_Portfolio/ForgeMM
bash scripts/setup_local_wsl.sh
bash scripts/run_local_lite_smoke.sh
```

只有 runtime audit 和一阶 SFT smoke 都通过，才执行 `run_local_lite_e0.sh`；其 val gate
返回 `advance` 后才允许 `run_local_lite_grpo.sh`。在此之前不存在模型性能结论。
