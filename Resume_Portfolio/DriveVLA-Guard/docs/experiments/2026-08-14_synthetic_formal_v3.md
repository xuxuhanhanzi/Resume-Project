# 实验记录：synthetic_formal_v3

> 历史运行：更严格 Ruff 修复改变了决策源码身份，发布证据已追加为 `formal_v4`。

## 1. 目标

在代码格式化与发布门增强后，重新冻结 B0/B1/E1–E4，证明配置/决策源码哈希能阻止旧结果被误续跑，并建立可自动重算的证据清单。

## 2. 环境

- 平台：Windows 11；
- GPU：不使用；
- Python：3.12.3；
- NumPy：1.26.4；PyYAML：6.0.1；
- 数据：`data/synthetic/smoke_manifest.jsonl`，12 个冻结场景；
- 上游 AutoVLA commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`。

## 3. 精确命令

```powershell
python scripts/run_synthetic_suite.py `
  --output-root artifacts/runs/formal_v3 `
  --manifest data/synthetic/smoke_manifest.jsonl `
  --scenes 12
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m drivevla_guard.cli freeze-evidence `
  --root . `
  --output artifacts/evidence_manifest.json
python -m drivevla_guard.cli verify-evidence
```

## 4. 结果

- 6 组运行、72/72 场景成功，无错误样本；
- B0 碰撞代理失败 6/12，E2 与 E4 均为 0/12；
- E3/E4 slow-route rate 均为 25%；
- E2/E4 平均延迟分别为 16.0/19.5 ms；
- 6 组 summary 与 3 组配对比较均由逐场景 JSONL 重算一致；
- 证据清单覆盖 manifest、6 个配置、72 条结果、6 个 summary、3 个 comparison、核心源码和结论文档。

## 5. 结论边界

本次运行证明发布后的工程链路、恢复身份和证据完整性，不是 AutoVLA 模型推理，也不是 NAVSIM PDMS。正式 B0/E2 仍需 24GB 以上 GPU、checkpoint、Qwen 权重、NAVSIM 数据和 metric cache。
