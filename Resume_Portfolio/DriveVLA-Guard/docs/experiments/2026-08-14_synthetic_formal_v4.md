# 实验记录：synthetic_formal_v4

> 历史运行：为保证 Windows 与 Linux 干净克隆哈希一致，发布证据以 LF 规范化后的 `formal_v5` 为准。

## 1. 目标

在完整 Ruff 规则修复后最终冻结 B0/B1/E1–E4，并验证任何决策源码变更都会使旧结果身份失效。

## 2. 环境

- 平台：Windows 11；Python 3.12.3；
- GPU：不使用；NumPy 1.26.4；PyYAML 6.0.1；
- manifest：`data/synthetic/smoke_manifest.jsonl`，12 个冻结场景；
- 上游 AutoVLA commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`。

## 3. 精确命令

```powershell
python scripts/run_synthetic_suite.py `
  --output-root artifacts/runs/formal_v4 `
  --manifest data/synthetic/smoke_manifest.jsonl `
  --scenes 12
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m drivevla_guard.cli freeze-evidence --root . --output artifacts/evidence_manifest.json
python -m drivevla_guard.cli verify-evidence
```

## 4. 结果

- 6 组运行、72/72 场景成功；
- B0 碰撞代理失败 6/12，E2/E4 为 0/12；
- E3/E4 slow-route rate 25%；
- E2/E4 平均代理延迟 16.0/19.5 ms；
- 6 个 summary 与 3 个 comparison 均从逐场景 JSONL 重算一致；
- manifest、配置、结果、结论文档与决策关键源码进入 SHA-256 清单。

## 5. 结论边界

这是确定性合成工程证据，不是 AutoVLA 模型推理或 NAVSIM PDMS。正式评测仍受 GPU、checkpoint、Qwen 权重、NAVSIM 数据和 metric cache 外部门禁约束。
