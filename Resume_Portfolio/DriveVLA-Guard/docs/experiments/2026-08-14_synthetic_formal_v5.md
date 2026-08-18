# 实验记录：synthetic_formal_v5

## 1. 目标

最终冻结跨平台 B0/B1/E1–E4 证据。所有生成器显式写 LF，避免 Windows Git 清理或 Linux 干净克隆改变 JSON/JSONL 哈希。

## 2. 环境

- Windows 11；Python 3.12.3；CPU 合成评测；
- NumPy 1.26.4；PyYAML 6.0.1；
- manifest：`data/synthetic/smoke_manifest.jsonl`，12 个场景；
- AutoVLA commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`。

## 3. 命令

```powershell
python scripts/run_synthetic_suite.py `
  --output-root artifacts/runs/formal_v5 `
  --manifest data/synthetic/smoke_manifest.jsonl `
  --scenes 12
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m drivevla_guard.cli freeze-evidence --root . --output artifacts/evidence_manifest.json
python scripts/check.py
```

## 4. 结果

- 6 组运行、72/72 场景成功；
- B0 碰撞代理失败 6/12，E2/E4 为 0/12；
- E3/E4 slow-route rate 25%；E2/E4 平均代理延迟 16.0/19.5 ms；
- summary 和 comparison 全部可从逐场景 JSONL 精确重算；
- frozen manifest 覆盖 stage00 preflight/upstream lock、配置、正式结果、关键源码与结论文档。

## 5. 结论边界

仅证明确定性合成工程链路与证据可复现；不代表 AutoVLA checkpoint 已运行，不代表 NAVSIM PDMS 或真实道路安全提升。
