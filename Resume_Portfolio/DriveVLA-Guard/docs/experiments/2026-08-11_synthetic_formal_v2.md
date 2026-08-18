# 实验记录：synthetic_formal_v2

> 历史运行：发布证据已由 2026-08-14 的 `formal_v3` 取代；本记录保留，不覆盖。

## 1. 目标

验证 B0/B1/E1–E4 的实现、路由门槛、逐场景记录、配对比较和可视化链路。

## 2. 环境

- 平台：Windows 11
- GPU：本实验不使用 GPU
- Python：3.12.3
- 当时的代码路径：`D:\Users\27475\Desktop\Resume_Project\DriveVLA-Guard`（历史路径）
- 数据：`data/synthetic/smoke_manifest.jsonl`

## 3. 实验变量

- 主变量：按 B0/B1/E1–E4 定义切换候选数量、重排序和路由；
- 固定项：12-scene manifest、seed 17、10 poses、0.5s、相同风险权重；
- 路由阈值：pilot 后冻结 risk 13.8、margin 0.05。

## 4. 命令

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
powershell -ExecutionPolicy Bypass -File .\scripts\run_synthetic_suite.ps1
```

## 5. 输出路径

- 正式逐场景结果：`artifacts/runs/formal_v2/`
- 配对结果：`artifacts/comparisons/formal_v2_*.json`
- 可视化：`artifacts/visualizations/formal_v2_e4/`

## 6. 结果

详见 [result_report.md](../result_report.md)。13 项 pytest 全部通过。

## 7. 失败与异常

- pilot 路由比例 50%，超过门槛；判断为 threshold 过低；
- 保留 pilot 文件，单变量调整 threshold 后运行 tuned；
- 轨迹初版横移过急，comfort 代理明显下降；改为 cubic smoothstep，并将 run identity 绑定源码后冻结 `formal_v2`。

## 8. 结论

核心工程链路验证通过。合成证据支持模块行为，不支持 NAVSIM 或真实驾驶性能结论。

## 9. 下一步

AutoDL 上运行官方 B0/E2；以 NAVSIM 子指标决定是否继续 E3/E4。
