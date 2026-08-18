# 实验记录：local_diagnostics_v1

## 1. 目标

完成本机可执行的 K 候选数、随机种子、路由风险阈值敏感性与 1,200 场景压力实验，确认
正式 NAVSIM 前的本地算法边界与稳定性。

## 2. 环境

- 平台：Windows 11；
- GPU：RTX 4070 Laptop 8GB（本实验为 CPU/NumPy 合成后端）；
- Python：3.12.3；
- 代码基线：`4282e91`，新增 `scripts/run_local_diagnostics.py`；
- 数据：脚本生成 1,200 个确定性合成场景，manifest SHA-256
  `fd91d064751cc176e7f2c8180494cd3f1b02fa6a54d69da094ef976f9759af85`。

## 3. 实验变量

- K sweep：`K=1/2/4/8`，种子 `17/42/2026`，rerank 开启、router 关闭；
- threshold sweep：K=1、seed=17，风险阈值 `4/8/13.8/16/20`；
- 固定项：同一 manifest、风险权重、SyntheticBackend、1,200 scenes/run。

## 4. 命令

```powershell
.\.venv\Scripts\python.exe -m ruff check scripts\run_local_diagnostics.py
.\.venv\Scripts\python.exe scripts\run_local_diagnostics.py `
  --output-root artifacts\runs\local_diagnostics_v1 --scenes 1200
```

## 5. 输出路径

- 原始逐场景结果：`artifacts/runs/local_diagnostics_v1/`（本地忽略）；
- 可提交摘要：`artifacts/evidence/local_diagnostics_v1_summary.json`；
- 原始 summary SHA-256：
  `f692fbd05e65f134f0cbbf283125e93019733248e26f704d92f7160410626f65`；
- 原始证据清单 SHA-256：
  `9e03ee69a79126bfab03bf213ee07564b51457f1b241c1e72f73381c823ec242`。

## 6. 结果

| K | 种子数 | 碰撞代理失败/1,200 | 声明延迟 | 结论 |
|---:|---:|---:|---:|---|
| 1 | 3 | 600 | 4 ms | 安全不足 |
| 2 | 3 | 300 | 8 ms | 部分改善 |
| 4 | 3 | 0 | 16 ms | 最小零碰撞代理 K |
| 8 | 3 | 0 | 32 ms | 无额外安全收益，延迟翻倍 |

阈值 13.8 与更激进的 4/8 得到相同 300/1,200 碰撞代理失败，但慢路由率从 50% 降至
25%、声明平均延迟从 11 ms 降至 7.5 ms；16/20 不触发慢路由并退化为 600 次失败。
17 个运行共 20,400 个 scene-run，全部成功，无异常。

## 7. 失败与异常

- 无执行失败；
- 这些场景只有四类重复结构，候选延迟是 SyntheticBackend 声明值，不能解释为真实模型时延。

## 8. 结论

K=4 是当前合成诊断中的最小充分候选数，K=8 不进入正式配置；预登记阈值 13.8 在该诊断
范围内保留。该结论只用于冻结云端 NAVSIM 配置，不允许声称 PDMS 或真实道路安全提升。

## 9. 下一步

本机合成实验已闭环。仅在云端资产齐备后运行官方 AutoVLA + NAVSIM B0/E2，并按门槛决定
是否继续 E3/E4。
