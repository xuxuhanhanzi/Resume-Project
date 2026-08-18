# 合成消融结果报告

> Run：`formal_v5`  
> 日期：2026-08-14  
> 范围：确定性合成场景，仅验证工程与假设，不是 NAVSIM PDMS

## 设置

- 12 个冻结场景：clear、stationary blocker、crossing actor、narrow lane 各 3 个；
- 轨迹：10 poses，0.5 秒间隔；
- B0/B1/E1–E4 使用相同 manifest；
- E3/E4 路由阈值在 pilot 后冻结为 risk `13.8`、margin `0.05`；pilot 产物保留在 `artifacts/runs/`；
- `formal_v5` 的 run identity 同时绑定 config 与决策关键源码 hash；
- 6 组 summary、72 条逐场景结果和 3 组配对比较已通过语义重算与 SHA-256 清单验收；
- 所有结果逐场景写入 JSONL，配置漂移时拒绝续跑。

## 主结果

| Run | 主变量 | 碰撞代理失败 | Mean comfort | Slow rate | Mean latency | Mean internal risk |
|---|---|---:|---:|---:|---:|---:|
| B0 | fast, K=1 | 6 | 1.000 | 0% | 4.0 ms | 6.878 |
| B1 | slow, K=1 | 3 | 0.784 | 0% | 14.0 ms | 6.558 |
| E1 | sampled K=4，不重排 | 6 | 0.905 | 0% | 16.0 ms | 7.021 |
| E2 | E1 + risk rerank | 0 | 0.718 | 0% | 16.0 ms | 4.412 |
| E3 | B0 + selective slow | 3 | 0.892 | 25% | 7.5 ms | 6.271 |
| E4 | E2 + selective slow | 0 | 0.727 | 25% | 19.5 ms | 4.386 |

## 配对结果

- E2 vs B0：6 个场景内部风险改善、6 个轻微回退；平均 risk delta `-2.465`。回退来自 clear/narrow 场景中采样噪声造成的舒适度损失。
- E3 vs B0：3 个场景改善、0 个回退；平均 risk delta `-0.607`。
- E4 vs E2：3 个场景改善、0 个回退；平均 risk delta `-0.026`，增量很小。

## 可以支持的结论

1. 多候选重排序实现能够在合成上下文中避开已知障碍；
2. 风险路由可以把 slow path 控制在 25%，并减少部分失败；
3. 重排序存在舒适度与延迟代价；
4. 在当前合成集上，E4 没有证明相对 E2 的额外安全收益，只呈现很小的内部风险改善。

## 不能支持的结论

- 不能声称 NAVSIM PDMS 提升；
- 不能声称真实道路更安全；
- 不能声称 AutoVLA 官方 checkpoint 已在本机复现；
- `proxy_score` 不是官方指标，不能与论文表格比较。

## 下一项判定实验

在 AutoDL 上固定 AutoVLA checkpoint 与 navtest manifest，先运行 B0/E2。只有 E2 的 NC/DAC/TTC 改善且 EP/Comfort 不发生不可接受回退，才运行 E3/E4。否则优先改进候选平滑与当前场景建模，而不是继续调总分权重。
