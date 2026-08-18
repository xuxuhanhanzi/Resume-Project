# 简历描述与面试问答

## 中文简历条目（当前证据版）

**DriveVLA-Guard｜自动驾驶 VLA 风险感知推理系统**

- 基于 AutoVLA/Qwen2.5-VL-3B 真实推理接口，实现多候选 Action Token 生成、轨迹解码、可解释风险分解与 risk-triggered fast/slow routing；采用顺序候选生成控制 KV-cache 峰值。
- 构建 manifest + config hash 驱动的可恢复评测框架，保存逐场景 token、轨迹、风险分量、路由原因与延迟，并实现严格前缀续跑和配对分析。
- 在 12 个冻结合成场景上完成 B0/B1/E1–E4 消融：组合方案将碰撞代理失败由 6 降至 0，slow path 比例 25%；同时识别出舒适度与延迟代价，未将代理结果表述为 NAVSIM 性能。
- 为 AutoVLA 官方 NAVSIM v2 agent 提供无未来信息泄漏的集成，当前障碍物仅由历史末帧 annotations 与 constant-velocity 外推构建；配套 21 项测试、GPU/依赖/路径/commit preflight、冻结证据校验、可视化与许可证隔离。

正式 NAVSIM 运行后，第三条应替换为官方 PDMS/NC/DAC/TTC/EP/Comfort 数值；运行前不要写“提升 NAVSIM PDMS”。

## 30 秒项目介绍

AutoVLA 已经能从三路视频和车辆状态生成 5 秒轨迹，但官方推理一次只给一条序列。我没有重新训练 3B 模型，而是在推理阶段生成多条 Action Token 候选，用当前可观测场景和轨迹动力学做可解释重排序；高风险或候选分歧场景才进入 slow CoT。工程重点是避免 future GT 泄漏、控制延迟，并用逐场景配对证据而不是只看平均分。

## 高频问答

### 为什么不用训练？

目标是短周期验证推理时扩展与安全选择。训练源码仍做了 SFT/GRPO 调用链审计，但复用官方 checkpoint 把时间投入候选生成、风险接口、评测和部署约束。后续只有正式结果证明瓶颈来自 policy 分布时才考虑 LoRA。

### 为什么不能直接用 NAVSIM metric cache 重排序？

metric cache 用于输出后的模拟评测，包含候选选择时不应访问的信息。把它放进 reranker 会产生 oracle 泄漏。当前集成只使用历史末帧 annotations 和速度外推，且在证据字段中明确标注。

### E4 为什么不一定比 E2 好？

E2 已经通过 K=4 找到安全轨迹，slow path 在合成集上只有很小增量，却增加延迟。因此可靠结论是“路由在部分 K=1 场景有帮助”，不是“所有模块叠加必然最好”。

### 风险总分会不会被手工权重投机？

会，所以分量全部保存，阈值只在 dev/pilot 冻结；formal 比较使用同一 manifest。同时最终判断看官方 NC/DAC/TTC/EP/Comfort/PDMS，而不是内部总分。

### 项目最大的工程难点是什么？

一是官方 `predict()` 不返回多候选和 log-prob，需要直接适配 `generate()`；二是 3B 多视频模型的显存，采用顺序生成换峰值显存；三是把评测信息和推理可见信息严格隔离。
