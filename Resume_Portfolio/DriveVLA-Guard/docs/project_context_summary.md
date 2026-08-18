# DriveVLA-Guard 项目上下文

> 冻结日期：2026-08-11  
> 当前阶段：事前规划  
> 目标岗位：中国车企/自动驾驶公司的 VLM、VLA、端到端驾驶或多模态后训练岗位

## 1. 任务定义

项目构建一个基于预训练驾驶 VLA 的风险感知推理系统。输入为自动驾驶场景的相机观测、车辆状态和导航信息，模型输出未来轨迹；系统在推理阶段生成多个候选，使用确定性驾驶风险指标重排序，并在高风险或高不确定性样本上触发慢思考路径。

核心研究问题：

> 在不重新训练基础 VLA 的前提下，多候选轨迹重排序和选择性快慢推理，能否在可控延迟开销内减少 NAVSIM 中的碰撞、越界或低 TTC 失败，同时不显著损害行驶进度？

## 2. 求职能力映射

| 项目模块 | 展示的岗位能力 |
|---|---|
| AutoVLA 源码审计 | VLM/VLA 架构、视觉 Token、Action Token、SFT、GRPO |
| 数据与坐标审计 | nuPlan/OpenScene/NAVSIM、车辆坐标系、轨迹格式 |
| 候选轨迹生成 | 自回归动作生成、解码策略、推理时扩展 |
| 风险评分器 | 碰撞、TTC、可行驶区域、舒适度、进度 |
| 快慢推理路由 | 不确定性、选择性计算、延迟—安全权衡 |
| NAVSIM 评测 | 正式基线、PDMS/EPDMS、逐场景失败分析 |
| 工程记录 | 固定 commit、配置、日志、可恢复评测、测试 |
| 可视化与部署审计 | 多相机展示、轨迹叠加、P50/P95 延迟、显存 |

## 3. 上游基线

### 3.1 官方主基线

- 模型与训练源码：[ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA)
- 主要 checkpoint：官方 AutoVLA checkpoint，具体 Hugging Face revision 在 Stage 0 冻结
- VLM 底座：AutoVLA 官方实现所用 Qwen2.5-VL-3B
- 评测框架：[autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- 数据来源：NAVSIM/OpenScene/nuPlan，具体 split 在 Stage 0 根据官方 checkpoint 兼容性冻结

### 3.2 工程 baseline

第一版工程 baseline 定义为：

```text
官方 AutoVLA checkpoint
+ 官方预处理
+ fast mode
+ greedy decoding
+ 单候选轨迹
+ 官方 NAVSIM 评测脚本
```

若官方仓库不能直接给出上述模式，Stage 0 必须记录实际默认模式，并以最小修改得到可复现 baseline；不能自行把修改后的结果称为“官方结果”。

## 4. 当前本地条件

工作区已有：

- `ForgeLLM`：LLM/VLM 结构、QLoRA、GRPO 与评测经验；
- `ForgeMM`：Qwen2.5-VL-3B、数据审计、清单驱动评测、逐样本结果和配对分析经验；
- `RepoPilot`：CLI、配置、可恢复运行和测试工程经验。

可复用的是工程方法，不直接复用 ChartQA 专用数据、reward 或模型结论。

现有 ForgeMM 文档报告本地设备为 RTX 4070 Laptop 8GB。正式 AutoVLA 多相机推理和 NAVSIM 评测默认规划在 AutoDL 24GB 级或更高显存 GPU 上完成；Stage 0 必须用实测替代该预估。

事前计划之后已经完成：

- AutoVLA 源码锁定到 `ba34eed74ce6729e7986592d0e66cbaca397b4fa`；
- 真实 2048-entry codebook 的加载与 10-token NumPy 解码；
- B0/B1/E1–E4 合成链路、21 项测试和 formal_v5 冻结证据；
- AutoVLA 多候选 adapter 与 NAVSIM agent；
- 本地官方 preflight 和许可证隔离。

当前仍缺失：

- AutoVLA 本地代码与冻结 commit；
- AutoVLA checkpoint 与文件哈希；
- NAVSIM/OpenScene 数据；
- 可运行环境版本；
- baseline 结果；
- 数据许可证和模型许可证审计记录；
- 可用显存和单场景延迟实测。

## 5. 主要变量与固定项

计划研究的变量：

1. 候选轨迹数量 `K`；
2. 是否启用确定性风险重排序；
3. 是否启用风险触发的 slow mode；
4. 路由阈值和候选分歧计算方式。

正式比较中必须固定：

- 模型 checkpoint 与量化方式；
- 上游代码 commit；
- 数据 split 与逐场景 manifest；
- 图像预处理、分辨率和帧设置；
- Action Token codebook；
- 车辆状态输入；
- 最大生成长度；
- 随机种子集合；
- NAVSIM 版本与指标实现；
- 推理 GPU、CUDA、PyTorch；
- warmup 次数和计时方法。

## 6. 主要风险

| 风险 | 影响 | 预防或退出策略 |
|---|---|---|
| 官方 reasoning data 未完整公开 | 无法完整训练复现 | 只声明使用官方 checkpoint 与源码审计 |
| 依赖冲突 | 延误项目 | 上游环境与数据预处理环境隔离，先 smoke |
| 8GB 本地显存不足 | 无法正式推理 | 本地只跑测试；AutoDL 跑模型与评测 |
| 多候选生成延迟过高 | 不具工程价值 | 先 K=2/4；限制 slow route 比例 |
| 重排序提升安全但损害进度 | 指标“投机” | 同时报告 NC、DAC、TTC、EP、Comfort、PDMS |
| NAVSIM 与真实道路存在差距 | 结论外推风险 | 结论限定为该 simulator/dataset setting |
| 个人贡献被上游淹没 | 简历说服力不足 | 自研模块独立、带测试、带消融和逐样本证据 |
| 为结果反复调测试集 | 数据泄漏 | 冻结开发集与正式评测集；测试集只运行冻结配置 |

## 7. 项目成功定义

项目成功分为三级：

1. **工程成功**：官方 checkpoint、候选生成、风险评分、路由和 NAVSIM 评测均可复现，结果可逐场景追溯。
2. **研究成功**：至少一个单变量模块在冻结评测上减少安全失败，且 PDMS/进度无不可接受退化。
3. **求职成功**：公开仓库能清晰区分上游资产和个人贡献，包含架构、实验、失败案例、效率报告和演示视频。

即使没有正向性能提升，只要工程与消融完整，项目仍可完成；此时必须将方法记录为“未在当前评测设置下得到稳定收益”。
