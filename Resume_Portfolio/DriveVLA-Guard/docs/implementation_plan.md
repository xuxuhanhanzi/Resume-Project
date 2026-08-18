# DriveVLA-Guard 完整实施计划

> 版本：v0.1  
> 冻结日期：2026-08-11  
> 计划周期：15 个工作日  
> 核心策略：预训练权重优先、训练源码精读、推理模块自研、评测证据优先

## 1. 目标和非目标

### 1.1 目标

1. 复现官方 AutoVLA checkpoint 的固定推理 baseline；
2. 独立实现 Action Token/轨迹解码的契约测试；
3. 实现多候选轨迹生成和无效轨迹过滤；
4. 实现确定性风险评分和候选重排序；
5. 实现风险/不确定性触发的快慢推理路由；
6. 在固定 NAVSIM 清单上完成单变量消融；
7. 形成延迟、显存、安全性、进度和失败类型报告；
8. 形成可供面试展示的可视化 Demo 和源码阅读笔记。

### 1.2 非目标

- 不从零训练或完整微调 VLA；
- 不复现论文的全部训练数据或全部榜单；
- 不在第一版实现世界模型、3DGS、CARLA 闭环或 TensorRT；
- 不将语言解释质量等同于规划质量；
- 不通过改变数据清单选择更有利的结果；
- 不将预训练模型、官方 codebook 或 NAVSIM 指标描述为个人实现。

## 2. 系统模块

### 2.1 `AutoVLAAdapter`

职责：

- 固定模型和 processor 加载；
- 显式暴露 fast/slow mode；
- 返回原始 token、token log-prob、轨迹和计时信息；
- 不在 adapter 中实现安全规则。

验收：相同输入、seed 和配置可复现相同 token/轨迹；上游 checkpoint 和 commit 可追溯。

### 2.2 `TrajectoryCodec`

职责：

- 使用官方 codebook 编解码；
- 验证 token 范围、轨迹长度、坐标轴、时间步和单位；
- 检查 encode/decode 或官方样例的一致性；
- 对非法输出返回稳定错误码，不静默修复。

验收：官方样例全部通过；随机/边界 token 测试不会崩溃；错误原因可统计。

### 2.3 `CandidateGenerator`

职责：

- 支持 greedy、top-k/top-p 或模型官方支持的采样方式；
- 生成 `K=1/2/4/8` 候选；
- 保存每条候选的 token、置信度、轨迹和生成耗时；
- 在重排序前仅执行格式/物理边界过滤。

验收：`K=1 + greedy` 与工程 baseline 一致；增大 K 不改变输入和模型权重。

### 2.4 `RiskScorer`

第一版仅使用确定性、可解释指标：

- collision risk；
- drivable-area violation；
- TTC risk；
- progress；
- comfort/jerk；
- trajectory validity；
- 可选模型置信度。

评分项必须独立记录；总分只用于选择，不替代官方 NAVSIM 指标。

验收：合成直行、急转、越界、静止、碰撞轨迹有符合预期的排序；数值边界有单元测试。

### 2.5 `ReasoningRouter`

触发候选：

- top-1 风险超过阈值；
- 所有候选均不合法或高风险；
- top-1/top-2 分数差低；
- token 平均熵高；
- TTC/越界门槛被触发。

职责：先运行 fast path，仅在触发时运行 slow path，并记录触发原因、额外延迟和最终是否改变轨迹。

验收：阈值固定后不得针对正式测试样本手调；所有触发都有机器可读原因。

### 2.6 `EvaluationHarness`

职责：

- manifest 驱动；
- 逐场景追加 JSONL；
- 断点恢复时验证已有记录是当前 manifest 的严格前缀；
- 报告官方指标、内部风险指标和效率指标；
- 生成配对比较与失败样本索引。

验收：中断后恢复不会重复或跳过场景；不同 checkpoint/config 不能误续跑。

## 3. 实验矩阵

| ID | 方案 | 相对比较时的唯一主变量 | 作用 |
|---|---|---|---|
| B0 | fast + greedy + K=1 | 官方/工程 baseline | 冻结基线 |
| B1 | slow + greedy + K=1 | 推理模式 | 测量慢思考收益和代价 |
| E1 | fast + sample + K=4，不重排序 | 相对 B0 启用“多候选生成子系统” | 测量多样性和有效率 |
| E2 | fast + sample + K=4 + risk rerank | 风险重排序 | 验证安全选择 |
| E3 | fast + greedy + risk router + slow | 快慢路由 | 验证选择性计算 |
| E4 | fast + sample + K=4 + rerank + router | 相对 E2 仅增加快慢路由；相对 E3 增加多候选重排序子系统 | 最终组合 |

主要结论只使用单变量配对：`B1↔B0`、`E2↔E1`、`E3↔B0`、`E4↔E2`。`E1↔B0` 只评估完整的多候选生成子系统，不把采样策略与 K 的影响拆开声称；如需拆分，另建 K=1 sampling 的诊断实验。

若 K=4 已产生不可接受延迟，不运行 K=8 正式评测；K=8 仅保留为小规模诊断。

## 4. 指标与成功门槛

### 4.1 官方任务指标

- PDMS/EPDMS；
- No Collision；
- Drivable Area Compliance；
- TTC；
- Ego Progress；
- Comfort；
- NAVSIM 当前版本要求的其他指标。

### 4.2 工程指标

- 单场景端到端 P50/P95 latency；
- 模型生成、评分和路由分项 latency；
- 峰值显存；
- invalid trajectory rate；
- slow-route rate；
- slow route 后轨迹改变率；
- 每场景有效候选数。

### 4.3 预注册门槛

最终 E4 若满足以下条件，可描述为正向结果：

1. 碰撞或越界失败数相对 B0 至少下降 10%；
2. PDMS 相对 B0 不下降超过 0.5 个百分点；
3. Ego Progress 不出现明显系统性退化；
4. 平均端到端延迟增幅不超过 35%；
5. slow-route rate 不超过 30%；
6. 正向差异不是由改变评测样本或过滤失败样本造成。

若只满足安全指标但延迟超标，结论限定为“离线安全重排序证据”；若只改善内部风险分数而官方指标不变，不声称规划提升。

## 5. 数据与评测协议

### 5.1 数据阶段

1. `smoke`：官方最小样例或 1～10 场景，只验证接口；
2. `dev`：固定小型开发清单，用于阈值选择和错误修复；
3. `formal`：冻结公共评测 split，阈值冻结后才运行；
4. `stress`：从 baseline 失败场景中构建诊断集，只用于失败分析，不替代 formal 指标。

### 5.2 泄漏控制

- formal split 不用于路由阈值选择；
- 不根据 formal 结果继续增加随机种子直到出现理想结论；
- 每个方案使用相同 manifest；
- 记录被上游 loader 跳过的所有样本和原因；
- 数据和 annotation 遵循各自许可证，不将受限数据提交 Git。

## 6. 15 个工作日执行表

| 日程 | 阶段 | 任务 | 退出条件 |
|---|---|---|---|
| Day 1 | S0 | 冻结上游 commit、环境、许可证、checkpoint revision | 环境审计文件完整 |
| Day 2 | S1 | 官方 checkpoint 单场景 smoke | 端到端推理可重复 |
| Day 3 | S1 | Action Token、SFT、GRPO 调用链阅读 | 六个关键问题有源码证据 |
| Day 4 | S2 | 冻结 B0 manifest 与配置 | B0 可断点恢复 |
| Day 5 | S2 | 跑 B0 dev，分类失败 | baseline 报告可生成 |
| Day 6 | S3 | TrajectoryCodec 与测试 | 官方/边界样例通过 |
| Day 7 | S3 | CandidateGenerator K=1/4 | K=1 一致性通过 |
| Day 8 | S3 | E1 dev | 多样性、有效率、延迟已知 |
| Day 9 | S4 | RiskScorer 与合成测试 | 风险排序符合预期 |
| Day 10 | S4 | E2 dev | 重排序链路可追踪 |
| Day 11 | S5 | ReasoningRouter | 每次触发有原因 |
| Day 12 | S5 | E3/E4 dev，冻结阈值 | formal 配置冻结 |
| Day 13 | S6 | B0/B1/E1-E4 formal | 全部方案同清单完成 |
| Day 14 | S6 | 配对分析、失败案例、效率报告 | 结论边界明确 |
| Day 15 | S7 | Demo、README、简历条目、视频 | 求职交付物齐全 |

## 7. 源码学习清单

只围绕以下问题定向阅读，不逐行通读整个项目：

| 问题 | 主来源 | 预期产物 |
|---|---|---|
| 视觉输入如何进入 VLM | AutoVLA/Qwen2.5-VL processor 与 model wrapper | 输入张量和视觉 token 流程图 |
| 车辆状态如何注入 | AutoVLA dataset/model adapter | 输入 schema 与位置说明 |
| 连续轨迹如何变成 token | AutoVLA action codebook 脚本 | 编解码说明和契约测试 |
| fast/slow mode 如何表示 | AutoVLA prompt/data/model config | 模式差异表 |
| SFT 监督什么 | AutoVLA SFT config/trainer | loss 区域和 label mask 说明 |
| GRPO 奖励什么 | AutoVLA RFT config/reward | reward 分量和优势计算说明 |
| NAVSIM 如何评分 | NAVSIM metric/agent interface | PDMS 子指标与输入输出表 |
| 3D空间能力从哪里来 | UniAD/SparseDrive/UniDriveVLA | 背景笔记，不在 v0.1 实现 |

## 8. AutoDL 运行规范

正式运行前记录：

```bash
pwd
git rev-parse HEAD
python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

每次运行记录绝对路径：代码、数据、checkpoint、配置、输出、日志、可视化。失败命令和 traceback 不删除；先判断属于依赖、显存、模型图、数据还是评测接口问题。

## 9. 计划中的求职交付物

- 可运行 GitHub 仓库；
- 上游依赖和许可证表；
- AutoVLA 源码阅读笔记；
- Action Token 编解码测试；
- B0/B1/E1-E4 主结果表；
- 逐场景配对比较；
- 安全、进度、延迟和显存联合分析；
- 5～10 个典型失败案例；
- 3 分钟演示视频；
- 中英文项目摘要；
- 简历项目描述和面试问答。

## 10. 后续扩展的进入条件

只有 v0.1 正式评测完成后，才从以下方向选择一个：

1. 小规模 LoRA smoke；
2. DriveLM 风格结构化解释；
3. OmniDrive/TensorRT 推理部署；
4. Bench2Drive/CARLA 闭环；
5. Qwen3-VL 或 UniDriveVLA backbone 迁移。

扩展前必须先提交新的单变量实验计划；不能在 v0.1 中并行展开。
