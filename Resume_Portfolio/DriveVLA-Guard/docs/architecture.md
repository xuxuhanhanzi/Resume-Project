# 架构与数据契约

## 设计原则

1. 模型生成和安全规则分离；adapter 不隐藏重排序逻辑。
2. 风险项分量保存，总分只负责选择，不冒充 NAVSIM PDMS。
3. 碰撞、TTC、可行驶区域只有在显式场景证据存在时计算；缺失时返回 `null`，不填零。
4. NAVSIM 集成只使用当前历史窗口可观测信息，不读取 future GT 或 metric cache 进行候选选择。
5. 正式比较由配置 + 决策关键源码的联合 hash、manifest 和严格前缀断点恢复约束。

## 运行流

```text
model_input + SceneContext
          |
          v
AutoVLA/Synthetic Backend -- K candidates --> Candidate(token, traj, logp, latency)
          |                                      |
          |                                      v
          |                         RiskScorer (lower is better)
          |                                      |
          +------------------------------- rerank fast candidates
                                                 |
                              threshold / margin / uncertainty
                                   |                     |
                                accept                 slow path
                                   |                     |
                                   +-------- final selection
                                                 |
                                      JSONL + summary + plot
```

## 关键契约

### `Trajectory`

- `poses`: `[T, 3]`，列为 ego 坐标系的 `x(m), y(m), heading(rad)`；
- `dt_s`: 正数；
- NaN/Inf 不在构造时静默修复，由风险评分标记 invalid。

### `SceneContext`

- `obstacles`: 每个障碍物包含与候选时间步对齐的 `[T,2]` 位置和保守圆半径；
- `lane_half_width_m`: 仅合成/显式车道走廊使用；NAVSIM v0.1 不伪造该值；
- `source`: 机器可读证据来源，例如 `trajectory_only` 或 `navsim_current_annotations_constant_velocity`。

### `RiskBreakdown`

`invalid/collision/ttc/drivable/comfort/progress/confidence` 独立输出。环境证据缺失时，对应字段为 `null`，且不参与总分。

### `PlanResult`

保存所有 fast/slow 候选、最终选择、route reasons、分项风险、延迟和 config hash，使每次决策可回放。

## 防泄漏边界

官方 NAVSIM runner 在模型输出之后才使用 metric cache 计算 PDMS。DriveVLA-Guard agent 的重排序器不接收 metric cache；可选碰撞上下文由当前帧 annotations 和当前速度做 constant-velocity 外推。该外推仍是简化假设，因此正式报告必须同时给出“trajectory-only”和“current-observation”设置。
