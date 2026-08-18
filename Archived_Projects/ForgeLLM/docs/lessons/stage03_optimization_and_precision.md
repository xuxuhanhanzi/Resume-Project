# Stage 3 完整讲义二：优化器、调度、梯度累积与混合精度

## 1. 参数更新不是“学习率乘梯度”这么简单

最朴素 SGD 为：

```text
theta <- theta - lr * gradient
```

AdamW 维护梯度的一阶动量 `m` 和平方梯度的二阶统计 `v`，使用归一化更新，并把 weight decay 作为独立的参数收缩。这里的关键字是 decoupled：weight decay 不应与 Adam 的梯度归一化混在一起。

**思考题：optimizer state 算模型参数吗？**

答案：不算，但它决定下一次参数更新。只保存模型权重而丢失 Adam 的 `m/v`，恢复后的第一步就不是原训练轨迹的下一步。

## 2. Weight decay 参数分组

Stage 3 的 AdamW 分组规则是：

- 线性层等普通矩阵权重：decay；
- bias：no decay；
- RMSNorm/LayerNorm 参数：no decay；
- embedding：no decay。

这不是宇宙唯一规则，而是冻结的教学基线。重要的是每个 trainable parameter 恰好属于一个组，不能重复或遗漏。

**思考题：为什么 norm scale 通常不做 decay？**

答案：它控制特征归一化后的缩放，不像普通矩阵那样表示复杂映射。对它施加同样收缩可能干扰归一化行为；真正结论仍应服从目标模型的基线和实验。

## 3. Warmup 与 cosine decay

训练初期参数和 optimizer state 都未稳定，直接使用峰值学习率可能产生大更新。线性 warmup 用前 `W` 次更新把学习率从较小值升到基准值。

之后 cosine decay 平滑降低到 `min_lr_ratio * base_lr`：

```text
scale = min_ratio + (1-min_ratio) * 0.5 * (1 + cos(pi * progress))
```

ForgeLLM 的 update 编号从 1 开始。若 warmup=2，base LR=1：第一次更新 0.5，第二次 1.0。scheduler 在 optimizer step 之后前进，为下一次更新准备 LR。

**思考题：为什么 scheduler 状态也要保存？仅用 global step 重算不行吗？**

答案：简单无分支 schedule 可以重算，但真实系统可能有多个参数组、动态调整或 plateau 逻辑。保存 state 是更通用的契约；同时用 global step 验证它没有错位。

## 4. 梯度裁剪

全局范数裁剪先计算全部梯度的联合 L2 norm：

```text
||g|| = sqrt(sum_i ||g_i||^2)
```

若超过阈值，就按同一比例缩小全部梯度。它改变的是更新幅度，不修复数据错误或 NaN。

正确顺序是：

```text
backward
→ FP16 时 unscale gradients
→ 检查 finite
→ clip global norm
→ optimizer step
```

**思考题：为什么不能先 clip 缩放后的 FP16 梯度？**

答案：此时梯度还乘着 loss scale，范数不是真实梯度范数，裁剪阈值失去含义。

## 5. 一个 optimizer step 的完整顺序

ForgeLLM 的 `train_step`：

1. `model.train()`；
2. `zero_grad(set_to_none=True)`；
3. 依次取 micro-batch；
4. autocast 前向并计算 main/MTP loss；
5. loss 除以 accumulation steps；
6. backward，梯度累加；
7. unscale、finite check、clip；
8. 所有 optimizer step；
9. scaler update；
10. 记录本次实际 LR；
11. scheduler 为下一次更新前进；
12. 增加 step 和 target tokens；
13. 写入原始 JSONL 指标。

`set_to_none=True` 表示没有梯度的参数保持 `grad=None`，通常更省内存，也能区分“真实零梯度”和“本轮未参与图”。

**思考题：scheduler 应该在 optimizer 之前还是之后调用？**

答案：两种约定都能实现，但必须冻结定义并测试。本项目先用当前 LR 更新，再让 scheduler 准备下一次 LR；记录的 LR 是本次真实使用值。

## 6. 梯度累积为什么能模拟大 batch

假设一个大 batch 被均分成 `K` 个相同大小 micro-batch。大 batch 平均 loss 的梯度为：

```text
grad((L1+...+LK)/K) = (grad L1 + ... + grad LK)/K
```

因此每个 micro loss 除以 `K` 后 backward、最后只 step 一次，就与大 batch 更新近似等价。

不等价来源包括：

- micro-batch 大小不相等却仍简单除以 K；
- dropout 或其他随机算子顺序不同；
- BatchNorm 依赖每个 micro-batch 统计；
- 每个 micro-batch 都错误调用 optimizer step；
- 梯度裁剪发生在每个 micro-batch 而不是累积完成后；
- 浮点加法顺序造成极小误差。

**思考题：accumulation=4 是否意味着看到 4 倍数据才增加 global step？**

答案：是。global step 表示 optimizer update 次数；tokens seen 则增加四个 micro-batch 的全部 target tokens。二者不能混为一个计数器。

## 7. FP32、BF16 与 FP16

| 格式 | exponent | fraction | 直观特点 |
|---|---:|---:|---|
| FP32 | 8 | 23 | 范围和精度都高，作为 oracle |
| BF16 | 8 | 7 | 范围接近 FP32、精度较低，训练通常稳定 |
| FP16 | 5 | 10 | 精度比 BF16 高一些，但动态范围小，易下溢/溢出 |

autocast 不是把整个模型永久转为低精度。它根据算子选择计算 dtype；参数和某些归约仍可保留 FP32。

FP16 的小梯度可能下溢为零。loss scaling 先把 loss 放大，backward 后再把梯度除回去；若检测到 Inf，就跳过更新并降低 scale。BF16 因 exponent 与 FP32 相同，通常不需要 scaler。

**思考题：BF16 fraction 更少，为什么仍常比 FP16 适合训练？**

答案：训练中的主要危险常是动态范围不足导致上溢/下溢。BF16 保留 8 位 exponent，能表示与 FP32 类似的数量级；代价是有效数字更少。

## 8. Validation 与训练指标

训练 step 记录：

- total/main/MTP loss；
- 本次实际 learning rates；
- clip 前 gradient norm；
- target tokens 和累计 tokens；
- step wall time、tokens/s；
- CUDA peak allocated memory。

原始 step loss 不应只保留平滑曲线。平滑有助阅读，但会隐藏尖峰；JSONL 必须保存原始值。

**思考题：训练 loss 一直下降而 validation 上升，首先应该做什么？**

答案：先检查 split 泄漏、validation 模式、分母和数据处理是否一致；若正确，再考虑过拟合。不要直接延长训练或只展示训练 loss。

## 9. NaN/Inf 的停止策略

本项目选择 fail-fast：

- total loss 非有限：不 backward 后续步骤；
- 任一 gradient 非有限：不 optimizer step；
- `clip_grad_norm_(error_if_nonfinite=True)` 再做一层保护；
- OOM 不自动改 batch，因为自动改动会破坏固定实验条件。

诊断顺序：先记录 step/batch/config，再检查输入 token 范围、loss、梯度、LR、precision，最后才考虑调整超参数。

**思考题：为什么“跳过坏 batch 继续训练”不是默认策略？**

答案：坏 batch 可能暴露数据或数值错误。静默跳过会改变数据轨迹、掩盖根因，并使恢复等价性失效。
