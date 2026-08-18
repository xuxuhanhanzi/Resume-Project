# Stage 5 讲义三：Policy Gradient、Baseline 与 PPO

## 1. 从监督学习切换到决策学习

监督学习知道每一步目标 token；强化学习只在动作或整条轨迹后收到 reward。设策略为 `πθ(a|s)`，目标是最大化期望回报：

```text
J(θ) = E_{a~πθ}[R(a)]
```

最小 categorical bandit 没有状态转移：一次从若干动作中采样，立即得到固定 reward。它足够验证 policy gradient 的符号、方差与 baseline，不需要先引入完整 LLM。

<details>
<summary>思考题：为什么 Stage 5 先做 bandit，而不直接在 Qwen 上看 loss 是否下降？</summary>

答案：bandit 的期望回报和精确梯度可以枚举，因此错误能被定位。Qwen 上 loss 下降可能来自 mask、版本、reward 或优化器多种因素，无法独立证明公式正确。
</details>

## 2. Score-function / REINFORCE 推导

利用 `∇π = π∇logπ`：

```text
∇J = Σ_a ∇πθ(a)R(a)
    = Σ_a πθ(a)R(a)∇logπθ(a)
    = E[R(a)∇logπθ(a)]
```

所以最小化的 Monte Carlo loss 写成：

```text
L_PG = -mean(R · log πθ(a))
```

实现中抽样动作和取得 log-prob 必须来自同一版本策略。reward 与 advantage 都 detach；否则 autograd 可能沿环境或估计器走入未定义路径。

<details>
<summary>思考题：为什么 loss 里有负号？</summary>

答案：优化器执行梯度下降，而目标是最大化回报。对 `-R logπ` 做下降，在 `R>0` 时提高该动作概率，在 `R<0` 时降低它。
</details>

## 3. 精确梯度与 Monte Carlo 梯度

对有限动作可以直接计算 `J=softmax(logits)·rewards`，再让 autograd 得到精确梯度。REINFORCE 用有限样本估计它，期望正确但有采样方差。测试通过逐动作 one-hot 展开再次计算 score-function，避免只依赖 autograd 一条路径。

样本数增加时，平均估计通常接近精确梯度，但单次误差不保证单调下降。方法实验使用 4,096 samples 和 3 seeds，只要求方向和统计量合理。

<details>
<summary>思考题：无偏估计是否意味着一次采样就准确？</summary>

答案：不是。无偏只表示无限重复后的均值等于真值；单次估计可能非常偏离。方差决定要多少样本才能稳定接近。
</details>

## 4. Baseline 为什么不改变期望梯度

把 reward 换成 advantage `A=R-b`：

```text
E[(R-b)∇logπ(a)]
= E[R∇logπ(a)] - b Σ_a π(a)∇logπ(a)
= E[R∇logπ(a)] - b∇Σ_aπ(a)
= E[R∇logπ(a)]
```

前提是 `b` 不依赖当前采样动作，并从梯度图 detach。常见 baseline 有批均值、移动均值、状态价值 `V(s)`。它可能降低方差，也可能选得不好反而增大方差。

本项目 v1 错把期望 reward 当成“必然降方差”的 baseline，实验没有通过；v2 使用该策略/奖励下的方差最优常数 baseline。三个 seed 的逐坐标平均方差约从 0.326–0.335 降到 0.246–0.252。v1 被保留为负例。

<details>
<summary>思考题：为什么 expected reward baseline 不一定是 score-gradient 的方差最优 baseline？</summary>

答案：梯度样本还乘有 `∇logπ(a)`，不同动作的 score 范数不同。最优常数 baseline 要按 score 平方加权，不只是 reward 的普通均值。
</details>

## 5. Importance ratio 与 old policy

PPO 的 rollout 来自 old/behavior policy，但优化时参数已成为 new policy。用：

```text
r_t(θ) = πθ(a_t|s_t) / π_old(a_t|s_t)
       = exp(logπ_new - logπ_old)
```

把 old 数据重用于新策略目标。若 old log-prob 的版本未知、生成配置变化或 Dropout 模式不同，这个 ratio 没有可信语义。`assert_policy_revision` 和 rollout SHA 是算法契约，不只是日志美化。

<details>
<summary>思考题：首个 on-policy 更新前 ratio 应是多少？</summary>

答案：应为 1，因为 new 尚未更新，必须与 behavior policy 相同。数值上 log-prob 差应接近 0；本项目 GRPO v2 的近似 KL 和 clip fraction 都精确为 0。
</details>

## 6. PPO clipped surrogate

PPO policy objective：

```text
L_clip = -mean(min(rA, clip(r,1-ε,1+ε)A))
```

这里最容易错的是负 advantage。设 `ε=0.2`：

| ratio | advantage | `rA` | clipped `rA` | min 选择 |
|---:|---:|---:|---:|---:|
| 1.5 | +1 | 1.5 | 1.2 | 1.2 |
| 1.5 | -1 | -1.5 | -1.2 | -1.5 |
| 0.5 | +1 | 0.5 | 0.8 | 0.5 |
| 0.5 | -1 | -0.5 | -0.8 | -0.8 |

因此 clipping 不是把所有 ratio 简单截断再乘 advantage，而是取悲观代理目标。正 advantage 限制过度增加概率；负 advantage 限制过度降低概率。

<details>
<summary>思考题：上表第二行为什么没有选择裁剪后的 -1.2？</summary>

答案：目标取 `min`，-1.5 更小、更悲观。对负 advantage，ratio 过大表示坏动作概率反而增加，不能用 clipping 给它更宽松的目标。
</details>

## 7. PPO 的其他组成部分

完整 PPO 通常还包含：

- value loss：拟合 return，项目实现了 clipped value loss；
- entropy bonus：防止策略过早坍缩；
- KL 到 reference：控制与原始语言模型的偏离；
- GAE：用价值函数构造低方差 advantage；
- 多 epoch/minibatch：提高样本利用，但增加 staleness。

本项目的 toy PPO 聚焦 ratio/clip/value 与真实更新链，三个 seed 的期望 reward 都从约 0.205 上升至约 1.491。它不包含大语言模型 PPO 的分布式 rollout、完整 critic 训练或生产稳定性。

<details>
<summary>思考题：PPO clipping 是否保证真实 KL 一定很小、性能一定不下降？</summary>

答案：不保证。它是局部代理约束，batch 外分布、多个 epoch、优势估计误差都可能导致较大变化；仍需监控 KL、clip fraction、entropy、reward 和独立评测。
</details>

## 8. 从 token 到 sequence 的困难

LLM 一次动作常被视为整个 response，但概率由 token 条件概率相乘：

```text
π(y|x)=Π_t π(y_t|x,y_<t)
logπ(y|x)=Σ_t logπ(y_t|...)
```

序列 reward 通常在回答末尾获得，却要分配给所有 response token；这就是 credit assignment。序列越长，log-prob sum 和 ratio 数值范围越大，也容易出现长度偏差和单 token outlier。下一讲的 GRPO、DrGRPO、DAPO 与 GSPO正是在不同层面处理这些问题。

## 9. 本讲结论边界

已经证明 exact gradient、REINFORCE 估计、baseline 期望不变、一个真正降方差的 baseline、PPO 正负裁剪分支及 toy 更新。没有证明这些小实验可预测大模型 RL 的超参数、稳定性或最终能力。
