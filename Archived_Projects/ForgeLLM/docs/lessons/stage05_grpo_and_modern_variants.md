# Stage 5 讲义四：GRPO、DrGRPO、DAPO、GSPO 与 VESPO

## 1. GRPO 为什么以“组”为单位

对每个 prompt 从同一 policy 采样 `G` 个回答，得到 reward `r_1...r_G`。标准教学实现按组标准化：

```text
mean = Σr_i/G
std  = sqrt(Σ(r_i-mean)^2/G)
A_i  = (r_i-mean)/(std+ε)
```

它用同题其他回答作为相对基线，不单独训练 critic，因此节省 value model 资源。但“无 critic”不等于“无 baseline”：组均值就是 baseline 结构的一部分。

<details>
<summary>思考题：为什么不同 prompt 的原始 reward 不应直接放在同一组标准化？</summary>

答案：题目难度和 reward 尺度不同。组内比较控制了 prompt 条件；混组会把“简单题高分”误当成某个回答更优。
</details>

## 2. RolloutRecord 是算法数据，不只是日志

每条 rollout 保存 prompt ID、policy revision、Adapter SHA、生成参数、seed、token IDs、逐 token old log-prob、reward components、verifier 版本、文本和 UTC 时间。缺任何关键身份，都不能可靠重算 ratio 或判断数据是否过期。

`old_log_probs` 必须在生成该 response 的同一 behavior policy 下计算。tokenizer、模板或 `max_new_tokens` 改变也需要新版本；只记录自然语言 response 不足以重现 token 级目标。

<details>
<summary>思考题：已有 response 文本，为什么还要存 token IDs？</summary>

答案：decode→encode 不一定保持同一 token 序列，特别是空格、特殊 token 与 tokenizer 版本变化时。log-prob 是针对具体 token 轨迹定义的，必须绑定原 token IDs。
</details>

## 3. 零方差组怎样处理

若一组 reward 完全相同，std=0；直接除法会放大数值噪声或产生 NaN。项目显式把该组 advantages 置零并标记 unusable。若全部组都无方差，本次 update fail-fast。

这不表示回答都“同样好”，只表示当前 reward 无法区分它们。解决路径可能是增加 group size、改进 reward 或采样多样性，而不是随意加噪声制造梯度。

<details>
<summary>思考题：给零方差 reward 随机加一点噪声是否合理？</summary>

答案：通常不合理。它会让随机数决定更新方向，掩盖 reward/采样无信息的问题。应显式统计零方差组并改进数据或判分器。
</details>

## 4. Token-level GRPO 目标

项目将每条 sequence advantage 广播到该回答的每个有效 token，计算 `ratio_t=exp(logπ_new,t-logπ_old,t)`，再用 PPO 风格 clip 和可选 KL 聚合。response mask 排除 padding。

这里要区分三层：reward 是 sequence 级；advantage 是每条 sequence 一个；ratio 和 loss reduction 是 token 级。把它们混称“token reward”会遮蔽 credit assignment。

<details>
<summary>思考题：同一回答中所有 token 共用一个 advantage 有什么局限？</summary>

答案：无法定位哪个 token 导致成功或失败，早期正确 token 与后期重复 token 收到同号信号。过程 reward、value/critic 或蒸馏型 dense signal 可改善粒度，但也带来新的监督偏差。
</details>

## 5. 首步 on-policy 门

首次 update 前，old 和 new 权重、模块模式必须相同。代码比较所有有效 token 的 log-prob，最大绝对差超过 `1e-5` 就拒绝。之后才允许计算 clipped loss。

v1 因 old 在 eval、new 在 train，LoRA dropout=0.05 造成近似 KL `3.83e-5`、clip fraction `0.0059`，不是严格首步 on-policy。v2 关闭所有 Dropout，梯度计算仍保持 eval，得到 KL=0、clip fraction=0。v1 原样留档，不能删除或覆盖。

<details>
<summary>思考题：近似 KL 很小，为什么仍要判 v1 不合规？</summary>

答案：本实验要验证身份契约，首步理论值可精确预期。若容忍已知模式不一致，就无法区分真正 staleness 与随机 Dropout；小偏差也可能在长序列和多步更新中累积。
</details>

## 6. Qwen GRPO v2 真实链路

使用 Stage 4 SFT Adapter，4 个 prompt、每题采样 4 条、每条最多 32 tokens，共 16 rollouts、508 response tokens。所有 4 组都有 reward 方差，执行恰好 1 个 optimizer step：

```text
sample → token IDs → old log-probs → shaped reward
→ group advantages → new log-probs → on-policy gate
→ clipped loss + KL → Adapter gradient → one update
```

v2 指标：reward mean `-0.03843`、std `0.02512`、gradient norm `1.7748`、peak allocated `4,011,605,504` bytes、Base 无梯度。此实验只证明接口和版本链可运行；一次更新不要求、也没有评价能力提升。

<details>
<summary>思考题：为什么 GRPO smoke 的训练 reward 可以为负仍执行更新？</summary>

答案：GRPO 使用组内相对 advantage。即使所有绝对 reward 都偏低，只要组内有差异，优于组均值的回答获得正 advantage，劣于均值的获得负 advantage。
</details>

## 7. DrGRPO：质疑两次归一化

标准 GRPO 组内除以 reward std，并常按每条 response 长度归一 token loss。DrGRPO 分析这些归一化可能引入 question-level difficulty 和 response-length bias。教学实现做两个最小改变：reward 只中心化、不除组 std；token loss 使用固定常数 normalizer。

它的价值是让“题目难度/长度改变梯度尺度”变得可观察，不代表所有任务都应无条件换成 DrGRPO。

<details>
<summary>思考题：去掉 std 归一化的代价是什么？</summary>

答案：不同 prompt 的 reward 尺度会直接影响梯度；若 verifier 标度不一致，大尺度组可能主导训练。它修复一种偏差，同时减少一种尺度稳定化。
</details>

## 8. DAPO 的四个独立机制

DAPO 不应被当成一个不可拆的名字。本项目分别实现：

1. Clip-Higher：下界 `1-ε_low`、上界 `1+ε_high`，给正优势概率提升更大空间；
2. Dynamic Sampling：过滤全对或全错等零信号 group；
3. token-level policy-gradient reduction：强调全局有效 token 口径；
4. overlong shaping：在长度上限前的 buffer 平滑衰减，避免硬截断突变。

frozen tensor 例中，ratio=1.4、advantage=1 时，标准上界得到 1.2，Clip-Higher 得到 1.28；这只证明分支公式，不证明训练收益。

<details>
<summary>思考题：Dynamic Sampling 会带来什么新分布偏差？</summary>

答案：训练更集中在当前策略“有时会、有时不会”的中等难度题，极易和极难题被降低权重。它提高有效梯度利用率，但改变了 prompt 训练分布，必须单独报告利用率。
</details>

## 9. GSPO：把 ratio 提升到 sequence 层

token ratio 容易被单个极端 token 主导。GSPO 风格权重先计算回答 token 的平均 log-ratio，再指数化：

```text
s = exp(mean_t(logπ_new,t-logπ_old,t))
```

随后在 sequence 层裁剪。平均 log-ratio 等价于几何平均 token ratio，长度尺度更稳定，也让同一序列的 token 共享一个权重；代价是可能隐藏局部坏 token。

<details>
<summary>思考题：为什么使用平均 log-ratio，而不是直接把 token ratio 做算术平均？</summary>

答案：sequence 概率是 token 概率乘积，log 空间对应求和；除以长度后指数化得到几何平均，与序列似然结构一致，也比直接乘积更数值稳定。
</details>

## 10. VESPO：面向 stale rollout 的软权重

VESPO 先使用真实 sequence importance weight，即有效 token log-ratio 求和后指数化，而不是 GSPO 的长度平均；再通过与 advantage 符号相关的 Gamma 型核：

```text
φ(w) = exp(λ + k log w - λw)
```

产生 detached soft weight。它试图让陈旧数据不是简单“硬剪掉或全保留”，同时抑制极端 importance weight。项目只做固定张量研究并报告 ESS；没有复现论文的大模型训练。

<details>
<summary>思考题：soft weight 为什么仍不能让任意 stale rollout 变得安全？</summary>

答案：当 behavior 与 current policy 支持集差异很大时，importance sampling 方差仍可爆炸，reward 分布也会变化。软权重是偏差—方差折中，不是消除 off-policy 风险。
</details>

## 11. 方法选择表

| 观察到的问题 | 首先检查的方法/机制 | 不应越界的说法 |
|---|---|---|
| 组内全同 reward | Dynamic Sampling / verifier | 不能说过滤后模型更聪明 |
| 长回答梯度尺度异常 | DrGRPO / token reduction | 不能只凭长度张量推生产收益 |
| 正优势提升被上界压住 | DAPO Clip-Higher | 不能忽略 KL 与坍缩风险 |
| 单 token ratio 极端 | GSPO sequence weight | 不能说局部错误消失 |
| rollout 明显过期 | VESPO / 重新采样 | 不能把 stale 数据称 on-policy |

## 12. 本讲结论边界

已实现并验证 GRPO 全链、零方差策略、首步 on-policy 门及四类现代变体的单变量张量行为。除了 1-step Qwen smoke，其余前沿变体都不是大模型效果复现。
