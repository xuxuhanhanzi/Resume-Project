# Stage 5 讲义二：Direct Preference Optimization

## 1. DPO 解决什么问题

经典 RLHF 通常要先训练 Reward Model，再在线采样并用 PPO 优化。DPO 利用带 KL 正则的最优策略与 reward 之间的闭式关系，直接把偏好 pair 写成分类式 loss。它省去显式 RM 和在线 rollout，但仍依赖偏好数据、冻结 reference 和正确的 log-prob 口径。

DPO 是离线偏好优化。没有“当前策略采样→环境奖励→再更新”的闭环，所以不得称作 on-policy RL。

<details>
<summary>思考题：DPO 不训练 RM，是否意味着它不含 reward 假设？</summary>

答案：不意味着。偏好概率模型、KL 正则和隐式 reward 关系仍在推导中；只是 reward 没有作为独立神经网络被显式拟合。
</details>

## 2. 四个 sequence log-prob

对同一 pair 需要：

```text
log πθ(y_w|x)    policy chosen
log πθ(y_l|x)    policy rejected
log πref(y_w|x)  reference chosen
log πref(y_l|x)  reference rejected
```

本项目使用回答 token 的 log-prob 求和。Causal LM 的第 `t` 个 logit 预测第 `t+1` 个 token，因此代码先做 `logits[:, :-1]` 与 `labels[:, 1:]`。只有 label 不等于 `-100` 的回答位置进入求和；prompt、role prefix 和 PAD 都不能进入。

<details>
<summary>思考题：为什么 prompt 即使在 chosen/rejected 中完全相同，也仍应显式 mask？</summary>

答案：理论上相同前缀项可能抵消，但实际有截断、padding、模板或数值路径差异。显式 response-only mask 直接对应条件概率 `π(y|x)`，也让 token count 与长度审计可信。
</details>

## 3. DPO logit 与 loss

先定义 policy 相对 reference 的 log-ratio：

```text
a_w = log πθ(y_w|x) - log πref(y_w|x)
a_l = log πθ(y_l|x) - log πref(y_l|x)
z   = a_w - a_l
L   = -log sigmoid(βz)
```

`z>0` 表示相较于 reference，policy 更偏向 chosen。初始化时 policy 与 reference 完全相同，`z=0`，所以首个 batch loss 必须是 `log 2`。本项目修正 Dropout 后 smoke v2 的首 loss 精确为 `0.693147`；这是身份一致性的强诊断。

<details>
<summary>思考题：如果 policy chosen 和 rejected 的 log-prob 都同时增加 5，DPO logit 是否变化？</summary>

答案：若 reference 不变且两边增量相同，`a_w-a_l` 不变。DPO关心相对偏好差，而不是把所有回答概率一同抬高。
</details>

## 4. beta 的含义

在标准 reverse-KL sigmoid DPO 中，`beta` 缩放隐式 reward 差和分类边界。它与原始 KL 约束强度相关，但不能简单口号化为“beta 越大越保守”而忽略具体约定。项目固定 `β=0.1`，不做调参搜索。

beta 变大时，同一个 `z` 会产生更饱和的 sigmoid；梯度集中到边界附近。beta 太小会让更新信号弱，太大可能使错误 pair 造成尖锐梯度。这里只实现并验证一个冻结配置，不得推出最佳 beta。

<details>
<summary>思考题：为什么本项目不根据 validation preference accuracy 搜 beta？</summary>

答案：任务目标是掌握算法，且小模板数据很容易被过拟合。搜索会花更多资源并把 validation 变成调参集；更重要的是 preference accuracy 本身不代表严格生成成功。
</details>

## 5. chosen/rejected 交换测试

若把 chosen 与 rejected 对调，原来的 `z` 变成 `-z`，优化方向也应反转。这是比“loss 能下降”更有力的局部正确性测试。实现还与 TRL 1.8.0 `DPOTrainer` 在固定 batch 上对齐 loss/梯度，避免只让自己的公式和自己的测试互相证明。

<details>
<summary>思考题：为什么需要框架对照，手写公式和单元测试还不够吗？</summary>

答案：手写实现可能和手写测试共享同一个错误，例如符号、shift 或 reduction 口径。独立框架固定 batch 对照提供不同实现路径；它仍不能证明数据和训练方案合理。
</details>

## 6. reference 到底是谁

正式运行的 reference 是 Stage 4 初始 SFT Adapter 在精确 Qwen Base revision 上的输出。代码先预计算并保存 reference chosen/rejected log-probs，训练期间 reference 不更新。policy 从同一个 SFT Adapter 初始化，但随后只更新 LoRA 参数；Base 始终冻结。

四个容易混淆的身份：

| 身份 | 会更新吗 | 用途 |
|---|---:|---|
| Base | 否 | 提供冻结主干参数 |
| initial SFT Adapter / reference | 否 | 定义相对偏移零点 |
| DPO policy Adapter | 是 | 被优化 |
| held-out evaluator | 否 | 用冻结数据评价前后行为 |

<details>
<summary>思考题：为什么不能每一步都把 reference 同步成当前 policy？</summary>

答案：那会移动比较零点，使已优化的相对变化被不断抵消，并偏离冻结的 DPO 目标。若要研究移动 reference，需要重新定义算法和实验，不能暗中更新。
</details>

## 7. Dropout v1 为什么不合规

Stage 4 LoRA 配置含 dropout=0.05。v1 在 reference/evaluation 时处于 eval，但训练 policy 处于 train，导致“相同权重”仍不是相同分布；首个 DPO batch 不再保证严格 `log 2`，GRPO old/new log-prob 也不一致。

修正方案是加载 Stage 5 栈时将所有 `nn.Dropout.p=0`，并在需要梯度时仍保持 `model.eval()`；eval 不会关闭 autograd。v1 Artifact 原样保留为实施偏差，v2 才进入正式证据。

<details>
<summary>思考题：`model.eval()` 是否等于“不计算梯度”？</summary>

答案：不是。eval 只改变 Dropout、BatchNorm 等模块行为；是否建图由 `torch.no_grad()`、参数 `requires_grad` 和上下文决定。GRPO v2 在 eval 模式下正常反向传播 Adapter。
</details>

## 8. 正式 DPO v2 应该怎样读

冻结预算：50 optimizer steps、200 pairs、4,214 response tokens、约 129.3 秒、峰值 allocated 1,638,436,352 bytes。Base 无梯度，Dropout 全关闭。

| 指标 | Before | After | 正确解释 |
|---|---:|---:|---|
| held-out implicit preference accuracy | 0.0 | 1.0 | pair 目标被强烈拟合 |
| mean DPO logit margin | 0.0 | 74.685 | 相对 reference 偏好间隔极大 |
| strict generation success | 0/16 | 0/16 | 行为任务没有通过 |
| Stage 4 assistant val loss | 1.1500 | 1.3584 | 原 SFT 分布拟合退化 |
| mean character 8-gram repetition | 0.0581 | 0.5528 | 重复明显恶化 |

因此，这次实验的价值不是“DPO 训练出了好模型”，而是展示代理目标过拟合、行为不改善和保留能力退化可以同时发生。大 margin 甚至是风险信号：小数据、固定模板和较大更新把 Adapter 推得过远。

<details>
<summary>思考题：preference accuracy=1 与 strict success=0 同时出现是否矛盾？</summary>

答案：不矛盾。前者用 teacher-forced log-prob 比较 pair，后者要求模型从 prompt 自回归生成完全匹配的字符串。模型可在给定回答 token 时更偏好 chosen，却仍在自由生成中追加文本、重复或偏离格式。
</details>

## 9. 你必须保留的结论边界

本项目已证明 response-only DPO 数学、TRL 固定 batch 对照、reference 冻结、Adapter-only 更新和 bounded Qwen 链路。它没有证明 DPO 提升通用能力，也没有比较 beta、数据规模或其他偏好方法。正式结果明确是一个负面的行为结果，这正是 Stage 6 需要综合评测的原因。
