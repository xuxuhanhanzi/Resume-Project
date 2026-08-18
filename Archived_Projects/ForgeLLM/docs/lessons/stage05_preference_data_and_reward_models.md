# Stage 5 讲义一：偏好数据、Verifier 与 Reward Model

## 1. 为什么 SFT 之后还需要偏好

SFT 学的是“给定前缀时，模仿目标 token”。如果一个 prompt 有多个语法正确的回答，SFT 不会天然知道哪一个更简洁、更安全或更符合格式。偏好数据把监督单位从一个目标回答改成同一 prompt 下的比较：

```text
x = prompt
y_w = chosen（更偏好）
y_l = rejected（较不偏好）
```

这只说明 `y_w` 在既定标准下优于 `y_l`，不说明 `y_w` 是唯一完美答案，也不说明模型在开放分布上会变好。

<details>
<summary>思考题：既然 chosen 是正确答案，为什么不直接把它继续做 SFT？</summary>

答案：可以，但信息不同。SFT 只增加 chosen 的似然；pair 同时告诉模型“chosen 应相对 rejected 更高”。当两个回答都常见、但质量不同，比较信号更直接。不过偏好信号仍可能有噪声，不能替代独立评测。
</details>

## 2. PreferenceRecord：先把数据身份说清楚

`PreferenceRecord` 保存 prompt、chosen、rejected、任务族、偏好来源、拒绝原因、verifier 版本和 metadata。构造时强制：

- prompt 非空并以 user 结束；
- chosen/rejected 非空且不同；
- prompt 不夹带 assistant 答案；
- 读取 JSONL 时字段集合必须完全一致；
- prompt 与完整 pair 的 SHA-256 必须重新计算并匹配。

`prompt_fingerprint` 只哈希任务身份；`pair_fingerprint` 还包含答案、来源和元数据。前者用于防止同一任务跨 split，后者用于审计整个 pair 是否被改写。

<details>
<summary>思考题：为什么不能只用 record_id 判断 train/test 泄漏？</summary>

答案：两个不同 ID 可能包装同一个 prompt。内容哈希按语义字段确定身份，能抓住“换 ID 不换题”的泄漏；它仍抓不到语义近重复，因此文档必须保留这个限制。
</details>

## 3. 为什么先切分任务，再生成 rejected

本项目先构造 512 个 `TaskSpec`，按任务内容哈希排序并切成 384/64/64，之后才为每题生成 rejected。这样 split 的归属不受某种错误答案模板影响，也避免同题不同 rejected 落入不同集合。

五个任务族是精确标识符、整数加法、紧凑 JSON、整数排序、ASCII 反转；六种 rejected 原因是错误答案、格式错误、额外解释、截断、重复、长度投机。它们适合学习数据契约和指标陷阱，但不代表自然人类偏好。

<details>
<summary>思考题：按哈希排序是否等于随机抽样？</summary>

答案：不是统计意义上的随机样本，但在输入固定时是稳定、可复现、与文件顺序无关的伪随机排列。这里追求的是教学可复现，不声称它代表真实用户分布。
</details>

## 4. Verifier、训练 reward 与独立测试

严格 verifier 的训练总 reward 只取 `exact`；`non_empty`、`no_extra`、`no_repetition`、`structural` 是审计组件。这样不会悄悄把“看起来像答案”并入成功标准。

攻击样本专门包括：正确答案后追加垃圾、包装成解释、重复、前导空格和尾随空格。若 reward 只检查 `expected in response`，这些攻击都会得高分，模型就可能学会利用漏洞。

在线 GRPO smoke 使用了独立标识的 shaped reward，为极弱 Base 提供相似度和格式部分分；held-out 成功仍用 strict exact。训练信号和验收信号必须在报告中分别命名。

<details>
<summary>思考题：为什么不能看到 shaped reward 上升就宣称任务成功率上升？</summary>

答案：两者定义不同。模型可能通过变长、复制正确前缀或迎合相似度取得部分分，却仍违反“只输出答案”。只有冻结的 held-out strict verifier 才能支持严格成功率结论。
</details>

## 5. Bradley–Terry 模型

Reward Model 为每个 `(x,y)` 输出标量 `r(x,y)`。Bradley–Terry 假设 chosen 胜过 rejected 的概率是：

```text
P(y_w > y_l | x) = sigmoid(r_w - r_l)
L_BT = -log sigmoid(r_w - r_l)
```

设 margin `m=r_w-r_l`：`m=0` 时 loss 为 `log 2≈0.6931`；margin 越正，loss 越小；margin 为负，说明排序错且 loss 增大。实现用 `-logsigmoid(m)`，比先 sigmoid 再 log 更稳定。

例：`r_w=1.2, r_l=0.2`，margin=1，偏好概率约 0.731，loss 约 0.313。

<details>
<summary>思考题：给 chosen/rejected reward 同时加 100，loss 会变化吗？这说明什么？</summary>

答案：不会，因为差值不变。这叫 reward shift 不可辨识：pairwise loss 只能约束相对排序，无法确定绝对零点。因此单独报告 reward 的绝对值通常没有可比意义。
</details>

## 6. TinyRewardModel 如何工作

教学 RM 将 UTF-8 byte 映射到 1–256，0 留给 padding；随后经过 Embedding、GRU，并取每条序列最后一个非 PAD hidden state，经线性头输出一个标量。它只用于验证：

```text
文本 → byte IDs + mask → GRU → last valid hidden → scalar reward
```

训练时 chosen 与 rejected 分别前向，送入同一个共享 RM，再计算 BT loss。共享参数很关键；若两边各有一套模型，比较尺度没有共同含义。

方法实验中，16 pairs、3 seeds 均达到 pair accuracy 1.0，loss 约 `3e-7` 到 `2e-6`。这只证明小模型能过拟合小集合和链路可学，不证明 reward 泛化。

<details>
<summary>思考题：训练 pair accuracy=1 是否表示 RM 是可靠裁判？</summary>

答案：不是。它可能记住数据、利用长度或模板特征。至少还要看 held-out pair accuracy、margin 分布、reward 与长度相关性、对抗样本和跨任务表现。
</details>

## 7. 必须配套报告的指标

- pair accuracy：`margin>0` 的比例；
- mean margin：排序置信间隔，但会受尺度漂移影响；
- loss：对错误和低 margin 更敏感；
- reward-length Pearson correlation：检查长度捷径，不等于因果；
- 分任务族、拒绝原因的切片结果；
- adversarial verifier 拒绝率。

本项目 320 个对抗响应全部被 strict reward 拒绝。tiny RM 三个 seed 的长度相关分别约 -0.207、-0.054、0.072；样本太小，不能据此断言“无长度偏差”。

<details>
<summary>思考题：相关系数接近 0 为什么仍不能证明不存在长度偏差？</summary>

答案：线性相关只能观察当前小样本上的线性关系。分任务混杂、非线性关系、分布外长回答都可能被遗漏；而且一个 RM 即使不偏好长度，也可能偏好特定模板。
</details>

## 8. 代码阅读检查表

阅读 `schema.py`、`preference_data.py`、`reward.py` 后，你应能回答：数据何时切分、哪些字段进入哈希、padding byte 是多少、取哪个 GRU 状态、BT loss 的唯一有效方向是什么、为什么 verifier 总分没有把审计组件相加。

本讲义的结论边界：已经证明 schema、哈希、防泄漏、verifier 攻击和 tiny RM 数学/训练链路；没有证明真人偏好质量、开放域 RM 泛化或安全对齐。
