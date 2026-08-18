# Stage 6 讲义三：Judge、盲评、位置偏差与鲁棒性

## 1. 三种 Judge 不是互相替代

Stage 6 把评分者分为三层：

1. **确定性规则**：exact、JSON、前缀、禁词、长度、重复；
2. **模型 Judge**：适合开放文本的相对质量，但会受提示、位置和自身能力影响；
3. **人工盲评**：可阅读语义和上下文，但昂贵、有主观差异，也会疲劳。

本地正式自动门以规则为主，另生成 30 对 Q1/Q2 人工盲评包。没有把免费模型 Judge 的主观分数塞进最终硬门禁，因为这批严格格式任务已有更直接的可执行规则。

<details>
<summary>思考题：为什么有规则后还需要人工盲评？</summary>

规则可稳定判断预定义约束，却可能漏掉可读性、误导性、语义等价但非 exact 的合理答案。人工盲评不是替换规则，而是审查规则没有表达的维度，并帮助发现评测设计问题。
</details>

## 2. Judge 也必须有协议

“让另一个模型打分”不是完整方法。至少要固定：

- rubric：正确性、指令遵循、重复、清晰度、有害性；
- A/B 输入顺序与交换策略；
- Judge 模型、revision、prompt、采样参数；
- 是否允许 tie；
- 无效回答怎样处理；
- 同一题是否多次评分；
- 与人工评分的一致性怎样报告。

若没有这些信息，Judge 分数只是新的不可复现输出。

<details>
<summary>思考题：为什么必须允许 tie？</summary>

强迫 Judge 在几乎相同或都失败的答案中选胜者，会制造虚假的细微差异，并放大位置偏差。Tie 是真实结果，不是评分失败。
</details>

## 3. Stage 6 的盲评隔离

`judge.py` 生成三个文件：

- `blind_review.jsonl`：响应 A/B 与空 rubric；
- `blind_review.html`：方便学习者逐对阅读；
- `private_key.jsonl`：A/B 到 Q1/Q2 的隐藏映射。

案例选择和 A/B 方向由 `seed + case_id` 的 SHA-256 决定，不依赖当前文件顺序。学习者必须先填写 public 文件，再打开 private key。若先看模型身份，盲评即失效，应另建新 seed 的包。

<details>
<summary>思考题：隐藏模型名但保留“回答风格”是否能保证完全盲？</summary>

不能。熟悉模型的人可能从风格猜测来源。盲化只降低显式品牌和期待偏差，不消除所有线索。因此报告应写“匿名 A/B”，而不是“绝对无偏”。
</details>

## 4. 一个可执行的人工评分流程

对每一对回答按固定顺序：

1. 先重新阅读题目与约束；
2. 独立判断 A、B 是否事实/格式正确；
3. 检查多写、截断和重复；
4. 再评价清晰度与潜在有害性；
5. 选择 A、B 或 Tie；
6. 写一句理由，不猜模型身份；
7. 每 10 题休息，避免疲劳造成后半段随意评分。

禁止在中途打开私钥，也不要因某答案更长就默认更好。

<details>
<summary>思考题：若 A 格式完全正确但解释少，B 内容丰富却违反“只输出 JSON”，谁胜？</summary>

在该任务 rubric 下 A 胜，因为指令遵循是任务定义的一部分。不能把个人对“丰富回答”的偏好放在显式约束之上；否则评分对象已经从原任务换成另一个任务。
</details>

## 5. 位置偏差和一致性

位置偏差指 Judge 倾向选 A 或第一个答案，与内容无关。最小审计是把 `(A,B)` 交换为 `(B,A)`，观察语义赢家是否保持。

`deterministic_rule_judge()` 把规则指标组成一个不含模型名的比较键；`position_consistency()` 对每对运行正序和逆序。规则 Judge 理应达到 100% 位置一致；若没有，说明实现把位置混入结果。

两个人工评分者可用 Cohen’s κ：

```text
κ = (observed agreement - chance agreement) / (1 - chance agreement)
```

κ 把偶然一致考虑进去，但会受类别分布影响，不能脱离原始混淆表孤立解释。

<details>
<summary>思考题：两个评分者 95% 一致，κ 是否一定很高？</summary>

不一定。若 95% 样本都属于一个极常见标签，两人即使只会选该标签也能得到高表面一致率，chance agreement 同样很高。应同时报告标签分布、原始一致率和 κ。
</details>

## 6. 等义鲁棒性变换是什么

Stage 6 不调用另一个模型临时改写题目，而是使用代码中的确定性前缀重述：保持原始内容和 expected response，只增加诸如“请严格回答这一等价请求”的表达。这样变换可复算，不引入外部 Judge 的新随机性。

这是一种**局部鲁棒性**测试，只覆盖指令措辞变化，不覆盖拼写噪声、多语言、对抗后缀、长上下文或事实扰动。

<details>
<summary>思考题：为什么 robustness case 必须保存 parent_case_id？</summary>

它建立原题与变换题的一对一关系，使我们能做 paired consistency 和成对统计。没有 parent，鲁棒性子集只剩一个独立平均数，无法定位哪道题在变换后翻转。
</details>

## 7. 多轮对话的失败教训

Stage 6 第一次运行发现两个 correctness 案例含历史 assistant turn。初版构造器删除了全部 assistant 消息，导致对话变成 `system → user → user`，既改变语义，也违反 Stage 4 的交替 Schema。

修正原则不是删除这些题，而是：只隐藏最后一个待预测 assistant answer，保留此前历史 assistant turn。v1 失败目录与错误堆栈保留，v2 使用新案例哈希和新 run 名称。

这个例子说明 Schema 失败是有价值的：如果代码静默拼接两个 user turn，评测可能继续跑完，却测了错误问题。

<details>
<summary>思考题：为什么修正后不能继续沿用 v1 的 Manifest？</summary>

案例语义和文件哈希已经变化。沿用 v1 会把两种不同输入混在同一 run identity 下。正确做法是保留失败证据，生成 v2 cases、v2 fingerprint 和新输出目录。
</details>

## 8. 近重复候选怎样人工复核

v2 污染审计出现 16 个 near-match，集中在两个多轮 correctness test 与同模板的训练案例，相似度约 0.85。它们没有规范化精确重合，但共享大量对话模板。

复核时要问：

- 标识符/答案是否相同？
- 任务结构相同是否正是我们想测的模板泛化？
- 训练和测试 ID 是否独立？
- 若删除，是否会选择性美化结果？

Stage 6 报告这些候选，不静默删除。最终硬门只要求精确重合为 0；near-match 留作结论限制。

<details>
<summary>思考题：模板近似为何仍可能高估泛化？</summary>

模型可能只学会固定模板中的槽位替换，而没有获得更广泛的指令能力。即使标识符不同，这类成功也只能支持“同模板内泛化”，不能外推到开放任务。因此结果必须按本地数据范围表述。
</details>

## 9. 本讲义代码路线

1. `judge.py::build_blind_comparisons`；
2. `judge.py::deterministic_rule_judge`；
3. `judge.py::position_consistency`；
4. `statistics.py::cohens_kappa`；
5. `tests/unit/test_evaluation_judge_systems_report.py`；
6. 最后才打开盲评 HTML，且暂时不要打开私钥。
