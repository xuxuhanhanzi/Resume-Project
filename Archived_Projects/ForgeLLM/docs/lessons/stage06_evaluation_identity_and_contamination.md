# Stage 6 讲义一：评测身份、冻结协议与数据污染

## 1. 为什么“跑出一个分数”还不是评测

一个分数至少依赖四类变量：被测模型、输入数据、推理协议、评分规则。只写“Qwen 得分 0.6”几乎不可复现，因为我们不知道：是 Base 还是 Adapter？哪个 revision？是否用了 Chat Template？采样温度是多少？答案是否在训练集中？

Stage 6 把一次评测定义成不可拆分的证据对象：

```text
run = models + tokenizers + adapters + cases + source hashes
    + generation settings + code revision
```

任何一项变化，都不是“同一次运行的小调整”，而是一个新 run。

<details>
<summary>思考题：只保存 Adapter 权重，为什么仍不能确认模型身份？</summary>

Adapter 的输出建立在特定 Base、Base revision、Tokenizer 和注入配置上。同一 Adapter 套到不同 Base revision，或者用不同 Chat Template 分词，都可能改变行为。完整身份至少需要 `model_id + revision + tokenizer_sha256 + adapter_sha256`；Base 的 `adapter_sha256` 为 null。
</details>

## 2. 训练指标、代理指标和最终指标

- 训练指标回答“优化器是否按目标移动”，例如 SFT loss、DPO loss；
- 代理指标回答“某个容易测的替代目标是否改善”，例如 chosen/rejected log-prob margin；
- 最终行为指标回答“模型生成结果是否满足冻结任务”，例如 exact match、JSON 合法性和重复率。

它们相关，但不存在必然蕴含。DPO 可以把 chosen 的对数概率推到 rejected 之上，却仍可能在自由生成时续写多余文本。Stage 4/5 的真实结果正是这种反例，因此 Stage 6 必须保存原始生成，而不是只读训练报告。

<details>
<summary>思考题：如果 Q2 的 DPO pair accuracy 是 100%，我们能否跳过生成评测？</summary>

不能。Pair accuracy 是 teacher-forced 的相对排序：给定两条完整答案后，比较其响应 log-prob。自由生成需要模型从一个 token 开始不断选择后续 token；局部选择、长度偏置、EOS 学习和重复会共同影响最终文本。两种实验回答的问题不同。
</details>

## 3. `EvaluationManifest` 怎样防止“无意换题”

`src/forgellm/evaluation/schema.py` 中的 Manifest 包含：

- 五个模型身份：Q0、Q1、Q2、Q3、M3；
- `cases_sha256` 与所有源数据哈希；
- `max_new_tokens/do_sample/temperature/top_p/seed`；
- Chat Template 哈希；
- Git commit 与 dirty/clean 状态；
- 由全部字段再次计算的 `run_fingerprint`。

加载 Manifest 时会重新计算指纹。若有人手工把 `run_name`、Adapter 哈希或生成参数改掉，却没有生成一份新 Manifest，解析器会拒绝该文件。

```python
fingerprint = sha256(canonical_json(manifest_without_fingerprint))
```

这里的 canonical JSON 固定 UTF-8、键排序和紧凑分隔符，避免相同语义因空格或键顺序得到不同哈希。

<details>
<summary>思考题：Git 状态为 dirty 是否意味着结果无效？</summary>

不一定。它意味着仅靠 commit 不能恢复当前代码，因此必须同时保存 dirty 状态，并以 Artifact、测试和具体文件作为证据。最理想是 clean commit；学习项目允许 dirty，但不能假装它是完全可复现的发布版本。
</details>

## 4. 冻结 64 例是怎样组成的

Stage 6 固定：

- 16 条 Stage 4 correctness test 原题；
- 16 条 Stage 5 preference test 原题；
- 每条原题对应一个确定性等义重述，共 32 条 robustness；
- 总计 64 条，每个 Q0/Q1/Q2 各生成一次，即 192 条原始输出。

为什么 preference test 原文件有更多记录，却只取排序后的前 16 条？因为实验计划预先规定总预算，避免看完结果后再挑“更有利”的题。`cases.py` 显式切片，并由测试断言 16+16+32。

<details>
<summary>思考题：为什么不在看到某类失败后再添加 20 道同类题？</summary>

失败后添加题适合形成下一版开发集，却不能悄悄并入当前确认性测试。否则题目选择已经受到结果影响，原来的统计解释被破坏。正确做法是冻结 v1 结论，再把新增题登记为 v2，使用新文件哈希与新 run。
</details>

## 5. 数据污染到底是什么

这里的污染指评测文本与训练文本发生未披露重合，使模型可能靠记忆而非泛化取得高分。Stage 6 做两层筛查：

1. 精确层：`NFKC → casefold → 合并空白 → SHA-256`；
2. 近重复层：规范化文本的字符 13-gram 集合 Jaccard，相似度阈值 0.8。

Jaccard 定义为：

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

对 2,880 条结构化训练记录执行两层筛查；对更大的 Stage 3 自然文本训练集执行规范化精确哈希筛查，避免为了一个小型本地评测制造不必要的二次方计算。

<details>
<summary>思考题：NFKC 和 casefold 会不会制造误报？</summary>

会有可能，所以匹配项是“需要报告和复核的候选”，不是自动宣判作弊。规范化故意忽略全角/半角、大小写和空白差异，提升召回率；若发现 near match，仍要阅读原文判断是模板共享、短语重合，还是真正答案泄漏。
</details>

## 6. 为什么模板重合不应静默删除

指令数据经常共享诸如“Return JSON”“Answer exactly”之类模板。若只要有共享短语就删题，会改变任务分布；若完全忽略，又会掩盖泄漏。Stage 6 的规则是：

- 精确重合进入硬门禁；
- 达阈值近重复完整列出；
- 低于阈值的常见模板不自动删除；
- 报告规范化、n-gram 和阈值，使读者能复算。

当前 v2 `preparation_report.json` 冻结结果为 0 个精确重合、16 个达到阈值的近重复候选。16 个候选集中在两个多轮 correctness 案例与同模板训练案例，已完整保留供复核，未静默删题。这只能说明在已纳入审计的本地训练文件中没有检测到精确污染，不能证明互联网上从未出现过相同问题。

<details>
<summary>思考题：0 个检测结果是否等于“绝对无污染”？</summary>

不等于。检测受数据可见范围、规范化规则和近重复阈值限制。诚实表述是“在列出的本地训练来源和冻结规则下未检测到污染”，而不是“数据绝对干净”。
</details>

## 7. 从代码追踪一次证据链

按以下顺序阅读，不要跳转：

1. `config.py`：为什么 TOML 只能有五个 section；
2. `cases.py`：原题与 robustness 如何生成；
3. `preparation.py`：如何组合训练文本并做污染筛查；
4. `schema.py`：案例与 Manifest 如何拒绝多余字段；
5. `data/processed/stage6_evaluation_v1/preparation_report.json`：真实哈希和计数。

这条链的核心不是哈希算法本身，而是**先冻结再观察**：先决定题、阈值和预算，再运行模型。

<details>
<summary>思考题：为什么严格 Schema 要拒绝未知字段，而不是忽略它们？</summary>

未知字段可能恰好是会改变结论的新控制量。静默忽略会让文件看似被支持，实际却没有执行其语义。严格拒绝迫使开发者明确升级 Schema 和版本，从而保留实验边界。
</details>
