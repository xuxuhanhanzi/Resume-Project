# Stage 6 讲义二：行为指标、语言建模公平性与统计不确定性

## 1. 为什么一个“准确率”不够

生成模型的失败形态很多。假设期望输出是 `{"ok":true}`：

- 完全相同：exact match；
- 输出合法 JSON 但值错误：格式通过、内容失败；
- 先输出正确答案再续写解释：correct-prefix-but-extra；
- 在 token 上限结束：truncation；
- 不断重复同一片段：repetition。

如果只报“JSON 合法率”，后四种失败中的多种会被掩盖。Stage 6 在 `behavioral.py` 中同时保留 exact、全部显式约束、正确前缀后多写、字符 8-gram、token 3-gram、长度与截断。

<details>
<summary>思考题：为什么 correct-prefix-but-extra 不能算 exact success？</summary>

对 API、JSON、标签或严格格式任务，多写文本会让下游解析失败。该指标单独记录，是为了区分“模型不会答案”和“模型知道答案但不会停止”，却不能擅自修改 exact 的语义。
</details>

## 2. 显式约束与隐含偏好

规则评测只能检查预先写入案例的约束，例如：

- `starts_with`；
- `required_substrings`；
- `forbidden_substrings`；
- `exact_word_count`；
- `require_json`；
- 冻结 expected response。

它不能可靠判断“文风优雅”“论证深刻”或未定义的安全性。指标实现若偷偷加入新偏好，就等于运行后换评分规则。

<details>
<summary>思考题：规则检查全部通过，是否意味着答案事实正确？</summary>

仅当规则本身完整编码了事实要求时才可能。JSON 合法、长度正确、包含某词都不能一般性保证事实正确。Stage 6 的小型任务使用冻结 expected response 提供强约束；开放问答仍需人工或经过验证的 Judge，并报告局限。
</details>

## 3. 两类重复率为什么同时存在

Whitespace token 3-gram 能发现“the answer is the answer is”一类重复，但会漏掉无空格、乱码或特殊符号循环。字符 8-gram 对这些情况更敏感。反过来，字符重复也可能把代码、JSON 键或自然的固定短语视为重复。

因此 Stage 6 不用一个指标替代另一个，而是并列报告：

```text
repetition = (n-gram occurrence count - unique n-gram count)
             / n-gram occurrence count
```

<details>
<summary>思考题：短于 n 的输出重复率为什么是 0，而不是“无法计算”？</summary>

实现把它定义为 0，因为根本不存在可重复的 n-gram occurrence。与此同时必须报告响应长度，防止一个空答案靠“重复率为 0”看起来稳定。任何单项指标都要和其他维度一起解释。
</details>

## 4. 原题成功与鲁棒性一致不是一回事

Stage 6 的 robustness case 是确定性等义重述。我们同时看：

- robustness 子集本身的成功率；
- 每对原题/重述的成功状态是否一致。

如果两边都失败，一致性为 1，但能力并不好。因此一致性不能替代成功率。

<details>
<summary>思考题：某模型原题和重述全部失败，一致性 100%，能说它鲁棒吗？</summary>

不能。只能说其失败状态对该变换稳定。鲁棒能力至少要求基线任务成功，再观察语义保持变换是否造成显著下降。
</details>

## 5. 从 NLL 到 PPL

对目标 token 序列，负对数似然为：

```text
NLL = -Σ log p(y_t | y_<t, x)
loss_per_token = NLL / target_tokens
PPL = exp(loss_per_token)
```

PPL 可以理解为平均不确定性的指数形式，但不能逐 batch 先算 PPL 再平均。正确做法是先累加所有 batch 的 NLL 与 token 数，最后除法和指数。

例子：batch A 有 2 token、loss 2；batch B 有 8 token、loss 4。正确总体 loss 是 `(2×2 + 8×4)/10 = 3.6`，不是 `(2+4)/2 = 3`。

<details>
<summary>思考题：为什么平均 batch loss 也可能错？</summary>

若各 batch 的有效 target token 数不同，等权平均会让短 batch 与长 batch 权重相同。必须用 NLL 总和除以 target token 总数。只有各 batch 分母完全相同，简单平均才巧合等价。
</details>

## 6. BPB 解决了什么

不同 Tokenizer 会把同一文本切成不同数量的 token，因此 token-normalized PPL 的分母改变。Bits per byte 使用 UTF-8 byte 数作为较稳定分母：

```text
BPB = NLL / (target_bytes × ln 2)
```

`LanguageModelingTotals` 同时保存 NLL、target tokens、target bytes 和 Tokenizer SHA。`assert_perplexity_comparable()` 在 Tokenizer 不同时直接拒绝 PPL 排名。

<details>
<summary>思考题：BPB 是否让任何两个模型都能公平比较？</summary>

不。它只改善分母可比性。两个模型仍需在相同原始文本、相同条件上下文、相同字节范围和一致的概率定义上评测。M3 与 Q0–Q2 的任务模板和模型路线不同，所以 Stage 6 仍把 M3 放在独立套件中。
</details>

## 7. Expected-response NLL 与自由生成

Stage 6 对 Q0–Q2 既算冻结答案的 assistant-only NLL，也运行自由 greedy generation。前者问“给定正确前缀时，模型给标准答案多少概率”；后者问“模型自己走完整条生成路径时会输出什么”。

这两者可能相反：NLL 改善，但首次错误 token 把后续生成带到完全不同的路径；或模型把多个合理答案的概率分散，exact 下降但开放质量未必下降。

<details>
<summary>思考题：为什么 NLL 仍然值得保留？</summary>

它是低方差、可加总、能定位训练影响的概率指标，并可与 token accuracy 等 teacher-forced 诊断结合。正确做法不是丢弃 NLL，而是不让它独占最终结论。
</details>

## 8. M3 为什么单独报告

M3 使用 Stage 1 Byte-BPE（词表 320）与自研 DecoderLM；Q0–Q2 使用 Qwen Tokenizer 与 ChatML。它们在参数量、训练语料、上下文协议、Tokenizer 和任务套件上都不同。

Stage 6 对 M3 运行固定 validation prefix，报告 NLL/token、PPL 与 BPB，但明确写入：不得用 M3 PPL 与 Qwen retention PPL 排名。该结果用于证明预训练评测链可复算，不用于宣称 M3 胜过或落后于 Qwen。

<details>
<summary>思考题：如果 M3 的 PPL 数值小于 Q0，我们为什么仍拒绝“更好”？</summary>

数值来自不同 token 单位和不同条件协议；更小可能只是 Tokenizer、文本切分或套件差异。没有共同实验问题，就没有有效排名。
</details>

## 9. Wilson 区间：点估计不是事实本身

若 16 题通过 5 题，点估计是 31.25%。但样本有限，真实任务分布成功率不可能被精确知道。Wilson 区间比简单的正态近似在小样本和 0/16、16/16 边界更可靠。

公式中 `p̂=k/n`、`z≈1.96`：

```text
center = (p̂ + z²/(2n)) / (1 + z²/n)
half   = z * sqrt(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
```

报告必须同时给 `k/n`、点估计和区间，不能只给区间或只给百分比。

<details>
<summary>思考题：0/16 的 95% 区间上界为什么不为 0？</summary>

因为有限样本中的零次观察不等于总体事件绝不会发生。Wilson 上界表达“在该样本量下，仍不能排除一个非零真实成功率”。这正是统计不确定性的价值。
</details>

## 10. 成对 Bootstrap：比较同一批题上的变化

Q0、Q1、Q2 回答完全相同的案例，因此比较时应先对每题计算：

```text
d_i = candidate_success_i - baseline_success_i
```

再以 case 为单位有放回抽样，重复 2,000 次，得到平均差的经验分布。若 95% 区间整个大于 0，才记为改善证据；否则结论为“不确定或未改善”。

不能分别打乱 baseline 和 candidate，因为那会丢掉题目难度的配对信息。

<details>
<summary>思考题：区间跨过 0，是否证明两个模型完全相同？</summary>

不是。它表示当前样本和协议不足以排除无差异，也不足以稳定支持方向性改善。结论是证据不充分，不是“数学证明相同”。增加预注册样本、提高任务代表性或降低噪声，才可能提高辨别力。
</details>

## 11. 本讲义的最小代码路线

1. `behavioral.py::evaluate_behavior`；
2. `language_modeling.py::LanguageModelingTotals`；
3. `statistics.py::wilson_interval`；
4. `statistics.py::paired_bootstrap_interval`；
5. `tests/unit/test_evaluation_metrics.py`。

先用测试中的小数字手算，再看正式报告。不要从一个最终百分比反推公式。
