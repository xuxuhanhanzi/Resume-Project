# Stage 1 G1-B 完整讲义：从经典 BPE 到现代 Tokenizer 方法

> 适用对象：已经学完 G1-A 经典 byte-level BPE，但尚未系统学习 pre-tokenization、Unigram、随机切分、Picky BPE、SuperBPE 与 tokenizer-free 方法的初学者。  
> 学习目标：理解并能追踪算法代码，不追求训练生产级 Tokenizer。  
> 使用方式：严格按第 1 节顺序学习；每一道思考题后紧跟答案。

## 0. 先建立正确目标

Tokenizer 学习有两个不同目标：

1. **产品目标**：为某个大模型训练压缩率高、覆盖领域广、推理快的正式 Tokenizer；
2. **方法目标**：理解文本在哪些边界被切分、候选 Token 如何产生、词表如何选择、编码路径如何决定。

G1-B 选择第二种。我们用很小的项目原创语料，因为算法内部状态在小语料上更容易观察。固定 `vocab_size=300`，不搜索“最优词表”。

这并不是降低学习质量。相反，如果直接使用几十 MiB 语料和成熟库，你通常只能看到最终 `tokenizer.json`，看不到每一次 merge、remove、动态规划与随机选择。

### 思考题 0：小语料实验得到的压缩率很差，是否说明算法错误？

<details>
<summary>答案</summary>

不说明。算法正确性首先由不变量判断，例如 byte 是否完整覆盖、merge 是否按规则执行、encode/decode 是否可逆。压缩率还受语料规模、重复模式和词表预算影响。小语料的压缩数字只用于比较同一夹具下的行为，不能推广到真实模型。

</details>

## 1. 学习顺序与文件顺序

本讲义分七站。不要同时打开全部文件。

```text
第 1 站 pre-tokenization
→ 第 2 站 BPE-dropout
→ 第 3 站 Unigram
→ 第 4 站 Picky BPE
→ 第 5 站 SuperBPE
→ 第 6 站特殊 Token、offset 与 entropy patching
→ 第 7 站统一实验与结论边界
```

对应代码阅读顺序：

1. `src/forgellm/tokenization/pretokenization.py`
2. `tests/unit/test_pretokenization.py`
3. `src/forgellm/tokenization/advanced_bpe.py` 中的 `train_classic_bpe` 与 `EducationalBPE.encode`
4. `tests/unit/test_advanced_bpe.py` 中的前三个测试
5. `src/forgellm/tokenization/unigram.py`
6. 回到 `advanced_bpe.py` 阅读 `train_picky_bpe`、`train_super_bpe`
7. `src/forgellm/tokenization/special_tokens.py`
8. `src/forgellm/tokenization/entropy_patching.py`
9. `src/forgellm/tokenization/method_lab.py`
10. `docs/experiments/2026-07-27_stage01_g1b_method_lab.md`

## 2. 第 1 站：Pre-tokenization 决定 BPE 能看见哪些 Pair

### 2.1 什么是 pre-tokenization

Pre-tokenization 可译为“预切分”。它发生在 BPE 学习之前，先把一篇文档切成若干 piece，并规定 BPE 是否允许跨越这些 piece 的边界。

例如：

```text
new york 123!
```

一种 Unicode-class 预切分可能得到：

```text
["new", " ", "york", " ", "123", "!"]
```

如果 BPE 只能在每个 piece 内统计 pair，那么：

- 可以学习 `n + e`；
- 可以学习 `1 + 2`；
- 不能学习 `new + 空格`；
- 不能直接学习 `new york`。

### 2.2 为什么它非常重要

经典 BPE 常被描述为“合并最高频相邻单元”，但“哪些单元算相邻”不是天然确定的。

比较三种方式：

| 模式 | 初始 piece | 可能效果 |
|---|---|---|
| `none` | 整篇文档一个 piece | 可以跨空格、数字、标点任意 merge |
| `whitespace` | 空白与非空白 run 分开 | 不跨空白，但标点仍可能与单词合并 |
| `unicode_class` | 字母、数字、空白、其他分别成组 | 更明确控制类别边界 |

ForgeLLM 的 `split_text()` 没有删除空格。它只增加边界，所有 piece 拼接后必须恢复原始 UTF-8 bytes。

### 2.3 阅读代码

打开 `pretokenization.py`，按顺序阅读：

1. `Pretokenization`；
2. `_character_class()`；
3. `split_text()`。

特别观察：

```python
if next_class != current_class:
```

这行决定何时关闭旧 piece 并开始新 piece。

### 思考题 1：为什么 pre-tokenization 不能简单地使用 `text.split()`？

<details>
<summary>答案</summary>

`text.split()` 会丢失空格数量、Tab、换行以及开头和结尾空白。Tokenizer 必须能够无损还原原文，因此 pre-tokenizer 必须保留每个 byte，只能添加边界，不能删除内容。

</details>

### 思考题 2：`none` 模式能学出跨空格 Token，这是否自动等于 SuperBPE？

<details>
<summary>答案</summary>

不等于。`none` 从第一次 merge 起就允许跨越全部边界；SuperBPE 先在受限边界中学习稳定子词，再在第二阶段开放跨空格 merge。两者的训练课程和 merge 顺序不同。

</details>

## 3. 第 2 站：BPE-dropout 与“一个字符串不只一种切法”

### 3.1 经典 BPE 为什么是确定的

训练完成后，经典 BPE 保存一组有顺序的 merge：

```text
(a, b) -> ab
(ab, ab) -> abab
```

编码时重放这些 merge。同一文本和同一模型得到同一 Token 序列。

### 3.2 BPE-dropout 做了什么

BPE-dropout 在训练语言模型时，对某次匹配到的 merge 以概率 `p` 暂时跳过。

它没有：

- 删除词表；
- 改写原始文本；
- 破坏 decode；
- 重新训练 merge rank。

它改变的是某一次 encode 的切分路径。例如 `banana` 可能被编码为：

```text
[banana]
[ban, ana]
[b, an, ana]
```

只要每个 Token 仍保存准确 byte payload，三种结果都能解码为同一个字符串。

### 3.3 为什么这是正则化

正则化（regularization）是防止模型过度依赖训练数据中单一模式的方法。随机切分迫使语言模型面对同一文本的多个合法边界，从而减少它对固定 Token 边界的依赖。

阅读 `advanced_bpe.py`：

1. `_replace_pair()` 中的 `dropout`；
2. `EducationalBPE.encode()` 中固定 `random.Random(seed)`；
3. `test_bpe_dropout_changes_segmentation_without_changing_bytes`。

### 思考题 3：为什么随机切分需要 seed？

<details>
<summary>答案</summary>

随机方法仍然需要可复现。相同模型、文本、dropout 和 seed 应得到相同切分，方便定位训练问题。改变 seed 才用于产生另一条合法路径。

</details>

### 思考题 4：推理时是否也应该默认启用 dropout？

<details>
<summary>答案</summary>

通常不默认启用。训练时随机切分用于正则化；推理时一般使用确定性切分，保证缓存、计费、上下文长度与复现稳定。特殊研究可以在推理时采样，但必须显式说明。

</details>

## 4. 第 3 站：Unigram——从贪心 Merge 转向概率模型

### 4.1 BPE 与 Unigram 的根本区别

BPE 的问题形式是：

> 现在出现次数最多的相邻 Pair 是哪个？

Unigram 的问题形式是：

> 给定一个候选 Token 词表，哪一种完整切分具有最大的概率？

假设 `banana` 可以切成：

```text
[banana]
[ba, nana]
[ban, ana]
[b, a, n, a, n, a]
```

Unigram 为每个 Token 保存一个概率 `P(token)`。一条切分路径的概率是各 Token 概率的乘积。为了避免大量小数相乘造成数值下溢，代码使用 log probability：

```text
log P(path) = Σ log P(token)
```

### 4.2 候选词表

教学实现先收集训练文本中的 byte substring，例如长度 2～8 的连续 bytes。256 个单 byte 永远保留，作为 byte fallback。

Byte fallback 的意义是：即使某个 Emoji 或罕见汉字从未出现在训练语料中，它仍能被拆成 UTF-8 bytes，因此不存在无法编码的输入。

### 4.3 Viterbi 最优切分

Viterbi 是动态规划算法。定义：

```text
best[i] = 编码前 i 个 bytes 能获得的最高 log probability
```

从位置 `i` 出发，枚举所有能够匹配的候选 Token。如果 Token 覆盖到位置 `j`：

```text
candidate = best[i] + log P(token)
```

如果 candidate 更大，就更新 `best[j]` 并保存 backpointer。最后从文本末尾沿 backpointer 反向恢复 Token 序列。

阅读 `unigram.py` 中：

1. `EducationalUnigram._edges()`；
2. `EducationalUnigram.encode()`；
3. `backpointer` 回溯部分。

### 4.4 EM：不知道切分时如何学习概率

EM 是 Expectation-Maximization，中文常译为“期望最大化”。

训练开始时，我们只有候选 Token 和粗略初始概率，却不知道每篇文本真正应该怎样切分。

一次 EM 包含：

1. **E-step（Expectation）**：在当前概率下，计算每个候选 Token 在全部可能切分中预计被使用多少次；
2. **M-step（Maximization）**：用预计使用次数重新估计 Token 概率。

教学实现使用 forward-backward：

- `forward[i]` 汇总从开头到位置 `i` 的全部路径；
- `backward[i]` 汇总从位置 `i` 到结尾的全部路径；
- 某条 Token edge 的 posterior 来自前半路径、该 Token 分数与后半路径之和。

代码使用 `_logsumexp()` 在 log 空间安全地求“多个概率之和”。

### 4.5 Unigram sampling

Viterbi 只取最高概率路径。Sampling 则按照路径 posterior 随机选择一条合法切分。`temperature` 越高，低概率切分越容易被采样。

这与 BPE-dropout 的共同点是都能产生多种切分；差别是：

| BPE-dropout | Unigram sampling |
|---|---|
| 随机跳过已有 merge | 从概率模型的合法路径中采样 |
| 随机性作用于 merge occurrence | 随机性作用于分段路径 |
| 基础模型仍是 BPE rank | 基础模型是 Token probability |

### 思考题 5：为什么不能直接枚举所有切分路径？

<details>
<summary>答案</summary>

文本变长后，可能的切分数量呈组合爆炸。动态规划把具有相同结束位置的子问题合并，只保存必要的累计分数或概率，因此能够在可接受时间内计算最优路径和路径总概率。

</details>

### 思考题 6：教学版 Unigram 与 SentencePiece 完全相同吗？

<details>
<summary>答案</summary>

不相同。教学版直接实现了候选生成、forward-backward EM、Viterbi 与 sampling，但最后的候选裁剪使用预计计数。SentencePiece 使用更成熟的 seed vocabulary、loss-change pruning、规范化与工程优化。我们掌握的是核心概率算法，不声称逐行复现 SentencePiece。

</details>

## 5. 第 4 站：Picky BPE——词表不只会增加，也可以删除

### 5.1 经典 BPE 的问题

经典 BPE 一旦创建 Token，就不会删除它。某个中间 Token 可能只在训练早期有用，后来几乎全部被更长 Token 吸收。

例如：

```text
ent + ucky -> entucky
k + entucky -> kentucky
```

如果 `entucky` 最终几乎不独立出现，它仍占据一个词表位置，未来还可能因为训练样本太少而得到质量不稳定的 embedding。

### 5.2 教学版 Picky BPE 的状态

我们为每个 learned token 保存：

- `creation_frequency`：创建它的 Pair 当时出现多少次；
- `current_usage`：当前训练序列中还出现多少次；
- `parents`：它由哪两个 Token 合并而来。

定义一个可观察的利用率：

```text
utilization = current_usage / creation_frequency
```

训练先达到固定 active vocabulary。每个 refinement step：

1. 添加当前最佳 merge；
2. active vocabulary 暂时多一个 Token；
3. 查找利用率不高于 threshold 的旧 Token；
4. 删除利用率最低者；
5. 把训练序列中该 Token 展开成它的左右 parent；
6. active vocabulary 回到原大小。

### 5.3 为什么 encoder 需要 REMOVE 事件

假设训练过程发生：

```text
MERGE (a, b) -> ab
MERGE (ab, c) -> abc
REMOVE ab -> (a, b)
```

编码不能只保存最终 merge 列表，否则无法解释被移除 Token 的历史作用。ForgeLLM 保存按时间排列的 event：

- 遇到 `MERGE` 就合并；
- 遇到 `REMOVE` 就把仍然存在的该 Token 展开；
- 最终输出只允许 active Token。

阅读 `advanced_bpe.py`：

1. `BPEEventKind` 与 `BPEEvent`；
2. `_expand_token()`；
3. `train_picky_bpe()` 的初始词表阶段；
4. refinement loop；
5. `EducationalBPE.encode()` 对 REMOVE 的处理。

### 5.4 与论文的边界

Picky BPE 论文采用更完整的 likelihood-aware removal。ForgeLLM 教学版用可解释的 frequency ratio 选择候选，目标是把“边训练边细化词表”和“merge/remove 共同决定编码”落实成代码。

因此实验只检查：

- 是否真的产生 remove；
- active vocabulary 是否保持固定；
- 事件重放是否可逆；
- 微型夹具上的低利用率 learned token 数如何变化。

不能声称已经复现论文中的模型效果。

### 思考题 7：删除 Token 后，包含它的更长 Token 是否也必须删除？

<details>
<summary>答案</summary>

不一定。更长 Token 已经保存自己的完整 byte payload。被删除 Token 可以继续作为历史构造步骤存在，但不能成为最终输出 ID。编码重放中，如果它已经被更长 Token 消耗，REMOVE 不会影响该位置；如果仍独立存在，REMOVE 会把它展开。

</details>

### 思考题 8：为什么不直接在训练结束后删除最低频 Token？

<details>
<summary>答案</summary>

训练后删除只改变最终表面词表，没有让后续 merge 在释放出的空间和改变后的序列上重新学习。训练过程中的 remove 会改变后续 pair count，因此属于词表细化，而不只是事后裁剪。

</details>

## 6. 第 5 站：SuperBPE——边界也可以采用 Curriculum

### 6.1 什么是 curriculum

Curriculum learning 可译为“课程学习”：先学习较简单或限制更强的任务，再逐步开放更复杂的任务。

SuperBPE 的核心不是“允许跨空格”这么简单，而是：

```text
Phase 1：先在词内/类别内学习 subword
Phase 2：再开放边界，学习跨空格 superword
```

### 6.2 为什么不从第一步就开放全部边界

完全无边界的 BPE 可能很早把空格、标点与局部字符结合，挤占有限 merge 预算。两阶段设计先确保常见词内结构得到表示，再让剩余预算学习常见短语。

### 6.3 代码中的边界变化

训练数据最初是：

```text
document -> pieces -> token IDs
```

Phase 1 对每个 piece 单独统计和替换。进入 Phase 2 时：

```python
whole_documents = [
    [token_id for piece in document for token_id in piece]
    for document in documents
]
```

piece 边界被移除，但空格 byte 本身仍然存在。后续 merge 因而可以逐步学习：

```text
"new" + " " -> "new "
"new " + "york" -> "new york"
```

Encoder 同样分两阶段重放：先对各 piece 应用 phase-1 event，拼接后再应用 phase-2 event。

### 思考题 9：为什么 phase-2 Token 可以跨空格，但仍然能无损 decode？

<details>
<summary>答案</summary>

因为空格没有被删除，而是成为 Token byte payload 的一部分。`b"new york"` 解码时仍然输出中间的空格 byte。

</details>

### 思考题 10：跨空格 Token 一定更好吗？

<details>
<summary>答案</summary>

不一定。它可能减少常见短语的 Token 数，也可能产生领域依赖强、低频或难以充分训练的大 Token。是否改善语言模型必须通过固定训练预算的下游实验判断；当前实验只证明课程和跨边界机制有效。

</details>

## 7. 第 6 站：特殊 Token、Offset 与 Tokenizer-free Patching

### 7.1 特殊 Token 不是普通字符串

`<eos>` 可以有两种含义：

1. 用户真的输入了六个普通字符；
2. 系统要求插入“序列结束”控制 ID。

如果 Tokenizer 无条件把用户文本中的 `<eos>` 转成控制 ID，恶意或意外输入可能改变模型协议。

`encode_with_special_policy()` 采用显式规则：

- 普通输入出现已知特殊 Token 字面量时默认报错；
- 调用者把它加入 `allowed_special` 后才转换成控制 ID；
- 未知 special 名称同样报错。

### 7.2 Byte offset 有什么用

Offset 是 Token 对应原文位置。中文字符可能占 3 个 UTF-8 bytes，所以 byte offset 和 Python 字符下标不是同一概念。

`encode_with_byte_offsets()` 返回半开区间：

```text
[start_byte, end_byte)
```

它可用于高亮 Token、对齐模型输出、数据标注，以及检查 normalize/pre-tokenize 是否改变位置。

### 7.3 从固定 Token 到动态 Patch

BPE 和 Unigram 都先训练一个固定词表。BLT（Byte Latent Transformer）路线则直接保留 byte，并动态把相邻 bytes 组成 patch。

Surprisal 可译为“惊讶度”：

```text
surprisal(byte) = -log2 P(byte | context)
```

- 预测概率高：surprisal 低；
- 预测概率低：surprisal 高。

教学版 `BigramEntropyPatcher` 使用前一个 byte 预测下一个 byte：

1. 统计 `previous_byte -> next_byte` 次数；
2. 加 smoothing，避免未见转移概率为 0；
3. 计算每个位置的 surprisal；
4. surprisal 超过 threshold 时开始新 patch；
5. 同时设置 `max_patch_bytes`，防止 patch 无限增长。

这不是 BLT：BLT 使用学习到的 entropy model、local encoder/decoder 和 latent Transformer。我们的代码只让动态边界规则可以手算和测试。

### 思考题 11：固定 BPE Token 与 entropy patch 的主要区别是什么？

<details>
<summary>答案</summary>

BPE 的切分由固定词表和 merge rank 决定；entropy patch 的边界由当前位置在上下文中的预测难度决定。相同 byte 片段在不同上下文中可能得到不同 patch 边界，而且 patch 不需要作为固定词表项拥有独立 embedding。

</details>

### 思考题 12：为什么 entropy 高的位置倾向于开启新 patch？

<details>
<summary>答案</summary>

高 entropy 表示下一个 byte 更难预测，通常需要更细的局部处理。把边界放在困难位置附近，可以让大模型对容易区域使用较长 patch、对困难区域保留更细粒度的信息。这是动态分配计算的直觉。

</details>

## 8. 第 7 站：统一实验应该怎样阅读

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\tokenizer_method_lab.py `
  --train tests\fixtures\tokenizer\train.jsonl `
  --evaluation tests\fixtures\tokenizer\method_lab_evaluation.jsonl `
  --output artifacts\stage01_tokenizer_method_lab\student_report.json
```

报告固定比较六条可直接评测的路径：

1. raw UTF-8 byte；
2. global classic BPE；
3. Unicode-boundary classic BPE；
4. Picky BPE 教学版；
5. SuperBPE curriculum；
6. Unigram EM。

评测文件专门提供 English、Chinese、code、numbers、whitespace、Emoji 和 mixed 七个 subset。另外单独报告 dropout、Unigram sampling、特殊 Token、offset 和 entropy patch。

### 8.1 先看硬不变量

所有方法首先检查：

- `round_trip_rate == 1.0`；
- 随机切分的所有样本都能解码；
- byte offsets 覆盖完整输入；
- entropy patches 拼接后恢复原文。

任何一项失败，都不能继续讨论压缩率。

### 8.2 再看结构差异

重点字段：

- `cross_whitespace_tokens`：词表中同时含空白和非空白的 Token 数；
- `learned_tokens_used_once_or_never`：训练夹具上利用率很低的 learned token；
- `removal_events`：Picky 实际发生多少次 remove；
- `phase_two_merge_events`：SuperBPE 有多少 merge 在开放边界后学习；
- `unique_segmentations`：随机方法产生多少种不同切分。

### 8.3 最后才看 bytes/token

微型夹具上的 bytes/token 只回答：

> 在这几篇项目原创文本上，该方法平均一个 Token 覆盖多少 UTF-8 bytes？

它不回答模型准确率、真实中英文公平性、生产吞吐、最优词表规模或在其他语料上的排名。

### 思考题 13：Picky 的低利用率 Token 数更少，能否直接声称它更好？

<details>
<summary>答案</summary>

不能。它只证明当前 removal heuristic 在该夹具上改变了词表利用情况。还需要真实语料、相同模型训练预算、下游指标和统计重复，才能讨论模型质量。甚至低频 Token 有时也可能对罕见但重要的文本有价值。

</details>

### 思考题 14：SuperBPE 的 Token 数更少，能否直接声称推理更快？

<details>
<summary>答案</summary>

不能直接声称。Token 数减少通常会减少主干序列长度，但实际速度还受词表大小、softmax、内核、batch、缓存和模型架构影响。当前没有训练语言模型，也没有做端到端推理基准。

</details>

## 9. 代码追踪练习

### 练习 A：追踪 pre-tokenization

输入 `A12 中文!`。请在纸上写出 Unicode-class pieces，然后运行 `split_text()` 核对。

<details>
<summary>答案</summary>

概念上应分为：字母 `A`、数字 `12`、空格、字母类的 `中文`、其他类 `!`。代码返回每段的 UTF-8 bytes；拼接后必须等于原始 UTF-8 bytes。

</details>

### 练习 B：追踪 Picky event

找到第一次 `REMOVE`，回答：

1. 被删除 Token 的 parent 是什么？
2. 当前训练序列出现它多少次？
3. encoder 在什么顺序应用对应 MERGE 和 REMOVE？

<details>
<summary>答案</summary>

具体 ID 随夹具和 tie-break 决定，应以运行结果为准。固定规则是先应用创建该 Token 的 MERGE，再按事件时间应用后续 merge，最后在遇到 REMOVE 时展开仍独立存在的该 Token。不能先把所有 remove 做完再做 merge。

</details>

### 练习 C：追踪 Unigram Viterbi

对一个短字符串打印或手写：

```text
position -> 可匹配 Token -> candidate score -> backpointer
```

<details>
<summary>答案</summary>

答案不是固定 Token 序列，而是正确追踪方式。每个 position 枚举 `_edges()`；用 `best[position] + token_score` 更新结束位置；到达最后一个 byte 后沿 backpointer 反向恢复，再 reverse。

</details>

## 10. 最终口述验收

请在不看讲义的情况下回答：

1. Pre-tokenization 改变了 BPE 的哪一个输入条件？
2. 为什么它必须可逆？
3. BPE-dropout 随机丢弃的是什么？
4. Unigram 的 Token 概率怎样决定整条切分路径？
5. Forward-backward 与 Viterbi 分别解决什么问题？
6. Picky BPE 为什么需要 remove event？
7. 删除中间 Token 后，为什么较长 Token 仍可存在？
8. SuperBPE 的两个 phase 分别允许哪些边界？
9. 用户文本中的 `<eos>` 为什么不能默认当控制 Token？
10. Byte offset 与字符下标为什么不同？
11. Entropy patch 与固定词表 Token 的差别是什么？
12. 为什么当前实验不能证明模型下游性能提升？

如果第 4～8 题无法回答，暂时不要进入 Stage 2；回到相应算法，手算一次最小例子并阅读对应测试。

## 11. 本阶段最终认识

Tokenizer 不只是一个 BPE merge 循环，而是四层共同决策：

```text
原文处理与边界
→ 候选词表怎样产生和更新
→ 给定词表怎样选择切分路径
→ Token ID 怎样安全地交给语言模型
```

经典 BPE、Unigram、Picky BPE、SuperBPE 和 entropy patching 的差别，都可以放回这四层定位。掌握这种定位能力，比在小项目里寻找一个“完美词表大小”更重要。
