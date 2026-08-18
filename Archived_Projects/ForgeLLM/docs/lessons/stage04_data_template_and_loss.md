# Stage 4 完整讲义一：对话数据、Chat Template 与 SFT Loss

## 1. Base、Pretrained、Instruct 和 Chat Model 不要混用

`Qwen3-0.6B-Base` 是预训练 Base：它学习下一个 token，但没有被保证遵循 system/user/assistant 协议。SFT 用“输入对话→目标回答”继续训练 Base；得到的模型可以称为 instruction-tuned，但是否已经是好用、安全、稳定的 Chat Model，还取决于偏好优化、安全训练、评测和部署协议。

```text
预训练 Base + SFT → 能学习指令格式和目标回答
再加偏好/RL/安全与系统评测 → 才逐步形成可交付 Chat Model
```

**思考题：为什么不能用 Qwen3 Instruct 作为本阶段训练前 Base？**

答案：Instruct 已经经过未知规模的后训练。把它当 Base 会让训练前能力包含既有 SFT/RL 影响，无法观察本项目 SFT 实际增加了什么，也无法公平解释失败。它可以只读参考模板和行为，不能替代 Base baseline。

## 2. 一条监督记录的最小契约

本项目不接受任意字典。`InstructionRecord` 固定为：

```json
{
  "id": "correctness-080",
  "messages": [
    {"role": "system", "content": "Follow the requested output format exactly."},
    {"role": "user", "content": "Reply with exactly: ITEM-080"},
    {"role": "assistant", "content": "ITEM-080"}
  ],
  "source": "forgellm-stage4-correctness-v1",
  "license": "project-original",
  "metadata": {"expected_response": "ITEM-080"}
}
```

`id` 用于追踪；`messages` 决定模型输入和目标；`source/license` 决定能否合法复现；`metadata` 只保存 JSON-compatible 的评测规则，不能暗藏 Python 对象。

**思考题：为什么 `expected_response` 不直接作为另一个顶层字段？**

答案：不同任务的规则不同，例如 JSON、前缀、禁用词或精确词数。稳定顶层 Schema 只表达训练记录，共性较弱的评测规则进入严格 JSON metadata；评估器仍必须逐项验证类型，不能盲信 metadata。

## 3. 角色状态机为什么要严格

允许的顺序是：可选的单个 system 必须位于 0；之后 user/assistant 交替；最后必须是 assistant。合法例子：

```text
user → assistant
system → user → assistant
user → assistant → user → assistant
```

以下全部拒绝：

- assistant 开头：缺少条件输入；
- 两个连续 user：无法确定哪个回答属于哪个 prompt；
- system 出现在中间：模板语义与多数聊天协议不一致；
- user 结尾：没有本条 SFT 的目标回答；
- 空 content：可能产生零监督 token 或歧义边界。

严格并不是说现实数据永远只有三种角色，而是 Stage 4 尚未实现 tool call 的独立状态机。与其把未知角色静默当文本，不如 fail-fast。

**思考题：多轮对话中，前一轮 assistant 是否也进入 loss？**

答案：当前契约会监督所有 assistant 内容，不只最后一轮。这相当于一条记录提供多个 assistant span。如果只想监督最后一轮，必须新增显式策略和测试，不能偷偷改变 mask。

## 4. 数据冻结、去重和切分

Stage 4 有两层数据：

| 数据 | 规模 | 用途 |
|---|---:|---|
| `stage4_correctness_v1` | 64/16/16 | 原创确定性任务、tiny overfit、严格行为测试 |
| SmolTalk `smol-constraints` | 2048/256/256 | 框架轨 SFT 与 held-out response loss |

公开数据固定到 revision `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`。处理顺序为：严格角色校验→总字符数不超过 1600→对 canonical messages 做 SHA-256 精确去重→按内容哈希排序→固定取前缀。train/validation/test 的内容哈希集合必须互斥。

这不是随机抽样统计代表性方案，而是小预算、可重复的教学冻结方案。它不能支持“代表全部聊天分布”的结论。

**思考题：为什么按哈希排序比直接取上游前 2048 条更可审计？**

答案：上游行顺序变化会改变直接前缀；内容哈希提供与输入顺序无关的稳定次序。它仍不等于分层随机抽样，但更适合本阶段的确定性复现。

## 5. Chat Template 是模型协议，不是美化字符串

Qwen ChatML 的一条消息形式为：

```text
<|im_start|>system\n内容<|im_end|>\n
<|im_start|>user\n内容<|im_end|>\n
<|im_start|>assistant\n目标<|im_end|>\n
```

本项目分段 tokenize，而不是先拼完整字符串再猜字符 offset：

1. role prefix：不监督；
2. message content：只有 assistant 监督；
3. `<|im_end|>`：assistant 消息监督，教模型停止；
4. 换行：不监督。

分段有一个明确代价：BPE merge 不跨段边界。因此它与“完整字符串一次 tokenize”的 token IDs 不保证完全相同。我们主动把这个行为定义为教学模板契约，以换取 assistant span 的可审计性；生产模板应优先使用官方 `apply_chat_template` 返回的 generation mask，并做逐 token 对照。

**思考题：为什么 role prefix 不进入 loss，却仍能影响梯度？**

答案：它们作为上下文参与 assistant logits 的计算。`-100` 只取消该位置的直接 CE 项，assistant loss 的梯度仍会通过注意力路径回传到处理 prompt 的激活；“不直接监督”不等于“对训练完全无影响”。

## 6. `input_ids`、`assistant_mask` 与 `labels`

对简化序列：

```text
位置       0      1       2      3       4
token      BOS   user    问题   asst    答案
mask       F      F       F      F       T
labels    -100   -100    -100   -100    答案ID
```

真正的 causal shift 发生后，是 `logits[:,3]` 预测 `labels[:,4]`。`TokenizedConversation` 在构造时强制：三个数组等长；mask 与 `label != -100` 完全一致；监督 label 必须复制对应 input token；shift 后至少存在一个监督位置。

截断只保留前 `max_length` token。如果截断砍掉全部 assistant 区域，就立即报错。本阶段第一次 tiny 实验用 128 长度正是因此失败；测得最长 fixture 需要 236，随后把该实验固定为 256。这个失败说明“能 tokenize”不等于“仍有训练目标”。

**思考题：为何不能从序列尾部截取，确保保留 assistant？**

答案：尾截可能丢掉 system/user 条件，让答案脱离问题。正确策略取决于任务，可以丢弃超长样本、截 prompt、截 response 或分层采样；必须显式选择并测试。本阶段采用简单前截并拒绝零监督样本。

## 7. assistant-only shifted cross-entropy

令 logits 为 `[B,T,V]`、labels 为 `[B,T]`：

```python
predictions = logits[:, :-1, :]
targets = labels[:, 1:]
selected = targets != -100
nll_sum = cross_entropy(predictions, targets, ignore_index=-100, reduction="sum")
loss = nll_sum / selected.sum()
```

必须只 shift 一次。若数据预处理先把 labels 左移，模型 loss 又左移一次，目标会错一位。分母是 shift 后非 `-100` 的真实 assistant targets，不是 batch size、序列长度或消息数。

举例：两条样本分别有 10 和 90 个 assistant targets。正确总 loss 为 `(NLL1+NLL2)/100`；先求每条均值再平均会让 10-token 样本与 90-token 样本权重相同，改变目标函数。

**思考题：EOS 是否应该监督？**

答案：本项目监督 assistant 的 `<|im_end|>`，因为准确停止属于目标行为。不监督会削弱停止信号；监督错误位置则可能让模型提前结束。这个选择必须与生成的 `eos_token_id` 一致并记录。

## 8. 代码追踪与本讲验收

按顺序追踪：

1. `Message.__post_init__`
2. `InstructionRecord.__post_init__`
3. `content_fingerprint` 与 `dataset_fingerprint`
4. `tokenize_qwen_record`
5. `TokenizedConversation.__post_init__`
6. `collate_tokenized_conversations`
7. `assistant_only_causal_loss`

完成后，你应该能拿一条多轮 JSONL，逐 token 指出哪些位置只是条件、哪些位置直接进入 loss，以及 shift 后的准确分母。
