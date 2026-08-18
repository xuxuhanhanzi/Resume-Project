# Stage 3 完整讲义一：预训练数据、Packing 与目标函数

## 1. 预训练真正消费的是什么

模型不会直接读取“故事”或“文档”。训练程序最终只看一个整数 Tensor：

```text
原始文档
→ 固定 Tokenizer
→ 每篇文档独立 encode
→ 文档末尾追加 EOS
→ 拼成 token stream
→ 切成固定长度窗口
→ 组成 [batch, sequence] Tensor
```

本项目先冻结 Stage 1 的 byte-BPE。这样 Stage 3 的变量是训练方法，而不是词表规模。Tokenizer 压缩率不理想会让相同文本产生更多 token，但不会破坏 byte fallback、可逆性和训练闭环。

**思考题：为什么不在 Stage 3 顺便把词表从 320 调到 8,000？**

答案：那会同时改变序列长度、样本数量、PPL 分母、embedding 参数量、吞吐和优化难度。我们将无法判断结果变化来自训练器还是 Tokenizer，违反单一主要变量原则。

## 2. 为什么每篇文档必须独立 encode

BPE merge 不应跨文档边界学习出一个“上一篇结尾+下一篇开头”的 token。实现中先对每篇文档执行 `tokenizer.encode(text, add_eos=True)`，再拼接结果：

```text
doc A tokens + EOS + doc B tokens + EOS + ...
```

EOS 的作用不是 padding。它是一个有语义的 target：告诉模型一个文档结束，并让模型学习结束后可能开始新文档。

**思考题：如果直接把全部字符串连接后再 BPE，会发生什么？**

答案：merge 可能跨越人为文档边界；数据来源和模型输入的语义边界不再一致。即使字符串中加入分隔符，也必须明确该分隔符是否属于词表和训练目标。

## 3. 为什么先切分文档，再 tokenize

正确顺序是：

```text
原始文档集合
→ 用内容哈希确定 train/validation/test
→ 每个 split 独立 tokenize 与 packing
```

错误顺序是先把所有 token 拼成一条流，再按位置切分。后一种做法可能把同一篇故事的一半放在 train，另一半放在 validation；模型已经见过上下文，验证 loss 会虚假变好。

Stage 3 使用内容 SHA-256 和固定 seed 做文档级切分。相同内容总会进入同一个 split，输入行顺序改变也不会改变归属。

**思考题：随机打乱行以后再按前 90% 切 train 是否足够？**

答案：只有同时冻结随机种子、打乱算法和输入顺序时才可复现，而且重复内容仍可能跨 split。内容哈希切分更容易审计，但仍需先做精确去重。

## 4. Packing 为什么重叠一个 token

设序列长度 `T=5`，token stream 为：

```text
[a, b, c, d, e, f, g, h, i]
```

若窗口完全不重叠：

```text
[a,b,c,d,e] 预测 b,c,d,e
[f,g,h,i,...] 预测 g,h,i,...
```

转移 `e→f` 永远不会被训练。ForgeLLM 使用 stride `T-1`：

```text
[a,b,c,d,e] 预测 b,c,d,e
[e,f,g,h,i] 预测 f,g,h,i
```

边界 token `e` 重复作为 context，但每个 target transition 只出现一次。尾部不足一个完整窗口的 target 数由 `dropped_tail_tokens` 记录，不被静默隐藏。

**思考题：重叠 token 是否算数据重复？**

答案：它作为输入 context 重复一次，但不作为 target 重复计数。训练预算使用 target token 数，而不是 Tensor 中所有 token 元素数。

## 5. Causal LM 的 label shift

模型输入 `[x0,x1,x2,x3]` 会产生四个位置的 logits。标准 next-token loss 使用：

```text
logits[:, 0] 预测 x1
logits[:, 1] 预测 x2
logits[:, 2] 预测 x3
logits[:, 3] 没有下一个 token，不计入
```

代码位于 `next_token_loss`：

```python
predictions = logits[:, :-1, :]
targets = input_ids[:, 1:]
```

因此一个 `[B,T]` batch 的 target token 数为 `B*(T-1)`。

**思考题：为什么不能让 `logits[:, i]` 预测 `input_ids[:, i]`？**

答案：当前位置的 token 已经作为输入进入模型，模型可以近似复制它；这不是自回归的下一个 token 学习目标。

## 6. 确定性 batch stream

普通 `DataLoader(shuffle=True)` 在多 worker、prefetch 和中断时很难仅靠“当前 step”恢复准确位置。本项目使用一个刻意简单的状态机：

- `seed + epoch` 生成当轮 permutation；
- `position` 表示该 permutation 中下一个样本位置；
- batch 跨 epoch 尾部时继续从下一轮取样，不丢弃尾部；
- checkpoint 只需保存 dataset fingerprint、batch size、seed、epoch、position。

恢复时重新生成同一 permutation，再把 cursor 放回同一位置。

**思考题：为什么只保存 epoch 不够？**

答案：中断通常发生在 epoch 中间。不保存 position 就只能从 epoch 开头重新训练，造成 batch 重复。

## 7. Token cache 不是普通性能缓存

纯 Python BPE 每次启动重新编码 19MB 语料会让启动开销超过短训练本身。因此 Stage 3 创建 token cache，但 cache 必须绑定：

1. JSONL 源文件 SHA-256；
2. Tokenizer fingerprint；
3. sequence length。

任何一项不匹配都拒绝加载，而不是悄悄使用旧 cache。cache 保存 token IDs、每个 token 对应的 UTF-8 byte 数、文档 ID 和数据集 fingerprint，并用临时文件加原子替换写入。

**思考题：为什么 sequence length 也要进入 cache 契约？token stream 不是一样吗？**

答案：当前 cache 的 dataset fingerprint 和窗口数量依赖 sequence length。若以后改成“只缓存原始 stream”，可以把窗口元数据分层；在 v1 中绑定它更简单且不容易误用。

## 8. Validation、PPL 与 BPB

Validation 必须满足：

- `model.eval()`；
- `torch.inference_mode()`；
- 不调用 backward/optimizer；
- 不推进训练数据 cursor；
- 结束后恢复之前的 train/eval 模式。

若平均 token 负对数似然为 `L`：

```text
PPL = exp(L)
```

PPL 的一个“事件”是 token。因此两个 Tokenizer 把同一文本切成不同 token 数时，PPL 不可直接公平比较。

BPB 使用原始 UTF-8 byte 做分母：

```text
BPB = total NLL in nats / (ln(2) * target bytes)
```

EOS 没有原始 byte payload，所以不增加 byte 分母，但它仍是一个 target token。

**思考题：PPL 低是否证明模型会生成更好的故事？**

答案：不证明。PPL 只衡量指定数据分布和 tokenization 下的平均预测概率。生成质量还受解码、覆盖范围、重复、事实性和任务需求影响。

## 9. 本讲义代码追踪

按顺序追踪：

1. `read_training_documents`
2. `PackedTokenDataset.from_documents`
3. `PackedTokenDataset.__getitem__`
4. `DeterministicBatchStream._next_indices`
5. `load_or_create_packed_jsonl`
6. `next_token_loss`
7. `Trainer.validate`
8. `bits_per_byte`

完成后，你应该能从一个 JSONL 文档一直解释到 loss 的每一个分母。
