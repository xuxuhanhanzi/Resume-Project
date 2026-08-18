# Stage 4 完整讲义二：SFT 训练闭环与可信评测

## 1. Collator 的职责只有组 batch，不应改语义

同一 batch 中样本长度不同，Stage 4 采用右侧 padding：

```text
input_ids:      [真实 token ........ PAD PAD]
attention_mask: [1 1 1 1 .......... 0   0]
labels:         [-100/目标 ........ -100 -100]
```

`attention_mask=0` 防止 padding 参与注意力；`labels=-100` 防止 padding 进入 CE。这两者解决不同问题，缺一不可。`pad_to_multiple_of=8` 只扩展 padding，不得改变 `supervised_tokens`。

**思考题：只有 `labels=-100`，不传 attention mask 可以吗？**

答案：PAD 虽不直接进入 loss，真实 token 仍可能注意到 PAD 表示并改变 logits。右 padding 的纯 causal mask 对较早 token 通常挡住未来 PAD，但批处理、缓存和不同实现下不能依赖这个偶然性质；协议上仍应显式传 attention mask。

## 2. 梯度累积最容易错的是分母

每个 micro-batch 返回 `negative_log_likelihood_sum` 和 `target_tokens`。连续 K 个 micro-batch 的正确目标是：

```text
L = (NLL_sum_1 + ... + NLL_sum_K) / (tokens_1 + ... + tokens_K)
```

实现先对每个 NLL sum 做 backward，使梯度累加；到 optimizer step 前，再把每个 Adapter gradient 除以累计目标 token 数。若对每个 micro-batch 的平均 loss 直接除 K，短回答 batch 会被过度加权。

**思考题：当每个 micro-batch 的监督 token 数完全相同时，两个方法是否等价？**

答案：等价；但对话回答长度通常不同。工程实现不能把“本次 fixture 碰巧等长”当算法假设。

## 3. 一个 Adapter optimizer step 的准确顺序

```text
加载 K 个 micro-batches
→ 每批 forward
→ 对 NLL sum backward
→ 累计 target token 分母
→ 除真实分母
→ 检查所有非 LoRA 参数 grad is None
→ 检查梯度范数有限并 clip
→ optimizer.step()
→ zero_grad(set_to_none=True)
→ 更新 assistant_tokens / step / elapsed
→ 检查 token、step、time 三个停止条件
```

Stage 4 使用 AdamW，只把 `requires_grad=True` 的 LoRA 参数交给优化器。Base 参数即使未传给 optimizer，也必须显式 `requires_grad=False` 并检查没有 grad；“optimizer 不更新它”弱于“它根本不产生梯度”。

本实现固定了 max steps、max assistant tokens、max duration，先到即停。正式运行因 100,293 tokens 在第 129 步触发 token 上限，没有继续到 500 步。

**思考题：为什么停止预算使用 assistant tokens，而不是输入 tokens？**

答案：assistant-only SFT 的直接学习信号只来自 assistant targets。相同输入长度可以有极不相同的回答长度；用总输入 token 会让有效监督预算不可比。系统吞吐仍应另行记录全部计算量或序列长度分布。

## 4. Tiny qualification 与方法实验回答不同问题

资格门只用两条记录重复 160 步，结果为：

- first loss `5.4090`；
- final loss `0.0045867`；
- assistant token accuracy `1.0`；
- 通过条件为 loss `<0.01` 且 accuracy `=1.0`。

这证明数据、mask、loss、backward 和更新链能工作，不证明泛化。

方法实验固定 8 条样本、80 步、seeds 41/42/43，只改变监督范围或更新参数：

| variant | 3 seeds 最终 loss 概况 | 解释 |
|---|---|---|
| full sequence | 约 0.36–0.40 | 同时背 prompt 和 response，任务更容易 |
| assistant only | 约 0.58–0.63 | 只优化回答，目标更严格 |
| hand-written LoRA | 约 4.85–4.93 | 仅 9,728 参数、80 步预算下明显欠拟合 |

LoRA 没有获胜仍是有效结果：它验证低秩路径可以训练，也说明参数效率不等于相同步数下更低训练 loss。继续调 rank/LR 直到获胜会破坏预先冻结的实验。

**思考题：能否用 full-sequence loss 更低证明它更适合聊天？**

答案：不能。它的一部分优化能力用于预测固定 prompt 和角色文本，两个 loss 的监督区域不同；数值不是同一个目标。必须在独立生成任务上比较行为。

## 5. 评测必须分四层

### 5.1 Held-out assistant loss

在 SmolTalk validation 上，用真实完整对话做 teacher forcing，只计算 assistant targets。它衡量模型给真实回答分配的概率，适合监测 SFT 拟合。

### 5.2 Correctness teacher-forced loss

在原创格式任务 test 上仍喂真实回答前缀，测答案 token 概率。它比生成容易，因为每一步都看到正确历史。

### 5.3 Greedy constrained generation

只提供 system/user 和 `assistant\n` 前缀，让模型自己生成。Verifier 检查 exact match、JSON、前缀、必含/禁含字符串和精确词数。它才直接回答“模型是否遵循输出约束”。

### 5.4 Pretraining retention loss

在固定 TinyStories 文本上测同一 Qwen tokenizer 下的 causal loss，只比较 Qwen Base 与其 Adapter。它是遗忘哨兵，不与 Stage 3 的 320-vocab PPL 横比。

**思考题：teacher-forced token accuracy 上升，generation 仍为 0%，矛盾吗？**

答案：不矛盾。teacher forcing 在正确历史条件下逐 token 预测；generation 一次早期错误会改变之后全部上下文，并且必须自己学会停止。两者暴露的能力不同。

## 6. 正式实验的五维结果

固定 Base revision、数据、模板、greedy 配置后，100k BF16 LoRA 得到：

| 维度 | 训练前 | 训练后 | 结论 |
|---|---:|---:|---|
| SmolTalk assistant loss | 1.5660 | 1.1500 | 对 SFT 分布拟合改善 |
| assistant token accuracy | 0.6478 | 0.6860 | teacher-forced 小幅改善 |
| correctness TF loss | 3.1000 | 1.8033 | 目标 token 概率改善 |
| correctness exact/constraint | 0/16 | 0/16 | 严格行为没有学会 |
| retention loss | 1.7770 | 1.8098 | 出现轻微回退 |

这组结果的正确表述是：“在单 seed、100k assistant tokens 的小型 SFT 下，held-out response likelihood 改善，但严格约束生成仍失败，并观察到小幅 pretraining-retention 退化。”不能写成“模型指令遵循能力显著提升”。

**思考题：为什么 correctness teacher-forced loss 大幅下降，测试又没在训练集出现？**

答案：SmolTalk constraints 教会了一些一般格式/回答模式，Base 本身也有能力；但概率改善不足以让正确短答案压过继续解释或退化重复的 token。严格 exact match 对停止和所有字符都敏感。

## 7. 指标也会失败：重复率审计

原指标把响应按空白切词再算 trigram。正式报告显示 word trigram 从 `0.3842` 降到 `0.0025`，看似巨大改善；人工查看却发现 `p��sito` 等无空格子串反复出现。因为整段重复被 `.split()` 当成一个长“词”，word trigram 根本看不到内部周期。

因此新增不覆盖原报告的 audit：移除空白后计算 character 8-gram 重复。结果：

```text
before: 0.2080
after:  0.4446
```

真实结论是字符级重复显著恶化。原指标仍保留，作为评测器失效的证据；不能事后改掉原报告假装第一次就设计正确。

**思考题：为什么不能看完输出后不断改 n，挑一个最符合直觉的值？**

答案：这会把测试集变成指标开发集。当前补充审计只用于揭示已观察到的盲区；Stage 5/6 应在新冻结数据上预先登记 word/token/character 多尺度指标与阈值。

## 8. 训练系统结果如何读

正式 BF16 LoRA：

- 606,142,464 总参数；10,092,544 可训练，占 1.665%；
- 100,293 assistant tokens，129 optimizer steps；
- 3% assistant-token warmup + cosine，最终 LR 为 0；
- 训练耗时 375.0 秒，整段 267.5 assistant tokens/s；
- 训练峰值 PyTorch allocated 1,939,151,360 bytes；
- Base gradients absent 全程为真；
- stop reason 为 `max_assistant_tokens`。

同数据顺序的恒定 LR v1 曾在后半段明显降速，整段只有 96.3 tokens/s；合规 v2 为 267.5。没有足够证据把差异单独归因于数据长度，系统负载、热状态和运行时缓存也可能参与。正式结论只报告各自总实测，不挑最快区间。`allocated` 也不是进程全部显存；它不包含驱动、CUDA context 和所有外部 allocator。

**思考题：训练 peak allocated 能直接与 `nvidia-smi` 数字比较吗？**

答案：不能当同一指标。PyTorch allocated、reserved、进程显存和系统 GPU 占用口径不同；同协议实验应比较同一个 API，并同时注明其边界。

## 9. 本讲代码追踪

1. `collate_tokenized_conversations`
2. `assistant_only_causal_loss`
3. `scale_gradients_by_token_count`
4. `run_tiny_variant`
5. `evaluate_assistant_loss`
6. `evaluate_generation`
7. `evaluate_retention_loss`
8. `train_bounded`
9. `stage4_repetition_audit.py`

完成后，你应能解释为什么本次实验“优化成功”和“行为失败”可以同时成立，并指出原重复指标为何给出相反印象。
