# Stage 3 完整讲义四：Muon、MTP 与有界预训练实验

## 1. 为什么 Stage 3 只选择 Muon 和 MTP

Stage 2 已学习 MoE、mHC、MLA、混合 attention 等架构方法，但预训练阶段如果同时替换注意力、残差、优化器和目标函数，任何结果都无法归因。

Muon 改变“怎样更新参数”；MTP 改变“给模型什么监督信号”。它们分别对应优化器和目标函数，适合做两个相互独立的教学实验。

**思考题：为什么不能运行一个“Muon+MTP”并与 AdamW baseline 比较？**

答案：结果变化至少有两个解释，无法知道哪个技术有效，是否存在交互也无法从单一组合判断。

## 2. Muon 的核心思想

Transformer 的许多权重是矩阵。Muon 先形成带 momentum 的矩阵 update，再用 Newton–Schulz 迭代近似其正交极因子，让不同方向的奇异值更均衡。

教学实现的核心：

```text
gradient
→ momentum / Nesterov update
→ reshape 为二维矩阵
→ Newton–Schulz zeroth-power/orthogonalization
→ aspect-ratio scaling
→ 参数更新
```

embedding、norm、bias 等参数不都适合矩阵正交化。因此本项目使用混合方案：合格矩阵交给 Muon，其余参数交给 AdamW。

Muon 学习率与 AdamW 学习率不是同一量级，不能简单复制。比较时固定模型、初始化、数据顺序、训练步数和 precision，只改变 optimizer strategy。

**思考题：本实验 Muon loss 更低，能否写“Muon 优于 AdamW”？**

答案：不能。当前证据只有一个 tiny 架构、100 步、三个种子和一组超参数。可以说“在冻结短程设置下，Muon 的 validation loss 均值更低且三种子方向一致”，不能推广到长训练、其他模型或最终质量。

## 3. MTP 的核心思想

普通 next-token prediction 在位置 `t` 只预测 `x[t+1]`。MTP 让同一 hidden state 额外预测更远的 token：

```text
offset 1: x[t+1]
offset 2: x[t+2]
...
```

本项目的 head 输出 `[B,T,K,V]`，每个 offset 只对仍有合法 target 的位置计算 CE。总目标：

```text
total_loss = main_next_token_loss + lambda * mtp_loss
```

MTP 可能提供更密集的未来监督，也能与推测解码思路关联；但它增加 head 参数和优化难度，并不保证小模型短跑的主 loss 更低。

**思考题：为什么报告必须同时保存 main loss、MTP loss 和 total loss？**

答案：total loss 因加入辅助项天然会改变数值，不能直接与 baseline 的 main loss 比较。分开记录才能判断主任务和辅助任务分别发生什么。

## 4. 三种子方法实验

固定项：tiny Decoder、同一正式 train/validation、FP32、100 optimizer steps、相同 schedule；种子为 41/42/43。

| Variant | 唯一主变量 | Validation loss 均值 | 标准差 |
|---|---|---:|---:|
| AdamW single | 基线 | 3.4075 | 0.0126 |
| Muon single | optimizer | 3.1926 | 0.0068 |
| AdamW MTP | objective/head | 3.4137 | 0.0080 |

观察：Muon 在当前短预算中更新更快；MTP 没有改善主 validation loss。可能解释包括 MTP 权重、head 初始化、模型容量、训练长度或目标竞争，但本阶段不做搜索来挑选有利结果。

**思考题：MTP 结果没有变好，是否说明实现失败？**

答案：不是。offset loss、梯度和 optimizer step 测试都通过。算法正确性与效果优越性是两个不同命题；负结果也应保留。

## 5. Smoke、Qualification 与 Bounded Run

### Smoke

20 步 tiny 模型检查：接口、loss finite、checkpoint、validation 和报告能否闭环。它不能决定正式配置。

### Qualification

100 步 5.36M/BF16 检查：

- loss 是否明显下降；
- 是否有 NaN/Inf/OOM；
- 显存和吞吐是否稳定；
- validation 和 checkpoint 是否正常。

结果：首 10 步 loss 5.1521，末 10 步 2.7290，最终 validation loss 2.6289；峰值显存约 171.6 MiB。

### Bounded Run

固定上限：1M target tokens、250 steps、3,600 秒，先达到者停止。第 125 步主动中断，恢复后第 247 步达到 token 上限。

| 指标 | 结果 |
|---|---:|
| target tokens | 1,003,808 |
| 首 10 步平均 loss | 5.3847 |
| 末 10 步平均 loss | 1.5995 |
| final validation loss | 1.6245 |
| PPL | 5.0759 |
| BPB | 2.0543 |
| 平均吞吐 | 24,811 tokens/s |
| 峰值 allocated memory | 235.9 MiB |
| 日志 step | 1–247 连续 |

**思考题：为什么 5M 模型显存只显示约 236 MiB，不等于显卡总占用？**

答案：这里记录 `torch.cuda.max_memory_allocated`，只统计 PyTorch 活跃 Tensor 分配，不含 CUDA context、驱动、缓存保留和桌面图形占用。报告必须写清指标定义。

## 6. 固定 prompt 的作用

固定 prompt 只用于查看训练前后是否出现明显变化和错误模式，例如：

- 是否只输出重复 token；
- 是否很快 EOS；
- byte-level 生成是否产生非法 UTF-8；
- 是否出现简单局部模式。

320 词表和 1M-token 短训练不应被期待生成高质量故事。若任意生成 token 组合成非法 UTF-8，报告保存 token IDs 和解码错误，而不是丢弃样本。

**思考题：看到一个漂亮样例能否证明模型质量？**

答案：不能。单样例可被随机性和挑选偏差影响。定性样例必须固定 prompt/seed，并与定量 validation 和错误分布一起报告。

## 7. 如何正确写结论

允许写：

- 实现了 deterministic packing、mixed precision、完整 checkpoint 和 bounded trainer；
- CPU/FP32 exact-resume 全状态一致；
- CUDA 正式运行在中断后保持 step 和 token 预算连续；
- 当前 1M-token 设置 loss 下降且没有未解释非有限值；
- 当前短实验中 Muon 比 AdamW 更快降低 validation loss；MTP 未显示主 loss 收益。

禁止写：

- 训练出了可用语言模型；
- Muon 普遍优于 AdamW；
- MTP 无效；
- PPL 可与不同 Tokenizer 的公开模型直接比较；
- TinyStories 指标代表真实世界能力。

**最终思考题：Stage 3 最重要的产物是 checkpoint 还是最低 loss？**

答案：都不是单独一个文件或数字。最重要的产物是可复现证据链：固定输入、明确状态转移、正确恢复、受控实验、原始指标和不越界结论。
