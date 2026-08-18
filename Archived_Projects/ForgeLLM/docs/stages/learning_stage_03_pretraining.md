# Stage 3：预训练闭环完成计划与验收状态

## 1. 阶段定位

Stage 3 的目标不是训练一个“好用的大模型”，而是掌握并证明一个可解释、可恢复、可评估的小模型预训练闭环。模型质量、长训练和词表调优不作为本阶段成功条件。

当前状态：**自动化实现、固定实验与学习者验收均已完成；G3 已关闭。**

唯一学习入口：`docs/lessons/stage03_learning_order.md`。不要从本计划、实验报告或源码目录随机开始。

## 2. 冻结基线

| 项目 | 冻结值 |
|---|---|
| Tokenizer | Stage 1 byte-BPE，实际词表 320，不重新调优 |
| 语料 | TinyStories 官方 19.4 MB 文件，SHA-256 `94e431...8cb38b4` |
| 许可证 | CDLA-Sharing-1.0 |
| 文档切分 | 内容哈希确定性 90%/5%/5%，先切分、后 tokenize/packing |
| 正式模型 | 5,361,856 参数 Decoder，7 层、d=256、GQA、QK-Norm、SDPA |
| 正式精度 | CUDA BF16；CPU FP32 作为确定性 oracle |
| 正式预算 | 1,000,000 target tokens 或 3,600 秒，先到者停止 |
| 正式优化器 | AdamW；Muon 只在单变量短实验中比较 |
| 正式目标 | 单 token next-token prediction；MTP 只在单变量短实验中比较 |

TinyStories 上游的 `valid` 文件只被用作一个体积合适的原始教学子集，项目会重新做文档级 train/validation/test 切分。因此本项目结果不是 TinyStories 官方 validation benchmark 结果。

## 3. 工作包与门禁

### G3-A：数据和目标语义

- 文档独立编码，并在文档末尾插入 EOS；
- 相邻训练窗口只重叠一个 context token，使每个保留 target transition 恰好出现一次；
- 尾部不足一个窗口的 token 数被显式记录；
- token cache 同时绑定源文件、Tokenizer 和序列长度指纹；
- train/validation/test 在 tokenize 之前完成文档级隔离；
- next-token label shift、PPL 和 BPB 分母有自动测试。

通过条件：packing、byte 计数、cache 和 cursor 测试全部通过。

### G3-B：训练更新正确性

- AdamW 参数按 decay/no-decay 分组；
- Muon 只接管合格矩阵参数，embedding、norm 和向量参数由 AdamW 兜底；
- warmup-cosine 的第一个更新、warmup 终点和最低学习率可测试；
- gradient accumulation 与等效大 batch 的 FP32 更新对齐；
- BF16/FP16 autocast 边界明确，FP16 scaler 状态进入 checkpoint；
- loss、gradient 出现 NaN/Inf 时在 optimizer step 之前失败。

通过条件：单元测试、累积等价测试、100-step qualification 全部通过。

### G3-C：Checkpoint 与恢复

checkpoint 必须保存：

- 模型和可选 MTP head；
- optimizer、scheduler、GradScaler；
- step、tokens seen、累计墙钟时间；
- Python、Torch CPU、全部 CUDA RNG；
- 数据集指纹、epoch 和下一个 batch cursor；
- 实验配置和 Tokenizer 指纹。

保存方式为同目录临时文件、flush/fsync、`os.replace` 原子替换。加载时先验证 schema 和指纹，再恢复运行状态。

通过条件：CPU/FP32 连续训练与 `K 步→保存→新进程恢复→N-K 步` 的 loss、参数、optimizer、scheduler 和下一批数据完全一致；正式 CUDA 运行真实中断并恢复。

### G3-D：有界训练与现代方法实验

- 20-step smoke；
- 100-step、5.36M、BF16 qualification；
- AdamW/Muon 三种子、100-step 单变量实验；
- single-token/MTP 三种子、100-step 单变量实验；
- 1M target-token 正式运行，在第 125 步主动中断并恢复；
- 固定 prompt 生成、validation、PPL、BPB、吞吐和显存记录。

现代方法的成功条件是“实现正确、训练稳定、结果可解释”，不是“必须胜过基线”。

## 4. 已完成自动化证据

| 证据 | 结果 |
|---|---|
| 正式语料 | 21,990 文档；train 19,747 / validation 1,128 / test 1,115；0 拒绝 |
| CPU 精确恢复 | loss、模型、optimizer、scheduler、trainer state、next batch 全部 exact |
| 100-step qualification | loss 5.1521→2.7290；验证 loss 2.6289；无非有限值 |
| Muon 短实验 | 三种子验证 loss 均值 3.1926；只作当前短程观察 |
| AdamW 短实验 | 三种子验证 loss 均值 3.4075 |
| MTP 短实验 | 三种子验证 loss 均值 3.4137；未观察到主 loss 收益 |
| 1M token 正式运行 | 247 步、1,003,808 tokens；第 125 步中断恢复；步号连续 |
| 正式最终指标 | val loss 1.6245、PPL 5.0759、BPB 2.0543、峰值 235.9 MiB |

完整数值与失败记录见 `docs/experiments/2026-07-27_stage03_implementation.md`。

## 5. 学习者验收

学习者必须完成 `stage03_learning_order.md` 的 16 站，并能够：

1. 手画一个 batch 的 input/target shift；
2. 解释为什么先切分文档再 tokenize；
3. 解释一个 token cache 至少绑定哪三个指纹；
4. 手算一次梯度累积；
5. 解释 AdamW no-decay 参数组；
6. 区分 FP32、BF16、FP16 与 loss scaling；
7. 列出完整 checkpoint 状态；
8. 说明只恢复模型权重会产生哪些不连续；
9. 复现实验脚本并阅读 JSONL 原始指标；
10. 对 Muon/MTP 结果作不越界解释。

用户已于 2026-07-28 确认完成学习、代码追踪、最小改动练习和口述验收，完整 G3 已标记为“已完成”。这不代表已训练出生产级模型。

## 6. 明确不做

- 不重新搜索词表大小；
- 不扩大到完整 TinyStories train 或其他大语料；
- 不启用付费云训练；
- 不把 Muon 与 MTP 同时加入同一个比较；
- 不宣称短实验能证明模型或方法的普遍质量；
- 不用自动降 batch 掩盖 OOM；
- 不把固定 prompt 的偶然文本当作定量评测。
