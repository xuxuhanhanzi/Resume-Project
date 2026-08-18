# 实验记录：Stage 1 G1-B Tokenizer 方法实验室

## 1. 目标

在不下载大语料、不搜索词表规模、不训练语言模型的条件下，用项目原创小型夹具验证现代 Tokenizer 主要方法的算法机制与 Python 实现：

- pre-tokenization；
- BPE-dropout；
- Unigram EM、Viterbi 与 sampling；
- Picky BPE 的词表细化；
- SuperBPE 的两阶段 pre-tokenization curriculum；
- 特殊 Token policy、byte offsets；
- tokenizer-free 路线的简化 entropy patching。

## 2. 参考来源与实现边界

- SentencePiece 支持 BPE、Unigram 与 subword regularization：[官方仓库](https://github.com/google/sentencepiece)；
- Picky BPE 在 BPE 训练中执行词表细化：[论文](https://arxiv.org/abs/2409.04599)；
- SuperBPE 先学习 subword，再学习跨空格 superword：[COLM 2025 论文](https://arxiv.org/abs/2503.13423)；
- BLT 使用 byte 与动态 entropy patch：[论文](https://arxiv.org/abs/2412.09871)。

ForgeLLM 的 Unigram 直接实现核心 forward-backward EM、Viterbi 和 sampling，但使用简化候选裁剪；Picky BPE 使用可解释的 frequency-ratio removal，而不是论文完整 likelihood-aware 规则；SuperBPE 复现两阶段边界课程；entropy patcher 使用 byte bigram，而不是 BLT 的学习型 entropy model。

## 3. 环境

- 平台：Windows
- Python：项目 `.venv`
- GPU：不需要
- 外部数据：无
- 训练夹具：`tests/fixtures/tokenizer/train.jsonl`，8 篇
- 评测夹具：`tests/fixtures/tokenizer/method_lab_evaluation.jsonl`，7 篇；覆盖 English、Chinese、code、numbers、whitespace、Emoji 和 mixed
- 数据许可证标记：`project-test-fixture`
- 固定词表：300；未做词表规模搜索

## 4. 主变量与固定项

- 主变量：Tokenizer 算法/边界方法；
- 固定项：训练和评测文本、`vocab_size=300`、`min_pair_frequency=1`、UTF-8 strict、相同评测器；
- 基线：raw UTF-8 byte；
- 主要比较：global BPE、Unicode-boundary BPE、Picky 教学版、SuperBPE 教学版、Unigram；
- 随机实验：BPE-dropout=0.35；Unigram sampling temperature=10.0；均使用显式 seed。

## 5. 命令

```powershell
.\.venv\Scripts\python.exe scripts\tokenizer_method_lab.py `
  --train tests\fixtures\tokenizer\train.jsonl `
  --evaluation tests\fixtures\tokenizer\method_lab_evaluation.jsonl `
  --output artifacts\stage01_tokenizer_method_lab\report.json
```

## 6. 输出

- 本地 JSON：`artifacts/stage01_tokenizer_method_lab/report.json`；
- JSON 被 `.gitignore` 排除，可由上述命令重建；
- 代码、测试、本记录与讲义进入仓库。

## 7. 总体结果

七篇评测文档共有 404 个 UTF-8 bytes。所有路径 round-trip 均为 1.0。

| 方法 | Token 数 | bytes/token | 结论范围 |
|---|---:|---:|---|
| Raw byte | 404 | 1.0000 | 无压缩基线 |
| Classic BPE global | 359 | 1.1253 | 无边界基线 |
| Classic BPE Unicode boundary | 362 | 1.1160 | 边界减少了可用 merge |
| Picky BPE 教学版 | 362 | 1.1160 | 事件机制通过；不声称压缩更优 |
| SuperBPE curriculum | 359 | 1.1253 | 第二阶段恢复部分跨边界压缩 |
| Unigram EM | 358 | 1.1285 | 概率切分与 byte fallback 通过 |

这些小数不是正式质量排名。

## 8. 结构证据

| 方法 | Learned Token | 跨空格 Token | 低利用率 Learned Token | Merge | Remove | Phase-2 Merge |
|---|---:|---:|---:|---:|---:|---:|
| Classic global | 44 | 10 | 16 | 44 | 0 | 0 |
| Classic Unicode boundary | 44 | 0 | 17 | 44 | 0 | 0 |
| Picky 教学版 | 44 | 0 | 11 | 56 | 12 | 0 |
| SuperBPE | 44 | 10 | 19 | 44 | 0 | 16 |
| Unigram | 44 | 0 | 16 | 不适用 | 不适用 | 不适用 |

可以支持的观察：

1. Unicode-class pre-tokenization 把跨空格 Token 从 10 降为 0，证明边界实际约束 merge。
2. Picky 教学版执行 56 次 merge 与 12 次 remove，最终仍保留 44 个 learned token；低利用率计数从相同边界经典 BPE 的 17 降至 11。
3. SuperBPE 有 16 次 phase-2 merge，并重新出现 10 个跨空格 Token，证明两阶段课程实际开放了边界。

不能据此声称 Picky 或 SuperBPE 在真实语料或语言模型上更好。

## 9. 随机切分、控制 Token 与动态 Patch

| 检查 | 结果 |
|---|---:|
| 20 个 BPE-dropout seed 的不同切分 | 20 |
| BPE-dropout 全部 round-trip | 通过 |
| 20 个 Unigram sampling seed 的不同切分 | 6 |
| Unigram sampling 全部 round-trip | 通过 |
| 未获许可的 `<eos>` 字面量 | 拒绝 |
| 显式允许 `<eos>` 后映射 ID 2 | 通过 |
| Byte offset 覆盖完整输入 | 通过 |
| Entropy patch 数 | 12 |
| Entropy patches exact round-trip | 通过 |

## 10. 失败与修正

1. 初版 Unigram sampling 在 temperature=1.5 的微型夹具上 20 个 seed 只产生一种切分。判断为 EM 后概率分布过尖，而不是 round-trip 错误。修正为在 sampling 的 backward 动态规划中正确应用 temperature，并在教学演示使用 temperature=10.0。
2. Picky 教学版不能宣称精确复现论文。代码与报告统一加入 `educational`/“教学版”边界，并记录具体 heuristic。
3. Global BPE 本来就能跨空格，因此 SuperBPE 的对照必须是“受限 Phase 1 → 开放 Phase 2”，不能只比较最终是否存在空格 Token。

## 11. 结论

G1-B 的自动化教学实现目标已经满足：方法不再停留在库名或论文摘要，而是具备可运行的边界切分、随机 merge、概率动态规划、merge/remove 事件、两阶段 curriculum、特殊 Token policy 和动态 patch 代码。

尚未完成的是学习者本人对讲义、代码追踪练习和口述题的验收。当前结果不能替代真实预训练模型上的 Tokenizer 消融实验。

## 12. 最终质量门

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
```

- Ruff format：通过；
- Ruff lint：通过；
- mypy strict：54 个 source file 无错误；
- pytest：100/100 通过。
