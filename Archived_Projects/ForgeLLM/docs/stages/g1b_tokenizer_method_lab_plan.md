# G1-B：Tokenizer 方法实验室计划

> 状态：代码、实验与学习者验收完成  
> 版本：2026-07-27  
> 前置门禁：G1-A 经典 byte-level BPE 已通过  
> 时间边界：实现与实验约 8–12 小时；不进行大语料训练

## 1. 范围修正

G1-B 的目标从“训练一个正式、接近生产使用的 Tokenizer”修正为：

> 理解现代 Tokenizer 的主要算法选择，并通过小规模、可解释的 Python 实现观察每一种方法如何改变训练和编码过程。

因此，本阶段明确取消：

- 64 MiB Wikipedia 获取与许可证工程；
- 4096/8192 等词表规模搜索；
- 生产级压缩率调优；
- Rust/C++ 性能重写；
- 用大模型下游效果证明某种方法更优。

这些工作在真正冻结预训练系统时再决定是否需要。当前全部实验使用项目原创小型夹具，固定 `vocab_size=300`，词表大小不是实验变量。

## 2. 可证伪假设

1. 不同 pre-tokenization 边界会改变 BPE 可以学习的 merge；Unicode-class 边界应阻止跨空格 Token。
2. BPE-dropout 和 Unigram sampling 能产生多个合法切分，同时保持 exact round-trip。
3. 教学版 Picky BPE 能以可重放的 merge/remove 事件替换一部分低利用率 learned token。
4. 教学版 SuperBPE 能先学习边界内子词，再在第二阶段学习跨空格 Token。
5. 固定词表不是唯一道路；简化 entropy patcher 能按局部 byte surprisal 动态建立 patch，并无损还原文本。
6. 所有方法都必须明确区分“算法行为已复现”和“论文所报告的大规模模型收益已复现”。

## 3. 实验矩阵

| 编号 | 方法 | 主变量 | 目的 |
|---|---|---|---|
| M0 | Raw UTF-8 bytes | 不合并 | 最低基线 |
| M1 | Classic BPE，global | 无 pre-token 边界 | 观察任意相邻 byte merge |
| M2 | Classic BPE，Unicode class | 字母/数字/空白/其他边界 | 观察边界约束 |
| M3 | BPE-dropout | 编码时随机跳过 merge | 学习 subword regularization |
| M4 | Unigram EM | 概率词表与动态规划 | 对比非贪心词表模型 |
| M5 | Picky BPE 教学版 | merge 后允许 remove | 学习词表细化与事件重放 |
| M6 | SuperBPE 教学版 | 两阶段边界 curriculum | 学习跨空格 superword |
| M7 | 特殊 Token policy/offset | 显式许可与 byte span | 补齐生产接口关键概念 |
| M8 | Bigram entropy patching | surprisal 动态分块 | 理解 tokenizer-free 路线 |

固定项：同一训练/评测夹具、UTF-8 strict、300 词表预算、相同评测函数；不根据结果调整词表规模。

## 4. 实现顺序

### Step 1：Pre-tokenization

- 实现 `none`、`whitespace`、`unicode_class` 三种无损切分；
- 验证所有 piece 拼接后与原始 UTF-8 bytes 完全一致；
- 用经典 BPE 比较全局 merge 与边界内 merge。

### Step 2：随机切分与 Unigram

- 在 BPE 编码时按固定 seed 随机跳过某次 merge；
- 从候选 byte piece 构建 Unigram 词表；
- 实现 forward-backward 期望计数、EM 更新、Viterbi 最优切分和 posterior sampling；
- 使用 256 个 byte piece 作为 fallback，保证未见文本可逆。

### Step 3：Picky BPE

- 先达到固定 active vocabulary；
- 每次 refinement 新增一个高频 merge；
- 根据“当前使用次数 / 创建时频率”选择低利用率旧 Token；
- 写入 remove 事件，并在训练序列中展开该 Token；
- 编码时严格按时间顺序重放 merge/remove。

本实现复现 add/remove 的核心机制，但没有复刻论文完整的 likelihood-aware removal，文件和报告必须使用“教学版”限定。

### Step 4：SuperBPE

- Phase 1：只在 Unicode-class piece 内学习 merge；
- Phase 2：移除 piece 边界，在完整文档序列上继续学习；
- 分别记录 phase 1/2 merge；
- 验证第二阶段能够生成同时包含空白与非空白 byte 的 Token。

### Step 5：生产概念与前沿演示

- 普通文本出现 `<eos>` 等控制字面量时默认拒绝；只有显式允许才映射到特殊 ID；
- 为普通 Token 返回原始 UTF-8 byte offsets；
- 训练 byte bigram 计数模型，用 surprisal threshold 和最大 patch 长度动态分块；
- entropy patching 只演示 BLT 的边界思想，不声称实现 BLT 模型。

### Step 6：统一实验与报告

- 固定运行 `scripts/tokenizer_method_lab.py`；
- 在全部评测文档和 english/chinese/emoji/edge 子集报告指标；
- 记录 learned token 使用率、跨空格 Token、Picky remove 数、SuperBPE phase-2 merge 数；
- 对随机算法验证多样性、固定 seed 可复现和 round-trip；
- 保存 JSON 报告与中文实验记录。

## 5. 验收门禁

### 自动化证据

- [x] 三种 pre-tokenization 均保持 UTF-8 bytes 不变；
- [x] 手写 Unigram 包含 EM、Viterbi 和 sampling，未见文本 round-trip；
- [x] BPE-dropout 产生多个合法切分，固定 seed 可复现；
- [x] Picky BPE 教学版产生 merge/remove 事件并可正确重放；
- [x] SuperBPE 教学版包含两个训练阶段并学习跨空格 Token；
- [x] 特殊 Token policy、byte offsets 和 entropy patching 有自动测试；
- [x] 统一报告包含六条 tokenizer 路径与分 subset 指标；
- [x] 项目完整 lint、mypy、pytest 与 smoke 通过。

### 学习者验收

- [ ] 能手算 pre-tokenization 如何改变 pair count；
- [ ] 能解释 BPE greedy merge 与 Unigram 概率切分的差别；
- [ ] 能沿代码说明 Picky remove 后为什么仍能编码；
- [ ] 能说明 SuperBPE 为什么需要两阶段而不是从一开始无边界；
- [ ] 能解释随机切分为什么是训练正则化，而不是损坏 Tokenizer；
- [ ] 能区分固定 Token、动态 patch 与语言模型主干计算。

自动化证据完成代表“G1-B 教学实现完成”；只有学习者验收也完成后，才声明完整 Stage 1 学习完成。

## 6. 产物

- `src/forgellm/tokenization/pretokenization.py`
- `src/forgellm/tokenization/advanced_bpe.py`
- `src/forgellm/tokenization/unigram.py`
- `src/forgellm/tokenization/special_tokens.py`
- `src/forgellm/tokenization/entropy_patching.py`
- `src/forgellm/tokenization/method_lab.py`
- `scripts/tokenizer_method_lab.py`
- `tests/unit/test_*token*`、`test_advanced_bpe.py`、`test_entropy_patching.py`
- `tests/integration/test_tokenizer_method_lab.py`
- `docs/lessons/stage01_tokenizer_method_lab.md`
- `docs/experiments/2026-07-27_stage01_g1b_method_lab.md`
- 本地 `artifacts/stage01_tokenizer_method_lab/report.json`（由 `.gitignore` 排除）

## 7. 结论边界

本阶段可以证明算法机制、代码契约、可逆性和微型夹具上的结构差异。它不能证明：

- Picky BPE 或 SuperBPE 在 ForgeLLM 的未来模型上提高准确率；
- 当前 300 词表适合正式预训练；
- 微型夹具的压缩排名可以推广到真实多语言语料；
- bigram entropy patcher 等价于 BLT；
- 教学 Python 实现具备生产吞吐。
