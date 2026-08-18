# ForgeLLM 模型优先 21 周学习与实现计划

> 状态：当前唯一有效主路线  
> 生效日期：2026-07-24（Asia/Singapore）  
> 最近修订：2026-08-09，完成全阶段清晰度审查，统一入口、执行步骤、产物、量化门槛、失败分支与当前状态
> 目标岗位：P1 LLM 算法/训练；P2 AI 全栈  
> 资源基线：每周 60 小时以上；本地约 8 GB NVIDIA GPU；外部资源总预算不超过 20 USD/月  
> 2026-07-25 已删除会干扰当前学习顺序的过时计划；历史路线变更证据保留在 Stage 0 讲义与实验记录中

> **当前完成口径：** Stage 0–5 的自动化门与学习者门已关闭；Stage 6 的自动化门 G6-A～G6-E 已关闭，G6-L 学习者最终答辩尚未关闭。当前 Q0/Q1/Q2 模型行为均为“未接受”，不能因训练和评测链路完成而写成“模型能力达标”。

## 1. 路线修订原因

Day 1–3 的学习表明，旧方案将 Python 包、CLI、配置、日志、CI、数据治理和完整证据链作为连续 14 天课程，会显著推迟 Tokenizer、模型构造、预训练和后训练。工程实现已经存在，继续逐日深入学习的边际收益低于尽早进入模型主线。

新路线采用以下原则：

```text
工程与数据支撑：够用、可查、遇到真实问题再补
模型与训练核心：系统学习、亲手实现、严格测试、受控实验
评测：在每个阶段开始前定义，不在训练结束后临时补
```

这不是取消工程质量，而是把工程从“学习目的”恢复为“模型实验的支撑工具”。

## 2. 优先级

| 等级 | 内容 | 学习与验收方式 |
|---|---|---|
| 核心 P0 | Tokenizer、PyTorch、Decoder-only Transformer、预训练、SFT、模型评测 | 完整讲义、亲手实现、正确性测试、实验与阶段门 |
| 核心 P0 | Checkpoint/Resume、混合精度、梯度累积、数据位置恢复 | 深入理解并完成等价/恢复实验 |
| 专项 P0 | C++/CUDA 自定义算子 | 完成一个 Python→C++ CPU→CUDA→Autograd→Benchmark 闭环 |
| 扩展 P1 | LoRA、QLoRA、DPO、Reward Model | 原理、最小实现、固定评测；正式运行受预算约束 |
| 扩展 P1 | Policy Gradient、PPO、GRPO | 理论和小型实验必须完成；大模型 RL 按资源触发 |
| 支撑 P2 | 包、CLI、配置、普通数据清洗、CI、日志、实验记录 | 执行卡和按需补学，不设整天课程 |
| 后移 P3 | 服务、vLLM、Docker、Kubernetes、Agent | 模型主线完成后再规划 |

## 3. 总体时间线

| 阶段 | 周期 | 核心交付 | 门禁 |
|---|---:|---|---|
| Stage 0：路线切换 | 0.5 周以内 | 新主计划、执行卡、质量门、详细讲义 | S0 |
| Stage 1：数据最小闭环与 Tokenizer | 2 周 | 手写 BPE、工程 Tokenizer、固定评测 | G1 |
| Stage 2：Transformer、现代架构与模型系统 | 7 周 | 5M–20M Decoder、前沿方法 reference、系统实验、自定义算子 | G2-Core / G2-Arch / G2-Systems |
| Stage 3：预训练闭环 | 3 周 | 训练循环、Checkpoint/Resume、正式小模型短跑 | G3 |
| Stage 4：监督式后训练基础 | 2 周 | Chat Template、SFT、LoRA/QLoRA、固定前后评测 | G4 |
| Stage 5：偏好优化与在线 RL | 3 周 | Reward Model、DPO、PG/PPO/GRPO 与前沿 RL 方法 | G5 |
| Stage 6：综合评测与最终验收 | 2 周 | 多维评测、Model Card、成本与结论边界 | G6 |
| Buffer：补弱与作品集 | 2 周 | 修复未过门禁、整理可展示证据 | Final |

核心学习约 19 周，缓冲 2 周。等待下载、云端排队和环境故障不得伪装成核心学习完成。

该时间线是项目启动时的基线工期，不是截至 2026-08-09 的剩余工期。实际状态以本节下方的“状态与证据”以及第 14 节为准。

### 3.1 统一执行口径

每个阶段必须按下面的同一顺序执行，任何阶段不得只凭“代码存在”或“测试数量增加”宣告完成：

1. **入口确认：** 前置门关闭；冻结数据、模型、Tokenizer、代码和配置版本；创建独立输出目录。
2. **协议预登记：** 写清可证伪假设、官方/工程基线、唯一主变量、固定项、预算、指标、成功门槛和停止条件。
3. **Smoke：** 用最小样本验证 Schema、shape、梯度、保存加载和失败保护；失败时不得进入 Quick Run。
4. **Quick Run：** 在有界预算内验证完整调用链和 Artifact；失败时不得进入正式运行。
5. **正式运行：** 只执行预登记方案，不依据中途结果静默改模型、数据、阈值或统计口径。
6. **自动化门：** 测试、原始输出、配置、环境、日志、指标和报告均可追溯。
7. **学习者门：** 完成唯一学习入口中的代码追踪、最小改动练习、手算/口述验收；该门只能由学习者确认。
8. **结论冻结：** 分开记录“实现完成”“实验完成”“能力达标”；负结果允许关闭工程门，但不能改写成能力提升。

门禁状态只使用四种值：`未开始`、`进行中`、`通过`、`阻塞`。`自动化通过 / 学习者待验收` 必须拆开写，不能合并为“基本完成”。

### 3.2 各阶段执行契约与状态

| 阶段 | 启动条件 | 本阶段唯一目标 | 必须产物 | 可判定退出条件 | 当前状态 |
|---|---|---|---|---|---|
| Stage 0 | 仓库可运行；发现旧路线与模型主线冲突 | 建立唯一当前路线，不新增模型功能 | 主计划、支撑卡、质量门记录、讲义、实验记录 | 主入口唯一；Ruff/mypy/pytest 通过；不误标后续阶段 | **通过** |
| Stage 1 | S0 通过；原创固定夹具可追溯 | 证明 Tokenizer 算法、序列化和评测闭环正确 | BPE/Unigram 等实现、候选 Artifact、统一报告、讲义 | 冻结夹具 round-trip 100%；byte BPE 普通文本 unknown rate=0；同输入同哈希；G1-A/G1-B/学习者门全过 | **通过** |
| Stage 2 | G1 通过；Tokenizer 接口冻结 | 证明 Decoder 主干、现代方法 reference 和系统边界正确 | 5.36M Decoder、架构/系统实验、自定义算子、报告与讲义 | 因果性、梯度、Tiny Overfit、cache 对齐、保存加载通过；manual/SDPA 在测试容差内；G2-Core/Arch/Systems/学习者门全过 | **通过** |
| Stage 3 | G2 通过；数据、Tokenizer、模型配置冻结 | 证明小模型预训练可恢复、可计量、可复现 | 数据 Manifest、训练日志、Checkpoint、Resume 对照、短跑报告 | CPU exact-resume；CUDA 恢复链可运行；达到 1M target tokens 或 60 分钟先到即停；无未解释 NaN/OOM/数据跳重；学习者门通过 | **通过** |
| Stage 4 | G3 通过；开源 Base 和 SFT 数据冻结 | 证明 SFT/LoRA/QLoRA 的目标、梯度和固定前后评测正确 | Adapter、数据/模型卡、显存实验、正式短跑、退化报告、讲义 | mask/loss 手算对齐；Base 冻结；FP32 merge/unmerge 最大绝对误差 ≤1e-6；真实 LoRA/QLoRA 反传；同协议报告所有改善与退化；学习者门通过 | **通过；行为未达标** |
| Stage 5 | G4 工程门关闭；固定 SFT Adapter 与评测可用 | 证明偏好/RL 算法契约和最小 on-policy 链正确 | preference Manifest、RM/DPO/PG/PPO/GRPO 实现、DPO/GRPO Artifact、讲义 | DPO loss 与独立公式/框架在 1e-6 口径内；toy policy 真实改善；首步 on-policy 不变量、Base/Reference 无梯度；reward hacking/回退已报告；学习者门通过 | **通过；能力收益未证明** |
| Stage 6 | G5 关闭；Q0/Q1/Q2/Q3 身份和评测集冻结 | 对训练链和模型行为作最终、可审计且不过度声明的验收 | 原始逐题输出、污染报告、统计/盲评/系统矩阵、Model/Evaluation Card、claim-evidence matrix | G6-A～E 自动门通过；学习者完成 30 对盲评、统计手算、代码追踪和答辩；行为门单独给出 accepted/rejected | **自动化通过；G6-L 待完成；行为 rejected** |
| Buffer / Final | G6-L 关闭 | 只修复未过门禁并形成作品集，不新增研究主线 | Release、复现命令、证据索引、简历 claim 清单、未完成项清单 | 陌生环境按 README 可运行最小闭环；每条公开 claim 能定位到配置/日志/报告；所有未证明能力明确排除 | **未开始** |

### 3.3 责任边界

- **自动化门负责人：** 代码、测试、配置、运行记录和报告的实现者；必须提供可复查证据。
- **学习者门负责人：** 学习者本人；口述、手算、盲评和最小改动练习不能由自动化测试代替。
- **最终声明负责人：** 对外发布者；只能使用 `docs/cards/` 与最终 claim-evidence matrix 支持的表述。
- **状态更新规则：** 每次阶段状态变化，同步更新本文件第 14 节、对应 `docs/stages/` 文件、`docs/month_01_progress.md` 和 `HANDOFF_SUMMARY.md`；各处不一致时按最保守状态处理并先修正文档。

## 4. Stage 0：切换到模型优先路线

### 目标

- Day 1–3 记录为用户已完成学习与测试；
- 取消 Day 4–14 逐日学习前置门槛；
- 保留旧文档和源码作为支撑知识库；
- 运行完整质量门；
- 下一步直接进入 Tokenizer。

### 执行契约

- **输入/前置：** 仓库已有最小工程入口，但旧课程与模型主线存在优先级冲突。
- **执行顺序：** 盘点入口 → 指定唯一主计划 → 把工程知识降为按需支撑卡 → 运行完整质量门 → 更新 README、进度和交接文件。
- **必须产物：** `docs/model_first_21_week_learning_plan.md`、`docs/reference/minimum_engineering_support_card.md`、Stage 0 讲义与实验记录。
- **通过含义：** 只证明路线切换和工程支撑可用，不证明 Tokenizer、模型或训练能力。
- **失败分支：** 若质量门或入口定位失败，只补齐具体缺口；若存在两个“当前主计划”，S0 直接失败。

### S0 门禁

- 新主计划是 README、进度台账和交接文件中的唯一当前路线；
- 支撑知识有最小执行卡；
- Ruff format/lint、mypy、pytest 全部通过；
- Stage 0 讲义与执行记录完成；
- 不把 Tokenizer、模型或训练标记为已完成。

## 5. Stage 1：数据最低闭环与 Tokenizer（2 周）

### 执行契约

- **输入/前置：** S0 通过；冻结项目原创训练/验证/测试夹具、规范化规则、特殊 Token ID 和随机种子。
- **执行顺序：** 手算 BPE → 最小 byte-level BPE → 序列化与 round-trip → 成熟库对照 → 现代方法单变量实验 → 统一报告 → 学习者验收。
- **必须产物：** 可加载 Tokenizer Artifact、训练 Manifest、固定 test 报告、方法实验报告、失败反例和唯一学习入口。
- **量化门槛：** 冻结夹具 round-trip 100%；byte-level BPE 普通文本 `unknown rate = 0`；相同输入/配置的词表与 merge 哈希一致；特殊 Token ID 与许可策略全部通过；所有比较同时报告分母和子集。
- **失败分支：** 保留最短反例并停止候选冻结；round-trip、确定性或特殊 Token 契约任一失败均不得进入 Stage 2。现代方法没有质量收益不算失败，但实现不变量、报告或学习者门缺失算失败。

### Week 1：Tokenizer 原理与手写 BPE

- Unicode、UTF-8、byte、character、token 的关系；
- 词表、特殊 Token、OOV 与基础符号单位；
- 手算 BPE pair frequency 与 merge；
- Python 实现确定性 pair 统计和 tie-break；
- 训练、保存、加载、encode、decode；
- 空串、中文、英文、Emoji、组合字符、空白和损坏模型测试。

### Week 2：现代 Tokenizer 方法实验

- none/whitespace/Unicode-class pre-tokenization；
- BPE-dropout 与 Unigram EM/Viterbi/sampling；
- Picky BPE merge/remove 与 SuperBPE 两阶段 curriculum；
- PAD/BOS/EOS/UNK 契约和固定 Token ID；
- 特殊 Token 显式许可、byte offset 与简化 entropy patching；
- 与成熟 Tokenizer 库进行行为对照；
- 固定小型原创夹具和词表大小，不执行词表规模搜索；
- 生成统一方法报告、完整讲义和代码追踪验收。

### 数据支撑的最低范围

- 使用项目原创并已标记来源的教学夹具；
- UTF-8/Unicode 规范化；
- 精确去重与最小长度过滤；
- 记录样本量、字节数和输入哈希。

正式大语料、许可证审计、词表规模调优、近似去重、复杂 PII 系统、多格式 Reader、完整数据平台和长时间规则校准不阻塞 G1；这些在 Stage 3 的真实预训练数据冻结前按需处理。

### G1 指标与门禁

- round-trip 成功率；
- bytes/token 与 chars/token；
- fertility，明确分母；
- unknown rate；
- 特殊 Token 正确率；
- encode/decode 吞吐；
- 语言/领域子集；
- 相同输入和配置生成相同词表/merge 哈希；
- 能手算并解释 BPE merge；
- 能解释 pre-tokenization、Unigram、Picky、SuperBPE 和动态 patch 的算法差异。

## 6. Stage 2：Transformer、现代架构与模型系统（7 周）

Stage 2 不以长时间训练或复现超大模型指标为目标。学习者应掌握稳定 Transformer 主干、当前前沿方法解决的问题与代码结构，以及从 PyTorch reference 到原生算子的系统边界。唯一学习入口是 `docs/lessons/stage02_learning_order.md`。

### 执行契约

- **输入/前置：** G1 通过；Tokenizer 接口和 5.36M 调试模型配置冻结；所有架构方法只使用小型 reference。
- **执行顺序：** PyTorch 基础 → 原子模块 → 完整 Decoder/生成 → 现代架构 → 模型系统 → C++/CUDA 算子 → 三条子门 → 学习者验收。
- **必须产物：** Decoder 和生成代码、G2-Core/Arch/Systems 测试与实验、`silu_mul` CPU/CUDA Artifact、固定 benchmark、讲义和边界说明。
- **量化门槛：** manual/SDPA 前向 `rtol=1e-5, atol=1e-6`，梯度 `rtol=1e-4, atol=1e-6`；KV Cache 按现有回归测试容差对齐；未来 token 不改变过去 logits；Tiny Set Overfit、保存/加载和全状态回归通过。
- **失败分支：** Core 任一正确性门失败时禁止 Stage 3 长训练；Arch 方法只允许标记为未验证，不得冒充官方复现；Systems 因平台缺失失败时记录为能力边界，不得用图捕获替代 Inductor 性能声明。

### Week 1：模型编程所需 PyTorch

- Tensor shape、stride、view/reshape、transpose 与 contiguous；
- broadcasting、dtype、device；
- Autograd 计算图、leaf tensor、梯度累积；
- `nn.Module`、Parameter、Buffer、state dict；
- 初始化、前向、反向、数值稳定性与 `gradcheck` 思想。

### Week 2：稳定原子模块

- Embedding 与 Linear；
- RMSNorm；
- RoPE；
- SwiGLU；
- Causal Mask；
- Multi-Head Attention 与 GQA。

每个模块必须有 shape、数值对照、梯度、边界与 dtype/device 测试。

### Week 3：完整 Decoder、生成与正确性

- Transformer Block 与残差；
- 多层堆叠；
- LM Head 与权重共享；
- shifted cross-entropy 与 teacher forcing；
- 参数量、FLOPs、激活显存估算。
- greedy、temperature、top-k/top-p；
- KV Cache；
- cache/no-cache 对齐；
- future token 不影响过去 logits；
- Tiny Set Overfit；
- 保存/加载和固定输入数值回归。

### Week 4：前沿 Attention、长上下文与混合架构

- Linear / Dynamic-NTK RoPE scaling：为什么固定训练长度之外的位置外推会失真；
- MLA：为什么压缩 KV Cache，以及 latent cache 与普通 MHA/GQA 的结构差异；
- MoBA-style block sparse attention：为什么从 token 级全连接转为可路由的 block；
- CSA/HCA-lite：压缩全局记忆与局部精确窗口如何组合；
- Gated DeltaNet 与 3:1 hybrid mixer：递归线性状态为何需要门控，以及为什么与 full attention 混合；
- 每个方法都完成“旧问题→核心原理→公式/shape→Python reference→测试→局限”的闭环。

### Week 5：MoE、残差、训练目标与优化器新方法

- Sparse MoE、top-k 路由、共享专家与负载辅助项；
- 确定性 Hash 路由与 Sinkhorn 近似平衡；
- mHC-lite 与 Attention Residuals：残差流从固定相加到受约束混合/多流读写；
- MTP：多个未来 offset 的辅助目标与推测解码关系；
- Muon：Newton–Schulz 矩阵更新的教学实现；
- 不用微型随机输入的结果替代官方模型的大规模训练结论。

### Week 6：模型系统 reference

- 手写 attention、PyTorch SDPA 与后端选择；
- FlashAttention 的 tiling、online max/normalizer 更新和精确分块 softmax reference；
- `torch.compile(fullgraph=True)`、graph break 与 eager/Inductor 边界；
- INT8 weight-only 量化的 scale、误差、存储账本与高性能 kernel 边界；
- 两进程 CPU/Gloo DDP 梯度同步，区分教学语义与多 GPU/NCCL 性能。

### Week 7：C++/CUDA 自定义算子

- C++ 最低知识：类型、引用、指针、`const`、RAII、编译与链接；
- ATen Tensor、operator schema、CPU/CUDA dispatch 与校验；
- `silu_mul` 的 Python→C++ CPU→CUDA 实现；
- Autograd、FakeTensor、`torch.library.opcheck()` 与 `gradcheck()`；
- grid/block/thread、内存层级、coalescing、dtype 与数值误差；
- 固定 shape/dtype/device、预热、同步和重复次数的公平 benchmark；
- 没有加速时如实保留结果，不为展示目的声称更快。

### 前沿模型滚动规则

- 已发布模型只按官方材料冻结；Kimi K3 官方报告已进入 Stage 5/6 技术地图，DeepSeek V4 因尚无可核验官方技术报告继续登记为“待核验”；
- 已发布模型优先使用官方技术报告、论文和仓库，逐项回答“为什么采用、解决什么旧问题、原理、代码映射、证据边界”；
- Stage 启动和验收各执行一次 `docs/frontier_model_technology_registry.md` 的 F0 新鲜度检查；
- 新方法先进入小型 reference 与单变量测试，不因名称更新而下载超大权重或启动长训练。

### G2 门禁

- **G2-Core：** 模型可前向、反向和生成；FP32 reference 对齐；因果性、梯度、Tiny Overfit、KV Cache、保存加载通过；
- **G2-Arch：** 每项方法都有来源、旧问题、原理、reference、测试和局限；MLA 缓存、稀疏/混合 attention 因果性、MoE assignment、MTP offset 等不变量可检查；
- **G2-Systems：** online-softmax 对齐 SDPA；compile 单图、INT8 误差/存储、真实双进程 DDP 有证据；`silu_mul` CPU/CUDA forward、gradcheck、opcheck 与 benchmark 通过；
- 学习者完成学习顺序中的代码追踪、最小改动实验与口述验收；
- 不以“代码能运行”替代数学正确性，不以教学 reference 冒充官方大模型复现。

## 7. Stage 3：预训练闭环（3 周）

Stage 3 只训练“掌握预训练系统”所需的最小预算，不追求生产级模型质量。唯一学习入口是 `docs/lessons/stage03_learning_order.md`；Tokenizer 固定为 Stage 1 产物，现代技术使用独立单变量短实验。

### 执行契约

- **输入/前置：** G2 通过；冻结数据 revision/许可证/哈希/split、Tokenizer 哈希、5.36M 模型配置和训练配置。
- **执行顺序：** 数据与目标语义 → 单步更新 → 混合精度/累积 → Checkpoint/Resume → 100-step 资格测试 → 有界正式短跑 → 单变量方法实验 → 报告与学习者验收。
- **必须产物：** 数据 Manifest、token cache 指纹、逐步结构化日志、完整 Checkpoint、连续/恢复对照、正式短跑报告和生成样例。
- **量化门槛：** CPU/FP32 连续与恢复的 loss、模型、optimizer、scheduler、trainer state 和 next batch 全部 exact；正式运行到 1,000,000 target tokens 或 3,600 秒先到者停止；tokens seen 单调且无已知重复/跳过；validation 不更新参数。
- **失败分支：** NaN/Inf、状态指纹不匹配或数据游标不连续时立即停止并保留 Checkpoint；先缩小规模复现，未恢复等价前不得进入 Stage 4。

### Week 1：训练循环

- 文档级 split、token stream、target-exact packing 与指纹 token cache；
- AdamW 和 weight decay 参数分组；
- warmup 与 cosine decay；
- gradient clipping；
- validation；
- train/validation loss、perplexity、bits-per-byte。

### Week 2：训练系统正确性

- gradient accumulation；
- FP32/BF16/FP16 与 loss scaling；
- NaN/Inf 检测；
- checkpoint schema 与原子保存；
- model/optimizer/scheduler/RNG/scaler/data cursor 恢复；
- 连续训练与中断恢复等价实验。

### Week 3：正式小模型短跑

- 5.36M 固定 Decoder；
- 100-step 资格测试；
- 1M target tokens 或 60 分钟的停止预算；
- 第 125 步真实中断并从新进程恢复；
- AdamW/Muon 与 single-token/MTP 两组独立三种子短实验；
- 固定 prompts 生成；
- 错误分析、吞吐、显存和成本。

### G3 门禁

- loss 下降可解释，无未解释 NaN/OOM；
- validation 不更新参数；
- Resume 曲线连续；
- 下一批数据无已知重复/跳过；
- 报告 tokens/s、step time、显存、tokens seen 和成本；
- 现代方法实验各自只有一个主变量，不要求新方法获胜；
- 结论严格限定为小规模预训练闭环。

## 8. Stage 4：监督式后训练基础（2 周）

原 Stage 4 的 C++/CUDA 自定义算子已经作为 Stage 2 第 7 周和 `G2-Systems` 完成，不在预训练后重复。2026-07-28 起，Stage 4 用于独立承接后训练的第一层：SFT、LoRA、QLoRA 和固定前后评测。详细、可直接执行的计划见 `docs/stages/learning_stage_04_supervised_posttraining.md`。

### 执行契约

- **输入/前置：** G3 通过；冻结 Qwen3-0.6B-Base revision、SFT 数据 revision/split、Chat Template、Tokenizer 和前后评测。
- **执行顺序：** Schema/Template → assistant-only mask 手算 → tiny SFT → 手写 LoRA/PEFT 对照 → LoRA/QLoRA 显存实验 → 唯一正式短跑 → 同协议评测 → 退化审计 → 学习者验收。
- **必须产物：** correctness 与正式数据 Manifest、Adapter-only Artifact、Data/Model Card、显存/吞吐对照、逐项评测和负结果记录。
- **量化门槛：** mask/token/loss 100% 对齐；Base 参数 byte-exact 不变；FP32 no-op 与 merge/unmerge 最大绝对误差 ≤`1e-6`；正式短跑以 100,000 assistant target tokens、500 steps 或 45 分钟先到者停止。
- **失败分支：** OOM 只允许调整 micro batch/accumulation 并重新登记；不得静默换模型。目标行为不提升时仍可关闭工程门，但状态必须写为“训练链路通过、行为未达标”，并禁止据此声称 SFT 能力提升。

### Week 1：数据、Chat Template、SFT 与手写 LoRA

- Base、Pretrained、Instruct 与 Chat Model 的区别；
- instruction、prompt-completion 与 conversational Schema；
- system/user/assistant 角色、BOS/EOS/PAD 与 Chat Template；
- assistant-only Label Mask、shifted CE、padding、truncation 与 packing；
- 自研 5.36M 模型 tiny overfit 和 full-sequence/assistant-only 单变量实验；
- 手写 `LoRALinear`、参数冻结、低秩更新、初始化与参数量；
- Adapter-only 保存/加载、merge/unmerge 和 PEFT 数值对照。

### Week 2：QLoRA、开源 Base 与冻结评测

- 4-bit、NF4、block-wise/double quantization 与 compute dtype；
- 冻结量化 Base 到 LoRA Adapter 的真实梯度路径；
- Qwen3-0.6B-Base BF16 LoRA 与 QLoRA 等步数显存实验；
- 100k assistant target tokens、500 steps 或 45 分钟先到即停的唯一正式短跑；
- held-out assistant loss、constraint exact match、格式遵循、长度/重复、保留能力、吞吐与显存；
- Qwen3 等公开多阶段后训练路线的原理映射；RL 部分后移到 Stage 5。

### G4 门禁

- 数据 revision、许可证、split、模板、Tokenizer 和模型 revision 均可追溯；
- SFT assistant-only mask 与手算 token/loss 100% 对齐，tiny overfit 通过；
- LoRA Base 参数冻结、参数量、梯度、merge/unmerge 与 Artifact 正确；
- 本机真实 QLoRA forward/backward 通过，并与 BF16 LoRA 使用同协议报告显存；
- 正式短跑前后使用同一评测，不要求结果必须为正，但必须报告退化和边界；
- 新手唯一入口为 `docs/lessons/stage04_learning_order.md`；完整讲义中的思考题与答案不得分离；
- 自动化门和学习者验收分开记录。

## 9. Stage 5：偏好优化与在线 RL（3 周）

Stage 5 只在 G4 的 SFT Base、Adapter Artifact 和固定评测成立后启动。

### 执行契约

- **输入/前置：** G4 工程门关闭；SFT Base、Adapter、preference split、verifier、reward 版本和固定保留集均冻结。
- **执行顺序：** Preference/RM → DPO 手算与框架对照 → toy PG/PPO → GRPO 最小 on-policy rollout → 前沿方法 frozen-tensor 单变量实验 → DPO/GRPO 有界运行 → 回退/reward hacking 审计 → 学习者验收。
- **必须产物：** Preference Manifest、rollout record、RM/DPO/PG/PPO/GRPO 实现、DPO 与 GRPO Artifact、固定前后评测、失败注入记录和讲义。
- **量化门槛：** DPO tiny 对照 `atol=1e-6, rtol=1e-6`；toy 环境至少一次真实 policy 改善；首个 on-policy step 的 KL/ratio/clip 关系可解释；Base/Reference 无梯度；所有质量、长度、重复、KL 和 reward hacking 指标按预登记口径报告。
- **失败分支：** 非有限值、身份/hash 不匹配、reward 常数、Base/Reference 出现梯度或 verifier 可被简单欺骗时立即停止。Pair 指标改善但 strict behavior 不改善时，只能声明“偏好目标被优化”，不能声明“模型对齐成功”。

### Week 1：偏好学习与 DPO

- preference pair、Bradley–Terry 与 Reward Model 基础；
- chosen/rejected、reference model、log-prob、DPO beta 与隐式 KL；
- 长度偏差、数据偏差、pairwise accuracy 和 reward margin；
- SFT→DPO 的同一评测集受控对照。

### Week 2：RL 基础、PPO 与 GRPO

- MDP、policy、reward、return、Policy Gradient、baseline 与 advantage；
- importance sampling、KL regularization、clip 与 credit assignment；
- PPO 与 GRPO 的 loss/gradient reference；
- 确定性数学/代码 Verifier 上的小 rollout 实验。

### Week 3：当前前沿方法与边界

- DAPO 的 Clip、动态采样与 token-level policy gradient；
- 领域专家培养与 on-policy distillation；
- 多教师 OPD、thinking preservation 与异步 RL 系统只在准入门满足后实现缩小实验；
- reward hacking、采样版本、rollout 数据契约与固定保留集。

完整大模型 PPO/GRPO 受 8GB 显存和预算约束。未正式运行时，只能声明实现/验证算法，不能声明完成大模型 RL 对齐。

### G5 门禁

- RM/DPO 的 log-prob、mask、reference 与 loss 有手算/框架对照；
- 至少一个小型环境完成真实 Policy Gradient 更新；
- PPO/GRPO 的 advantage、clip、KL 与采样版本可检查；
- 前沿方法只在稳定 SFT/GRPO 基线后作为单变量引入；
- DPO/RL 的实现、实验和结论边界分开记录；
- reward hacking、长度偏差和 Base 能力回退有检查。

## 10. Stage 6：综合评测与最终验收（2 周）

### 执行契约

- **输入/前置：** G5 关闭；Q0 Base、Q1 SFT、Q2 DPO、Q3 GRPO 身份和 generation config 冻结；T0/T1/T2/T3 数据分层确定。
- **执行顺序：** Schema/Manifest → 污染审计 → 原始生成 → 行为与 LM 指标 → 统计区间 → 盲评包 → 系统矩阵 → claim-evidence 审计 → 学习者最终答辩。
- **必须产物：** 64 例/192 输出原始 Artifact、污染报告、Wilson/paired bootstrap、30 对盲评、18 格系统矩阵、Model/Evaluation Card 和最终证据报告。
- **自动化通过条件：** 同输入 Manifest byte-exact；任一模型/生成配置变化都会改变 run fingerprint；exact 污染为 0；逐题原始输出可回放；改善和退化均有区间/分母；每条 claim 能定位到证据。
- **学习者通过条件：** 完成 30 对盲评、至少一项统计手算、关键调用链追踪、限制条件解释和最终答辩；未完成前 G6 只能写“自动化通过”。
- **行为通过条件：** 与工程门分离。当前 Q0/Q1/Q2 strict 均为 `0/32` 且截断率为 `1.0`，因此行为状态是 `rejected`；不得因 G6 自动化通过而改为 accepted。
- **失败分支：** OOM、NaN、身份 hash 不匹配、答案泄漏或输出目录冲突立即停止；修复后创建新版本和新目录，禁止跨版本拼接结果。

### 指标体系

| 阶段 | 核心指标 |
|---|---|
| Tokenizer | round-trip、fertility、bytes/token、unknown rate、子集表现 |
| 预训练 | train/val loss、PPL、BPB、tokens seen、tokens/s、显存、Resume |
| SFT | held-out loss、任务准确率、格式遵循、遗忘、长度分布 |
| DPO/RL | pairwise win rate、reward margin、KL、长度、reward hacking、多样性 |
| 系统 | 参数量、checkpoint 大小、吞吐、延迟、峰值显存、失败率、成本 |

Perplexity 只在相同 Tokenizer/分词口径下比较；不同 Tokenizer 必须同时报告 BPB 或其他可比口径。

### G6 门禁

最终报告必须回答：

- 与哪个 Base/Baseline 比较；
- 数据和评测集是否固定；
- 训练只改变了什么；
- 指标改善与退化分别是多少；
- 是否存在污染、遗忘和长度偏差；
- 结果能否重复；
- 硬件、时间和费用是多少；
- 哪些能力仍未验证。

生成样例只能作为定性材料，不能替代量化评测。

## 11. 每周 60 小时分配

| 工作类型 | 时间 | 比例 |
|---|---:|---:|
| 模型/算法实现与调试 | 30h | 50% |
| 数学、原理与源码阅读 | 12h | 20% |
| 训练与受控实验 | 10h | 17% |
| 模型正确性测试 | 5h | 8% |
| 文档与工程维护 | 3h | 5% |

核心模型与算法约占 70%，实验与评测约占 20%，一般工程和文档约占 10%。

## 12. 教学输出规则

### 核心模块

Tokenizer、PyTorch 模型、预训练、C++/CUDA、SFT、RL 和评测使用完整讲义：

- 原理和必要数学；
- 真实代码调用链；
- 最小实现；
- 正确/边界/失败测试；
- 实验与指标；
- 12–20 道题，每题后紧跟可折叠答案；
- 阶段门与结论边界。

### 支撑模块

配置、CLI、Git、CI、日志和普通数据处理默认只生成：

- 1–2 页执行卡；
- 当前任务所需的最小概念；
- 精确命令和排错顺序；
- 3–5 道题；
- 30–120 分钟的按需补学。

用户明确要求详细讲义时，可以覆盖该默认规则。

## 13. 实验纪律

- 每个 Stage 开始前定义假设、主变量、固定项、指标、门槛、失败处理和产物；
- Smoke 通过后才能 Quick Run，Quick Run 通过后才能正式运行；
- 每次实验尽量只有一个主变量；
- 低成本正确性实验优先 3 seeds；
- 性能实验固定硬件、shape、batch、dtype、预热与重复次数；
- 失败若改变下一步决策，必须保留记录；
- 所有简历结论绑定配置、数据/Tokenizer、代码版本、硬件和结果。

## 14. 当前状态与下一步

截至 2026-08-09（实验结果均沿用原始运行日期，不因本次文档审查重跑）：

- Day 1–3 学习与测试由用户确认完成；
- Day 1–10 工程和数据夹具实现已经存在；
- Day 4–14 的旧学习课程不再是前置门槛；
- Stage 0 路线切换已经完成；
- Stage 1 的 G1-A 已完成：手写 byte BPE、固定特殊 Token、保存/加载、CLI、评测与 Hugging Face 对照；
- G1-B 已按学习目标改为方法实验室：pre-tokenization、BPE-dropout、Unigram、Picky/SuperBPE、特殊 Token policy 与 entropy patching 的代码、实验和讲义完成，100 项测试通过；
- 用户已确认 Stage 1 和 Stage 2 学习与测试完成；
- G2-Core 与 G2-Arch 的 Python 实现、自动化测试、实验和完整讲义已经形成；
- G2-Systems 的 SDPA/online-softmax、`torch.compile` 单图、INT8 reference 与双进程 DDP 已通过；
- `silu_mul` 已使用 MSVC 19.42/CUDA 12.4 完成 CPU/CUDA 编译、forward、gradcheck、opcheck、空 Tensor 与固定 benchmark；
- G2-Core、G2-Arch、G2-Systems 的自动化工程门全部通过；
- Stage 2 自动化门与学习者验收全部完成；
- Stage 3 的确定性数据、训练循环、混合精度、完整 checkpoint、CPU exact-resume、CUDA 实际恢复、Muon/MTP 方法实验和 1M-token 有界训练已经完成；
- Stage 3 自动化与学习者验收均已完成；
- Stage 4 的 G4-A～G4-E 自动化实现、真实 BF16 LoRA/NF4 QLoRA、100,293-token 正式实验、四份完整讲义和 16 站入口已完成；用户已确认 G4-L 学习者验收；
- Stage 4 正式结果为 held-out loss 改善但 strict generation 0/16，retention 与字符级重复退化；负结果和指标审计均已保留；
- Stage 5 三周偏好/RL 方案已完成 G5-A～G5-L：偏好数据/RM/DPO/PG/PPO/GRPO、前沿张量实验、正式 DPO v2、真实 1-step GRPO v2、五份讲义、18 站入口和学习者验收均已落地；
- Stage 5 DPO 的 pair 指标改善但 strict generation 仍为 0/16，并伴随保留与重复退化；该负结果进入 Stage 6 主验收问题；
- Stage 6 三项默认决策已获批准，G6-A～G6-E 已完成：64 例/192 输出、0 exact/16 near 污染审计、Wilson/paired Bootstrap、30 对盲评包已生成（人工评分待完成）、18 格系统矩阵、E0–E15、最终证据报告与五份完整讲义；G6-L 待学习者完成；
- 原 5 周后训练总预算已拆分为 Stage 4 两周 SFT/LoRA/QLoRA 与 Stage 5 三周偏好/RL，总周期没有增加。

### 当前唯一下一步

1. 按 `docs/lessons/stage06_learning_order.md` 完成 18 站学习；
2. 在不查看私钥的前提下完成 30 对盲评并保存个人评分 Artifact；
3. 完成 Wilson/paired difference 手算、关键调用链追踪和最终答辩；
4. 关闭 G6-L 后再进入 Buffer / Final，整理 Release、复现入口和简历证据。

不得重复 Stage 3 的 1M-token、Stage 4 的 100k-token、Stage 5 的 DPO/GRPO 或 Stage 6 正式运行。确需复现时，必须另立协议并写入新输出目录。Q0/Q1/Q2 当前行为状态保持 `rejected`，直至新的预登记实验提供相反证据。
