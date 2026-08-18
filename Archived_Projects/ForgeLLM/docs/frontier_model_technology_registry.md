# ForgeLLM 前沿模型与技术滚动注册表

> 基准日期：2026-07-28（Asia/Singapore）  
> 适用范围：Tokenizer、模型结构、预训练、后训练与推理验证  
> 维护原则：官方来源优先、阶段内冻结、阶段间滚动升级、单变量验证、资源约束优先  
> 下一次强制复审：Stage 6 启动前；DeepSeek V4 保持 pending，Kimi K3 已通过官方仓库/报告核验

## 1. 为什么需要滚动注册表

模型家族和训练方法的版本更新速度已经快于本项目的完整开发周期。把 `DeepSeek-V3`、`Qwen3` 或 `Gemma 3` 直接写死在长期计划中，会同时带来两种风险：

1. **版本陈旧风险**：新版本已经替换关键结构，但项目仍把旧结构称为“当前前沿”；
2. **追新失控风险**：每次出现新版本就改训练基线，使实验无法复现、结论无法比较。

本注册表把参考对象分成三层：

- **基础层**：BPE/Unigram、Decoder-only Transformer、RMSNorm、RoPE、SwiGLU、AdamW 等稳定知识，不因新版本自动删除；
- **演化层**：DeepSeek-V3、Gemma 3、Qwen3 等仍能解释技术演进的版本，保留为历史对照，不再称为最新；
- **滚动前沿层**：在最近一次审计中通过官方来源核验的当前模型和方法，允许在新 Stage 开始时替换。

“尽量使用最新”在本项目中的准确含义是：**用最新公开证据更新候选模块，用项目资源能承担的最小实验验证它，而不是宣称复现整个前沿模型。**

## 2. 信息源与准入规则

### 2.1 来源等级

| 等级 | 可接受来源 | 用途 |
|---|---|---|
| S0 | 官方技术报告、官方模型卡、官方仓库、官方发布记录、许可证 | 确认版本、结构、权重、发布日期和许可；可作为采用依据 |
| S1 | 作者论文或顶会论文，并有作者/机构代码 | 补充方法细节；可进入实验候选 |
| S2 | Hugging Face Transformers、PyTorch、vLLM、SGLang 等主流框架的正式实现文档 | 确认工程接口和交叉验证结构 |
| S3 | 新闻、榜单、社交媒体、二手解读、社区实现 | 只用于发现线索，不用于冻结版本或支撑项目结论 |

同一项技术至少需要一个 S0/S1 来源；声称“已支持”还必须有可执行代码或主流框架实现。只存在新闻名称、没有官方模型卡/报告/仓库的版本，状态统一为 `待核验`。

### 2.2 候选模块准入条件

候选模块必须同时回答：

- 它替换或补充哪个经典模块？
- 官方来源、代码、许可证和发布日期是什么？
- 在 8GB GPU 或 CPU Smoke 上能否构造缩小实验？
- 对照组、唯一主变量、指标和失败判据是什么？
- 即使结果为负，是否能形成可解释的工程或研究证据？

不能回答上述问题的技术进入观察池，不进入当前实现计划。

## 3. 2026-07-27 官方前沿快照

| 家族/方向 | 当前可核验版本 | 官方发布时间 | 当前应参考的内容 | 项目定位 | 官方入口 |
|---|---|---:|---|---|---|
| DeepSeek | DeepSeek-R1 / V3.2；V4 待核验 | 以官方仓库为准 | R1 的 cold start、GRPO 与多阶段后训练；V3 系列的 MLA/MoE/MTP | 已核验历史/当前参考；不把第三方框架支持当作 V4 技术报告 | [R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1) / [R1 报告](https://arxiv.org/abs/2501.12948) |
| Qwen | Qwen3.6；架构说明继承 Qwen3.5 | 2026-04-16/22 | Gated Delta Networks + 稀疏 MoE、原生多模态、可扩展异步 RL、Thinking Preservation | 当前开放模型与混合架构参考 | [官方仓库](https://github.com/QwenLM/Qwen3.6) |
| Gemma | Gemma 4；含 MTP 变体与 12B Unified | 2026-03-31 至 2026-06-03 | 专用 Draft/MTP 模型、推测解码、统一模型变体 | 当前 MTP 与低延迟推理参考；Gemma 3 降为演化对照 | [版本记录](https://ai.google.dev/gemma/docs/releases) / [文档](https://ai.google.dev/gemma/docs) |
| OLMo | OLMo 3；Olmo Hybrid | 2025-12 / 2026-04 | 全生命周期开放证据；Gated DeltaNet 混合结构 | 可复现训练流程主参考；混合线性注意力对照 | [OLMo 3 论文](https://arxiv.org/abs/2512.13961) / [Olmo Hybrid 论文](https://arxiv.org/abs/2604.03444) |
| Moonshot/Kimi | Kimi K3；Kimi K2.5；Kimi Linear | 以官方仓库为准 | K3 的 SFT cold start、9 个 RL 专家、partial rollout/staleness、MOPD 与部署感知 QAT；Kimi Linear 的混合线性注意力 | K3 进入 Stage 5 后训练教学；超大权重不作为本地训练基线 | [K3 官方仓库/报告](https://github.com/MoonshotAI/Kimi-K3) / [K2.5](https://github.com/MoonshotAI/Kimi-K2.5) / [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear) |

注意：这张表只声明“截至审计日可由官方材料核验的当前参考”。它不声明这些模型彼此的绝对能力排名，也不把厂商自报 Benchmark 当作本项目实验结论。

2026-07-28 Stage 5 F0 增量审计发现 Moonshot AI 已发布 Kimi K3 官方仓库与完整技术报告，因此其已从 `待核验` 转为 `已核验`。相反，未找到 DeepSeek 官方发布的 V4 技术报告；第三方推理框架或模型别名只能证明接口线索，不能支撑训练技术归因，因此 DeepSeek V4 转为 `待核验`。

## 4. DeepSeek 技术演化与证据修正

### 4.1 V3、V3.2、R1 与 V4 的证据关系

| 版本 | 在本项目中的身份 | 仍值得保留的学习点 | 已被当前版本更新的部分 |
|---|---|---|---|
| DeepSeek-V3 | 历史/演化基线 | MLA、辅助损失自由的 MoE 负载均衡、MTP、FP8 大规模训练思路 | 不再把 MLA 单独称为 DeepSeek 当前注意力方案 |
| DeepSeek-V3.2 | 演化基线 | DSA 稀疏注意力与服务演化 | 只按其官方材料记录，不推断 V4 |
| DeepSeek-R1 | 已核验后训练参考 | cold start、GRPO 与多阶段后训练；可读性/语言混合失败 | Stage 5 只学习方法，不复现大规模结果 |
| DeepSeek-V4 | 待核验名称 | 当前没有足够官方技术材料可登记具体机制 | 不得归因 CSA/HCA、mHC、Hash-MoE 或 on-policy distillation |

### 4.2 立即采用的修正

- 文档中出现“DeepSeek V4 使用某技术”的旧表述一律视为未核验，不作为实施依据；
- V3/V3.2 不删除，作为 MLA、MTP、MoE 与稀疏注意力的演化对照；
- Stage 5 后训练只引用已公开的 DeepSeek-R1 报告；
- 若未来出现 V4 官方报告，先做增量审计，再决定是否修改下一阶段，绝不追溯改写已有实验。

## 5. Tokenizer 路线：经典主线与 2025–2026 候选

### 5.1 稳定主线

Day 16–20 仍先完成手写 BPE 与工程 Tokenizer。这不是版本落后，而是为了建立可测试的词表训练、编码/解码、特殊 Token、确定性和 Artifact 契约。没有这条基线，后续新方法无法进行公平消融。

### 5.2 新方法候选

| 方法 | 年份 | 相对经典 BPE 的变化 | 本项目实验 | 优先级/可行性 |
|---|---:|---|---|---|
| SuperBPE | 2025 | 通过预分词课程先学习子词，再学习跨空白的“superword” | 在相同语料、词表大小和下游模型配置下，只替换词表学习策略；比较 fertility、压缩率、长尾、中文/代码边界和训练吞吐 | P1 / 4 |
| BLT | 2024–2025 | 取消固定子词词表，按下一字节熵动态形成 Patch | 先复现字节→熵→动态 Patch 的 CPU/小模型路径，再与 BPE 比较鲁棒性和每样本计算量 | P2 / 2 |
| Fast BLT | 2026 | 以 BLT-D、Self-speculation、Diffusion+Verification 缓解逐字节生成慢 | 当前只做论文与接口设计；BLT 基线成立且代码/资源允许后再做生成实验 | 观察池 / 1 |

来源： [SuperBPE](https://arxiv.org/abs/2503.13423)、[BLT 论文](https://arxiv.org/abs/2412.09871)、[BLT 官方代码](https://github.com/facebookresearch/blt)、[Fast BLT](https://arxiv.org/abs/2605.08044)。

Tokenizer 阶段只允许 BPE→SuperBPE 一项进入本周期 P1。BLT 会改变模型输入与计算单元，不应伪装成普通 Tokenizer 插件；它需要独立 Stage。

## 6. 预训练与模型结构路线

| 候选技术 | 当前来源 | 最小可验证实现 | 对照和指标 | 资源判断 | 决策 |
|---|---|---|---|---|---|
| Muon 优化器 | Muon 独立方法与 Kimi K2 | 在相同初始化、Token 数、Batch 和 LR 预算下，与 AdamW 比较 | loss/token、梯度范数、NaN、step time、峰值显存 | 8GB 可做小模型 | Stage 2 reference 与 Stage 3 三种子短训练对照完成；不作普遍优越性声明 |
| mHC-lite | 独立 mHC 候选 | 为残差支路增加小倍数并行流与受约束混合，先做 shape/梯度/数值测试 | 收敛、稳定性、额外参数、吞吐、显存 | 可缩小；完整实现有复杂度 | Stage 2 教学 reference 完成；不归因 DeepSeek V4 |
| CSA/HCA-lite | 历史候选组合 | 局部窗口 + 压缩长程分支；先做 reference kernel | 与全注意力比较因果性、输出误差、显存、长序列速度 | 8GB 只适合 1K–8K 小模型 | Stage 2 教学 reference/因果性测试完成；不归因 DeepSeek V4 |
| Hash-MoE bootstrap | 哈希路由教学候选 | 固定 token-id→expert-id 映射与学习路由做结构对照 | 负载分布、路由稳定、loss、专家利用率 | 小 MoE 可做 | Stage 2 确定性 reference 完成；不归因 DeepSeek V4 |
| Gated DeltaNet hybrid | Qwen3.5/3.6、Olmo Hybrid | 每 N 层插入一个全注意力层，其余使用简化 Gated DeltaNet | perplexity、长程检索、吞吐、状态内存 | Kernel 复杂；先 reference | Stage 2 的 3:1 reference/因果性测试完成 |
| KDA hybrid | Kimi Linear | 以简化 gated delta state 学习 3:1 线性/全局注意力调度思想 | 同上，额外比较有限状态更新稳定性 | 官方高性能 KDA kernel 复杂 | Stage 2 只完成思想近似；不得称为 KDA 复现 |
| MTP / Draft head | DeepSeek-V3、Gemma 4 | 主干增加未来预测头；训练时加入加权未来 Token loss | 主任务 loss、接受率、解码速度、额外训练成本 | 小模型可做 | Stage 3 三种子辅助目标短实验完成；当前未见主 loss 收益，解码收益待后续 |

优先顺序不是按论文热度，而是按“信息增益 ÷ 实现代价”：

1. `Muon vs AdamW`；
2. `MTP head vs 单 Token 预测`；
3. `mHC-lite vs 标准残差`；
4. `CSA/HCA-lite vs 标准注意力`；
5. `Hash-MoE bootstrap vs 学习路由`；
6. Gated DeltaNet/KDA 仅在前述实验稳定后推进。

每个实验保持相同数据、Tokenizer、参数量级、Token 预算、精度和评测集；一次只改变一个主变量。完整 V4 的 1M Context、超大 MoE 和官方吞吐数字不在本项目可复现声明范围内。

## 7. 后训练路线：从经典闭环升级到 2025–2026 方法

### 7.1 稳定主线

监督式后训练先完成：

```text
冻结 Base 与评测集
→ SFT / LoRA 基线
→ 格式、能力、遗忘和安全回归
→ 可验证奖励任务
→ 再决定偏好优化或 RL
```

SFT 与 LoRA/QLoRA 仍是必要基线；存在合格偏好数据时，再把 DPO 作为经典偏好优化对照。没有稳定 SFT 和冻结评测，无法判断 GRPO/DAPO 或蒸馏是否真正改善。

### 7.2 Stage 4 冻结采用项

| 项目 | 2026-07-28 决策 | 为什么采用 | 证据边界 |
|---|---|---|---|
| 可执行 Base | `Qwen/Qwen3-0.6B-Base`，实施时固定 revision | Apache-2.0、0.6B、28 层、BF16、32K；适合 8GB 下的短程 Adapter 实验 | 只是资源友好的执行基线，不代表当前最大或最新 Qwen 家族 |
| SFT 数据 | 96 条原创 correctness + SmolTalk `smol-constraints` 小子集 | 前者可逐 token 审计，后者提供 Apache-2.0 的真实 instruction 数据 | 只训练 constraint/format 基础，不宣称通用 Chat 数据覆盖 |
| SFT 目标 | assistant-only NLL | 明确排除 system/user/pad，和当前 TRL conversational 接口对照 | 模板必须产生正确 generation mask；先关闭 packing |
| PEFT | 手写 LoRA + PEFT 对照，正式用 BF16 LoRA | LoRA 是最小可解释低秩基线，便于检查 Base freeze、参数量与 merge | 不搜索 rank/alpha，不逐一训练所有 LoRA 变体 |
| 量化微调 | NF4 + double quant + BF16 compute 的真实 QLoRA smoke | 用同协议直接测 4-bit Base 的梯度路径与显存收益 | 不把理论权重大小当进程显存；本机 backward 未通过前不声明支持 |
| 当前 SFT 扩展 | DFT 单 batch/极短实验，P2 | 当前 TRL 已暴露 `loss_type="dft"`，适合学习 token reweighting | 不替代稳定 NLL，不阻塞 G4，不作方法排名 |

官方入口：[`Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)、[Qwen3 后训练路线](https://qwenlm.github.io/blog/qwen3/)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)、[PEFT 量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)、[bitsandbytes 安装/平台支持](https://huggingface.co/docs/bitsandbytes/installation)、[SmolTalk 数据卡](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)。

Qwen3 官方公开的四阶段后训练是长 CoT cold start、reasoning RL、thinking mode fusion 和 general RL。Stage 4 只学习其中的监督式 cold start/数据融合基础；RL 阶段与 Kimi K3 官方报告中的 MOPD 放入 Stage 5，避免在 SFT 基线尚未成立时混入 rollout 变量。

### 7.3 新方法候选

| 方法 | 主要价值 | 本项目最小实验 | 准入条件 | 状态 |
|---|---|---|---|---|
| GRPO | 无需单独 Value 模型的组相对策略更新 | 在确定性数学/代码 Verifier 上，用 Qwen 4×4 rollout 跑一步 | SFT 稳定；Reward 可单测；预算可控 | Stage 5 链路完成；不作能力结论 |
| DAPO | 解耦 Clip、动态采样、Token-level policy-gradient 等，改善长推理 RL 稳定性 | 逐项实现/对齐 loss 和采样规则，不直接复现 32B 结果 | GRPO 数学基线已通过 | Stage 5 frozen-tensor 完成 |
| 9 专家 + MOPD | Kimi K3 以 3 领域×3 effort 专家向 student 提供 dense delta | 多教师选择 + clipped stop-gradient delta 的最小张量原型 | K3 官方报告已核验；不加载外部教师 | Stage 5 reference 完成 |
| VESPO stale weighting | 对 stale rollout 的 sequence IS 做 Gamma 软重加权 | 固定 log-ratio/advantage 张量与 ESS | 官方论文与代码可核验 | Stage 5 reference 完成 |
| 异步 Agent RL | Qwen3.5/3.6 的系统级方向 | 先做 rollout/reward/training 解耦接口和队列 Smoke | 单机同步 RL 已可复现，再考虑系统优化 | P2 |

来源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[DAPO](https://arxiv.org/abs/2503.14476)、[VESPO](https://arxiv.org/abs/2602.10693)、[Kimi K3 官方仓库/报告](https://github.com/MoonshotAI/Kimi-K3)。

后训练采用“算法正确性优先于规模”的原则：先验证 log-prob、mask、KL、advantage、clip、reward、采样版本和 rollout 数据契约，再谈能力提升。8GB GPU 与 20 USD/月预算下，不承诺大规模 RL。

## 8. 分阶段采用矩阵

| 项目阶段 | 冻结的稳定基线 | 最多允许的前沿变量 | 本阶段不做 |
|---|---|---|---|
| Stage 1 Tokenizer | 手写 BPE + 工程 Tokenizer | SuperBPE/Unigram/Picky 等方法 reference | BLT 完整架构、Fast BLT |
| Stage 2 G2-Core | RMSNorm + RoPE + SwiGLU + MHA/GQA/SDPA + 标准残差 | QK-Norm、生成与 KV Cache | 正式预训练、1M Context |
| Stage 2 G2-Arch | Stable Decoder | MLA、MoBA、CSA/HCA-lite、DeltaNet Hybrid、MoE、mHC、MTP、Muon 的独立 reference | 同时堆叠新模块、官方整模效果复现 |
| Stage 2 G2-Systems | SDPA、固定模型/shape/dtype | online-softmax、compile、INT8、DDP、自定义 `silu_mul` | 宣称复现官方 kernel 效率 |
| Stage 3 预训练闭环 | AdamW + 标准单 Token loss | Muon 与 MTP 各自作为独立单变量短实验 | 同一实验同时替换优化器、残差、注意力和目标函数 |
| Stage 4 监督式后训练 | assistant-only SFT + BF16 LoRA + 固定前后评测 | QLoRA 真实内存实验；DFT 只做 P2 梯度实验 | RM/DPO/RL、rank/LR 搜索、无基线多方法堆叠 |
| Stage 5 偏好/RL | RM/DPO + 可验证的小型 PG/GRPO | DAPO/DrGRPO/GSPO/VESPO/MOPD 单变量 reference | 无 Verifier 的大模型 RL、跳过 SFT 基线；已自动化完成 |

## 9. F0 前沿新鲜度门禁

F0 是研究参考门禁，不替代各 Stage 的工程门禁。它在 Stage 启动、Stage 验收和每个后训练 Stage 启动前执行。

### 9.1 检查清单

- [ ] 搜索目标家族的官方发布页、模型卡、仓库和技术报告；
- [ ] 记录 `family`、`selected_version`、`release_date`、`last_verified_at`；
- [ ] 记录报告、模型卡、代码、框架支持、许可证的精确 URL；
- [ ] 写明它替代的旧版本、旧版本仍保留的教学/对照价值；
- [ ] 对候选模块做 1–5 级可行性评分；
- [ ] 明确唯一主变量、对照、预算、指标、停止条件；
- [ ] 在 Stage 计划中冻结版本；
- [ ] Stage 中途的新发布只进入下一 Stage 的审计队列。

### 9.2 版本记录模板

```yaml
family: DeepSeek
selected_version: DeepSeek-R1
release_date: 2025-01
last_verified_at: 2026-07-28
sources:
  repository: https://github.com/deepseek-ai/DeepSeek-R1
  report: https://arxiv.org/abs/2501.12948
pending_name: DeepSeek-V4
supersedes: null
retained_historical_modules: [MLA, MTP, auxiliary-loss-free MoE balancing]
candidate_modules: [GRPO, cold-start-SFT, multi-stage-RL]
stage_decision: frozen | candidate | watch | rejected
decision_reason: "..."
```

## 10. 更新触发器与防过时机制

出现以下任一事件时触发增量审计：

- 官方家族出现新 major/minor 版本；
- 官方报告披露替换核心结构或训练目标；
- Transformers/vLLM/SGLang 增加正式支持；
- 原候选代码、权重或许可证发生变化；
- 新论文能够直接替换当前待实现模块，并且具备官方代码。

每次审计只做差分：`新增了什么 → 替代什么 → 是否可缩小验证 → 是否改变当前 Stage`。默认不改变已经开始的 Stage；只有发现正确性、安全、许可证或 API 停用风险时，才允许中途修订，并写 ADR。

## 11. 当前决策结论

1. DeepSeek V3/V3.2 与 R1 保留为已核验参考；DeepSeek V4 因缺少可核验官方技术报告转为待核验。
2. Qwen 与 Gemma 条目保留其各自最近一次官方审计身份；开始新 Stage 时必须重新核验，不用第三方别名升级。
3. OLMo 3 负责“全流程可复现”参考，Olmo Hybrid/Kimi Linear 负责混合线性注意力参考。
4. Tokenizer 不只做经典 BPE：增加 SuperBPE 的可控消融；BLT/Fast BLT 作为独立研究 Stage，不挤占 G1。
5. Stage 2 已完成 Muon、MTP、mHC-lite、CSA/HCA、Hash 路由与 Gated DeltaNet hybrid 的缩小 reference；Stage 3 已将 Muon 和 MTP 分成两个互不混用的单变量短实验。
6. 原 5 周后训练拆为 Stage 4 两周 SFT/LoRA/QLoRA 与 Stage 5 三周偏好/RL，总周期不变；Stage 4 采用自研 5.36M + Qwen3-0.6B-Base 双轨。
7. Stage 5 在稳定 SFT 基线后完成 GRPO/DAPO 等机制，并按 Kimi K3 官方报告加入多教师 MOPD 最小 reference。
8. 所有“最新”结论绑定 `last_verified_at`；没有官方证据的传闻版本不得进入实现计划。
9. Kimi K3 已由 Moonshot AI 官方仓库/技术报告确认；DeepSeek V4 仍待 DeepSeek 官方材料，不采信第三方接口线索作技术归因。
