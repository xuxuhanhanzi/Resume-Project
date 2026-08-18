# Stage 4：监督式后训练基础实现计划

> 计划状态：自动化 G4-A～G4-E 与 G4-L 学习者验收均已完成  
> 冻结日期：2026-07-28（Asia/Singapore）  
> 预计周期：2 周，按 120 小时上限规划  
> 阶段门：G4-A / G4-B / G4-C / G4-D / G4-E / G4-L  
> 实施前置：用户已于 2026-07-28 确认 Stage 3 学习者验收完成并批准本实验方案

## 1. 阶段定位

Stage 4 的目标不是训练一个“完美聊天模型”，也不是通过反复调 rank、学习率、数据规模或 epoch 来追求排行榜数字。本阶段只回答五个核心问题：

1. 一条多轮对话如何变成严格可检查的 token、label 和 loss mask？
2. SFT 与预训练的目标函数哪里相同，监督区域又哪里不同？
3. LoRA 为什么能把权重更新限制到低秩子空间，代码怎样注入、保存和合并？
4. QLoRA 如何在冻结 4-bit Base 的同时，把梯度传给 BF16 LoRA Adapter？
5. 如何用同一组冻结评测区分“学会目标任务”“只记住训练样本”和“损伤 Base 能力”？

原路线中名为 Stage 4 的 C++/CUDA 自定义算子已经并入 Stage 2 的 `G2-Systems` 并完成。自 2026-07-28 起，Stage 4 编号用于预训练之后的第一个新阶段：**监督式后训练基础**。历史证据不删除，也不重复实现。

本阶段结束后的路线是：

```text
Stage 3 预训练
→ Stage 4 监督式后训练：SFT / LoRA / QLoRA / 固定前后评测
→ Stage 5 偏好优化与在线 RL：RM / DPO / PG / PPO / GRPO / DAPO / OPD
→ Stage 6 综合评测与最终验收
```

## 2. 设计原则与不做事项

### 2.1 设计原则

- **算法正确性优先于训练规模：** 先验证模板、mask、loss、低秩更新和量化边界，再启动短训练。
- **双轨实现：** 自研 5.36M 模型用于可完全检查的算法实验；开源 0.6B Base 用于框架与真实 Adapter 工作流。
- **一次一个主变量：** full-sequence/assistant-only、full fine-tune/LoRA、BF16 LoRA/QLoRA 分成独立实验。
- **冻结后再训练：** 模型 revision、数据 revision、split、模板、生成配置和评测先冻结。
- **负结果也可完成阶段：** 方法实现正确但短训练未提高能力时，保留结果并收窄结论，不靠继续调参制造正结果。
- **完整讲义模式：** 学习顺序先于文件列表；每道思考题后立即给出答案。

### 2.2 明确不做

- 不做 LoRA rank、alpha、学习率、词表、数据规模或 epoch 网格搜索；
- 不训练 1.5B 以上模型，不下载 DeepSeek/Kimi 超大权重；
- 不使用付费 LLM Judge 作为主评测；
- 不在没有固定 Base 评测的情况下只展示生成样例；
- 不把 QLoRA 等同于“把整个模型用 4-bit 训练”；
- 不把 SFT 的短程收益写成通用对话、推理或安全能力提升；
- 不在 Stage 4 实现 Reward Model、DPO、PPO、GRPO、DAPO 或 on-policy distillation；这些属于 Stage 5。

## 3. F0 审计结论与冻结候选

### 3.1 开源 Base

主候选冻结为 [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)：Apache-2.0、0.6B 参数、28 层、32K 上下文、BF16 权重。它不是“当前最大或最新家族”的代名词，而是当前资源下最小、稳定、可执行且有官方模型卡的 Base 基线。实施 Phase 0 必须把实际下载 revision 写入 Manifest；不能只记录浮动的 `main`。

双轨职责固定如下：

| 轨道 | 模型 | 用途 | 不允许的结论 |
|---|---|---|---|
| A：算法轨 | Stage 3 的 5.36M 自研 Decoder | mask、SFT loss、tiny overfit、手写 LoRA、全量微调对照 | 不代表真实开源聊天模型质量 |
| B：框架轨 | Qwen3-0.6B-Base | Chat Template、PEFT、BF16 LoRA、QLoRA、Adapter Artifact | 不代表复现 Qwen 官方后训练配方 |
| 只读参考 | Qwen3-0.6B Instruct | 检查模板和行为边界 | 不作为 Base 的训练前基线，不与本项目 Adapter 做不公平能力排名 |

若 Qwen 模型卡、许可证或框架支持在实施前发生不兼容变化，Phase 0 才允许替换候选，并必须记录替换原因。不能因为出现更大的新模型就中途改基线。

### 3.2 数据

数据分两层，不把大规模数据收集变成新的学习阻塞：

| 数据集 | 计划规模 | 作用 | 冻结规则 |
|---|---:|---|---|
| `sft_correctness_v1` | 96 条项目原创样例 | 模板、mask、overfit、边界与失败测试 | 64/16/16，人工逐条审计，固定 SHA-256 |
| `smol_constraints_v1` | 2,048 train + 256 validation + 最多 256 test | 有界真实 SFT 与 held-out loss | 从 SmolTalk `smol-constraints` 的固定 revision 按内容哈希抽样；train/test 隔离、精确去重 |
| `retention_v1` | TinyStories held-out 固定子集 + 项目原创通用 prompt | Base 能力回退检查 | Qwen Base 与 Adapter 使用同一 Qwen Tokenizer 比较；不与 Stage 3 的 320-vocab PPL 横比 |

SmolTalk 官方卡说明 `smol-constraints` 等新增子集使用 Apache-2.0。实施时仍需保存数据卡、revision、原始字段、许可证、样本哈希、拒绝原因与处理脚本；没有这些证据，不得开始框架轨正式短跑。

### 3.3 当前框架事实

- TRL 的 SFT 接口支持 conversational / prompt-completion 数据、`assistant_only_loss` 和 packing；assistant-only 模式依赖能返回 assistant generation mask 的模板。
- PEFT 的标准 LoRA 初始化把 B 矩阵置零，因此 Adapter 初始应为 no-op；QLoRA-style 配置可将 Adapter 注入所有 Linear 层。
- QLoRA 的必学机制是冻结 4-bit Base、NF4、double quantization、BF16 compute 与 LoRA 梯度路径。
- bitsandbytes 官方安装表已列出 Windows x86-64 CUDA wheel 和 NF4/FP4 支持；本机仍必须通过真实 import、4-bit load、forward/backward 和峰值显存门，不能只凭文档宣称可用。

## 4. 冻结实验协议

### 4.1 默认正式配置

以下参数用于消除调参搜索，不等于最优参数：

| 项目 | 冻结值/规则 |
|---|---|
| Base | `Qwen/Qwen3-0.6B-Base`，精确 revision 在 Phase 0 写入 |
| 最大序列长度 | 512；不在本阶段做长上下文训练 |
| 监督范围 | assistant-only，EOS 是否监督必须显式记录 |
| Packing | baseline 关闭；只在独立语义测试通过后做可选吞吐实验 |
| LoRA | `r=16`、`alpha=32`、`dropout=0.05`、`bias=none`、`target_modules=all-linear` |
| LoRA 精度 | Base BF16；Adapter BF16/FP32 由实现记录，梯度有限 |
| QLoRA | NF4、double quantization、BF16 compute、冻结 Base、同一 LoRA 配置 |
| 优化器 | AdamW；只训练 Adapter 参数 |
| 学习率 | `2e-4` 固定教学起点；不搜索 |
| 调度 | 3% warmup + cosine；按监督 token 记录进度 |
| 生成 | greedy 为量化主指标；固定 `max_new_tokens`；采样只作定性附录 |
| seed | 正式 0.6B 运行 1 个 seed；低成本算法实验 3 个 seeds |

Micro batch、gradient accumulation 和 gradient checkpointing 只允许根据 Phase 0 显存探针做一次资源调整，调整目标是“能运行”，不是提高指标。最终值必须写入配置与报告。

### 4.2 预算和停止条件

| 运行 | 上限 | 目的 |
|---|---:|---|
| CPU/5.36M 单元与 tiny overfit | 每项 ≤10 分钟 | 正确性 |
| Qwen BF16 LoRA smoke | 20 optimizer steps | 接口、显存、梯度、Artifact |
| QLoRA smoke | 与 BF16 smoke 相同 batch/sequence/steps | 真实 4-bit 梯度路径和显存对照 |
| 唯一正式短跑 | 100,000 assistant target tokens、500 steps 或 45 分钟，先到者停止 | 冻结条件下的完整闭环 |
| Stage 4 新增磁盘 | 10 GiB 软上限 | 防止 checkpoint 和 cache 无界增长 |
| Stage 4 外部费用 | 默认 0 USD | 本地优先；任何云端回退先单独审批并登记 |

OOM 不触发自动模型替换或无界降配置。先停止、记录实际显存，再只调整 micro batch/accumulation；仍失败才进入资源复审。

## 5. 工作包与子门

### G4-A：数据、角色与 Chat Template

计划实现：

- `InstructionRecord` / `Message` 严格 Schema，角色仅允许 system/user/assistant；
- 单轮与多轮 conversational 数据的规范化；
- 明确 BOS、EOS、PAD、generation prompt 和工具角色的阶段边界；
- 模板序列化、tokenize、assistant span 定位和可读审计表；
- 长样本截断时，不产生零监督 token 的样本；
- 文档级 split、内容哈希去重、来源/许可证/PII 最小审计；
- Data Manifest 绑定数据 revision、模板文本、Tokenizer revision 和 split hash。

测试至少覆盖：空消息、未知角色、连续同角色、空 assistant、只有 system、Unicode、特殊 Token 注入、尾部截断、模板缺少 generation mask、多轮 assistant span。

通过条件：

- 96 条 correctness 数据人工审计完成；
- 固定样例的模板字符串、token IDs、assistant spans 和 hash 可重复；
- train/validation/test 无内容哈希交叉；
- 每个训练样本至少有一个受监督 assistant target token；
- 被拒绝样本都有机器可读原因。

### G4-B：SFT 目标、Collator 与训练闭环

计划实现：

- 从 Causal LM shifted CE 推导 SFT loss；
- `labels=-100` 只排除 system/user/padding，不误伤 assistant；
- EOS 监督策略、padding side、attention mask 与 loss 分母显式化；
- baseline 默认不 packing；独立实现 packing 时保留样本边界和 assistant mask；
- held-out response-only loss、token accuracy 和监督 token 数；
- 复用 Stage 3 的 optimizer、precision、finite check、metrics 和 checkpoint 经验，但为 Adapter 状态建立独立 schema；
- 与 TRL `SFTTrainer` 的一个固定 batch 做 token、mask 和 loss 对照。

核心不变量：

```text
labels[j] 通常复制 input_ids[j]，CausalLM 内部再做 shift
logits[:, j-1] 预测 labels[:, j]，即位置 j 的 assistant token
system/user/pad 的 label → -100
assistant content / 显式选定的 EOS 的 label → token id
loss 分母 → shift 后非 -100 的 assistant target tokens
```

通过条件：

- 手工构造 batch 的 token-level loss 与 PyTorch CE 对齐；
- prompt/padding 位置没有直接 CE 项，padding 也不能通过 attention 影响有效 token；prompt 仍可作为 assistant 的条件产生间接梯度，讲义必须明确区分；
- 32 条以内 tiny fixture 重复训练后，监督 token accuracy ≥95%，loss 相对初始下降 ≥80%；
- assistant-only 与 full-sequence 实验只改变 label mask，并报告差异；
- 空监督、NaN/Inf、超长截断和错误模板 fail-fast。

### G4-C：手写 LoRA 与 PEFT 对照

手写 `LoRALinear` 必须覆盖：

- 原权重冻结，`ΔW = scaling × B @ A`；
- A/B shape、rank、alpha、dropout 和初始化；
- B 零初始化使训练前 LoRA 为 no-op；
- 只选择目标 Linear，禁止静默漏注入；
- trainable/total parameter 计数；
- Adapter-only `state_dict`、保存、加载、merge、unmerge；
- tied embedding/LM head 不在本阶段隐式改写。

实验：

1. 5.36M 模型全量微调 vs 手写 LoRA，固定初始化、数据和 assistant tokens；
2. 手写 LoRA vs PEFT，在同一 FP32 Linear 和同一 A/B 权重上对齐 forward/gradient；
3. merge 前后、保存加载前后做数值回归。

通过条件：

- Base 参数在 Adapter 训练前后 byte-exact 不变；
- trainable 参数量等于公式计算值；
- FP32 no-op 和 merge/unmerge 最大绝对误差 ≤`1e-6`；BF16 单独使用预先登记容差；
- Adapter round-trip 输出一致；
- 框架对照差异在固定容差内，无未解释漏注入。

### G4-D：QLoRA 机制与真实内存实验

计划学习和验证：

- 线性量化的 scale/zero-point 与 NF4 codebook 的区别；
- block-wise quantization、double quantization 和反量化计算；
- 为什么 4-bit Base 不接收梯度，而输入梯度仍能穿过反量化路径到 LoRA；
- compute dtype、storage dtype、Adapter dtype 和 optimizer state 的四本内存账；
- `prepare_model_for_kbit_training`、全 Linear 注入和梯度 checkpoint 边界；
- BF16 LoRA 与 QLoRA 在同一 20-step 协议下的峰值 allocated/reserved memory、tokens/s、loss 和 checkpoint 大小。

通过条件：

- 本机真实加载 4-bit Qwen Base，并完成 forward/backward/optimizer step；
- 量化 Base 全部冻结、Adapter 梯度有限且非零；
- 报告实际 dtype 和峰值显存，不用理论 4-bit 大小冒充进程显存；
- QLoRA 峰值 allocated memory 低于同协议 BF16 LoRA；若不满足，必须定位测量或配置原因；
- 本机官方 wheel 若失败，保留完整失败证据并进入资源复审；未经审批不自动购买云资源。

### G4-E：冻结评测与有界正式短跑

训练前必须保存 Base 结果，训练后使用完全相同的 prompt、模板、Tokenizer、generation config 和 metric code。

| 维度 | 指标 | 解释边界 |
|---|---|---|
| SFT 拟合 | held-out assistant loss、token accuracy | 只在同模型/Tokenizer/模板下比较 |
| 目标任务 | 项目原创 constraint exact match、格式遵循率 | 使用确定性 verifier，不用主观 Judge 代替 |
| 行为 | 空回答率、重复率、响应 token 长度分布 | 防止靠变长获得假提升 |
| 保留能力 | TinyStories 固定文本 loss、通用 prompt 回归 | 只比较 Qwen Base vs 同一 Qwen Adapter |
| 系统 | trainable params、tokens/s、step time、峰值显存、checkpoint 大小 | 固定硬件、batch、sequence、dtype |
| 可复现 | config/data/model/template/code fingerprints | 任一缺失都不能形成正式结论 |

正式短跑默认选择 **BF16 LoRA**，因为它是解释最简单的稳定基线；QLoRA 用等步数真实 smoke 检查内存机制。只有 BF16 LoRA 本地无法运行而 QLoRA 已通过全部正确性门时，才允许把 QLoRA 改为正式短跑，并记录路线变化。

阶段不要求所有质量指标上升。通过条件是：冻结协议完整执行、至少一个目标指标和所有回退指标被报告、失败或退化有解释、结论没有越过单 seed/小数据/短训练边界。

### G4-L：学习者验收

自动化实现通过后，用户仍需按 Stage 4 唯一学习入口完成：

- 逐 token 手画一条多轮对话的 `input_ids/labels/assistant_mask`；
- 从 CE 推导 assistant-only SFT loss；
- 手算一个 Linear 的 LoRA 参数量与 `B @ A` shape；
- 解释为什么 B 零初始化使 Adapter 初始为 no-op；
- 解释 merge 为什么不能在训练中反复无记录执行；
- 画出 QLoRA storage/compute/gradient 路径；
- 阅读一次 BF16 LoRA 和 QLoRA 显存账；
- 对正式短跑结果做不越界口述。

用户确认讲义、代码追踪、最小改动实验和口述验收完成后，完整 G4 才标记为已完成。

## 6. 单变量实验矩阵

| ID | 假设/问题 | Baseline | 唯一主变量 | 预算 | 成功定义 |
|---|---|---|---|---:|---|
| E0 | 模板是否确定 | 同一 messages | 重复序列化 | CPU | 文本/token/span/hash 完全一致 |
| E1 | assistant-only mask 是否正确 | 手算 labels | 实现 labels | CPU | 每个 token 100% 对齐 |
| E2 | SFT 更新链是否有效 | 初始 tiny model | 训练 steps | ≤10 min | token accuracy/loss 达门 |
| E3 | mask 改变了什么 | full-sequence | assistant-only | 3 seeds × 短跑 | 只改变监督区域，差异可解释 |
| E4 | LoRA 是否真为低秩更新 | full fine-tune | 手写 LoRA | 3 seeds × 短跑 | Base 冻结、更新可测、参数账正确 |
| E5 | 手写与 PEFT 是否同义 | 手写 LoRA | PEFT module | 单 batch | forward/grad/merge 对齐 |
| E6 | 0.6B BF16 LoRA 是否可运行 | Base eval | 20-step Adapter | 本地 | finite、Artifact、显存证据齐全 |
| E7 | QLoRA 是否降低显存 | E6 | 4-bit Base | 同一 20 steps | 真实反向通过且峰值显存降低 |
| E8 | 有界 SFT 改变了什么 | 冻结 Base | BF16 LoRA 100k tokens | ≤45 min | 全套前后评测与边界报告完成 |
| E9（P2） | DFT 与普通 NLL 梯度如何不同 | assistant-only NLL | DFT loss | 单 batch/极短跑 | 公式、梯度、稳定性可解释；不阻塞 G4 |

E9 只用于理解当前框架已经暴露的新 SFT loss，不升级为新的长期调参项目。DoRA、rsLoRA、LoftQ、PiSSA/EVA 等变体进入讲义的“方法地图”，不在本阶段逐一训练。

## 7. 计划实现清单

### 7.1 源码与配置

```text
src/forgellm/post_training/
├── __init__.py
├── schema.py              # messages / instruction records
├── chat_template.py       # serialize / tokenize / assistant spans
├── collator.py            # padding / labels / masks
├── sft.py                 # objective 与 tiny SFT loop
├── lora.py                # 手写 LoRALinear、注入、merge/unmerge
├── adapters.py            # PEFT/Qwen 适配与 Artifact
└── evaluation.py          # 冻结前后评测和 rule-based metrics

configs/post_training/
├── stage4_tiny_sft.toml
├── stage4_qwen_lora_smoke.toml
├── stage4_qwen_qlora_smoke.toml
└── stage4_qwen_lora_bounded.toml
```

计划脚本：数据获取/冻结、Stage 4 smoke、bounded run、LoRA/QLoRA memory lab 和统一评测。所有脚本必须支持 `--help`、明确输出目录和停止条件。

### 7.2 测试

至少形成：

- Schema/模板/特殊 Token/截断单元测试；
- assistant-only labels 与手算 loss 对照；
- collator padding 与零监督失败测试；
- LoRA no-op、参数计数、Base freeze、gradient、merge/unmerge、round-trip；
- PEFT 数值对照；
- tiny overfit 集成测试；
- Qwen/QLoRA 标记为可选重依赖 smoke，不让离线基础质量门误下载模型；
- 统一评测的 metric denominator、长度、重复和 retention 回归测试。

### 7.3 讲义与学习入口

实施时创建，且只把 `docs/lessons/stage04_learning_order.md` 作为新手入口：

1. `stage04_sft_data_and_chat_template.md`；
2. `stage04_sft_objective_and_training.md`；
3. `stage04_lora_and_peft.md`；
4. `stage04_qlora_and_evaluation.md`。

每份讲义采用完整讲义模式：词汇表、旧问题、原理、公式、shape、真实代码调用链、手写实现、框架对照、正确/边界/失败测试、实验、局限；12–20 道思考题中的每道题后立即提供可折叠答案。

## 8. 16 站学习顺序

计划阶段先冻结顺序，实施完成后再把精确文件行号和函数名写入唯一入口：

1. Base、Pretrained、Instruct、Chat Model 的区别；
2. instruction / prompt-completion / conversational 三种数据格式；
3. system/user/assistant 角色与多轮边界；
4. Chat Template、BOS/EOS/PAD 与 generation prompt；
5. 从字符串到 token spans，再到 assistant mask；
6. Causal LM shift 与 response-only SFT loss；
7. padding、truncation、packing 和 loss denominator；
8. tiny batch 手算与 tiny overfit；
9. 全量微调的参数/optimizer/显存账；
10. LoRA 低秩假设、公式与 shape；
11. 手写注入、参数冻结、初始化与梯度；
12. Adapter 保存、加载、merge/unmerge；
13. PEFT 对照与开源 Base 适配；
14. 4-bit、NF4、double quantization 与 QLoRA 梯度路径；
15. BF16 LoRA/QLoRA 显存实验与冻结前后评测；
16. Qwen 等开放模型的多阶段后训练映射、结果分析与口述验收。

不能从源码目录随机开始，也不能把计划、实验报告和四份讲义作为并列入口。

## 9. 与当前开放模型技术的对应关系

Stage 4 不复刻厂商秘而不宣的数据配比和大规模算力，而是提取可验证的共同结构：

| 公开路线 | Stage 4 学什么 | 后移内容 |
|---|---|---|
| Qwen3 四阶段后训练 | 长 CoT cold-start SFT、thinking/non-thinking 数据混合为什么仍依赖监督微调 | reasoning RL 与 general RL 进入 Stage 5 |
| DeepSeek 当前领域专家路线 | 领域 SFT、Adapter 隔离和固定保留集为什么是专家培养基础 | on-policy distillation 进入 Stage 5 |
| PEFT/QLoRA 工程路线 | 低秩更新、全 Linear 注入、NF4/double quant 与内存账 | 多 Adapter 路由和复杂初始化只作观察 |
| TRL 当前 SFT 接口 | conversational mask、assistant-only loss、packing、DFT 方法入口 | DFT 只做 P2 梯度实验，不替代稳定 NLL 基线 |

这样既回答“最新模型为什么采用这些技术”，又不让追逐每个新缩写重新拉长主周期。

## 10. 两周排期

| 日程 | 主要任务 | 当日出口 |
|---|---|---|
| Day 1 | F0、依赖/许可证/显存/磁盘审计，冻结 revision | 模型/数据/资源 Manifest 草案 |
| Day 2 | Schema、Chat Template、assistant span | G4-A 单元测试 |
| Day 3 | Collator、labels、shifted SFT loss | 手算 batch 100% 对齐 |
| Day 4 | tiny SFT loop、overfit、full/assistant-only 实验 | G4-B 自动化证据 |
| Day 5 | 手写 LoRA、注入、参数账、梯度 | no-op/Base freeze/gradient 测试 |
| Day 6 | save/load、merge/unmerge、PEFT 对照 | G4-C 自动化证据 |
| Day 7 | QLoRA 原理、bitsandbytes 环境门、内存账 | 真实 4-bit forward/backward |
| Day 8 | 冻结评测、Qwen BF16/4-bit Base baseline | G4-D、Base 报告 |
| Day 9 | BF16 LoRA 正式有界短跑与相同评测 | G4-E 原始 Artifact |
| Day 10 | 分析、质量门、四份讲义、16 站导航、实验记录 | 自动化 G4 完成；转入 G4-L |

时间分配上限：实现与测试 45h、原理/讲义 30h、实验与分析 25h、审计/文档 10h、故障缓冲 10h。缓冲耗尽后停止扩张范围，不从 Stage 5 借时间。

## 11. 总体验收清单

### 自动化 G4

- [x] G4-A：数据、模板、spans、hash 和 split 可审计；
- [x] G4-B：assistant-only SFT loss、tiny overfit 和框架 batch 对照通过；
- [x] G4-C：手写 LoRA、PEFT 对照、Base freeze、merge 和 Artifact 通过；
- [x] G4-D：真实 QLoRA backward 与 BF16/4-bit 显存账完成；
- [x] G4-E：冻结 Base、唯一有界短跑、相同前后评测和报告完成；
- [x] Ruff format/lint、mypy strict、基础 pytest 与重依赖 smoke 分层通过；
- [x] 数据卡、模型卡、实验记录、成本和所有 fingerprint 完整；
- [x] 没有把短训练或单 seed 结果写成普遍能力结论。

### 完整 G4

- [x] 自动化 G4 全部通过；
- [x] `stage04_learning_order.md` 的 16 站完成；
- [x] 最小改动实验与即时问答完成；
- [x] 学习者能独立讲清 SFT mask、LoRA、QLoRA 与评测边界；
- [x] 用户明确确认 Stage 4 学习与测试完成。

## 12. 实施结果与下一决策点

自动化实施已完成，详细证据见 `docs/experiments/2026-07-28_stage04_implementation.md`。正式 BF16 LoRA 在第 129 步达到 100,293 assistant tokens：held-out loss 改善，但 strict generation 仍为 0/16，retention 略退化，字符级重复审计确认生成重复恶化。2026-07-28 用户明确确认已完成 Stage 4 学习者验收，G4 全部门关闭。Stage 5 随后获批并完成自动化实现。
