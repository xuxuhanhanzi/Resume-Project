# ForgeLLM 模型优先学习路径

> 2026-07-28 修订：当前唯一详细主计划为 [`model_first_21_week_learning_plan.md`](model_first_21_week_learning_plan.md)。本文件只提供高层导航。

这条路径服务于 LLM 算法/训练岗位。学习时间优先投入 Tokenizer、PyTorch 模型构造、预训练、自定义 C++/CUDA 算子、后训练和模型评测；Python 工程、CLI、CI、日志和普通数据治理降级为按需支撑知识。

## Stage 0：模型优先路线切换（0.5 周以内）

- Day 1–3 学习与测试完成；
- Day 4–14 旧课程降级为查阅资料；
- 完成质量门和支撑知识执行卡；
- 下一步直接进入 Tokenizer。

## Stage 1：数据最低闭环与 Tokenizer（2 周）

- 当前状态：G1-A、G1-B 实现与学习者验收均已完成；
- 新手唯一学习入口：`docs/lessons/stage01_learning_order.md`；不要把阶段计划、规格、源码和实验记录作为并列起点；
- Unicode/byte/token、特殊 Token 和 BPE；
- 手写确定性 BPE、保存/加载、encode/decode；
- pre-tokenization、BPE-dropout、Unigram、Picky BPE 与 SuperBPE；
- 特殊 Token policy、byte offset 与简化 entropy patching；
- 工程 Tokenizer 与固定小型夹具评测，不进行词表规模优化；
- round-trip、fertility、bytes/token、unknown rate 和子集指标。

## Stage 2：Transformer、现代架构与模型系统（7 周）

- 当前状态：G2-Core、G2-Arch、G2-Systems 与学习者验收全部完成；
- 新手唯一学习入口：`docs/lessons/stage02_learning_order.md`；
- Tensor、Autograd、Module、dtype/device 和数值稳定性；
- Embedding、RMSNorm、RoPE、SwiGLU、MHA/GQA/QK-Norm、残差；
- Causal LM、loss shift、生成与 KV Cache；
- shape、因果性、梯度、数值对照和 Tiny Overfit；
- MLA、MoBA、CSA/HCA-lite、DeltaNet/Hybrid、MoE、mHC、MTP 与 Muon 的教学 reference；
- SDPA/online-softmax、`torch.compile`、INT8、DDP；
- `silu_mul` Python→C++ CPU→CUDA→Autograd→opcheck→benchmark；
- Kimi K3 已按官方仓库/报告进入 Stage 5；DeepSeek V4 仍只做待核验登记。所有新版本只使用官方来源更新。

## Stage 3：预训练闭环（3 周）

- 当前状态：自动化实现、方法实验、1M-token 有界运行与学习者验收均已完成；
- 新手唯一入口：`docs/lessons/stage03_learning_order.md`；
- 文档级 split、target-exact packing、确定性 cursor 和指纹 token cache；
- AdamW、调度器、gradient accumulation、混合精度；
- validation、Checkpoint/Resume、RNG 和数据位置恢复；
- Muon/AdamW 与 MTP/single-token 独立短实验；
- 5.36M 小模型、1M target-token 正式短跑并真实中断恢复；
- loss/PPL/BPB、吞吐、显存和成本。

## Stage 4：监督式后训练基础（2 周）

- 当前状态：自动化实现、正式短跑、完整讲义与学习者验收均已完成；
- 历史学习入口：`docs/lessons/stage04_learning_order.md`；
- 原 C++/CUDA Stage 4 已并入 Stage 2 并完成，不再重复；
- conversational Schema、Chat Template、assistant-only Label Mask 与 SFT loss；
- 自研 5.36M 模型算法轨：tiny overfit、full/response-only 和手写 LoRA；
- Qwen3-0.6B-Base 框架轨：PEFT、BF16 LoRA、QLoRA 与 Adapter Artifact；
- 固定前后评测：held-out loss、constraint exact match、格式、长度/重复、保留能力、吞吐和显存；
- 正式运行限于 100k assistant tokens、500 steps 或 45 分钟先到即停，不做参数搜索。

## Stage 5：偏好优化与在线 RL（3 周）

- 当前状态：自动化实现、正式 DPO/一步 GRPO、完整讲义与学习者验收均已完成；
- 历史学习入口：`docs/lessons/stage05_learning_order.md`；
- Reward Model、Bradley–Terry、DPO 与长度/数据偏差；
- MDP、Policy Gradient、PPO、GRPO 和 reward hacking；
- 满足稳定基线后再学习 DAPO、领域专家培养与 on-policy distillation；
- 大模型 RL 正式运行由 Verifier、显存与预算共同触发。

## Stage 6：综合评测与最终验收（2 周）

- 当前状态：自动化实现、正式质量/系统评测和最终报告均已完成；G6-L 待学习者完成；
- 当前唯一学习入口：`docs/lessons/stage06_learning_order.md`；
- Tokenizer、预训练、SFT、偏好/RL 与系统指标统一；
- 固定 Baseline、数据、分母和硬件；
- 检查遗忘、污染、长度偏差和 reward hacking；
- 完成 Model Card、Evaluation Card、成本表、盲评包和结论边界；
- 不设综合总分；M3 不与 Qwen 跨 Tokenizer 比 PPL；Q3 只验收流程。

## Buffer：补弱与作品集（2 周）

- 只修复未过门禁或证据缺口；
- 整理可复现命令、图表、报告和简历表述；
- 服务、部署和 RepoPilot 在模型主线完成后另行规划。

## 学习方法

每个知识点使用同一套闭环：

1. **预习（30–60 分钟）：** 写下输入、输出、核心不变量和三个疑问；
2. **最小实现：** 不追求性能，只让原理可执行；
3. **测试：** 至少包含正常、边界、失败和确定性测试；
4. **框架对照：** 与 PyTorch/Hugging Face 等成熟实现做数值或行为比较；
5. **实验：** 固定基线，只改变一个主变量；
6. **讲解：** 用 5 分钟口述“为什么、怎么做、证据、局限”；
7. **归档：** 更新 README、实验记录、卡片和进度台账。

掌握标准不是“看完课程”，而是同时满足：能解释、能实现、能测试、能测量、能复现、能说明边界。

## 求职能力映射

| 项目证据 | 对应能力 | 更匹配的岗位 |
|---|---|---|
| 数据、Tokenizer、从零模型、预训练 | LLM 原理与训练闭环 | LLM 算法/训练工程师 |
| 性能 Profile、DDP/FSDP2、恢复 | 训练系统与性能分析 | LLM Systems / 训练平台 |
| LoRA/SFT/DPO 受控评测 | 后训练与评测 | Fine-tuning / Research Engineer |
| vLLM、API、负载测试、Compose | 推理与 AI 后端 | 推理工程师 / AI Backend |
| RepoPilot 检索、Agent、沙箱、Benchmark | Agent 与应用工程 | Agent / Applied AI Engineer |

所有简历数字必须等实验完成后再填写，并能追溯到 Run ID、配置、Commit、硬件和报告。
