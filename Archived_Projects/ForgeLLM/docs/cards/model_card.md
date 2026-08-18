# Model Card：Stage 3 5.36M 教学预训练 Checkpoint

## 模型与训练

- Decoder-only Transformer，5,361,856 参数；
- vocab 320、d_model 256、7 层、8 query heads、2 KV heads；
- RMSNorm、RoPE、SwiGLU、GQA、QK-Norm、SDPA；
- context 128、BF16、AdamW、single-token causal objective；
- 247 optimizer steps、1,003,808 target tokens；
- 第 125 步主动中断并从完整 checkpoint 恢复；
- 外部付费成本：0 USD。

## 数据与结果

数据见 `docs/cards/data_card.md`。

- final validation loss 1.6245；PPL 5.0759；BPB 2.0543；
- 峰值 PyTorch allocated CUDA memory 235.9 MiB；
- checkpoint SHA-256：`d9e363fb5d5ce4ccb5cbae2cc005f680500b62932f789ec1e1a34f77570491db`。

## 适用范围与限制

仅用于学习训练循环、优化器、混合精度、checkpoint/resume、指标和实验设计。不适合作为聊天、知识问答、代码、安全或生产模型。

参数、数据和 token 预算极小，数据分布单一，320 词表压缩效率有限；未做事实性、安全、偏见、记忆和成员推断评测；定性生成有重复和不完整语句。不得用本模型支持 Muon、MTP 或大模型架构的普遍效果声明。

---

# Model Card：Stage 4 Qwen3-0.6B BF16 LoRA Adapter

## Base 与 Adapter

- Base：`Qwen/Qwen3-0.6B-Base`；
- revision：`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；
- Base 许可证：Apache-2.0；
- 总参数：606,142,464；
- LoRA：all-linear、r=16、alpha=32、dropout=0.05、bias=none；
- 可训练参数：10,092,544（1.665%）；
- Adapter safetensors：40,422,168 bytes；
- Adapter SHA-256：`d301bea28462ecdfbecfe248578c0367cafc7fd7a374da712ef27e8dc441e9ff`。

## 训练

- BF16 Base + PEFT LoRA，AdamW，LR 2e-4；
- max length 512、micro batch 1、gradient accumulation 8；
- assistant-only CE，真实 assistant target token 分母；
- 129 optimizer steps、100,293 assistant tokens；
- 3% assistant-token warmup + cosine，最终学习率 0；
- 训练耗时 375.0 秒，267.5 assistant tokens/s；
- PyTorch peak allocated 1,939,151,360 bytes；
- stop reason：`max_assistant_tokens`；
- Base gradients 全程缺席；外部付费成本 0 USD。

## 结果

- SmolTalk validation assistant loss：1.5660 → 1.1500；
- correctness teacher-forced loss：3.1000 → 1.8033；
- strict constraint generation：0/16 → 0/16；
- TinyStories retention loss：1.7770 → 1.8098；
- 补充字符 8-gram 重复率：0.2080 → 0.4446。

因此只证明固定小数据/预算下的 SFT 拟合和 Adapter 工程链路；没有证明通用指令遵循、聊天质量、推理、安全或生产可用性。生成经常输出正确开头后继续解释或退化重复，严格任务门未通过，并观察到轻微 retention 回退。

## 产物边界

`adapter/` 只含 Adapter 与 tokenizer 配置，加载必须使用上述精确 Base revision。没有保存可完整恢复 Adam/RNG/data cursor 的训练 checkpoint；不得声称 Stage 4 支持中断续训。合规 v2 报告 SHA-256 为 `6e6ebbdb7a81412578cc2a428ee7f4be1a815484398890d621ed38251e4fc003`；补充 repetition audit 不覆盖原报告。恒定 LR 的 v1 作为实施偏差证据保留，不是本卡主模型。

---

# Model Card：Stage 5 Qwen3-0.6B DPO Adapter v2

## 身份与训练

- Base/revision：与 Stage 4 相同的 `Qwen/Qwen3-0.6B-Base@da87bfb...c0cd`；
- reference：冻结的 Stage 4 SFT Adapter，SHA-256 `d301bea2...e9ff`；
- 初始 policy：与 reference 完全相同；所有 Dropout 关闭；
- DPO：standard reverse-KL sigmoid、beta=0.1、response log-prob sum；
- 50 optimizer steps、200 pairs、4,214 response tokens、129.26 秒；
- Base gradients absent；peak allocated 1,638,436,352 bytes；
- final Adapter SHA-256：`01f46228...e019`；
- report SHA-256：`dbf34452...ba0`。

## 结果与验收状态

- held-out preference accuracy：0.0 → 1.0；
- mean DPO margin：0.0 → 74.6852；
- strict preference-task generation：0/16 → 0/16；
- Stage 4 assistant validation loss：1.1500 → 1.3584；
- preference generation character 8-gram repetition：0.0581 → 0.5528。

该 Adapter 只证明冻结小数据上的 DPO 目标与工程链路；行为验收未通过，不适合聊天、推理、安全或生产。极大 margin 与退化指标提示代理目标过拟合/偏移风险。Dropout 模式不一致的 v1 原样保留，但不是本卡模型。

## GRPO v2 说明

Stage 5 另保存一个 1-step GRPO Adapter，SHA-256 `3e0d21ba...01f`。它由 4×4 共 16 条真实 rollout 产生，首步 KL/clip fraction 均为 0，只用于证明 on-policy 数据链与 Adapter 梯度；没有进行能力前后排名，不应作为“GRPO 改善模型”的模型卡。

---

# Model Card 补充：Stage 6 冻结综合评测 v5

## 评测身份

- Q0：Qwen3-0.6B Base；
- Q1：Stage 4 SFT Adapter v2；
- Q2：Stage 5 DPO Adapter v2；
- Q3：Stage 5 GRPO one-step，仅审计 pipeline；
- M3：Stage 3 自研 DecoderLM，独立 Tokenizer/套件；
- 主运行指纹：`d2a26e41768f751e23dd8138ac2279744611b4a4f6b1c1ae67d8a0eb6ce013a1`；
- Stage 6 可执行源码/配置聚合 SHA-256：`0754b0c62cb7fcb43857e569a2b7b17c2a29a76fc18c813112d4380c22bf3516`；
- 64 个冻结案例，每个 Q0/Q1/Q2 各生成一次，greedy、32-token 上限。

## 质量结果

| 指标 | Q0 Base | Q1 SFT | Q2 DPO |
|---|---:|---:|---:|
| 原题 strict pass | 0/32 | 0/32 | 0/32 |
| 95% Wilson 上界 | 0.1072 | 0.1072 | 0.1072 |
| correct-prefix-but-extra（全部 64） | 0.0000 | 0.4531 | 0.6250 |
| mean character 8-gram repetition | 0.1156 | 0.1716 | 0.4426 |
| truncation rate | 1.0000 | 1.0000 | 1.0000 |
| expected-response loss/token | 1.9255 | 0.5414 | 0.5420 |
| expected-response BPB | 1.5975 | 0.4492 | 0.4496 |
| retention loss/token | 1.8411 | 1.8829 | 2.0124 |
| retention / Q0 | 1.0000 | 1.0227 | 1.0930 |
| held-out pair accuracy（64） | 0.7656 | 0.9844 | 1.0000 |
| mean chosen−rejected log-prob | 11.8437 | 17.2949 | 93.9996 |

Q1 显著降低冻结答案的 teacher-forced NLL，并经常先生成正确前缀，但没有一次在 32-token 协议下正确停止；Q2 把 pair accuracy 推到 1.0、margin 推得极大，却没有改善 strict generation，并把字符重复从 Q1 的 0.1716 恶化到 0.4426。两个成对 strict comparison 的差值和 95% Bootstrap 区间均为 `[0, 0]`，因为三者都是 0/32。

## 验收结论

Q0/Q1/Q2 都未通过 correctness 门，因此状态均不能写成“model behavior accepted”。Q2 还未通过 stability 门。Q3 仍只证明 4×4 rollout 与一步 on-policy 更新链可运行，不参加能力排名。M3 在自己套件上的 loss/PPL/BPB 为 1.6245/5.0759/2.0543；Tokenizer 与协议不同，禁止同 Q0–Q2 横向比较 PPL。

## 数据与结论边界

污染审计发现 0 个精确重合和 16 个同模板 near-match 候选；这限制了向开放任务外推。评测集规模小、模板化、主要为英语严格格式任务，不覆盖开放知识、安全、多语言、代码执行或长推理。完整逐题输出、Manifest、盲评包和统计结果位于 `artifacts/stage06/quality_v5/`。
