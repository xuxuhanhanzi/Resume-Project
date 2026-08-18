# 2026-07-28 Stage 4 监督式后训练实施记录

## 1. 结论先行

G4-A～G4-E 自动化门已完成：严格对话 Schema、ChatML assistant mask、token-normalized SFT、手写 LoRA、PEFT、真实 NF4 QLoRA、冻结前后评测和唯一一次 100k-token BF16 LoRA 运行全部落地。G4-L 学习者门仍需用户按 16 站完成。

合规正式 v2 改善 held-out teacher-forced loss，但 16 条严格生成任务仍全部失败，retention 略退化，并在补充审计中发现字符级重复恶化。恒定 LR 的 v1 被识别为不符合冻结 scheduler 的实施偏差并原样保留；Stage 4 完成依据是方法和证据闭环，不是模型质量达标。

## 2. 冻结身份与环境

| 项目 | 冻结值 |
|---|---|
| Base | `Qwen/Qwen3-0.6B-Base` |
| Base revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| SmolTalk revision | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB |
| torch | 2.6.0+cu124 |
| transformers | 5.14.1 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| datasets | 5.0.0 |
| accelerate | 1.14.0 |
| trl | 1.8.0 |
| 外部费用 | 0 USD |

Transformers 5.14.1 要求 tokenizers 0.22.x，因此 Stage 1 当前联合依赖从 0.23.1 固定为 0.22.2；Stage 1 历史实验仍如实记录当时 0.23.1。

## 3. 实现范围

### G4-A 数据与模板

- `schema.py`：严格 Message/InstructionRecord、canonical hash、JSONL 读写、ID/内容重复拒绝；
- `correctness_data.py`：96 条项目原创确定性记录，64/16/16；
- `prepare_stage4_public_data.py`：SmolTalk 校验、1600 字符上限、精确去重、哈希排序、2048/256/256；
- `chat_template.py` 与 `adapters.py`：dependency-free 与 Qwen ChatML 两条分段模板；
- 截断导致零 shifted assistant target 时 fail-fast。

### G4-B SFT

- 右 padding collator，attention 与 loss mask 分开；
- shifted CE 明确返回 NLL sum、target token 数、accuracy；
- gradient accumulation 按真实 assistant tokens 归一化；
- tiny qualification 与 full-sequence/assistant-only/LoRA 三种子方法实验。

### G4-C LoRA/PEFT

- 手写 Linear 低秩分支、B 零初始化、Base freeze；
- 显式注入、参数计数、Adapter-only round-trip；
- merge/unmerge 数值回归；
- PEFT all-linear、初始化 no-op、Base gradient 缺席检查。

### G4-D QLoRA

- bitsandbytes NF4 + double quantization + BF16 compute；
- k-bit prepare 与 `use_reentrant=False` checkpoint；
- 真实 4-bit load/forward/backward/optimizer/save；
- 2-step 同协议和 20-step memory lab。

### G4-E 评测与正式运行

- assistant validation loss/accuracy；
- correctness teacher-forced loss；
- greedy strict verifier；
- TinyStories retention loss；
- word trigram、响应长度、参数/显存/吞吐；
- 发现 word trigram 盲区后新增不覆盖原报告的 character 8-gram audit。

## 4. 数据证据

| 数据 | 记录数 | 关键 SHA-256 |
|---|---:|---|
| correctness train/val/test | 64/16/16 | train `ca2197cd...db322` |
| SmolTalk frozen train/val/test | 2048/256/256 | train `fe86a098...34f5` |
| Stage 3 retention validation | 1128（评测取固定前缀） | `1a7c7ac3...a00a` |

SmolTalk 上游 train/test 为 34,424/1,812；过长拒绝为 4,358/222。冻结三个 split 内容 fingerprint 无交叉。

## 5. Tiny 证据

两记录 qualification：first loss 5.4090，160 步后 loss 0.0045867，assistant-token accuracy 1.0，门通过。

8 记录、80 步、3 seeds 方法实验中，full sequence 最终 loss 约 0.36–0.40，assistant only 约 0.58–0.63，LoRA 约 4.85–4.93。LoRA 在该随机 tiny Base 和预算下欠拟合；不继续调参。

第一次 tiny 运行使用 max length 128，被零 assistant supervision 检查拒绝。测得 fixture 最长 236 后，实验固定为 256。该失败改变了安全配置，保留在实施记录中。

## 6. 框架烟雾与 QLoRA

首次 Qwen BF16 下载阶段进程中断，只写入 `qwen_lora_smoke_v1/provenance.json`；未删除该失败目录。缓存完成后 `qwen_lora_smoke_v2` 成功。

| 实验 | steps/tokens | load peak | train peak | tokens/s | 结果边界 |
|---|---:|---:|---:|---:|---|
| BF16 LoRA smoke | 2 / 699 | 1,232,928,256 B | 1,647,464,960 B | 40.57 | 冷启动管线门 |
| NF4 QLoRA smoke | 2 / 699 | 1,161,793,536 B | 1,627,442,688 B | 34.28 | 同协议 4-bit 门 |
| BF16 LoRA memory | 20 / 14,864 | 1,232,928,256 B | 1,939,151,360 B | 272.40 | token scheduler 公平对照 |
| NF4 QLoRA memory v2 | 20 / 14,864 | 1,161,793,536 B | 1,929,152,000 B | 203.78 | token scheduler 公平对照 |

20-step QLoRA measured peak 比 BF16 低 9,999,360 bytes（约 9.5 MiB），但吞吐慢约 25%；差值远小于理论 Base 权重节省，不能宣传显著训练显存收益。QLoRA validation loss 1.6801→1.2970，4 条严格生成仍 0%。旧 QLoRA memory v1 使用恒定 LR，作为协议修订前证据保留，不参与最终公平表。

## 7. 正式 BF16 LoRA 与协议修正

首次 bounded v1 完成后，审计发现训练器遗漏了计划冻结的 3% warmup + cosine，实际为恒定 LR。没有覆盖或删除 v1；新增 token-progress scheduler、边界测试和 TOML 字段后，在全新目录执行 v2。v2 才是主证据。

停止条件为 100,000 assistant tokens、500 steps、2,700 秒训练时间先到即停。合规 v2 实际：

- 129 optimizer steps / 1,032 micro steps；
- 100,293 assistant tokens；
- 3% assistant-token warmup + cosine，final LR 0；
- 375.0 秒；267.45 assistant tokens/s；
- peak allocated 1,939,151,360 bytes；
- stop=`max_assistant_tokens`；
- Base gradients absent；
- 10,092,544 trainable / 606,142,464 total。

| 指标 | Before | After | 差异解释 |
|---|---:|---:|---|
| assistant val loss | 1.56595 | 1.15002 | SFT 分布拟合改善 |
| assistant accuracy | 0.64783 | 0.68026 | teacher-forced 小幅改善 |
| correctness TF loss | 3.09999 | 1.80335 | 正确 token 概率改善 |
| correctness exact/constraint | 0/16 | 0/16 | 行为门失败 |
| retention loss | 1.77702 | 1.80980 | 轻微遗忘 |
| word trigram repeat | 0.38416 | 0.07275 | 对无空格重复仍失真 |
| character 8-gram repeat | 0.20800 | 0.44459 | 补充审计确认退化 |

训练后多条输出以正确答案开头，却继续生成解释或异常循环，且均跑满 64 tokens。结论不得越过“短程单配置 SFT likelihood 改善，但严格指令行为未通过”。

## 8. Artifact

- 合规 v2 report SHA-256：`6e6ebbdb7a81412578cc2a428ee7f4be1a815484398890d621ed38251e4fc003`；
- v2 repetition audit SHA-256：`2bec4b49317dc8150fc4cb203f018d4f2ab1e2b5d72d573715dbada1112282a3`；
- v2 Adapter SHA-256：`d301bea28462ecdfbecfe248578c0367cafc7fd7a374da712ef27e8dc441e9ff`；
- Adapter safetensors 大小：40,422,168 bytes。

原始报告不可覆盖；补充审计作为独立文件保留。Stage 4 没有保存 optimizer/RNG/cursor 完整 checkpoint，不声明训练可恢复。

## 9. 全仓质量门

最终执行：

```powershell
$env:FORGELLM_TEST_EXTENSION='1'
.\.venv\Scripts\python.exe scripts\dev.py check
```

结果：125 个文件通过 Ruff format、Ruff lint 与 mypy strict；pytest `208/208 passed`。两条 warning 来自 Windows C++ 扩展编译环境的编译器版本探测和 setuptools 私有接口提示，不是测试失败。

## 10. 完成边界与下一步

自动化 G4-A/B/C/D/E 完成；完整讲义和 16 站入口已生成。用户仍需完成 G4-L。Stage 5 计划从 Stage 4 这个“loss 改善但行为失败”的 Adapter/评测出发，学习 RM、DPO、PG/PPO/GRPO 与当前可核验方法，而不是先假定偏好/RL 一定会修复问题。
