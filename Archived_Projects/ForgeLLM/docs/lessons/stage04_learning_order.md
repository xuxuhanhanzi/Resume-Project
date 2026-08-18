# Stage 4 唯一学习入口：SFT、LoRA 与 QLoRA 16 站

## 使用规则

Stage 4 的自动化实现已经完成，但“代码存在”不等于你已经掌握它。请严格按下表学习：每站先读指定讲义，再只追踪指定代码，随后完成最小检查。不要一开始通读 `post_training/`，也不要先看 PEFT 源码。

所有思考题都在知识点后立即附答案，不设独立答案文件。正式 100k-token 训练已经完成，不要求再次烧卡；学习时复现 CPU 单元测试、tiny qualification 和至多 2-step smoke 即可。

## 开始前只记住一条主线

```text
messages
→ 严格对话 Schema
→ Chat Template
→ input_ids / assistant_mask / labels
→ shifted assistant-only CE
→ 冻结 Base、只更新 LoRA
→ BF16 或 NF4 Base
→ 同口径训练前后评测
→ 报告能力改善、退化与指标缺陷
```

## 新手术语预热

| 术语 | 在本项目中的准确含义 |
|---|---|
| Base model | 只完成预训练、尚未针对对话指令训练的模型 |
| SFT | 用已知目标回答做监督微调；本阶段只监督 assistant 区域 |
| Chat Template | 把角色和内容确定性地序列化为模型 token 的协议 |
| label mask | 用 `-100` 标出不直接进入 CE 的 token 位置 |
| target token | shift 后真正进入 loss 分母的 assistant token |
| LoRA | 冻结原权重，用低秩矩阵 `B @ A` 表示增量 |
| Adapter | 可单独保存、加载或合并的 LoRA 参数集合 |
| PEFT | Hugging Face 的参数高效微调库；本项目用它做框架轨 |
| QLoRA | 冻结 4-bit Base，同时训练较高精度 LoRA；不是训练 4-bit Base |
| NF4 | 针对近似正态分布权重设计的 4-bit 表示 |
| teacher forcing | 评估/训练时将真实前缀喂给模型预测下一个真实 token |
| greedy generation | 每步选最大概率 token；行为评测不使用真实答案作前缀 |

## 16 站学习路线

| 站 | 主题 | 先读 | 代码追踪 | 必做验收 |
|---:|---|---|---|---|
| 1 | Base、SFT 与对话记录 | 讲义一 1–2 | `schema.py` 的两个 dataclass | 手写一条合法多轮记录 |
| 2 | 角色顺序与 fail-fast | 讲义一 3 | `InstructionRecord.__post_init__` | 解释 4 种非法顺序 |
| 3 | 数据来源、去重和切分 | 讲义一 4 | 两个 `prepare_stage4_*data.py` | 核对 manifest 哈希 |
| 4 | ChatML 分段序列化 | 讲义一 5 | `tokenize_qwen_record` | 画出 user/assistant 段 |
| 5 | assistant mask 与截断 | 讲义一 6 | `TokenizedConversation` | 逐 token 标 `-100` |
| 6 | shifted SFT loss | 讲义一 7 | `assistant_only_causal_loss` | 手算 4-token loss 位置 |
| 7 | Padding 与 Collator | 讲义二 1 | `collate_tokenized_conversations` | 验证 PAD 不入 loss |
| 8 | 真 token 分母的累积 | 讲义二 2–3 | `scale_gradients_by_token_count`、`train_bounded` | 解释为何不能除 micro-step 数 |
| 9 | tiny overfit 与 mask 对照 | 讲义二 4 | `method_lab.py`、两个 tiny 脚本 | 解读 3 seeds 负结果 |
| 10 | 训练前后评测 | 讲义二 5–7 | `hf_experiment.py` 的四个 evaluator | 区分 loss 与 generation |
| 11 | LoRA 数学与 shape | 讲义三 1–3 | `LoRALinear.forward` | 手算参数量与矩阵 shape |
| 12 | 注入、冻结和梯度 | 讲义三 4 | `inject_lora`、Base 梯度检查 | 证明 B 初始梯度非零 |
| 13 | Adapter 保存与 merge | 讲义三 5–6 | `save_adapter`、`merge/unmerge` | 解释为何 merge 后不能继续无记录训练 |
| 14 | PEFT 对照 | 讲义三 7 | `load_stage4_hf_stack` | 核对 trainable fraction |
| 15 | QLoRA 内存与梯度路由 | 讲义四 1–5 | `BitsAndBytesConfig` 与 k-bit prepare | 画 storage/compute/gradient 三路 |
| 16 | 正式实验与结论边界 | 讲义四 6–9 | 三份报告和 repetition audit | 完成五维口述验收 |

## 文件阅读顺序

只按以下顺序打开，不按目录字母顺序阅读：

1. `docs/lessons/stage04_data_template_and_loss.md`
2. `src/forgellm/post_training/schema.py`
3. `src/forgellm/post_training/chat_template.py`
4. `src/forgellm/post_training/adapters.py` 中两个 tokenize 函数
5. `src/forgellm/post_training/sft.py`
6. `docs/lessons/stage04_sft_training_and_evaluation.md`
7. `src/forgellm/post_training/collator.py`
8. `src/forgellm/post_training/method_lab.py`
9. `src/forgellm/post_training/hf_experiment.py`
10. `docs/lessons/stage04_lora_and_peft.md`
11. `src/forgellm/post_training/lora.py`
12. `src/forgellm/post_training/adapters.py` 的模型加载部分
13. `docs/lessons/stage04_qlora_and_experiment_analysis.md`
14. `configs/post_training/` 的四个 TOML
15. `docs/experiments/2026-07-28_stage04_implementation.md`
16. `artifacts/stage04/qwen_lora_bounded_v2/report.json` 与 `repetition_audit.json`

## 固定复现命令

先运行 Stage 4 自动测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_post_training_*.py `
  tests\integration\test_stage4_tiny_sft.py
```

查看 tiny qualification，不要覆盖已有证据：

```powershell
Get-Content -Raw -Encoding UTF8 `
  artifacts\stage04\tiny_qualification\report.json
```

若你要亲自运行一次新的 tiny 资格门，必须使用新输出路径：

```powershell
.\.venv\Scripts\python.exe scripts\stage4_tiny_qualification.py `
  --tokenizer artifacts\stage01_tokenizer_candidate\model\tokenizer.json `
  --train data\processed\stage4_correctness_v1\train.jsonl `
  --output artifacts\stage04_student\tiny_qualification\report.json
```

正式 100k-token 运行无需重复。2-step Qwen smoke 也只有在你想追踪 PEFT 调用链且显存空闲时才运行，并必须使用新目录。

## 四个最小改动练习

1. 新增一条三轮对话 fixture，逐 token 打印 `input_ids/labels/assistant_mask`，不得改模板。
2. 将一个 batch 的 padding 长度改成 8 的倍数，证明 supervised token 数不变。
3. 在手写 LoRA 中把 B 从零初始化临时改成随机初始化，只运行单元测试并解释 no-op 为什么失败；随后恢复代码。
4. 给重复审计构造一个无空格重复字符串，比较 word trigram 与 character 8-gram，不得用正式测试集选阈值。

## 学习者完成判定

以下项目都完成后，Stage 4 的 G4-L 才能由你确认关闭：

- 按顺序完成 16 站和即时问答；
- 能从一条 `messages` 记录复述到 loss 分母；
- 能手算 LoRA 参数量并解释 A/B 初始化的梯度次序；
- 能画出 QLoRA 的 4-bit storage、BF16 compute 和 Adapter gradient；
- 能同时报告正式实验的正结果、退化、失败行为和指标缺陷，不把低 loss 写成“模型已对齐”。
