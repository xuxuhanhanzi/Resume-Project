# Stage 3 唯一学习入口：预训练闭环 16 站

## 使用方式

请严格按站点顺序学习。每一站均执行“读指定讲义章节→追踪指定函数→完成一个最小实验→立即回答题目”的闭环。不要先通读整个 `training/` 目录，也不要把实验报告当作入门讲义。

本阶段所有思考题都在对应知识点之后立即给出答案，不设独立答案区。

## 开始前只记住一句话

预训练不是“调用 `loss.backward()` 很多次”，而是一条需要保持语义连续的状态机：

```text
冻结数据和 Tokenizer
→ 确定性 batch
→ 前向/反向/更新
→ validation
→ 保存全部状态
→ 中断后从同一个 next batch 继续
→ 在固定预算内解释指标
```

## 术语预热

| 术语 | 新手解释 |
|---|---|
| target token | 真正被模型预测并计入 loss 的 token |
| packing | 把多个文档 token 拼成固定长度训练窗口 |
| cursor | 数据流下一次应从哪里继续的位置 |
| optimizer state | 如 Adam 的动量，不是模型权重 |
| scheduler | 根据训练进度改变学习率的规则 |
| autocast | 在安全范围内自动选择较低精度算子 |
| loss scaling | 放大 FP16 loss，减少过小梯度下溢 |
| checkpoint | 可让完整训练状态继续运行的快照 |
| PPL | `exp(平均 token NLL)`，只宜在相同 Tokenizer 下比较 |
| BPB | 每个原始 UTF-8 byte 需要的平均 bit，更适合跨 Tokenizer 对照 |

## 16 站学习路线

| 站 | 主题 | 先读 | 再追踪代码 | 必做验收 |
|---|---|---|---|---|
| 1 | 文本到 token stream | 数据讲义 1–2 | `read_training_documents`、`from_documents` | 解释 EOS 位置 |
| 2 | 文档切分与泄漏 | 数据讲义 3 | `prepare_stage3_corpus.py` | 验证 split 文档 ID 不交叉 |
| 3 | packing 与 label shift | 数据讲义 4–5 | `PackedTokenDataset.__getitem__`、`next_token_loss` | 手画长度 5 的 input/target |
| 4 | 确定性 batch cursor | 数据讲义 6 | `DeterministicBatchStream` | 保存后复现下一批 |
| 5 | token cache 与指纹 | 数据讲义 7 | `load_or_create_packed_jsonl` | 说出三个绑定项 |
| 6 | AdamW 参数分组 | 优化讲义 1–2 | `partition_parameters`、`build_optimizer` | 列出 decay/no-decay |
| 7 | warmup/cosine/裁剪 | 优化讲义 3–4 | `WarmupCosineScheduler` | 手算前两次 LR |
| 8 | 最小训练 step | 优化讲义 5 | `Trainer.train_step` | 标注更新顺序 |
| 9 | 梯度累积 | 优化讲义 6 | accumulation 集成测试 | 对齐大 batch |
| 10 | FP32/BF16/FP16 | 优化讲义 7 | `_autocast_dtype`、`GradScaler` | 解释 BF16 为何通常不需 scaler |
| 11 | validation/PPL/BPB | 数据讲义 8、优化讲义 8 | `Trainer.validate`、`metrics.py` | 检查状态不变 |
| 12 | checkpoint schema | 恢复讲义 1–3 | `_checkpoint_payload` | 不看代码列全状态 |
| 13 | 原子保存与恢复 | 恢复讲义 4–6 | `save_checkpoint_atomic`、`load_checkpoint` | 复现 exact-resume |
| 14 | 故障诊断 | 恢复讲义 7 | `_check_gradients_finite` | 注入 NaN 并定位 |
| 15 | Muon 与 MTP | 现代方法讲义 1–4 | `Muon`、`MultiTokenPredictionHead`、method lab | 分别指出主变量 |
| 16 | 有界正式训练 | 现代方法讲义 5–7 | `stage3_train.py`、正式报告 | 作结论边界口述 |

## 文件阅读顺序

只按以下顺序打开文件：

1. `docs/lessons/stage03_pretraining_data_and_objective.md`
2. `src/forgellm/training/data.py`
3. `src/forgellm/model/decoder.py` 中的 `next_token_loss`
4. `docs/lessons/stage03_optimization_and_precision.md`
5. `src/forgellm/training/optimization.py`
6. `src/forgellm/training/trainer.py` 的 `train_step` 和 `validate`
7. `docs/lessons/stage03_checkpoint_and_recovery.md`
8. `src/forgellm/training/checkpoint.py`
9. `Trainer.save_checkpoint/load_checkpoint`
10. `docs/lessons/stage03_modern_methods_and_experiments.md`
11. `src/forgellm/model/optim.py`、`frontier_layers.py` 的 MTP 部分
12. 三个 Stage 3 脚本和实验报告

## 固定复现命令

先运行自动测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_training_*.py tests\integration\test_stage3_training.py
```

再运行 CPU 精确恢复：

```powershell
.\.venv\Scripts\python.exe scripts\stage3_resume_equivalence.py `
  --config configs\training\stage3_smoke.toml `
  --tokenizer artifacts\stage01_tokenizer_candidate\model\tokenizer.json `
  --train data\processed\stage3_tinystories\train.jsonl `
  --validation data\processed\stage3_tinystories\validation.jsonl `
  --output-dir artifacts\stage03_student\resume
```

正式 1M-token 结果已经存在，不要求重复烧卡。学习时优先复现 20-step smoke；只有你主动想验证恢复流程时，才新建一个短 run 目录。

## 最小改动练习

完成以下四项即可，不做超参数搜索：

1. 将 smoke 的 accumulation 从 1 改为 2，同时把 micro batch 减半，检查更新差异；
2. 把 validation batch 数从 2 改为 4，观察估计方差而不修改训练状态；
3. 将 `mtp_loss_weight` 从 0.1 改为 0.05，只运行 20 步并解释主 loss/总 loss；
4. 在第 7 步停止、恢复到第 12 步，检查日志 step 是否连续。

## 完成判定

只有以下四项都完成，才向进度台账确认学习者验收：

- 16 站讲义与即时问答完成；
- 能沿代码复述一个 optimizer step；
- CPU exact-resume 或等价的短恢复实验通过；
- 能对 AdamW/Muon/MTP 和正式 1M-token 报告作不越界解释。
