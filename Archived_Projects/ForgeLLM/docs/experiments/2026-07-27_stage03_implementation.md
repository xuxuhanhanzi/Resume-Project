# 2026-07-27 Stage 3 预训练闭环实现与实验记录

## 1. 目标与结论边界

目标：在本地 RTX 4070 Laptop 8GB 上证明一个小模型预训练系统具备确定性数据、正确更新、完整 checkpoint、真实中断恢复、现代方法短实验和固定预算评估。

结论边界：所有结果只支持“小规模预训练闭环正确且可复现”，不支持通用模型质量或方法普遍优越性声明。

## 2. 数据冻结

- 来源：[roneneldan/TinyStories 官方数据仓库](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-valid.txt)
- 研究背景：[TinyStories 论文](https://arxiv.org/abs/2305.07759)
- 许可证：CDLA-Sharing-1.0
- 原始文件：19,447,282 bytes
- 固定 SHA-256：`94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`
- 原始文档：21,990
- 重划分：train 19,747 / validation 1,128 / test 1,115
- 拒绝：0

上游文件名是 `valid`，但这里只把它作为 19.4MB 原始教学子集，并重新执行确定性文档级 90/5/5 切分。因此这里的 validation 不是官方 TinyStories validation benchmark。

| Split | Bytes | SHA-256 |
|---|---:|---|
| train | 19,850,060 | `9f37dbad...5d3885e` |
| validation | 1,139,060 | `1a7c7ac3...7ebba00a` |
| test | 1,117,254 | `518b4668...568eb8d` |

Tokenizer 固定为 Stage 1 320-vocab byte-BPE，fingerprint `82ccbedc...0305223`。不训练新 Tokenizer，不调整词表规模。

## 3. 新增实现

- 文档独立 BPE、EOS、target-exact packing、byte 计数；
- epoch+position 确定性 batch cursor；
- 源数据/Tokenizer/sequence length 指纹绑定 token cache；
- heap-based BPE encode，与旧逐 rank replay 完全一致；
- AdamW 分组、Muon+AdamW 混合 optimizer、warmup/cosine；
- gradient accumulation、finite check、gradient clipping；
- FP32/BF16/FP16 autocast 与 GradScaler；
- validation、PPL、BPB、tokens/s、峰值 allocated memory；
- MTP auxiliary head 进入真实 optimizer/checkpoint；
- 原子 checkpoint 与 model/optimizer/scheduler/scaler/cursor/RNG 全状态恢复。

## 4. 固定运行入口

```powershell
.\.venv\Scripts\python.exe scripts\fetch_stage3_corpus.py
.\.venv\Scripts\python.exe scripts\prepare_stage3_corpus.py

.\.venv\Scripts\python.exe scripts\stage3_train.py `
  --config configs\training\stage3_bounded.toml `
  --tokenizer artifacts\stage01_tokenizer_candidate\model\tokenizer.json `
  --train data\processed\stage3_tinystories\train.jsonl `
  --validation data\processed\stage3_tinystories\validation.jsonl `
  --output-dir artifacts\stage03\bounded_1m `
  --stop-after-step 125

.\.venv\Scripts\python.exe scripts\stage3_train.py `
  --config configs\training\stage3_bounded.toml `
  --tokenizer artifacts\stage01_tokenizer_candidate\model\tokenizer.json `
  --train data\processed\stage3_tinystories\train.jsonl `
  --validation data\processed\stage3_tinystories\validation.jsonl `
  --output-dir artifacts\stage03\bounded_1m `
  --resume
```

## 5. Correctness 与恢复结果

CPU/FP32 exact-resume 中，loss sequence、model、optimizer、scheduler、trainer state 和 next batch indices/tokens 全部 exact。

真实 CUDA 中断：第 125 步、508,000 target tokens 停止；新进程从 step 125 加载；下一条 train step 为 126；最终 JSONL step 1–247 连续。

## 6. Qualification

配置：5,361,856 参数、sequence 128、micro batch 4、accumulation 2、BF16、AdamW、100 steps。

| 指标 | 结果 |
|---|---:|
| target tokens | 101,600 |
| 首 10 步平均 loss | 5.1521 |
| 末 10 步平均 loss | 2.7290 |
| final validation loss | 2.6289 |
| PPL / BPB | 13.8584 / 3.2767 |
| mean tokens/s | 12,808.8 |
| peak allocated memory | 171.6 MiB |

无 NaN/Inf/OOM，资格门通过。

## 7. Muon 与 MTP 单变量实验

固定 tiny 模型、正式数据、FP32、100 steps；每个 variant 使用 seeds 41/42/43。

| Variant | 主变量 | Validation loss mean | stdev |
|---|---|---:|---:|
| AdamW single | baseline | 3.4075 | 0.0126 |
| Muon single | optimizer | 3.1926 | 0.0068 |
| AdamW MTP | objective/head | 3.4137 | 0.0080 |

解释：在当前短程冻结设置下，Muon validation loss 更低；MTP 未改善主 validation loss。未调参、未长训，不能推广为方法排名。

## 8. 1M-token 有界运行

配置 fingerprint：`132b77a0...65ef64`。

| 指标 | 结果 |
|---|---:|
| 完成 step / target tokens | 247 / 1,003,808 |
| stop reason | `max_tokens` |
| 首 / 末 10 步平均 loss | 5.3847 / 1.5995 |
| final validation loss | 1.6245 |
| PPL / BPB | 5.0759 / 2.0543 |
| mean tokens/s | 24,811.1 |
| peak allocated memory | 235.9 MiB |
| accumulated training wall time | 43.70 s |
| 日志连续 | 是，1–247 |

Checkpoint 大小 64,450,806 bytes，SHA-256 `d9e363fb...70491db`。固定 prompt 只出现简单局部语法并有明显重复，不作生成质量声明。

## 9. 失败与修复

### F1：BPE 启动开销

第一次启动时，逐 merge 全文 replay 的时间超过短训练。实现 heap-based ranked merge 并逐文本对齐旧 naive replay；加入指纹 token cache，token IDs 不变。

### F2：CUDA RNG 恢复 device 错误

`torch.load(map_location=cuda)` 将 RNG ByteTensor 映射到 GPU，而 `set_rng_state_all` 要求 CPU ByteTensor。修复为验证后逐个 `.cpu()`；重新启动后成功从 step 125 继续。

该失败保留，因为它改变了 checkpoint 实现，并证明真实恢复测试不可省略。

## 10. 状态

G3 自动化工程门通过。完整 Stage 3 仍等待学习者完成 16 站讲义、代码追踪、最小实验和口述验收。
