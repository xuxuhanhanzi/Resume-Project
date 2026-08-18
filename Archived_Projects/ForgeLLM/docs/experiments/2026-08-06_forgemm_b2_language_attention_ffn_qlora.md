# ForgeMM B2 语言注意力 + FFN QLoRA 实验记录（2026-08-06）

## 1. 目标

在 B1 的语言注意力 QLoRA 基础上，只增加语言 FFN 的 `gate_proj/up_proj/down_proj` LoRA，验证更大的语言侧适配容量能否改善 ChartQA 图表问答。

## 2. 假设与主变量

假设：图表问答包含数值、比较和关系变换，仅调整注意力的信息路由可能不够；允许 FFN 的非线性变换参与适配，可能更适合该任务。

唯一主变量：

- B1 目标：`q_proj/k_proj/v_proj/o_proj`。
- B2 目标：B1 四个目标加 `gate_proj/up_proj/down_proj`。

数据、抽样顺序、提示词、Answer-only 标签、量化、LoRA rank/alpha、学习率、训练步数、验证样本与评测规则均保持不变。

## 3. 环境与正式设置

- GPU：RTX 4070 Laptop GPU，8 GB。
- 模型：`D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct`。
- 数据：`D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset`。
- 训练池：`train_human` 前 512 条；种子 42；实际处理 200 条不同样本。
- 验证：固定 `val_human` 前 50 条。
- QLoRA：NF4 4-bit、FP16 compute、rank 8、alpha 16、dropout 0。
- 优化：AdamW，峰值学习率 `2e-4`，5 步 warmup + cosine decay。
- 训练：50 个优化步，每步梯度累积 4 次。

## 4. 失败的首次 smoke 与修复

第一次 B2 smoke 在训练前被边界断言中止。PEFT 使用短模块名 `gate_proj` 进行匹配时，不仅命中语言 FFN，也命中了视觉编码器中的同名模块：

```text
base_model.model.model.visual.blocks.0.mlp.gate_proj.lora_A.default.weight
```

这会破坏“只扩大语言侧 LoRA”的单变量实验。修复方式是改用完整路径正则，只匹配：

```text
language_model.layers.<n>.self_attn.{q_proj,k_proj,v_proj,o_proj}
language_model.layers.<n>.mlp.{gate_proj,up_proj,down_proj}
```

新增测试同时验证语言目标可以命中、视觉同名模块不能命中。失败目录保留在：

`artifacts/forgemm/b2_language_attention_ffn_qlora_smoke_20260806/`

## 5. 通过的 smoke

命令：

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_b1_language_qlora_smoke.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\b2_language_attention_ffn_qlora_smoke_v2_20260806" `
  --target-profile attention-ffn --train-limit 8 --eval-limit 4 `
  --max-steps 2 --gradient-accumulation-steps 1 --checkpoint-interval 2 `
  --rank 8 --alpha 16 --learning-rate 0.0002 --seed 42
```

结果：

- 可训练参数 14,966,784，约占 4-bit-aware 总参数的 0.3970%。
- 可训练参数张量名称 504 个，全部属于指定语言层 LoRA。
- 252/252 个 LoRA-B 张量发生非零更新。
- 两步 loss 为 `2.041615`、`0.003483`，均为有限值。
- 峰值已分配显存约 3.930 GiB，无 OOM。
- 保存后的 Adapter 正则配置复核通过。

## 6. 正式运行命令

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_b1_language_qlora_smoke.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\b2_language_attention_ffn_qlora_bounded50_20260806" `
  --target-profile attention-ffn --train-limit 512 --eval-limit 50 `
  --max-steps 50 --gradient-accumulation-steps 4 --warmup-steps 5 `
  --eval-interval 25 --checkpoint-interval 25 `
  --rank 8 --alpha 16 --learning-rate 0.0002 --seed 42
```

## 7. B1/B2 结果

| 方法 | 可训练参数 | 第 0 步严格 | 第 25 步严格 | 第 50 步严格 | 第 50 步宽松 |
|---|---:|---:|---:|---:|---:|
| B1：Attention QLoRA | 3,686,400 | 46% | 42% | 42% | 50% |
| B2：Attention + FFN QLoRA | 14,966,784 | 46% | 42% | 40% | 48% |

B2 的其他结果：

- 宽松准确率：第 0/25/50 步分别为 54%/50%/48%。
- 前 10 步平均 loss：`0.3814`；后 10 步平均 loss：`0.3807`。
- 252/252 个目标 LoRA-B 张量均为非零。
- 峰值已分配显存约 7.986 GiB，未 OOM，但几乎触及 8 GB 上限。
- 最佳严格准确率仍是第 0 步的 46%，因此没有训练后 Adapter 能替代初始模型；修正后的模型选择逻辑正确记录 `best_optimizer_step = 0`，且不生成伪“最佳 Adapter”。

## 8. 样本级变化

相对第 0 步：

- 修正 1 条原本错误的样本。
- 使 4 条原本正确的样本退化。
- 19 条保持正确，26 条保持错误。

B1 与 B2 都修正了索引 5，但都使索引 7、8、47 退化。B2 另外使索引 30 从正确答案 `12` 变成错误答案 `57`，所以 B2 比 B1 少答对 1 条，没有任何 B2 独有的新增正确样本。

## 9. 结论与声明边界

B2 工程实现通过，但实验假设没有得到当前证据支持。将可训练参数扩大约 4.06 倍没有改善 50 条验证集表现，反而从 B1 的 42% 降到 40%；训练 loss 也没有呈现足以支持更好拟合的稳定下降。不能声称 FFN QLoRA 提升了 ChartQA 能力。

50 条验证集每条对应 2 个百分点，因此 B1 与 B2 的 2 个百分点差异本身不具备强统计结论；但 B2 同时占用更多参数和接近全部显存，当前没有继续选择它的工程理由。

## 10. 下一步判断

暂不继续扩大 LoRA 到视觉编码器。B1 和 B2 使用同一批 200 条 Answer-only 样本都出现“修正少数错误、破坏更多原本正确答案”的现象，更值得优先检验的是训练配方和数据规模，而不是继续增加可训练模块。

建议下一项保持 B1 的较小注意力 LoRA 边界，只改变一个因素：将峰值学习率从 `2e-4` 降低到 `5e-5`。这可以直接检验当前退化是否由小数据下更新过强导致，并且显存安全、成本低于 B2。若降低学习率仍无改善，再进入更大训练数据规模与完整验证集实验。

## 11. 产物

- 计划：`docs/multimodal/forgemm_b2_language_attention_ffn_qlora_plan.md`
- 正式报告：`artifacts/forgemm/b2_language_attention_ffn_qlora_bounded50_20260806/report.json`
- 第 25/50 步检查点：`artifacts/forgemm/b2_language_attention_ffn_qlora_bounded50_20260806/checkpoints/`
- 最终 Adapter：`artifacts/forgemm/b2_language_attention_ffn_qlora_bounded50_20260806/adapter/`
- 通过的 smoke：`artifacts/forgemm/b2_language_attention_ffn_qlora_smoke_v2_20260806/`
