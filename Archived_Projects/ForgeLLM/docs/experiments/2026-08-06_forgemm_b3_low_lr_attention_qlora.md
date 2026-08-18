# ForgeMM B3 低学习率 Attention QLoRA 实验记录（2026-08-06）

## 1. 目标

检验 B1/B2 的验证性能退化是否主要由小数据条件下 `2e-4` 学习率过高、模型更新过强导致。

## 2. 实验变量

唯一主变量是峰值学习率：

- B1：Attention QLoRA，`2e-4`。
- B3：Attention QLoRA，`5e-5`。

模型、LoRA 目标、rank/alpha、Answer-only 标签、训练样本及顺序、优化步数、warmup、图像预算、提示词、验证集和评测规则均保持一致。

## 3. 环境与设置

- GPU：RTX 4070 Laptop GPU，8 GB。
- 模型：`D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct`。
- 数据：`D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset`。
- LoRA：仅语言 `q_proj/k_proj/v_proj/o_proj`，rank 8，alpha 16，dropout 0。
- 量化：NF4 4-bit，FP16 compute。
- 训练池：`train_human` 前 512 条，固定种子 42，实际处理相同顺序的 200 条样本。
- 训练：50 个优化步，梯度累积 4，5 步 warmup + cosine decay。
- 验证：固定 `val_human` 前 50 条，第 0/25/50 步执行。

## 4. Smoke

命令：

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_b1_language_qlora_smoke.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\b3_attention_qlora_lr5e5_smoke_20260806" `
  --target-profile attention --train-limit 8 --eval-limit 4 `
  --max-steps 2 --gradient-accumulation-steps 1 --checkpoint-interval 2 `
  --rank 8 --alpha 16 --learning-rate 0.00005 --seed 42
```

结果：

- 可训练参数 3,686,400，约占 4-bit-aware 总参数的 0.0981%。
- 144/144 个 LoRA-B 张量发生非零更新。
- 两步 loss 为 `2.041615`、`0.003413`，均为有限值。
- 峰值已分配显存约 3.804 GiB，无 OOM。
- 目标模块与保存 Adapter 配置复核通过。

## 5. 正式运行命令

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_b1_language_qlora_smoke.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\b3_attention_qlora_lr5e5_bounded50_20260806" `
  --target-profile attention --train-limit 512 --eval-limit 50 `
  --max-steps 50 --gradient-accumulation-steps 4 --warmup-steps 5 `
  --eval-interval 25 --checkpoint-interval 25 `
  --rank 8 --alpha 16 --learning-rate 0.00005 --seed 42
```

## 6. 总体结果

| 方法 | LoRA 范围 | 学习率 | 第 0 步严格 | 第 25 步严格 | 第 50 步严格 | 第 50 步宽松 |
|---|---|---:|---:|---:|---:|---:|
| B1 | Attention | `2e-4` | 46% | 42% | 42% | 50% |
| B2 | Attention + FFN | `2e-4` | 46% | 42% | 40% | 48% |
| B3 | Attention | `5e-5` | 46% | 44% | 44% | 52% |

B3 的宽松准确率在第 0/25/50 步分别为 54%/52%/52%。第 25 步和第 50 步结果相同，继续训练到 50 步没有恢复初始分数。

其他结果：

- 前 10 步平均训练 loss：`0.4177`。
- 后 10 步平均训练 loss：`0.3974`。
- 144/144 个目标 LoRA-B 张量均发生非零更新。
- 峰值已分配显存约 7.819 GiB，无 OOM。
- 最佳严格准确率仍为第 0 步的 46%，训练器正确记录 `best_optimizer_step = 0`，没有把更差的训练后 Adapter 标记为最佳。

## 7. 样本级分析

相对初始模型，B3：

- 修正 0 条原本错误的样本。
- 使 1 条原本正确的样本退化。
- 22 条保持正确，27 条保持错误。

发生退化的是索引 7：

- 问题：`What is the median value of favourable line in the graph?`
- 参考答案：`40`
- 初始模型：`40`
- B1/B2/B3：均输出 `31`

低学习率保护了 B1/B2 曾破坏的另外三条样本：

- 索引 8：B3 保持正确答案 `Disapprove`。
- 索引 30：B3 保持正确答案 `12`。
- 索引 47：B3 保持正确答案 `0.68`。

代价是 B1/B2 曾修正的索引 5，在 B3 中保持初始错误答案。因此 B3 比 B1 多答对 1 条，比 B2 多答对 2 条，但仍比原始模型少答对 1 条。

## 8. 结论与声明边界

降低学习率确实减少了 Answer-only SFT 对底座能力的破坏：严格准确率由 B1 的 42% 改善到 44%，并保护了三条高学习率下退化的样本。因此“`2e-4` 更新过强”得到部分证据支持。

但是 B3 没有修正任何初始错误，最终仍低于初始模型的 46%，未达到最低成功门槛。不能声称低学习率 QLoRA 提升了 ChartQA 能力，也没有理由把训练后 Adapter 作为当前最佳模型。

50 条验证集每条对应 2 个百分点，B3 相对 B1/B2 的优势只能作为工程方向证据，不是稳定统计结论。

## 9. 下一步

停止继续搜索 LoRA 模块范围或学习率。三组实验共同显示：当前 200 条 Answer-only 训练样本主要改变少数输出，却没有产生稳定的净新增正确答案。

下一阶段先进行数据诊断，不立即训练：

1. 审计 ChartQA 训练/验证问题类型、答案类型和数值范围。
2. 检查当前前 512 条训练池是否与前 50 条验证集任务分布失配。
3. 将问题划分为直接读取、极值/比较、算术、median 等推理类别。
4. 根据诊断结果再选择“扩大训练暴露量”或“任务均衡采样”，一次只验证一个数据变量。

## 10. 产物

- 计划：`docs/multimodal/forgemm_b3_low_lr_attention_qlora_plan.md`
- 正式报告：`artifacts/forgemm/b3_attention_qlora_lr5e5_bounded50_20260806/report.json`
- 第 25/50 步检查点：`artifacts/forgemm/b3_attention_qlora_lr5e5_bounded50_20260806/checkpoints/`
- 最终 Adapter：`artifacts/forgemm/b3_attention_qlora_lr5e5_bounded50_20260806/adapter/`
- Smoke：`artifacts/forgemm/b3_attention_qlora_lr5e5_smoke_20260806/`
