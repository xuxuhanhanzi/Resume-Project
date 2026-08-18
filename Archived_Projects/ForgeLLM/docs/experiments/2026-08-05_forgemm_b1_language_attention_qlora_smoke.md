# B1 语言注意力 QLoRA 冒烟记录（2026-08-05）

## 目标

验证 Qwen2.5-VL-3B-Instruct 能够使用 ChartQA 多模态样本进行 Answer-only SFT，并保证唯一可训练部分是语言模型注意力层中的 LoRA 参数。

## 环境与数据

- GPU：RTX 4070 Laptop GPU，8 GB。
- 模型：`D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct`。
- 训练：`train_human` 前 2 条，共 2 步。
- 验证：`val_human` 前 4 条，仅作链路检查。
- 图像上限：401,408 pixels。

## 主变量与固定项

- 主变量：在语言注意力的 `q_proj/k_proj/v_proj/o_proj` 注入 rank-8 LoRA。
- 方法：Answer-only SFT、NF4 4-bit QLoRA、FP16 compute、AdamW。
- 固定：视觉编码器、Merger、语言 FFN、提示词、图像预算和评测规则。

## 命令

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_b1_language_qlora_smoke.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\b1_language_attention_qlora_smoke_v2_20260805" `
  --train-limit 2 --eval-limit 4 --max-steps 2
```

## 结果

| 项目 | 数值 |
|---|---:|
| 可训练参数 | 3,686,400 |
| 4-bit-aware 总参数 | 3,758,309,376 |
| 可训练比例 | 0.0981% |
| 可训练参数张量名称数 | 288 |
| 第 1 步损失 | 0.034026 |
| 第 2 步损失 | 0.685302 |
| 每步答案监督 token | 3 / 3 |
| 峰值 PyTorch 分配显存 | 3.83 GiB |
| Adapter 权重大小 | 14,787,968 bytes |

训练前后 4 条验证样本的归一化精确率均为 50%，宽松准确率均为 75%。两步训练的目的不是提高能力，因此该结果不构成质量结论。

## 边界核验

- 288 个可训练参数名称全部位于 `language_model.layers.*.self_attn.*.lora_*`。
- 视觉编码器、Merger、语言 FFN 和底座权重均未训练。
- 图像与问题 token 标签均为 `-100`；只有答案后缀参与损失。
- Adapter 已保存，PEFT 能重新读取 rank、alpha 和四个目标模块。

## 异常与修正

第一次冒烟报告使用普通 `numel()` 统计 4-bit 参数，会低估量化权重对应的逻辑参数量。第二次运行改用 PEFT 的 4-bit-aware 统计，正式记录以 v2 为准；第一次产物仅保留为问题证据。

## 结论

B1 工程闭环通过，但尚未进行足够训练，不能声称准确率改善。下一阶段应将同一方案扩展为受限 quick run，并加入独立训练/验证抽样、训练时长和稳定性记录，再决定是否执行完整 B1。

## 追加：20 步 quick run

随后以 64 条训练池、20 个不同训练样本、20 条验证样本运行 20 步：

- 20 个损失均为有限值；前 5 步平均损失约 0.5597，后 5 步约 0.1715。由于样本不同，该变化只能说明优化过程可工作，不能解释为收敛曲线。
- 144 个 LoRA-B 张量均从零变为非零，确认优化器真实更新了全部目标模块。
- 训练栈内的验证精确率训练前后均为 50%，宽松准确率均为 55%，20 条预测没有变化。
- 峰值 PyTorch 分配显存约 3.89 GiB，无 OOM。

正式产物为 `artifacts/forgemm/b1_language_attention_qlora_quick20_v2_20260805/`。

复核同时发现，`prepare_model_for_kbit_training` 会改变部分非量化层的精度，因此训练栈的零增量 Adapter 初始输出和原始冻结推理基线有少量差异。后续报告必须同时保留：

1. 原始冻结模型基线；
2. 同一训练栈中的 Adapter 训练前基线；
3. Adapter 训练后结果。

20 步没有改变验证输出，不代表 B1 无效，只说明需要用更合理的训练规模和验证抽样完成正式 quick experiment。
