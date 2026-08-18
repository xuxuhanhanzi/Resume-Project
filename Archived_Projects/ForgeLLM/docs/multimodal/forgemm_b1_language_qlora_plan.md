# B1：语言注意力 QLoRA 冒烟计划

## 目标

验证 ChartQA 的“图像 + 问题 → 短答案”样本能够通过 Answer-only SFT，只更新 Qwen2.5-VL 语言模型注意力层中的 LoRA 参数。

## 唯一主变量

在已完成的冻结底座基线上启用语言注意力层 QLoRA。视觉编码器、Merger、语言 FFN、数据格式、提示词、图像像素上限和评测规则保持不变。

## 冒烟设置

- 数据：2 条 `train_human`，4 条 `val_human`。
- 步数：2。
- LoRA：rank 8，alpha 16，dropout 0。
- 目标：`q_proj`、`k_proj`、`v_proj`、`o_proj`。
- 量化：NF4 4-bit；FP16 compute。
- 优化器：AdamW，学习率 `2e-4`。

## 成功门槛

- 所有可训练参数必须同时满足：位于 `language_model.layers`、位于 `self_attn`、名称包含 `lora_`。
- 图像与问题 token 的标签必须为 `-100`，至少保留一个答案监督 token。
- 两步损失均为有限值，能够保存并重新识别 LoRA adapter，且无 CUDA OOM。
- 训练前后评测只用于验证链路，不作为能力提升结论。
