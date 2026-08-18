# ForgeMM 清洁分层 val250 四模型复评计划

## 目标

在排除跨划分重复图片、任务与答案类型分布接近完整 val human 的 250 条固定样本上，重新比较原始 Qwen2.5-VL-3B-Instruct、B1、B2、B3，检验 val50 上的排序是否稳定。

## 固定项

- 同一 Qwen2.5-VL-3B-Instruct 本地底座。
- 同一 NF4 4-bit、FP16 compute 推理栈。
- 同一提示词、图像像素预算、greedy decoding 与最大生成长度。
- 同一 `proposed_stratified_val250_manifest.csv` 顺序。
- Adapter 推理不调用训练准备函数，四个模型的底座推理精度保持一致。

## 对象

- `frozen_raw`：无 Adapter。
- `b1_attention_lr2e4`：B1 最终 Adapter。
- `b2_attention_ffn_lr2e4`：B2 最终 Adapter。
- `b3_attention_lr5e5`：B3 最终 Adapter。

## 指标

- 主指标：归一化严格准确率。
- 辅助指标：ChartQA 5% 数值宽松准确率。
- 按启发式任务类型、答案类型统计。
- 相对原始模型统计修正、退化和保持数量。

## 工程门槛

- 每个模型逐样本写入 JSONL 并支持断点恢复。
- 每个模型独立进程串行运行，避免显存叠加。
- 评测清单 SHA-256、Adapter SHA-256、推理配置写入报告。
- 先以 4 条样本执行 smoke，再运行完整 250 条。

## 决策门槛

- 如果某 Adapter 严格准确率高于原始模型至少 1 个百分点，并且净修正数为正，则进入更大验证或多种子复核，仍不直接声明稳定提升。
- 如果所有 Adapter 不高于原始模型，则进入 B4：保持 B3 配方，只替换为任务×答案联合分层训练 200 条。
