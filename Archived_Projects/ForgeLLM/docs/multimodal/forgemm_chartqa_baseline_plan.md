# ForgeMM ChartQA 基线：第一阶段

## 目标

建立未微调 Qwen2.5-VL-3B-Instruct 在 ChartQA 验证集上的可复现基线。模型输入仅为图像和问题，不输入 CSV、标注框或测试集数据。

## 假设与边界

- 这一步不验证 PEFT 方法，只记录底座模型的准确率、显存峰值和可复核预测。
- 默认只运行 20 条 `val_human` 样本作为冒烟试验；通过后才扩大到完整验证集。
- 主指标为归一化短文本精确匹配；同时报告 ChartQA 常用的 5% 数值宽松指标。
  后者会把相近年份视为正确，因此仅用于与既有工作比较，不作为项目主结论。
  两个指标都不从解释文本中抽取数字。

## 固定设置

- 模型：本地 Qwen2.5-VL-3B-Instruct
- 推理：NF4 4-bit、FP16 compute、贪心生成
- 图像上限：401,408 pixels
- 输出：`artifacts/forgemm/chartqa_baseline_smoke/`，包含逐条预测与汇总 JSON

## 成功门槛

20 条样本全部完成推理，生成 `predictions.jsonl` 和 `summary.json`，且没有 CUDA OOM 或数据路径错误。该门槛不对准确率设要求。
