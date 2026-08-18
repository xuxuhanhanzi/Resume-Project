# ForgeMM ChartQA 基线冒烟记录（2026-08-05）

## 目标

验证本地 Qwen2.5-VL-3B-Instruct 能否在不使用 CSV、标注框或测试集的前提下，完成 ChartQA 验证集图像问答，并建立第一份可复核的基线产物。

## 固定设置

- 模型：`D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct`
- 数据：`ChartQA Dataset/val/val_human.json` 的前 20 条；只输入对应 PNG 与问题。
- 推理：NF4 4-bit，FP16 compute，贪心生成，最多 32 个新 token。
- 图像上限：401,408 pixels。
- GPU：RTX 4070 Laptop GPU（8 GB）。

## 命令

```powershell
.\.venv\Scripts\python.exe scripts\forgemm_chartqa_baseline.py `
  --model-path "D:\Users\27475\Desktop\models\Qwen2.5-VL-3B-Instruct" `
  --data-root "D:\Users\27475\Desktop\datasets\ChartQA\ChartQA Dataset" `
  --output-dir "artifacts\forgemm\chartqa_baseline_smoke_metric_v2_20260805" `
  --limit 20
```

## 结果

| 指标 | 数值 |
|---|---:|
| 样本数 | 20 |
| 归一化精确答案数 | 9 |
| 归一化精确率（主指标） | 45.0% |
| ChartQA 宽松答案数 | 10 |
| ChartQA 宽松准确率（辅助比较） | 50.0% |
| PyTorch 分配峰值显存 | 2.54 GiB |

产物：

- `artifacts/forgemm/chartqa_baseline_smoke_metric_v2_20260805/predictions.jsonl`
- `artifacts/forgemm/chartqa_baseline_smoke_metric_v2_20260805/summary.json`

## 指标审计

首个实现只输出单一的 5% 数值宽松分数。复核发现第 1 条样本将预测年份 `2006` 与参考年份 `2018` 判为正确，因为相对误差小于 5%。因此脚本已改为同时报告：

- 归一化精确率：项目的主指标；
- ChartQA 宽松准确率：仅用于与采用该传统协议的工作比较。

不会把 20 条冒烟结果写成最终基准成绩，也不会只引用宽松指标。

## 结论与下一步

本地端到端 VLM 链路、4-bit 量化和数据路径均已通过。错误样本涉及数值读取、颜色/图例绑定、时间定位与比较推理，足以支持下一步先设计“固定参数预算下的模块适配位置”对照，而不是直接加入 RL 或视觉路由。
