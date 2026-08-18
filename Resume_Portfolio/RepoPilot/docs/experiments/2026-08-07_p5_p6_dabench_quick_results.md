# 实验记录：P5/P6 DABench Quick-10

## 1. 目标

补齐 DABench 10-task Agent 门禁，并只改变最大尝试次数，验证执行反馈恢复的收益与成本。

## 2. 固定项

- 任务：manifest `dabench_agent_smoke10_v1.json` 中 10 条任务（3 easy / 5 medium / 2 hard）
- 模型 revision：`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- Docker image：`sha256:fc82eedb08c10ba2471b62781d34edf8a5e70bc788bf5defb38fc6546ee302dd`
- 网络：deny；temperature：0；官方闭式答案评分
- 主变量：B1 `max_attempts=3`；A1 `max_attempts=1`

## 3. 命令

```powershell
python scripts/run_dabench_agent_smoke.py --run-id 20260807_dabench_p5_b1_agent_qwen2_5_7b_quick10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-attempts 3
python scripts/run_dabench_agent_smoke.py --run-id 20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_quick10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-attempts 1
```

## 4. 结果

| Run | Accuracy | Easy | Medium | Hard | Attempts | Recoveries | Input tokens | Output tokens | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 feedback | 9/10 | 3/3 | 4/5 | 2/2 | 12 | 2 | 10,886 | 2,629 | 97.438s |
| A1 no feedback | 8/10 | 3/3 | 4/5 | 1/2 | 10 | 0 | 7,990 | 1,938 | 71.687s |

## 5. 结论

反馈恢复额外解决 hard 任务 `dabench-dev-0007`，准确率提高 10 个百分点；代价为约 35.9% 额外耗时和 36.2% 额外输入 tokens。任务 `0006` 在两种配置均失败，应归入答案格式/分组边界专项审计。该单轮结果需要重复实验确认。
