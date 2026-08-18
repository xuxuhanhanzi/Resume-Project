# 实验记录：P5/P6 FRAMES Quick-10

## 1. 目标

比较同一 Qwen2.5-7B 在直接回答、完整 Agent、无 Planner 和无 Reviewer 四种配置下的准确率、F1 与成本。每次消融只改变一个组件。

## 2. 固定项

- 任务：`frames-test-0000` 至 `frames-test-0009`
- 模型 revision：`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- temperature：0；Answer 最大输出 512 tokens
- Agent 使用同一冻结语料、BM25 参数、评分器和任务顺序

## 3. 命令

```powershell
python scripts/run_frames_direct_smoke.py --run-id 20260807_frames_p5_b0_direct_qwen2_5_7b_quick10 --count 10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --max-output-tokens 512
python scripts/run_frames_oracle_smoke.py --run-id 20260807_frames_p6_a4_no_planner_qwen2_5_7b_quick10 --count 10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --reasoned --reviewed --reasoning-output-tokens 512
python scripts/run_frames_oracle_smoke.py --run-id 20260807_frames_p6_a5_no_reviewer_qwen2_5_7b_quick10 --count 10 --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e --planned --reasoned --reasoning-output-tokens 512
```

B1 使用先前冻结运行：`20260807_frames_agent_qwen2_5_7b_smoke10_v1`。

## 4. 结果

| Run | Accuracy | Avg F1 | Input tokens | Output tokens | Tool calls | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| B0 direct | 1/10 | 0.1333 | 830 | 36 | 0 | 11.032s |
| B1 full Agent | 2/10 | 0.2222 | 49,144 | 5,522 | 39 | 178.344s |
| A4 no Planner | 1/10 | 0.1462 | 51,344 | 4,713 | 39 | 129.016s |
| A5 no Reviewer | 2/10 | 0.2154 | 25,598 | 3,688 | 39 | 118.313s |

## 5. 结论

- 完整 Agent 相比同模型直接回答增加 1 个正确任务，但成本显著上升。
- 移除 Planner 后少 1 个正确任务，是 Planner 可能有效的初步信号。
- 移除 Reviewer 后正确任务不变，耗时下降约 33.7%，输入 tokens 下降约 47.9%；Reviewer 暂不进入候选正式配置。
- 只有单次 10 题 quick run，以上均不是统计性结论；下一步对 B1/A4/A5 做 3 次重复后再决定。
