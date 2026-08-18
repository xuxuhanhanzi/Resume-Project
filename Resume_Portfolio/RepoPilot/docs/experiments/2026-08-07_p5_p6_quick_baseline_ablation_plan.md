# 实验计划：P5/P6 快速基线与单变量消融

## 1. 阶段目标

在不越过工程门禁的前提下，先对已完成 10-task smoke 的 FRAMES 运行同模型基线和机制消融；DABench 与 SWE-bench-Live 分别补齐 10-task、3-task Agent smoke 后再进入对比。

## 2. 假设

- H1：离线证据检索、规划和 Reviewer 的完整 Agent 相比同模型直接回答提高 FRAMES 准确率。
- H2：单独移除 Planner 或 Reviewer 会降低准确率或答案 F1，但减少 token 与耗时。

## 3. 主变量与运行

| Run | 主变量 | Planner | Retrieval | Reviewer |
|---|---|---:|---:|---:|
| B0 | 直接模型 | 0 | 0 | 0 |
| B1 | 完整 Agent | 1 | 1 | 1 |
| A4 | 仅移除 Planner | 0 | 1 | 1 |
| A5 | 仅移除 Reviewer | 1 | 1 | 0 |

## 4. 固定项

- 数据：FRAMES revision `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 任务：`frames-test-0000` 至 `frames-test-0009`
- 模型：`qwen2.5:7b`，revision `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- temperature：0
- Agent 语料 manifest：`3200861c8dd642730b5464ce48a2c9fcb77a412df0fdef7d72e6688eecd97fa3`
- Agent 检索：每个 oracle 页面 2 个 BM25 chunk；chunk 2000 chars，overlap 250 chars
- Answer 最大输出：512 tokens；Planner 96 tokens/来源；Reviewer 512 tokens
- 评分：同一 `FramesEvaluator`；分别报告 accuracy、answer token F1、tokens、工具调用与耗时

## 5. 成功门槛

- 每个运行 10/10 完成、无空结果和框架异常。
- 每次消融只改变表中一个组件。
- 本轮单次 quick run 只用于筛选；正式结论需要固定配置重复 3 次。

## 6. 失败处理

- 保留失败 artifact，不覆盖 run directory。
- 若准确率相同，使用 F1 与成本指标诊断，不宣称机制无效。
- DABench/SWE-bench-Live 未满足 Agent smoke 前，不加入跨域对比表。

## 7. 预计产物

- `artifacts/benchmarks/20260807_frames_p5_b0_direct_qwen2_5_7b_quick10/`
- `artifacts/benchmarks/20260807_frames_agent_qwen2_5_7b_smoke10_v1/`（B1，已完成）
- `artifacts/benchmarks/20260807_frames_p6_a4_no_planner_qwen2_5_7b_quick10/`
- `artifacts/benchmarks/20260807_frames_p6_a5_no_reviewer_qwen2_5_7b_quick10/`
- 汇总实验记录与失败分析
