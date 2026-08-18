# 实验记录：FRAMES Agent Qwen2.5-7B Smoke-10

## 1. 目标

把 3 条链路验证扩展到固定前 10 条 FRAMES 任务，验证规划、离线检索、多跳推理、Reviewer、证据记录与统一评分器的稳定性。本轮为 oracle-document smoke，不作为开放检索主结果。

## 2. 固定设置

- 模型与规划器：`qwen2.5:7b`，revision `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 语料 manifest SHA-256：`3200861c8dd642730b5464ce48a2c9fcb77a412df0fdef7d72e6688eecd97fa3`
- 任务：`frames-test-0000` 至 `frames-test-0009`
- temperature：0；逐来源查询规划；BM25；显式多跳推理；Reviewer

## 3. 结果

- Accuracy：`2/10 = 0.20`；正确任务为 `0004`、`0007`
- 平均 answer token F1：`0.222`
- evidence recall / precision：均为 `1.0`
- 39 次检索调用；输入 `49,144` tokens，输出 `5,522` tokens
- Agent 总耗时：`178.344s`
- 全部 10 个任务均完成并写入 query plan、retrieval context、model response 与 reviewer response，无运行异常

## 4. 判断

完整 Agent 链路通过 10 条稳定性门禁，且相对无文档基线 `0/10` 提升到 `2/10`。错误主要发生在证据齐全后的关系组合、计数和答案格式，不是证据覆盖失败；正式实验前应优先改进推理/验证，而不是继续扩大检索召回。

证据：`artifacts/benchmarks/20260807_frames_agent_qwen2_5_7b_smoke10_v1/results.json`。
