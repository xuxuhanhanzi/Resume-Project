# 实验记录：P2 FRAMES 协议 Smoke

## 1. 目标

固定 FRAMES 官方数据 revision，验证真实 TSV 的加载、gold/public 隔离、oracle/retrieval 模式分离和确定性评分。

## 2. 数据与环境

- 数据源：`google/frames-benchmark`
- revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- test.tsv SHA-256：`4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff`
- 规模：824 个 test 任务
- 网络策略：deny；本轮不抓取实时 Wikipedia 页面

## 3. 主变量与固定项

- 主变量：FRAMES adapter 与 deterministic evaluator
- 固定项：统一 runner contracts、数据 revision、3 个预注册任务
- 评分：normalized exact match、token F1、evidence recall/precision、invalid citation rate

## 4. 门禁

- 3 个真实任务通过协议 smoke。
- retrieval 模式不暴露官方相关页面。
- dataset hash 不一致时 fail closed。
- 本轮结果不得作为模型性能结果。

## 5. 结果

- 官方 revision 与 TSV SHA-256 校验通过，共加载 824 条任务。
- 3 个真实任务通过 deterministic evaluator smoke。
- retrieval 模式未暴露 gold Wikipedia URLs。
- Agent smoke 未运行：本地模型端点不可用，离线 Wikipedia 语料尚未物化。
