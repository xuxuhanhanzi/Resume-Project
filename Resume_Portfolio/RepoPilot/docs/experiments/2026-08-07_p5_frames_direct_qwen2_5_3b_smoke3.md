# 实验记录：FRAMES Direct Qwen2.5-3B Smoke-3

## 1. 目标

验证非思考型本地模型能否通过统一 runner 产生可评分答案。本实验仍是无工具、无文档的固定基线，不代表 Agent 主结果。

## 2. 固定设置

- 模型：`qwen2.5:3b`
- 模型 manifest SHA-256：`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`
- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 任务：`frames-test-0000` 至 `frames-test-0002`
- temperature：0
- 最大输出：128 tokens
- 工具/文档：无

## 3. 成功门槛

- 三个任务均产生非空答案
- runner、scorer、artifact 全链路无异常

## 4. 结果

- 运行成功：`3/3` 均产生非空答案，runner、scorer、artifact 无异常
- Accuracy：`0/3 = 0.0`
- 输入/输出：`253/13 tokens`
- 总任务耗时：`14.141s`（包含首次模型加载）
- 结论：作为无文档负对照有效；0 分不属于链路故障
