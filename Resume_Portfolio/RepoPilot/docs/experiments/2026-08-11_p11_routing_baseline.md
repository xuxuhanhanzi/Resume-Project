# 实验记录：P11 可审计路由基线

- 日期：2026-08-11
- 状态：实现及运行时绑定完成，待负载对照
- 目标：实现 fixed/rule/model/hybrid 四种路由及可定位的级联降级。
- 单一主变量：新增任务路由决策层；不改变 AgentRuntime、工具、Verifier 或评测集。

## 验收边界

1. 规则路由确定、可解释；
2. 模型路由只接收 PublicTaskSpec 字段，tools 为空；
3. 模型超时、非法 JSON、非法 target、越界 confidence、工具调用均降级到规则结果；
4. hybrid 对高置信度规则直接接受，对模糊任务调用模型；
5. 每个结果保存 target、mode、reason、confidence 和完整 cascade。
6. coordinator 根据 target 选择完整 executor/planner/reviewer binding，并在 checkpoint
   中持久化选择；恢复时复用原决定而不重新路由。

## 当前证据

`tests/unit/test_routing.py` 覆盖 fixed/rule/model/hybrid、只读约束、非法 JSON 和工具调用降级；
`tests/integration/test_multi_agent_runtime.py` 额外验证规则选择 escalated runtime、未调用 standard
executor，并将理由写入 checkpoint。

统一质量门结果：132 files formatted；Ruff 通过；strict mypy 103 source files / 0 errors；
pytest 80 passed。路由功能本身已有实现证据，但准确率、成本和延迟收益仍未验证。
本阶段尚未声称路由提高正确率或降低成本；只有在相同冻结任务、模型、预算和 evaluator 的
至少三次成对对照后，才能决定是否进入默认配置。
