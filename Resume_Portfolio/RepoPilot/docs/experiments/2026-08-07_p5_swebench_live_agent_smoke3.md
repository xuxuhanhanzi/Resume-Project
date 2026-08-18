# P5 SWE-bench-Live Agent 三任务烟雾实验

## 固定条件

- 数据 revision：`a637bd46829f3132e12938c8a0ca93173a977b8e`
- 官方 evaluator：`ad79b850f15e33992e96f03f6e97f05ddf9aa0be`
- 模型：`qwen2.5:7b`，revision `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- Docker 网络：关闭；只读取公开问题描述，判分阶段才加载官方隐藏测试。

## 结果

| 实例 | Agent 生成 | 官方结果 | 主要失败 |
|---|---|---|---|
| cfn-lint-3767 | 迭代耗尽，空补丁 | empty patch | 未完成有效编辑 |
| babel-1141 | 生成补丁并通过可见测试 | unresolved | 目标隐藏测试出现 `TypeError` |
| mesa-2394 | 生成补丁，可见验证失败 | unresolved | 目标测试失败且破坏 17 个回归测试 |

官方 `resolved` 为 `0/3`。这不是框架异常：两个非空补丁均成功应用且官方评测完整运行，说明执行链路已打通，但当前 7B 本地模型与循环策略的代码修改质量不足。

## 下一门槛

暂不扩大 SWE-bench-Live 样本。先改进失败反馈摘要、强制新增针对性测试，并在同一三任务集上复测；只有 smoke 明显改善后才进入正式规模。
