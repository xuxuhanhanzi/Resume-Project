# 实验记录：P3 DABench 协议 Smoke

## 1. 目标

绕过 Hugging Face 对异构 CSV 的自动 schema 合并，按问题 ID 显式关联官方问题、标签与单个 CSV，并验证闭式评分。

## 2. 数据

- revision：`b455d578e30fee513abd79936cbdf7a6de026cb5`
- 问题与标签：完整 257 条 dev 任务
- 当前表格快照：`test_ave.csv`
- smoke IDs：0（easy）、5（medium）、7（hard）

## 3. 门禁

- 三种难度各 1 个真实任务通过 evaluator smoke。
- 公共任务不包含 `common_answers`。
- 每次 prepare 只复制当前任务的 `input.csv`。
- CSV、问题或标签哈希不一致时 fail closed。
- 真实生成代码只能使用无网络 Docker 环境；本轮 oracle 协议 smoke 不执行生成代码。

## 4. 结果

- 完整加载 257 条问题与标签，未使用自动 CSV schema 合并。
- ID 0、5、7 三个真实任务通过闭式 evaluator smoke。
- prepare 仅复制当前任务的 `input.csv`。
- 早期状态：Docker daemon 不可用时未执行 Agent 代码。

## 5. Docker 恢复与 Agent smoke

- Docker Server：`29.6.1`
- 最小容器：`hello-world` 通过
- 无网容器：`python:3.11-alpine --network none` 通过
- DABench 镜像：`repopilot-dabench:py311-v1`
- 镜像 ID：`sha256:fc82eedb08c10ba2471b62781d34edf8a5e70bc788bf5defb38fc6546ee302dd`
- 模型：`qwen2.5:7b`
- 运行：`20260807_dabench_qwen2_5_7b_final_smoke3`

| 难度 | 输出 | 尝试次数 | 工具调用 | 耗时 | 官方评分 |
|---|---|---:|---:|---:|---:|
| easy | `@mean_fare[34.65]` | 1 | 2 | 4.64s | 通过 |
| medium | `@correlation_coefficient[0.21]` | 1 | 2 | 4.77s | 通过 |
| hard | `@prediction_accuracy[0.78]` | 3 | 4 | 29.59s | 通过 |

hard 任务先后经历 NaN 执行失败和错误算法替换；分类反馈与执行前语义策略使第三次代码满足 `LinearRegression` 约束并通过官方闭式评分。最终 smoke 为 `3/3 = 100%`。该结果仅证明工程链路与恢复机制通过门禁，不作为正式模型性能结论。
