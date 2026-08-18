# Stage 4–5 CPU contract gate

## 目标

在不安装 GPU 训练栈的前提下，关闭训练数据协议、ms-swift reward plugin 字段透传、Chart-FGRPO 数值与 dual checkpoint、冻结评测统计的本地工程门。

## 结果

- 1,763 条唯一严格 EvidenceStore 记录；
- 每条两个可追踪模板视图，Structured SFT 3,526 条；
- GRPO 1,763 prompts；
- SFT SHA-256：`476c9a8af0b216e7a03bd06beba88380d3e1f4563f93ebad3574730d89da5559`；
- GRPO SHA-256：`c80e0b62fc136257cdcfa413eb88e72f67204543ce10b1a7062aac2ae7e3330e`；
- 42 个源码/测试/脚本文件通过 Ruff 与 strict mypy；56/56 tests 通过。

## 已验证契约

1. SFT target 可由 ForgeMM Parser 重新解析，operation 引用顺序保持不变；
2. GRPO 的 `reference_answer`、`gold_evidence`、两个 mask 位于 messages 外，可由 ms-swift ORM kwargs 接收；
3. 三个 reward 分别注册为 `forgemm_task/evidence/operation`；
4. group advantage 对 task/evidence/operation 独立标准化，mask 不适用样本不会获得约束 advantage；
5. 约束低于阈值时 lambda 增加，高于阈值时下降并 clip；恢复后 update step 与乘子轨迹连续；
6. 统一评测分开报告 answer、Evidence F1、operation、FCR、inconsistency，并提供配对 bootstrap 与 exact McNemar。

## 结论边界

3,526 是模板序列数，不是 3,526 个独立 QA；唯一严格记录仍为 1,763。该门不等于 ms-swift、Qwen2.5-VL、QLoRA 或 GRPO 已在 GPU 上运行，E2/E3/E4/E5 质量结果仍为空。

