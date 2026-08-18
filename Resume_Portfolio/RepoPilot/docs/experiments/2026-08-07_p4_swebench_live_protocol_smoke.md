# 实验记录：P4 SWE-bench-Live 协议 Smoke

## 1. 目标

从固定 lite parquet 中分离公开任务和 evaluator-only 补丁/测试，并确保 RepoPilot 不使用内部 verifier 冒充官方 resolved 结果。

## 2. 数据

- revision：`a637bd46829f3132e12938c8a0ca93173a977b8e`
- lite parquet SHA-256：`7ee0a75c41bfc954fd441b67ce738fc5c1cbae00721c4e30e7db4d893057c9ab`
- smoke：cfn-lint、babel、mesa 三个 Python 实例
- 官方 evaluator：`microsoft/SWE-bench-Live` 的 `python-only` 分支，提交 `ad79b850f15e33992e96f03f6e97f05ddf9aa0be`
- 通过镜像：`starryzhang/sweb.eval.x86_64.python-babel_1776_babel-1141@sha256:b9847102c2e7d0f82738cd1a73fe11b242ca2152dc9c28091430ec5a12a7bd70`

## 3. 门禁

- public extraction 不含 gold patch、test patch、FAIL_TO_PASS 或 PASS_TO_PASS。
- 没有实例镜像时 prepare 必须 fail closed。
- 只有官方 evaluator 的结构化结果可以生成 resolved 指标。
- Windows 主机只对上游 `patch.diff` 和 `eval.sh` 强制 LF；不修改数据、测试或评分逻辑。

## 4. 结果

- 固定并校验 300-task lite parquet，抽取 3 个 Python 协议任务。
- public/evaluator 数据拆分通过，缺少实例环境时 fail closed。
- wrapper 拒绝非官方评分输入，并可直接规范化上游 instance report。
- `python-babel__babel-1141`：gold patch 成功应用，目标测试及回归测试通过，官方 `resolved=true`；容器内 `7119 passed, 8 skipped`，19.42 秒，官方总运行 36.21 秒。
- `aws-cloudformation__cfn-lint-3767`：目标测试通过，但 4 个 PASS_TO_PASS 失败，官方 `resolved=false`。作为数据/镜像负例保留，不用于证明模型能力。
- 证据：`artifacts/benchmarks/20260807_swebench_live_official_gold_smoke/` 与 `logs/run_evaluation/`。

## 5. 结论

官方 Docker 评测与判分链路已打通。该结果仅是 evaluator gold smoke，不是 Agent 的 SWE-bench-Live 成绩；下一门禁是生成 Agent patch 后走同一官方评测器。
