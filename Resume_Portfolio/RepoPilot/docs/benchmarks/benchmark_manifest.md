# Benchmark Manifest

## `repopilot_python_micro_v1`

- Manifest：`evaluation/micro_benchmark.yaml`
- 任务数：10
- 语言：Python
- 类型：单文件确定性 Bug Fix
- Harness：RepoPilot `1.0.3`（历史 run 使用各实验记录中的冻结 commit/version）
- 模型：正式本地 Qwen run 前登记 checkpoint、量化与服务版本
- Prompt：`src/repopilot/context/builder.py`
- 工具和预算：各 Task Spec 冻结
- 开发验证：提交的 `verify.py`
- 隐藏验证：必须位于 Agent 工作区外，当前正式集合待建设

该集合用于技术组件消融与回归，不用于声称 SWE-bench 或真实工程能力。
