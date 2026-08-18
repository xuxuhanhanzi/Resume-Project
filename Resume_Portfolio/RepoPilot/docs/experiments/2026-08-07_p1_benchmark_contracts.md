# 实验记录：P1 公共评测骨架

## 1. 目标

验证知识研究、数据分析和软件工程任务能否在不共享领域专用字段的前提下，使用同一套 `load → prepare → execute → evaluate` 协议，并保证 evaluator-only 数据不进入 Agent 输入。

## 2. 环境

- 平台：Windows，本地离线开发
- Python：项目要求 3.11–3.12
- 代码路径：`D:\Users\27475\Desktop\Resume_Project\RepoPilot`
- 数据集：仅使用合成契约 fixture，不下载真实数据

## 3. 实验变量

- 主变量：新增 domain-neutral benchmark contracts 与 runner
- 固定项：现有 AgentRuntime、Python 修复任务接口、模型与工具实现
- 对比对象：现有代码专用 `PublicTaskSpec` 与兼容 adapter

## 4. 计划命令

```powershell
python -m pytest tests/unit/test_benchmark_contracts.py tests/integration/test_benchmark_runner.py
python -m pytest
python -m ruff check .
python -m mypy src tests scripts
```

## 5. 预计产物

- `src/repopilot/benchmarks/` 公共契约、注册中心、runner、manifest
- 原有 Python 修复任务兼容 adapter
- 三领域契约测试与 evaluator secrecy 测试

## 6. 结果

- 新增契约与 runner 定向测试：7 passed。
- 全仓测试：40 passed。
- strict mypy：64 个源文件无错误。
- 当前 Python 环境未安装 Ruff，因此本轮未取得 Ruff 结果；pytest 与 mypy 均通过。

## 7. 结论

P1 门禁通过。三个领域的合成任务已共享同一 runner，evaluator case 仅在 Agent 执行结束后读取；原有 `PublicTaskSpec` 通过兼容 adapter 保留。下一步进入 P2 FRAMES 真实数据接入。
