# 阶段计划：Stage 01 Engineering Baseline

## 1. 阶段目标

将当前文档骨架变成可安装、可检查、可测试、可在陌生环境复现的最小 Python 项目，形成 `v0.1-engineering` 候选。

## 2. 可证伪假设

采用最小 Python 包、统一 CLI、分层测试、环境快照和 Linux CI，可以在不提前引入训练框架与容器的情况下，让 Windows 开发环境和 Linux CI 使用相同命令完成 CPU Smoke。

若 Windows 与 Linux 必须维护不同核心命令、或新环境仍需要未记录的手工修改，则假设失败，G0 不通过。

## 3. 官方/公认基线

- Python 包元数据使用 `pyproject.toml`；
- 测试由 `pytest` 执行；
- lint/format 使用同一工具链；
- 类型检查、单元测试与 CPU Smoke 由 CI 自动运行；
- 运行结果关联配置、环境、Git 状态和 Run ID。

具体工具版本在 Day 2 建立环境时固定；版本未锁定前不宣称可复现。

## 4. 当前工程基线

- 只有目录、README、空 `pyproject.toml` 和占位 Makefile；
- 没有 `src` 包、CLI、依赖、测试、CI 命令或运行 Artifact；
- 本地存在 Conda、Git、CUDA 与 8 GB NVIDIA GPU；
- Docker 不存在，WSL 没有 Linux 发行版；
- 仓库尚无实验结果。

## 5. 唯一主变量与固定项

**主变量：** 是否加入首个完整工程闭环（包 + CLI + 质量工具 + 测试 + Smoke + CI + 文档）。

**固定项：**

- 不实现数据业务逻辑、Tokenizer 或模型；
- 不运行付费任务；
- 不修改已有 `SCI` 环境；
- 不引入 Docker、vLLM、DeepSpeed、数据库或服务端；
- 所有测试使用仓库内极小夹具或临时目录；
- Windows 本地与 Linux CI 使用同一个 Python CLI。

## 6. 指标与成功门槛

| 指标 | 成功门槛 |
|---|---|
| 安装 | 新独立环境按 README 一次完成 |
| CLI | `--help` 返回 0，非法参数返回非 0 且信息清晰 |
| lint/format | 无错误 |
| typecheck | 项目源代码无错误 |
| unit/integration/smoke | 全部通过 |
| 确定性 | 固定输入与种子生成一致的解析配置和关键元数据 |
| 隔离 | 测试不污染正式 `artifacts/`，不修改已有 Conda 环境 |
| CI | Linux CI 使用文档中的同一入口通过 |
| 复现 | 新环境无需未记录的手工步骤 |

G0 必须全部满足，不使用覆盖率百分比替代行为验收。

## 7. 预算、停止条件与失败处理

- 时间预算：最多 1 周，按 60 小时上限；核心实现目标 35 小时；
- 费用预算：0 USD；
- 单个依赖兼容问题超过 4 小时，先删除非必要依赖；
- Docker/WSL 安装问题不阻塞 G0，转入 P1 风险队列；
- Windows/Linux 差异导致 Makefile 不一致时，以 Python CLI 为权威，平台脚本仅做代理；
- 任何质量工具若迫使大量无价值配置，优先缩小工具范围并写明代价。

## 8. 预期 Artifact 与报告

- 完整 `pyproject.toml`；
- `src/forgellm/` 最小包和 CLI；
- `tests/unit`、`tests/integration`、`tests/smoke`；
- 本地与 CI 统一质量命令；
- 环境快照和 CPU Smoke Artifact；
- 一份 G0 复现记录；
- 更新后的 README、进度台账与 `v0.1-engineering` 候选说明。
