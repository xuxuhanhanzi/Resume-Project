# 实验记录：Stage 01 Day 04 Cross-platform Quality Commands

## 1. 目标

建立单一跨平台质量入口，并验证 unit、integration、smoke 三层能够独立运行，任一子步骤失败时立即返回非零退出码。

## 2. 环境

- 平台：Windows 11 / win32
- Python：3.12.3
- 项目环境：`.venv`
- Git 基线：`f934b00538238991c136c5726845ca3b34bb150d` + dirty diff
- GPU/数据：未使用

## 3. 实验变量

- 主变量：新增 `scripts/dev.py` 统一命令和测试目录分层
- 固定项：同一依赖、同一代码、CPU-only
- 对比对象：Day 3 分散执行 ruff/mypy/pytest 的命令

## 4. 命令

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
.\.venv\Scripts\python.exe scripts\dev.py unit
.\.venv\Scripts\python.exe scripts\dev.py integration
.\.venv\Scripts\python.exe scripts\dev.py smoke
```

## 5. 输出路径

- 统一入口：`scripts/dev.py`
- Makefile 代理：`Makefile`
- 测试策略：`CONTRIBUTING.md`

## 6. 结果

| 检查 | 结果 |
|---|---:|
| format-check | 通过 |
| Ruff lint | 通过 |
| mypy strict | 通过 |
| 全部测试 | 19/19 通过 |
| unit | 17/17 通过 |
| integration | 1/1 通过 |
| smoke | 1/1 通过 |
| 未知命令 | 返回 2，测试通过 |

## 7. 失败与异常

- 首次 `check` 在 format-check 停止，发现 `config.py` 与统一格式不一致；
- 使用锁定版本 Ruff 格式化后重新执行，所有步骤通过；
- 该失败证明统一入口能够在后续检查前阻止格式漂移。

## 8. 结论

Day 4 通过。Python 入口是权威命令，Makefile 只做代理，避免 Windows/Linux 维护不同核心逻辑。

## 9. 下一步

建立 Ubuntu CI，并在第二个全新环境复现安装和全部质量检查。

