# 实验记录：Stage 01 Day 05 Fresh Environment Reproduction

## 1. 目标

验证 README 所述安装方式能否在第二个全新虚拟环境中完成 editable install、统一质量检查与 CPU CLI Smoke，并建立 Linux CI 定义。

## 2. 环境

- 平台：Windows 11 / win32
- Python：3.12.3
- 新环境：`.venv-repro`
- 开发工具：mypy 1.20.2、pytest 8.4.2、ruff 0.15.21
- Git 基线：`f934b00538238991c136c5726845ca3b34bb150d` + dirty diff
- GPU/数据：未使用

## 3. 实验变量

- 主变量：在不复用 `.venv` 和 `SCI` 的新环境安装与复现
- 固定项：同一代码、`requirements-dev.lock`、CPU-only
- 对比对象：原开发环境 `.venv`

## 4. 命令

```powershell
python -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install -e . -r requirements-dev.lock
.\.venv-repro\Scripts\python.exe scripts\dev.py check
.\.venv-repro\Scripts\python.exe -m forgellm doctor
```

## 5. 输出路径

- 新环境：`.venv-repro/`（Git 忽略）
- CI：`.github/workflows/ci.yml`
- 安装说明：`README.md`

## 6. 结果

| 指标 | 结果 |
|---|---:|
| 新环境创建 | 通过 |
| editable install | 通过 |
| format/lint/typecheck | 全部通过 |
| pytest | 19/19 通过 |
| `python -m forgellm doctor` | 通过 |
| CI YAML 解析 | 通过 |
| Ubuntu 外部 CI | 待运行 |

## 7. 失败与异常

- 沙箱内首次安装因 `WinError 10013` 无法访问依赖索引；
- 经受控网络授权后，使用配置的清华 PyPI 镜像安装成功；
- 当前 ForgeLLM 仓库没有 Git 远端，无法触发 GitHub Actions；不能将 CI 配置存在写成 CI 已通过。

## 8. 结论

Day 5 的本地陌生环境复现通过。G0 的本地部分满足；Linux CI 定义已经完成，但外部运行证据待仓库设置远端后补齐。因此 `v0.1-engineering` 目前是本地候选，不是已发布 Release。

## 9. 下一步

进入 Day 6 数据契约与 Manifest；外部 CI 不阻塞纯标准库数据实现，但在正式发布前必须补跑。

