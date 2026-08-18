# 实验记录：R7 Windows 干净克隆与发布

- 日期：2026-08-14
- 成功冻结对象：`v1.0.2` / commit `bc3ee9cceccb2f70340fa19c3c54759736b06520`
- 成功克隆目录：独立临时目录 `repopilot-clean-clone-bc3ee9c`
- 工具环境：全新 `.venv`，Python 3.12.3；从 `requirements-dev.lock` 安装。

## 成功结果

| 门槛 | 结果 |
|---|---|
| Editable package | `repopilot-1.0.2` 构建并安装 |
| Ruff format | 141 files already formatted |
| Ruff lint | all checks passed |
| strict mypy | 112 source files，0 errors |
| pytest | 90 passed，1 live-Docker test 按设计 skipped |
| 离线 demo | 修改 subtract、运行 verifier、输出 completed |
| Git | `status --short` 空；exact tag=`v1.0.2` |

live-Docker test 不在默认 suite 隐式跳过后冒充已验证；它已在主工作区以固定镜像单独执行，
结果 6/6，且 CI 有独立 `security-container` job。

## 干净克隆发现并保留的失败

### v1.0.0

- 首次非提权运行因 Codex 临时目录写权限无法创建 Ruff cache/artifacts；以相同命令获得
  临时目录写权限后继续；
- 真实缺陷：Windows `core.autocrlf` 将 DABench/FRAMES 冻结数据改为 CRLF，11 个 hash
  测试失败（79 passed）；离线 demo 本身成功。

### v1.0.1

- 全局强制 JSONL/TSV LF 后，DABench/FRAMES 恢复；
- 两个历史 SWE smoke JSONL 的冻结 hash 原本基于 Windows CRLF，出现 4 个 hash 失败
  （86 passed）；离线 demo 成功。

### v1.0.2 修复

- 默认代码/数据/manifest 使用 LF；
- 仅对两个明确的 SWE smoke JSONL 设置 `eol=crlf`；
- 干净克隆显示 DABench/FRAMES `w/lf`、SWE 两文件 `w/crlf`，全套测试通过。

## 发布策略

旧 annotated tags 不移动、不删除。`v1.0.3` 只增加本复现记录并更新版本号，不改变执行
代码、测试或数据字节；功能冻结基线仍是已完成干净克隆的 `v1.0.2`。
