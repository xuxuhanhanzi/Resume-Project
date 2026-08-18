# Local datasets

本目录保存 ForgeMM 的本地数据集 payload。大文件已由项目 `.gitignore` 排除，
不得提交 Git。

- `ChartQA/`：应直接包含 `train/`、`val/`、`test/`。
- `ChartQAPro/`：使用 `chartqapro_test.parquet` 作为唯一规范数据源。
- 每个数据集目录中的 `DATASET_CARD.md` 和 `manifest.json` 用于记录来源、结构与
  完整性信息，可以提交 Git。

项目代码应优先读取命令行参数或 `FORGEMM_DATASETS_ROOT`，不得硬编码个人绝对路径。

