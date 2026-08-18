# R7：v0.2.1 干净克隆发布验收

- commit：`63b047ca9d5ab36bc274fe9abe4ff62858811bf8`；tag：`v0.2.1`；
- 克隆目录：`D:\Temp\forgemm-clean-63b047c-20260814`；
- Python 3.12.3；按 `requirements-dev.lock` 安装并使用标准隔离构建安装 ForgeMM 0.2.1；
- Ruff format/lint 通过；strict mypy 42 files 通过；pytest 56/56 通过；
- 未提交数据 payload 时 audit 返回 `dataset_identity_status=unavailable`，而非错误的 hash failure；
- `git status --short` 为空，精确匹配 `v0.2.1`。

该验收证明 CPU 工程发布可从 Git 重建，不证明 ms-swift/GPU 训练和模型效果。
