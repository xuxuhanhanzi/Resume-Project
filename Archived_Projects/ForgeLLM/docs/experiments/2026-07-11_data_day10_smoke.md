# 实验记录：Data Day 10 Deterministic Fixture Smoke

## 1. 目标

在自建 JSONL 夹具上验证 Reader、Schema、Unicode 规范化、质量过滤、精确去重、内容哈希切分、稳定 Writer、Manifest 和报告的完整 CPU 闭环。

## 2. 环境

- 平台：Windows 11 / win32
- Python：3.12.3
- 项目环境：`.venv`
- GPU/PyTorch/CUDA：未使用
- Git 基线：`f934b00538238991c136c5726845ca3b34bb150d` + dirty diff
- 数据：`tests/fixtures/data/sample_documents.jsonl`
- 数据性质：项目自建测试夹具，不是正式训练数据

## 3. 实验变量

- 主变量：完整确定性数据流水线
- 固定项：NFC、min 12、max 1000、80/10/10 basis points、自建 9 行夹具
- 对比对象：无数据实现的仓库骨架

## 4. 命令

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
.\.venv\Scripts\forgellm.exe data-pipeline --config configs\data\smoke.toml --input tests\fixtures\data\sample_documents.jsonl --output-dir artifacts\data_pipeline\win_data_fixture_v2_20260711 --source-name forgellm-test-fixture --source-license project-test-fixture
```

## 5. 输出路径

- 正式 Smoke：`artifacts/data_pipeline/win_data_fixture_v2_20260711/`
- 初次覆盖不足运行：`artifacts/data_pipeline/win_data_fixture_v1_20260711/`
- 配置：`configs/data/smoke.toml`
- 设计：`docs/data_pipeline_design.md`
- 输出文件：train/validation/test/rejects JSONL、report、manifest

## 6. 结果

### 质量检查

| 指标 | 数值 |
|---|---:|
| Ruff / format | 通过 |
| mypy strict | 通过，27 个源文件 |
| pytest | 33/33 通过 |
| 同配置独立重复运行 | 6/6 输出文件 SHA-256 一致 |

### 数据结果

| 指标 | 数值 |
|---|---:|
| 输入 | 9 |
| Schema 解析成功 | 8 |
| 保留 | 4 |
| 拒绝 | 5 |
| train / validation / test | 2 / 1 / 1 |
| 保留字符 / UTF-8 bytes | 220 / 283 |
| 配置 SHA-256 | `151dd20cc148c16fe0663a56ab5c393d645ede5bfc619ac4ff9f49748a8d0ffe` |
| 输入 SHA-256 | `49eb01a3155b3a019d2c71861360814718e1164c62cce611e30bb166025d4e40` |

### 输出哈希

| 文件 | 记录数 | SHA-256 |
|---|---:|---|
| train.jsonl | 2 | `ca7b1f621a5199b2182ea7eb3e755d4993f2934e7476d1eb2bc10aa7655e9834` |
| validation.jsonl | 1 | `380fbb51aee31ebb5ecbbc6aff71e755a3f75189c77d894a70da0f783c868a42` |
| test.jsonl | 1 | `0693fb073887ead522eed9a109d22d08be6723a1f43b8124d393a5074f9535d2` |
| rejects.jsonl | 5 | `89ab032979ffd68cba5543dd28c8bd527c6eda7cae36d0249514471cc1aeee39` |

拒绝原因各 1 条：控制字符、精确重复、空文本、Schema 无效、过短。

## 7. 失败与异常

1. 首次确定性测试失败：原文件 CRLF、重排文件 LF，`record_ref` 将行尾计入哈希。修复为先移除 CR/LF 再计算行引用，重排后的四类 JSONL 哈希一致。
2. 首次 Smoke 使用 seed 1337，4 条保留记录全部进入 train，未真正覆盖非空 validation/test Writer。保留 v1 产物，将固定 Smoke seed 改为 6，v2 得到 2/1/1，并增加精确断言。
3. `mypy` 首次发现 unit/integration 同名测试模块冲突；增加测试包 `__init__.py`，避免模块名碰撞。

## 8. 结论

Day 6–10 的测试夹具闭环通过。可以声称“数据流水线接口与确定性行为已在自建 9 条夹具上验证”，不能声称正式数据已经清洗、许可证已经审核、质量规则适合真实训练或具备大规模性能。

## 9. 下一步

用户先学习 Day 1–10 知识并完成自测。之后从 Day 11 开始，最多审计三个真实候选数据源，只批准一个来源清晰且许可证可用的数据源。
