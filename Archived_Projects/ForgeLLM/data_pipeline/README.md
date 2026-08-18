# Data Pipeline

该模块提供可追踪的 JSONL 读取、规范化、过滤、精确去重、确定性切分、写入与报告流程。实现代码位于 `src/forgellm/data/`，本目录保留为后续大规模 Reader/Filter/Dedup 扩展入口。

CPU Smoke：

```powershell
forgellm data-pipeline --config configs/data/smoke.toml --input tests/fixtures/data/sample_documents.jsonl --output-dir artifacts/data_pipeline/local-smoke --source-name forgellm-test-fixture --source-license project-test-fixture
```

输出包含 train/validation/test/rejects JSONL、`report.json` 与 `manifest.json`。输出目录必须不存在，流水线不会覆盖已有结果。

当前只验证自建测试夹具；尚未完成正式数据源许可证、PII、凭据、污染、近似去重或规模性能审计。
