# Contributing

任何实现变更须先明确 Task Spec、工具策略、沙箱边界、验证方式和安全测试。

提交前运行：

```text
python scripts/dev.py check
```

新增模型、Prompt、工具、预算或 Benchmark 时必须更新 Agent Card/Benchmark Manifest；
不可信代码不得通过 `LocalTrustedRunner` 执行。
