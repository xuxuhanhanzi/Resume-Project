# Contributing

当前处于 P0 工程与数据阶段。提交实现前应先创建阶段计划、补齐测试策略，并在实验记录中登记数据、配置和基线。

统一质量入口：

```bash
python scripts/dev.py check
```

测试按目录分为：

- `tests/unit/`：纯函数、Schema、边界与失败行为；
- `tests/integration/`：多个模块和文件系统的组合行为；
- `tests/smoke/`：从用户入口验证最小闭环。

测试必须使用 `tmp_path` 或等价临时目录，不得写入正式 `artifacts/`。失败测试应保留可诊断的退出码和错误信息。
