# ForgeMM v0.2.1

发布修复：环境审计现在把未提交的受限数据 payload 缺失标记为 `unavailable`，不会让普通 CI 失败；正式数据审计使用 `--require-datasets`，任何缺失或哈希错误仍会失败。
