# DriveVLA-Guard v1.0.0

这是首个本地可发布版本。核心库、合成评测、证据验证和官方运行交接均可由仓库命令复现。

本版本的冻结数字来自 12 个确定性合成场景：B0 碰撞代理失败 6/12，E2/E4 为 0/12，E4 slow route 25%、平均代理延迟 19.5 ms。它们不是 NAVSIM 或真实道路指标。

正式 NAVSIM 尚未执行，因为当前机器只有 8GB 显存并缺少 checkpoint、Qwen 权重、NAVSIM 数据和 metric cache。脚本会在这些门禁未满足时以 `ready=false` 停止。

