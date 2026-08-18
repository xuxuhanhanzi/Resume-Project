# 3 分钟演示脚本

1. 0:00–0:25：说明 AutoVLA 单轨迹基线与“不重训基础模型”的项目边界；
2. 0:25–1:00：展示 `GuardPipeline` 的 K 候选、风险分解、重排序和 slow routing；
3. 1:00–1:35：运行 `python scripts/check.py`，展示 21 项测试与新目录完整复现；
4. 1:35–2:10：对比 B0 与 E2/E4 的逐场景轨迹、风险分量、slow reason 和延迟；
5. 2:10–2:35：运行 `verify-evidence`，展示 config/source hash 与 SHA-256 防漂移；
6. 2:35–3:00：展示官方 preflight 的 `ready=false` 和阻塞项，明确合成代理与 NAVSIM/真实道路的结论边界。
