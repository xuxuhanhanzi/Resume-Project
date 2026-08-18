# Changelog

## 0.2.3 - 2026-08-14

- 修复迁移后训练数据绝对图片路径失效，v2 改用可移植的项目相对路径；
- 全量验证 1,763 个 group、3,526 个 SFT gold completion 与三通道 reward；
- 完成 E3/E4/E5 200 steps × 3 seeds CPU 固定张量实验和精确 checkpoint 恢复。

## 0.2.2 - 2026-08-14

- 固化 `v0.2.1` 干净克隆、锁定依赖与无数据 CI 环境验收记录。

## 0.2.1 - 2026-08-14

- 区分数据 payload `verified`、`unavailable` 与 `failed`，修复干净克隆/CI 对受限数据的错误依赖；
- 增加 `--require-datasets` 正式数据门，payload 存在但身份错误时仍强制失败。

## 0.2.0 - 2026-08-14

- 新增 traceable Structured SFT/GRPO 数据构建器；
- 新增官方 ORM 形式的 ms-swift 三通道 reward plugin；
- 新增 Chart-FGRPO group advantage、dual state 与 checkpoint 恢复；
- 新增 faithful metrics、paired bootstrap 和 exact McNemar；
- 本地质量门提升到 56 tests、42 files strict mypy。
