# ForgeMM 状态

## v0.2.3 本地实验门

- [x] 56 tests、Ruff、strict mypy；
- [x] ChartQA/ChartQAPro loader、全量审计与数据身份；
- [x] 1,763 条严格 EvidenceStore、Parser、安全 Executor、三通道 Reward；
- [x] 3,526 SFT 模板序列与 1,763 GRPO prompt，输出哈希冻结；
- [x] 修复迁移前绝对图片路径：v2 使用 `datasets/...` 相对路径，1,763/1,763 图片存在，3,526/3,526 gold completion 的三通道 reward 全为 1；
- [x] ms-swift ORM reward plugin CPU 字段契约；
- [x] Chart-FGRPO advantage、mask、dual update/clip/checkpoint/restore；
- [x] E3/E4/E5 CPU 固定张量对照：200 steps × 3 seeds，所有 advantage 有限，E5 checkpoint 恢复最终 dual state 精确一致；
- [x] FCR、inconsistency、paired bootstrap 与 McNemar 统计；
- [x] 数据 payload 缺失返回 `unavailable`，存在但哈希错误返回 `failed`，CI 不依赖受限数据；
- [x] `v0.2.1` 干净克隆按固定锁文件重建，标准隔离安装、Ruff、strict mypy、56 tests 与 clone-aware audit 全通过；
- [ ] Linux/RTX 4090 上 ms-swift 4.2.2 推理、QLoRA SFT、GRPO 与字段透传 smoke；
- [ ] E0–E5/A1–A2 云端训练、三种子正式评测；最佳 Adapter 产生后回本机完成 4-bit 演示；
- [x] 干净克隆与本地 release；
- [ ] 远程发布。
