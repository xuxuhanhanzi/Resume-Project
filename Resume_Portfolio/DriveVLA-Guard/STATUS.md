# DriveVLA-Guard 状态

## v1.0.2 本地发布门

- [x] 配置校验、风险重排序、快慢路由与严格前缀续跑；
- [x] AutoVLA 多候选 adapter 与 NAVSIM agent；
- [x] 21 项测试、Ruff、compileall；
- [x] 跨平台 B0/B1/E1–E4 合成复现；
- [x] `formal_v5` 72/72 场景、摘要/配对语义重算、SHA-256 冻结清单；
- [x] preflight 校验路径类型、依赖、GPU/显存、上游文件与 commit；
- [x] CI、一键发布检查、运行手册、结果报告、简历与面试口径；
- [x] `v1.0.0` 独立干净克隆：锁定依赖、Ruff、21 tests、compileall、证据校验和新 72 场景矩阵全部通过；
- [x] 本地 1,200 场景压力诊断：K=1/2/4/8 × 3 seeds 与 5 档路由阈值共 20,400 scene-runs，0 失败；K=4 为最小零碰撞代理配置，K=8 无额外安全收益；
- [ ] 官方 NAVSIM B0/E2：2026-08-15 AutoDL RTX 4090 严格 preflight 已实际运行并返回
  `ready=false`；缺少官方 AutoVLA checkpoint/checkout、NAVSIM/nuPlan 数据与 metric cache，
  且 24,564 MiB 比当前 24 GiB 门少 12 MiB。详见
  `docs/experiments/2026-08-15_autodl_official_preflight.md`；
- [ ] 远程 release 与演示视频：等待用户选择仓库和发布渠道。

合成结果只证明工程行为。正式 NAVSIM 指标缺失时，简历不得声称 PDMS 提升或真实道路安全提升。
