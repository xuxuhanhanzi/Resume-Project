# 四项目 100% 完成路线图

> 冻结日期：2026-08-14  
> 口径：只有可复现、可定位到命令/日志/产物的验收项才计为完成。外部资产缺失时，
> “脚本和运行手册就绪”不等于“实验完成”。

## 统一完成定义

每个项目达到 100% 必须同时满足：

1. 干净环境可重建，格式、Lint、类型检查、单元/集成测试全部通过；
2. 核心功能存在端到端演示，不依赖未记录的本机状态；
3. 性能或算法数字有冻结输入、完整命令、环境、日志、结果与哈希；
4. README、状态文档、简历条目和面试材料使用同一证据口径；
5. 有独立 Git 历史、干净克隆复现记录、版本标签和发布说明；
6. 不把合成、smoke、模拟负载或开发集结果描述为正式性能。

## 当前总览

| 项目 | 当前可验证状态 | 100% 仍缺少的硬门槛 | 下一阶段 |
|---|---|---|---|
| Hospital Workforce Platform | Java 21 与前端生产镜像、MySQL 8.4 下 7/7 测试、七服务 Compose、真实 Keycloak P0 演示、1,000 条排班 k6、独立 Git、干净克隆与 `v1.0.1` 本地发布均通过 | 无剩余实验；视频与远程发布属于交付动作 | 实验闭环 |
| RepoPilot | HTTP/SSE、真实 AgentRuntime 1/4/8、固定镜像安全、三领域指标、真实 Qwen2.5-7B 1/4/8 容量、90 tests 均已冻结 | 无剩余实验；仅远程发布 | 本地实验闭环，`v1.0.4` |
| DriveVLA-Guard | `formal_v5` 72/72；1,200 场景 K/seed/threshold 诊断共 20,400 scene-runs；严格 preflight 与质量门通过 | 云端官方 AutoVLA + NAVSIM B0/E2，判定后 E3/E4 | 本地实验闭环，`v1.0.2` |
| ForgeMM | 1,763 EvidenceStore、v2 3,526 SFT/1,763 GRPO、全量 reward 回放、E3/E4/E5 三种子 CPU 算法诊断、56 tests 均通过 | 云端 ms-swift smoke、E0-E5/A1-A2 训练与冻结评测 | 本地实验闭环，`v0.2.3` |

## Hospital Workforce Platform

### H0：构建基线（已完成）

- [x] `npm run build` 成功；
- [x] Docker Desktop 29.6.1 可用；
- [x] `docker build --network=host -t hospital-workforce-platform:verification .` 成功；
- [x] Java 21 编译 55 个主源文件、3 个测试文件，生成可运行 Spring Boot JAR/镜像；
- [x] Dockerfile 使用 BuildKit Maven cache，并为不稳定下载设置有限重试。

### H1-H4：进行中

- [x] H1：运行 Payslip、Spring Modulith、MySQL Testcontainers 测试并保存 Surefire 报告；
- [x] H2：补充 401/403/角色授权、幂等重放、并发冲突和审计记录自动测试；
- [x] H3：Compose 启动 MySQL/Redis/Keycloak/backend/frontend/Prometheus/Grafana，完成健康检查和 P0 API 演示；
- [x] H4：冻结 50 名员工、1,000 条排班后运行 5 分钟 k6；16,746 请求、0% 错误、55.70 req/s、p95 5.83 ms，并保存原始 JSON 与哈希；
- [x] 建立独立 Git 仓库；`v1.0.0` 干净克隆中前端构建、后端 7/7 测试与 Compose 配置通过；以不改写旧标签的 `v1.0.1` 固化验证记录；
- [ ] 按 `docs/demo-storyboard.md` 录制 2-3 分钟视频，并在用户选择远程仓库后发布 release（需要外部写入授权）。

## RepoPilot

- [x] 实现 Bearer 鉴权 HTTP JSON API 与可续传 SSE，覆盖限流、请求上限、task path allowlist 和断连恢复；
- [x] 用真实 AgentRuntime + 文件修改 + 子进程 Verifier 运行每档 120 个任务的 1/4/8 并发，保存 P50/P95/P99、吞吐和进程内存；模型为 ScriptedProvider，不能当作 Qwen 吞吐；
- [x] 用固定 digest `qwen2.5:7b` Q4_K_M 运行真实模型服务 1/4/8 并发各 24 请求；72/72 非空，吞吐 5.52/16.80/18.62 req/s，并与 AgentRuntime 调度容量分开报告；
- [x] 覆盖模型不可达、模型异常、service/node timeout、worker 中断、checkpoint 恢复、取消、SSE 断开与硬并发排队；
- [x] 完成固定 digest Docker 现场验收：非 Root、只读 RootFS、cap-drop、no-new-privileges、断网、无 Docker socket、CPU/内存/PID 限制；覆盖路径/符号链接逃逸与 hidden-test contract 隔离；
- [x] 依据 5 个 fresh task 的 0/5（4 空 patch、1 不充分 patch）形成明确负结果停止决定；没有新干预时不继续消耗/污染 validation，重新开启条件已冻结；
- [x] 冻结卡片、runbook 与证据索引；`v1.0.2` 干净克隆通过格式/Lint/strict mypy、90 tests 和离线演示，以 `v1.0.3` 固化验证记录与本地 release tag；远程发布等待用户选择仓库。

## DriveVLA-Guard

- [x] 完成配置边界校验、AutoVLA 多候选 adapter、无未来信息泄漏 NAVSIM agent 和 B0/E2/E3/E4 选择脚本；
- [x] preflight 记录路径类型、模块版本、GPU/显存、上游文件契约、commit 与 blocker；本机 8GB 显存明确 `ready=false`，不启动无效正式实验；
- [x] `formal_v5` 在冻结 12 场景上完成 B0/B1/E1-E4 共 72/72 场景，summary/comparison 全部语义重算一致并由 SHA-256 清单保护；
- [x] 完成 1,200 场景 K=1/2/4/8 × 3 seeds 与五档风险阈值诊断，共 20,400 scene-runs、0 执行失败；K=4 是最小零碰撞代理 K，K=8 只增加声明延迟；
- [x] Ruff、21 tests、compileall、一键发布检查和独立 Git 仓库完成；`v1.0.0` 干净克隆按锁文件重建并重跑完整矩阵，以 `v1.0.1` 固化记录；
- [ ] 在 AutoDL 记录 GPU、Python/PyTorch/CUDA、代码 diff、模型/数据哈希；
- [ ] 通过 checkpoint、NAVSIM、metric cache 和 agent preflight；
- [ ] 固定 navtest manifest 运行官方 B0 与 E2；
- [ ] 只有 NC/DAC/TTC 改善且 EP/Comfort 可接受时才运行 E3/E4；
- [ ] 保存逐场景预测、官方 PDMS 分解、延迟、显存和失败案例；
- [x] README/状态/简历/面试材料已统一当前合成证据和 claim boundary；本地 release 完成；
- [ ] 官方结果回填与远程 release 等待外部资产和用户选择的仓库。

## ForgeMM

- [x] 本地 Ruff、strict mypy 与 57 tests 通过；
- [x] 1,763 条唯一严格标签生成 v2 的 3,526 条双模板 SFT 序列和 1,763 条 GRPO prompt；图片使用可迁移相对路径，1,763/1,763 存在，3,526/3,526 gold reward 回放全通过；
- [x] 完成官方 ORM 形式的三通道 reward plugin CPU 字段契约；
- [x] 完成 group advantage、constraint mask、dual update/clip、checkpoint 恢复、FCR、paired bootstrap 与 exact McNemar；
- [x] 完成 E3/E4/E5 200 steps × 3 seeds CPU 固定张量实验；所有 advantage 有限，中途恢复的最终 dual state 精确一致；
- [x] `v0.2.1` 干净克隆按锁文件和标准隔离构建通过 56 tests/strict mypy，以 `v0.2.2` 固化本地发布证据；
- [x] 云端 RTX 4090 固定 ms-swift 4.2.2，并依次通过推理、QLoRA SFT、4×2 GRPO、三路字段透传与 checkpoint smoke；
- [x] Structured SFT 序列门达到 3,526，同时保留唯一严格记录数 1,763 和 group key；
- [x] 完成 reward plugin、Chart-FGRPO advantage、dual state checkpoint 与恢复测试；
- [ ] 云端按 E0-E5/A1-A2 矩阵运行 quick 和正式实验；E3/E4/E5 使用 17/42/2026 三种子；
- [ ] 冻结 ChartQA test 与 ChartQAPro，报告 FCR、Evidence F1、Operation Consistency、准确率和统计区间；
- [ ] 云端产生最佳 Adapter 后进行 4-bit 交付验收；该项依赖云端训练产物，不是当前可独立执行的本地实验。

## 仅剩云端实验矩阵

本机可独立执行的实验为 **0 项待办**。以下项目都被当前 8GB 显存、缺失模型/数据资产或
上游训练产物硬阻断：

| 项目 | 云端实验 | 最低前置条件 | 启动/停止门 |
|---|---|---|---|
| DriveVLA-Guard | AutoVLA + NAVSIM B0、E2；满足门槛后 E3、E4 | Linux、24GB+ GPU、16.3GB checkpoint、Qwen2.5-VL、NAVSIM/navtest、metric cache | preflight `ready=true`；B0/E2 无 NC/DAC/TTC 改善则停止 E3/E4 |
| ForgeMM | ms-swift 推理/32-sample QLoRA/4×2 GRPO smoke；E0-E5、A1-A2；三种子冻结评测 | Linux、24GB+ GPU、Qwen2.5-VL-3B、ms-swift 4.2.2、训练输出目录 | smoke 全通过才训练；E2 不达格式/可执行率门槛则停止 RL |

Hospital 与 RepoPilot 没有剩余云端实验；它们只剩远程发布等外部交付动作。ForgeMM 的
本机 Adapter 演示必须等待云端训练产生 Adapter 后才能执行，属于下游验收依赖。

## 执行原则

- 先完成本机可闭环的工程门，再申请/使用远程算力；
- 每个正式实验先写计划，保留失败 run，不覆盖旧结论；
- 项目百分比只在本文件对应硬门槛完成并有证据后更新；
- 任何项目都不能仅通过修改 README 被标记为 100%。
