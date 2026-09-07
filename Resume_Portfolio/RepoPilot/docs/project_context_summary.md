# RepoPilot 实验上下文审计（2026-08-23）

## 任务与基线

RepoPilot 是一个本地优先、受权限和验证边界约束的 Coding Agent。现有代码已具备 BM25、可选本地
dense/hybrid 检索、结构化工具、确定性验证器、trace/receipt、checkpoint、reviewer 与 Docker
边界。原有 FRAMES、DABench、SWE smoke 和训练产物均保留为历史诊断，不用作本实验计划的正式结论。

本次新增的研究边界是：

- R1：可独立消融的 AdaptiveCodeHybrid（RRF、查询路由、Python symbol/test 邻接、固定预算选择）；
- R2：Plan--Execute--Verify 状态机、一次有界 replan 与可解释风险 reviewer gate；
- R3：可选接入 runtime 的 JSON-schema、路径范围、阶段 allowlist 与结构化 recovery code；
- R4：固定 schema 的 typed provenance memory、内容哈希 cache、LongMemEval 离线指标；
- P0：不可覆盖 split manifest、run receipt、Wilson 区间和配对 bootstrap。

## 固定外部源

| 资源 | 本地 checkout / digest | 用途 |
|---|---|---|
| LongMemEval | 9e0b455f4ef0e2ab8f2e582289761153549043fc；data d6f21e...a442 | R4 数据与官方评测器接口核对 |
| CodeRAG-Bench | f9e100ca9ed94b8f1983b356ae81966e30210cf4 | R1 BEIR/RepoEval 管线核对 |
| BFCL | 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 | R3 离线数据与 evaluator 核对 |

## 可运行性与风险

- Python 3.12.3、pytest 8.4.2、Docker Server 29.6.1 与本地
  nomic-embed-text/Qwen 模型可用；
- 2026-08-24 的 E: 预检为 148.20 GiB 空闲，且已在 E: 的隔离 SWE-bench checkout 中通过官方
  `--gold` 1/1 evaluator smoke 与单实例 generation smoke；Windows CRLF 兼容补丁见 ADR 0006；
- Docker Desktop 的 image VHDX 仍位于 C:。正式 SWE-bench holdout 仍须为所有配对配置、重复、
  workspace 与镜像缓存预留足够空间；若迁移到 Linux/云端，须固定同一代码 commit、数据 digest 与
  Docker digest；
- LongMemEval-S 的严格“任一共享 session 即同组”规则使 500 题连成单一泄漏组，不能构造计划要求的
  20/40/40 split。R4 不得在此规则下开始调参或产生 final 数字。

## 当前结论边界

本仓库现在可以产生可复查的实验配置、数据 receipt、离线指标、工具契约和状态机回归测试；尚未产生
R1/R2/R3/R4 任一正式 final holdout 的能力比较结果。因此任何简历中“提升”“优于”“降低”数字仍
必须留空。
