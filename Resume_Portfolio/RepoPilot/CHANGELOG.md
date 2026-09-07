# Changelog

## 1.19.0 - 2026-08-23

- P61 新增非破坏性 release-scope inventory，按产品候选、生成证据、参考材料和本地状态分类 Git
  条目，保持人工选择 commit 的边界；P62 新增 CI/Docker preflight，实机 Docker sandbox 安全门已在
  本机重新构建并通过，远程 CI 仍要求配置 remote 后由维护者推送；
- P63 用固定无工具标记完成一次 DeepSeek 流式验收，并将取消、跨 runtime 恢复、revision/undo 与权限
  交由确定性本地测试覆盖；P64 增加离线 `/status`、`/workflow`、`/changes`、`/context` 转录验收；
- P65 为本地 wheel smoke 增加 SHA-256 receipt 和离线 verifier；P66 新增开发 receipt reliability
  dashboard 和不含 task/gold/hidden payload 的外部 holdout 元数据契约。Qwen 未配置或调用。

## 1.18.0 - 2026-08-23

- P52/P53 新增非破坏性 Git release-baseline 与显式 Docker 实时安全门；Docker 守护进程缺失会作为
  blocker 退出，绝不回退至宿主机执行；
- P54–P57 覆盖取消 checkpoint 跨 runtime 重启恢复，DeepSeek SSE 终止标记验证、`/changes` 本地 revision
  概览、语义保持的 acceptance 源码脱敏，以及既有 preview/undo/rewind 冲突保护；
- P58 对三份指定 provider 文件完成 DeepSeek 无工具只读验收并分类外部建议；首次因源码脱敏失真而不采信，
  修复后复验完成，未自动修改代码；P59 增加 Windows CI acceptance job，P60 固化发布步骤。Qwen 未配置或调用。

## 1.17.0 - 2026-08-23

- P44–P47 将 P40 只读外部建议逐项本地分级，只修复可复现的取消竞态、跨线程通知、POSIX 进程组、
  Docker 工作区默认可写和 UTF-8 截断问题；未把未复现的符号链接说法当作事实；
- P48/P49 将无可见最终文本分类为可操作的 provider 失败，并覆盖取消 checkpoint 后继续会话；
- P50/P51 增加不改 PATH/Profile/凭据/旧元数据的 Windows 手动恢复指南，以及显式 opt-in、保留证据的
  干净虚拟环境 wheel 安装 smoke 检查。Qwen 未配置或调用。

## 1.16.0 - 2026-08-23

- P38 增加 `auth probe deepseek`：仅发送固定公开标记，输出无内容的 HTTP/SSE 完成诊断；DeepSeek 的
  有界 probe/acceptance 请求关闭隐藏推理，以避免小 token 预算只产生内部推理而没有最终文本；
- P39/P40 增加 acceptance 的发送前 intent receipt、成功/空响应/传输失败 final receipt 与离线
  `acceptance show|verify`。经用户授权，P40 对此前相同三份文件完成一次无工具、只读、无修改验收；
  receipt 完整性和本地取消/恢复测试均通过，外部发现没有自动转化为代码修改；
- P41–P43 增强 `/status`、`/workflow`、`/trace` 的运行时可见性，增加不执行修复的
  `doctor --fix-plan`、项目固定 `scripts/repopilot.ps1` 启动器和 receipt 篡改/流式空响应回归覆盖；
  未修改 PATH、PowerShell profile、凭据或 Anaconda 安装，Qwen 未配置或调用。

## 1.15.0 - 2026-08-23

- P31/P32 为新 synthetic coding-development receipt 绑定 `outcomes.redacted.jsonl` 的 SHA-256，记录固定的
  fixture 验证命令，并提供离线 `eval coding-dev verify-receipt`；历史 receipt 仍可读取，但会标记为
  `legacy_unbound`；
- P33–P36 强化交互取消后的 checkpoint 提示、DeepSeek-first `/model` 引导、Windows `shell-doctor` JSONL
  输出，以及 `python -m repopilot` 下正确的项目虚拟环境 launcher 诊断；没有修改 PATH、PowerShell profile
  或 pip 元数据；
- P37 增加显式文件清单、最大 4 文件/48 KB、无工具无编辑的 DeepSeek 只读验收命令。凭据、会话、`.git`、
  `.repopilot`、虚拟环境和 artifacts 均被拒绝；真实外传仍需逐文件授权。

## 1.14.0 - 2026-08-23

- P28 增加离线 `repopilot eval coding-dev compare`：逐案与漏斗类别对比两个公开合成运行；只有
  suite digest、case set、provider、model 和 execution profile 全部一致才标记为 `controlled`，
  旧 P21 到 P23 的变化明确是 `directional_only`；
- P29 收紧 coding-development receipt 的字段、大小、计数、运行 profile 和 outcome 对齐校验；
  report/compare 拒绝未知字段，不会将任意篡改字段带入输出；
- P30 补充 compare/receipt 回归覆盖和本地质量门禁。未配置或测试 Qwen，未传输真实项目内容至任何 provider。

## 1.13.0 - 2026-08-23

- P21 在公开四任务合成开发集上完成一次 DeepSeek `deepseek-v4-flash` 诊断：3/4 通过确定性验证，
  剩余一例在测试已通过后请求可选 diff 并达到 12,000 token 预算；该结果不是能力 benchmark；
- P22 为受信任的公开 fixture 增加“成功精确 patch + 不可变测试通过后立即确定性验证”的 opt-in
  完成路径，普通交互会话与非 fixture 任务保持原行为；P23 在相同 suite/model/case set 上复跑为 4/4；
- Qwen 未配置或调用；全局 Anaconda PATH 与旧 pip metadata 未自动修改。真实仓库云端只读验收需另行
  授权文件内容发送至 provider；发布前本地门禁与回归测试仍可执行。

## 1.12.0 - 2026-08-23

- P16 增加合成 coding-development run receipt 与离线 `eval coding-dev report`：记录可比较、
  脱敏的状态/计数/失败原因，并按实际漏斗类别生成 P17 诊断队列；不生成能力分数，也未自动调用云端模型；
- P18 增加 `shell-doctor powershell` 和隔离 PowerShell launcher 验收，明确诊断 Anaconda/PATH
  冲突而不修改 PATH、`$PROFILE`、凭据或 pip 元数据；
- P19 增加本地只读 `/workflow`，将计划、变更、验证和 repair readiness 汇总为安全的下一步提示；
- P20 增加不改状态的 Windows release gate，检查 package fixture、session-only launcher、依赖、
  whitespace、lint/type/test。没有新增 SWE 或综合 coding 分数。

## 1.11.0 - 2026-08-23

- 完成 CLI 后续 P11–P15：增加只读的 Windows 启动验收、冻结的合成 coding-development
  fixture 运行器、会话本地证据视图，以及验证日志不再自动进入后续云端模型上下文；
- supervisor 增加项目队列配额、标签、单任务超时、启动/超时诊断和离线 `show`；
- MCP 增加显式 probe 的本地脱敏审计、离线 `history`、受策略过滤后的 capability drift 和
  1–120 秒探测超时。没有新增 SWE 或综合 coding 分数。

## 1.10.0 - 2026-08-23

- 增加不破坏既有环境的 Windows 项目内 bootstrap、独立 coding development-funnel 评测和
  patch conflict 的强制重新读取恢复；
- 扩展 `/status`、显式本地 supervisor（不自动重放中断任务）与 MCP 服务级工具策略/命名 probe；
- P6 不报告新的 SWE 或综合 coding 分数，正式 holdout 保持冻结。

## 1.0.4 - 2026-08-14

- 新增通过 RepoPilot provider 的固定 digest Qwen2.5-7B 本地容量基准；
- 完成 1/4/8 并发各 24 请求，72/72 非空，并冻结原始结果 SHA-256 与声明边界。

## 1.0.3 - 2026-08-14

- 固化 v1.0.2 全新 venv 干净克隆：format/lint/type、90 tests 与离线 demo 全部通过；
- 记录 v1.0.0/v1.0.1 的换行/hash 失败，不移动或删除旧标签；无执行代码或数据变更。

## 1.0.2 - 2026-08-14

- 为两个历史 SWE-bench-Live smoke JSONL 增加精确 CRLF override；它们的冻结 SHA-256
  在历史上基于 Windows CRLF，而 DABench/FRAMES 哈希基于 LF；
- 避免用全局 JSONL 规则把 1.0.1 的 11 个已修复测试变成 4 个 SWE hash 回归。

## 1.0.1 - 2026-08-14

- 增加 `.gitattributes`，强制代码、manifest、JSONL/TSV/CSV 与文档以 LF 入库和检出；
- 修复 Windows `core.autocrlf` 在干净克隆中改变冻结数据字节、导致 DABench/FRAMES
  SHA-256 校验失败的问题；保留 v1.0.0 标签不移动。

## 1.0.0 - 2026-08-14

- 冻结本地优先 AgentRuntime、工具/策略、checkpoint/journal、确定性 verifier 与三领域 adapter；
- 完成可执行 Planner → AgentRuntime → Verifier → Reviewer 图、路由级联与恢复；
- 增加 Bearer 鉴权 JSON API、可续传 SSE、限流、请求上限、取消与 timeout；
- 完成 360 个真实 runtime/verifier 任务的 1/4/8 负载实验与资源记录；
- 固定 Docker 基础镜像 digest，现场验证非 Root、只读根、断网、cap/PID/CPU/内存限制；
- 冻结 FRAMES 11/60、DABench 73.3% mean、SWE 0/5 三个独立指标及 Adapter 负结果；
- 保留真实 Qwen 服务容量、多租户与 gVisor/微虚拟机为明确非目标/未验证项。
