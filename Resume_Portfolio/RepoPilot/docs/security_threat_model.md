# Security Threat Model

必须覆盖：Prompt Injection、Secret 泄漏、命令注入、路径穿越、符号链接逃逸、恶意依赖、资源耗尽、外网回连、测试投机与 CI/部署文件篡改。

P0 要求：临时工作区、默认断网、非 Root、只读根文件系统、资源限制、无宿主 Docker Socket、无真实 Secret、路径规范化与高风险文件审批。

## 已实现控制

- Context 将仓库、Issue、工具结果和 Diff 标记为不可信数据；
- 所有模型路径经过 canonicalize、workspace containment 和 Task allow/deny list；
- 默认工具不包含任意 Shell，测试命令来自不可变 Task Spec token list；
- `apply_patch` 只执行精确文本替换并限制修改文件数；
- CI、部署、依赖和权限文件触发结构化审批；
- Trace 对常见 secret/token/password/authorization 字段脱敏；
- 本地 Runner 对不可信任务 fail-closed；Docker Runner 生成断网、非 Root、只读 RootFS、
  capabilities drop、no-new-privileges 和资源限制命令；
- PublicTaskSpec 拒绝 `hidden_tests`，HiddenTestGrader 在独立数据边界中使用它们。
- 固定 digest 的 Docker 沙箱已在 Docker Desktop 现场验证：uid/gid 65534、只读 RootFS、
  CapEff 全零、NoNewPrivs=1、断网、无 Docker socket、1 GiB 内存、1 CPU、64 PID；
- 路径测试覆盖 `..`、绝对路径和指向工作区外的符号链接；HTTP task path 也必须位于
  allowlisted task root。

## 尚未证明

- 未执行敌对依赖安装、真正的 fork bomb、磁盘填满或容器逃逸利用；资源/mount 限制已
  通过 cgroup 与内核状态读取验证；
- Prompt Injection 只能通过权限与数据流降低风险，不能证明模型永不受影响；
- 尚未接入 gVisor、Firecracker 或独立网络审计；
- 微型基准中的 `verify.py` 全部是公开开发检查；SWE 官方 evaluator 的隐藏边界由独立
  task contract 和容器评测路径维护，不对 Agent 暴露。
