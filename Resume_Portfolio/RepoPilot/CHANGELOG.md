# Changelog

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
