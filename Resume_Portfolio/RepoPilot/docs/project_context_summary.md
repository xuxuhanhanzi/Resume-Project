# Project Context Summary

> ⚠️ This is a Stage 1–6 era snapshot. For the authoritative current state (P7–P12
> complete, gates green, metrics), see [`docs/STATUS.md`](STATUS.md).

## P0 goal

研究一个本地优先的 Agent 系统，如何通过规划、检索、工具、执行反馈、验证、记忆和治理机制，
在知识研究、数据分析、软件工程三类任务形态下提高成功率，同时控制 tokens、耗时、安全风险
和失败恢复成本。三领域分别报告各自官方主指标，不合成跨领域总分。

## Current state (as of 2026-08-09; superseded by STATUS.md)

Stage 1–6 教学型实现 + P7–P12 全部完成。三领域协议和评测链路已跑通：

- 知识研究 (FRAMES)：R2 hybrid 三次重复均值 **20.0%**（20/30/10），仅供参考，未达可引用规模。
- 数据分析 (DABench)：val35 base **71.4% (25/35)** —— 唯一可引用主指标（哈希冻结、官方评分）。
  早期 10 题 smoke 的 90% 高估约 19pp，不可引用。
- 软件工程 (SWE-bench-Live)：Agent 官方 resolved = **0/3**，不可引用（三题已被开发污染，仅作 dev smoke）。

本地 Qwen2.5-7B (Q4_K_M, digest 845dbda0ea48...) 已冻结为正式重复实验模型，
temperature=0, seed=7, 网络 deny。Docker daemon 29.6.1 已验证可用，DABench 隔离镜像已物化。

**模型训练（P10 已完成）**：对 Qwen2.5-1.5B 做了 QLoRA SFT + 手写 DPO（adapter 已生成），
但 **尚未接入评测链路**，因此不宣称任何下游准确率提升。

## Open risks

- FRAMES 10 题结果不能外推；需扩到 ≥50 题做 3 次重复才可能升为"可引用"。
- SWE-bench-Live 现有三题已被开发过程污染，后续必须冻结新的未查看 holdout。
- FRAMES 绝对准确率低，证据合成（非检索）仍是主要瓶颈（P9 A2 已证）。
- P10 adapter 未接入评测，下游收益未知。
- 不得把自训练模型作为 RepoPilot MVP 的前置依赖。
