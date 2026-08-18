# Evaluation Card：ForgeLLM Stage 6 quality v5 / systems v5 / final v6

## 评测问题

Stage 6 评估现有训练链是否产生满足本项目冻结约束的模型，并验证评测过程是否可追溯。它不尝试证明通用聊天、推理、安全或生产能力。

## 被测对象

- Q0：Qwen3-0.6B Base；
- Q1：Stage 4 SFT Adapter v2；
- Q2：Stage 5 DPO Adapter v2；
- Q3：一步 GRPO Adapter，只审计流程；
- M3：Stage 3 DecoderLM，单独套件。

## 数据与协议

64 个冻结案例（16 correctness、16 preference、32 robustness），Q0/Q1/Q2 各 64 条 raw generation。主协议为 greedy、32-token 上限、同一 Qwen Tokenizer 与 ChatML。精确污染 0，near-match 16。Manifest 同时绑定 Base revision、Adapter/Tokenizer/data hash、生成设置、Git 状态、运行源码聚合 SHA 和环境版本。

## Primary 结果

| 模型 | strict 原题 | retention/Q0 | char-8 repetition | 系统矩阵 | 行为状态 |
|---|---:|---:|---:|---|---|
| Q0 | 0/32 | 1.0000 | 0.1156 | 完整 | 未接受 |
| Q1 | 0/32 | 1.0227 | 0.1716 | 完整 | 未接受 |
| Q2 | 0/32 | 1.0930 | 0.4426 | 完整 | 未接受 |

Q1/Q2 的 expected-response loss 和 preference ranking 改善，但都没有 strict success；Q2 还未通过 stability 门。所有模型均 64/64 跑满 token 上限。不存在综合总分。

## Secondary 证据

- Q1 有 45.31% 输出为正确前缀后多写；Q2 为 62.50%；
- preference accuracy：Q0 0.7656、Q1 0.9844、Q2 1.0；
- paired strict Bootstrap 差值区间均 `[0,0]`；
- 规则 Judge 位置交换一致率 1.0；
- E10：Q1 32→64 token 的 median 约 2.214→4.437 秒，strict 均 0；
- M3：loss/PPL/BPB=1.6245/5.0759/2.0543，只在自身套件解释。

## 人工审计

30 对 Q1/Q2 匿名盲评包已生成，含 Prompt 与 rubric，模型映射独立保存。当前未完成人工标签，不报告 Cohen’s κ。学习者在评分前不得打开 private key。

## 系统边界

18 格系统矩阵在单台 RTX 4070 Laptop GPU、BF16、单进程下运行。结果受执行顺序、GPU 频率/电源和微型重复数影响，只用于本机教学验收。理论 KV 只计算 K/V 元素，不含权重、激活、workspace 或 allocator。

## 支持与不支持的声明

支持：Stage 6 的冻结身份、数据污染、质量、统计、盲评、系统矩阵和证据审计链路可运行；Q1/Q2 提高了冻结答案概率/偏好排序；三个模型的严格行为均未达标。

不支持：SFT/DPO 提升了通用模型能力；Q2 已对齐；Q3 的 GRPO 改善能力；M3 PPL 可与 Qwen 比较；本地微基准代表生产吞吐；该小型模板集代表开放用户分布。

## 证据入口

- 学习入口：`docs/lessons/stage06_learning_order.md`；
- 实验记录：`docs/experiments/2026-07-28_stage06_implementation.md`；
- 质量：`artifacts/stage06/quality_v5/`；
- 系统：`artifacts/stage06/systems_v5/report.json`；
- 最终门禁：`artifacts/stage06/final_v6/report.json`；
- 学习者验收：pending。
