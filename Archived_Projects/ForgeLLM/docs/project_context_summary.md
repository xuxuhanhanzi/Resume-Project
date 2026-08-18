# Project Context Summary

## P0 goal

形成模型优先的 LLM 学习与项目证据链：Tokenizer、5M–20M Decoder-only Transformer、预训练与恢复、自定义 C++/CUDA 算子、Stage 4 SFT/LoRA/QLoRA、Stage 5 DPO/RL 最小闭环，以及贯穿各阶段的固定评测。

## Current state

工程与数据夹具基线和 Stage 0–5 的自动化/学习者验收已完成。Stage 6 已完成 G6-A～G6-E：64 个冻结案例、192 条 Q0/Q1/Q2 原始生成、污染与统计审计、30 对盲评包、M3 独立语言建模、Q3 pipeline-only 审计、18 格系统矩阵、E10 32/64-token 审计、最终证据报告与五份完整讲义。自动化门通过，G6-L 等待学习者按 18 站完成。

## Open risks

- 本地约 8 GB GPU、每周 60 小时和约 71.6 GiB 工作盘余量已登记；外部预算精确口径在首次付费任务前确认；
- Stage 3/4/5 数据来源、许可证、revision、split 与 hash 均已冻结；所有结果只支持小规模方法结论；
- Stage 4 目标是可规则验证的 constraint/format following，不扩张为未知领域适配；正式行为门未通过；
- Qwen3-0.6B-Base BF16 LoRA 和 Windows bitsandbytes NF4 QLoRA 已实测；QLoRA measured peak 只略低于 BF16，不能宣传显著节省；
- 原 word trigram 指标漏检无空格重复；Stage 5 已预登记 character 8-gram 并捕获 DPO 重复退化；
- Stage 6 复核 DPO v2：pair accuracy 1.0，但 strict 0/32、截断率 1.0、字符重复 0.4426；三个模型行为均未接受；
- 大模型 RL 仍受 Verifier、8GB 显存和外部预算约束；Stage 5 只运行一步真实 GRPO，不作能力结论；
- Kimi K3 官方报告已进入事实注册表和 MOPD 教学；DeepSeek V4 因无可核验官方技术报告继续 pending；
- Windows 官方环境没有可工作的 Triton，因此尚未验证 Inductor 性能；这不影响已通过的自定义 C++/CUDA 算子门；
- 不得让 CI、服务或平台工程重新成为模型主线的全量学习阻塞。
- Stage 6 污染审计有 0 exact、16 个同模板 near-match；结果不能外推到开放任务；
- 系统矩阵只代表单台 4070 Laptop GPU 的固定顺序 microbenchmark；Q0 的异常慢 batch=1 不支持 Adapter 普遍加速声明；
- 当前唯一未关闭项是 G6-L：人工盲评、统计手算、代码追踪和结论边界答辩。
