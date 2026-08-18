# Configurations

配置按数据、模型、训练、后训练、评测与服务分层。每一次正式运行应保存解析后的只读配置副本。

Stage 3 训练配置按成本阶梯排列：

1. `training/stage3_smoke.toml`：20-step 接口检查；
2. `training/stage3_qualification.toml`：5.36M/BF16、100-step 资格门；
3. `training/stage3_bounded.toml`：1M target tokens 或 60 分钟的正式上限。

不要跳过 smoke/qualification 直接扩大预算，也不要在同一次对照中同时修改 optimizer 和 MTP 目标。

Stage 5 配置：

1. `alignment/stage5_qwen_dpo_smoke.toml`：验证 reference/policy 首 batch `log 2`、Dropout 关闭和 Adapter-only 梯度；
2. `alignment/stage5_qwen_bounded.toml`：正式 DPO 50-step/20k pair-token/20-minute 三重上限，以及 GRPO 4 prompts × group 4 × 32 tokens、恰好一步的冻结协议。

两份配置都绑定 Qwen Base revision 与 Stage 4 Adapter 路径。DPO/GRPO 输出目录必须是新目录；不得覆盖 `artifacts/stage05/*_v1` 或 `*_v2`。

Stage 6 配置：

1. `evaluation/stage6_final.toml`：冻结 Q0–Q3/M3 身份、64 例任务来源、greedy 32-token 质量协议、2,000 次 paired bootstrap、30 对盲评，以及 batch 1/4 × prompt 32/128/256 的系统矩阵；
2. Stage 6 不训练或搜索超参数；看到 test 结果后不得调整生成设置；
3. v1 因多轮历史 assistant turn 被错误删除而失败；v2 完整但 expected-response BPB 混入 ChatML 终止 token；v3 因源码身份不充分而早停；v4/v4/v5 保留为格式统一前的有效历史证据；当前正式质量版本为 `quality_v5`，系统为 `systems_v5`，最终报告为 `final_v6`。所有历史证据保留，禁止拼接版本。
