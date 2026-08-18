# Stage 5 讲义五：Kimi K3、在线蒸馏与失败分析

## 1. 前沿模型学习的正确姿势

对于最新模型，应把“官方披露事实”“本项目实现”“合理推断”分开。截止 2026-07-28，Kimi K3 已有官方仓库和技术报告；DeepSeek V4 没有找到可核验的 DeepSeek 官方技术报告，因此仍标为 pending，不猜测其训练方法。

学习一项前沿技术时固定回答四个问题：旧问题是什么、核心公式/数据流是什么、项目实现到哪一层、它没有解决什么。模型榜单分数不能替代机制证据。

<details>
<summary>思考题：社交媒体或框架“已支持 DeepSeek V4”能否证明官方模型使用某种 RL 算法？</summary>

答案：不能。框架支持可能只涉及配置、权重格式或推理算子；没有官方报告/模型卡/仓库，就不能把训练技术归因给该模型。
</details>

## 2. Kimi K3 的后训练总图

Kimi K3 官方报告披露的后训练不是“只跑一个 GRPO”：先做 SFT cold start，再把强化学习按三个领域和三种 reasoning effort 组织为 9 个专家，随后蒸馏/融合回最终模型。报告还讨论 partial rollout、staleness regularization、effort budget、Agentic GRM 的 verbosity control，以及部署感知量化训练。

```text
SFT cold start
→ 3 domains × 3 reasoning efforts = 9 RL experts
→ 专家产生相对 Base 的 dense delta signal
→ 多教师在线策略蒸馏（MOPD）
→ 最终通用 policy
```

这解决的是多领域、多推理强度专家如何合并，而不只是在单一 reward 上继续训练。项目没有 K3 权重和规模资源，只实现核心 dense reward 张量语义。

<details>
<summary>思考题：为什么训练 9 个专家后还要蒸馏回一个模型？</summary>

答案：部署时同时路由和维护多个大专家成本高，且不同领域能力需要共存。蒸馏尝试把专家相对 Base 的行为增量转移到一个 student；代价是教师冲突和容量瓶颈。
</details>

## 3. OPD：学习 teacher 相对 Base 的 delta

普通蒸馏常让 student 直接匹配 teacher 分布。On-Policy Delta Distillation 更关注 teacher 相对一个共同 Base 的改进信号，避免把教师全部背景分布重复复制。K3 报告中的 per-token 核心可抽象为 teacher/student log-prob 差的 stop-gradient、裁剪 dense reward：

```text
r_t = stopgrad(clip(log p_teacher(y_t|h_t)
                    - log p_student(y_t|h_t), -Rmax, Rmax))
```

项目 `opd_delta_reward` 实现这一最小张量契约：shape 对齐、reward 截断、teacher/student 差 detach。它不是 K3 训练系统复现。

<details>
<summary>思考题：为什么 delta reward 要 stop-gradient？</summary>

答案：它被当作外部学习信号，而不是让优化器通过 reward 公式同时移动教师或制造二阶捷径。policy gradient 应沿 student 的 log-prob 路径更新，reward 本身不反传。
</details>

## 4. MOPD：多教师如何进入同一 student

Multi-teacher OPD 为每条样本选择对应领域/effort 教师，再计算 dense delta。项目输入形状为教师 `[batch, teachers, tokens]`、student `[batch,tokens]` 和每条样本一个 teacher index；先选择教师，后调用同一 OPD reward。

关键难点不是写出 `gather`，而是教师路由、不同教师尺度、冲突样本和 student 容量。项目的 frozen tensor 实验得到平均 dense reward 0.35，只证明选择与裁剪代码正确，绝不代表 9 专家能力已合并。

<details>
<summary>思考题：把 9 个教师 logits 简单平均是否等价于 MOPD？</summary>

答案：不等价。简单平均忽略样本所属领域与 reasoning effort，也可能把互相冲突的分布混成低质量目标。MOPD 需要明确教师选择/组合策略和对应身份。
</details>

## 5. Partial rollout 与 staleness

完整长轨迹昂贵；partial rollout 从已有前缀继续采样可降低成本，但前缀可能来自旧策略。policy 更新越快，旧前缀/old log-prob 与当前策略越不一致。K3 报告使用 staleness regularization 处理这种训练系统矛盾。

本项目用两种最小证据学习这个问题：`RolloutRecord` 把数据绑定到 policy revision；GRPO 首步检查 old/new log-prob；VESPO tensor study 展示 stale sequence 的软重加权与 ESS。没有实现异步 actor 或 partial-rollout 集群。

<details>
<summary>思考题：提高 rollout 吞吐与保持 on-policy 为什么会冲突？</summary>

答案：actor 生成越慢或 learner 更新越快，采样完成时 policy 已经变化。等待同步会浪费算力；继续消费旧数据会增加 off-policy 偏差。生产系统需要在同步、丢弃、重加权和正则之间折中。
</details>

## 6. Reasoning effort 与 verbosity 不是同一件事

K3 报告把低/中/高 reasoning effort 作为可控制预算，并用 Agentic GRM 处理 agent 任务中的 verbosity。更长输出不自动等于更多有效推理：它可能只是重复、解释性填充或工具调用冗余。

本项目 Stage 4/5 正好观察到这一点：模型常以正确答案开头后继续生成，DPO 后字符重复率明显恶化。Stage 6 必须将正确性、长度、重复、格式和成本拆开，不用单一 reward 吞并。

<details>
<summary>思考题：如果高 effort 模型准确率更高，能否只比较准确率？</summary>

答案：不能。还要报告 token/延迟成本，并在相同预算或形成 Pareto 曲线后比较；否则“用更多计算得到更高分”与“方法更有效”混在一起。
</details>

## 7. 部署感知 QAT 为什么出现在训练报告里

K3 官方报告披露了面向 MXFP4/MXFP8 的部署感知量化训练。原因是训练后再量化可能破坏分布，尤其 MoE、激活和长上下文路径对数值误差敏感。QAT 在训练中模拟部署数值，使模型适应目标格式。

这与本项目 Stage 4 QLoRA 不同：QLoRA 是用 4-bit 冻结 Base 节省微调存储并训练较高精度 Adapter；部署 QAT 是让最终模型适应推理数值格式。两者都出现“4-bit”，目标和梯度路径完全不同。

<details>
<summary>思考题：完成 QLoRA 是否意味着得到可直接 MXFP4 部署的模型？</summary>

答案：不意味着。QLoRA 的量化 Base 通常不是最终 merge 后的部署格式，也未校准目标硬件算子、激活和精度。部署量化需要独立导出、校准、内核与质量评测。
</details>

## 8. VESPO 与 K3 staleness 的关系边界

二者都关心策略陈旧，但不能说 K3“使用 VESPO”，除非官方材料明确如此。VESPO 给出了 sequence importance weight 的 Gamma 型软核；K3 报告描述自己的 partial rollout/staleness regularization 系统。项目把两者放在同一问题地图中学习，而不作算法归因。

<details>
<summary>思考题：两篇方法解决类似问题，为什么不能互相替换名称？</summary>

答案：问题相似不代表目标函数、数据流和理论假设相同。准确技术写作必须引用各自公式与实验，不能用主题相近代替证据。
</details>

## 9. Stage 5 两类关键失败

### 9.1 实施失败：模式不一致

DPO/GRPO v1 的 Dropout 模式破坏了 reference/old/new 身份。处理方式是保留原报告、找出机制、添加自动门、在新目录重跑。不能删除失败记录或只在文字里说“已修复”。

### 9.2 方法失败：代理目标成功但行为失败

DPO v2 偏好准确率 1.0、margin 74.685，却 strict 0/16，SFT loss 和重复指标退化。这不是代码失败，而是一次有价值的实验结论：小模板 pair + 短程 DPO 能极快优化 teacher-forced 相对似然，却未修复自由生成。

可能机制包括小数据过拟合、sum log-prob/长度效应、更新过强、训练/生成鸿沟、Base 本身能力不足、严格 verifier 与训练 pair 不完全同分布。当前单次实验无法确定唯一因果。

<details>
<summary>思考题：看到 margin=74.685，下一步应该继续训练让 margin 更大吗？</summary>

答案：不应该。margin 已极端，而行为和保留指标退化；继续优化同一代理大概率扩大偏移。应先做学习率/早停等受控实验或改善数据与评测，但本阶段冻结计划不再调参，交由 Stage 6 审计。
</details>

## 10. 如何写不越界结论

合格表达：

> 在冻结的 384 条训练 pair、Qwen3-0.6B-Base + Stage 4 Adapter、50-step 配置上，DPO 提高 held-out pair 的隐式偏好准确率，但未提高 16 条严格生成成功率，并伴随验证损失与重复退化。

不合格表达：

> DPO 无效；GRPO/K3 方法一定更好。

单配置负结果不能否定算法族，1-step GRPO 也没有能力对比。Stage 5 的完成标准是正确实现、可复现证据和诚实边界，而非训练出完美 tokenizer 或大模型。

## 11. 一手资料

- DPO：<https://arxiv.org/abs/2305.18290>
- PPO：<https://arxiv.org/abs/1707.06347>
- DeepSeek-R1：<https://arxiv.org/abs/2501.12948>
- DAPO：<https://arxiv.org/abs/2503.14476>
- DrGRPO：<https://arxiv.org/abs/2503.20783>
- GSPO：<https://qwenlm.github.io/blog/gspo/>
- VESPO：<https://arxiv.org/abs/2602.10693>
- Kimi K3 官方仓库与报告入口：<https://github.com/MoonshotAI/Kimi-K3>

## 12. 综合口述验收

你应能不看代码完成以下陈述：DPO 为什么不是在线 RL；四个 log-prob 各是谁；baseline 为什么不改期望；PPO 对正负 advantage 如何裁剪；GRPO 的零方差组如何处理；DrGRPO/DAPO/GSPO/VESPO 的主问题各是什么；K3 MOPD 为什么是多教师 dense delta；以及为什么本项目 DPO v2 同时是“工程成功”和“行为失败”。
