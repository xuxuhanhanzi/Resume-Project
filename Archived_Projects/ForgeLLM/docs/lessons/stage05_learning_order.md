# Stage 5 唯一学习入口：偏好优化与在线 RL 18 站

## 使用规则

Stage 5 的自动化实现已经完成，但本阶段的目标是掌握方法，不是背诵算法名称，也不是重复烧卡。请严格按站学习：先读讲义指定章节，再追踪一个小范围代码，最后完成手算或最小测试。所有思考题的答案都紧跟题目，不存在独立答案文件。

正式 DPO 与 GRPO 证据已经保存在 `artifacts/stage05/`。学习时默认只运行 CPU 单元测试、方法实验和至多一次新目录的 smoke；不要覆盖现有 Artifact，也不要重新做长训练。

## 先记住整条主线

```text
prompt + chosen/rejected
→ 可审计 PreferenceRecord
→ Reward Model 或 DPO 离线目标
→ policy / reference 身份冻结
→ 环境采样 rollout
→ reward 与 group advantage
→ old/new ratio、clip、KL
→ Adapter update
→ 独立行为评测与退化审计
```

## 新手术语预热

| 术语 | 本项目中的准确含义 |
|---|---|
| preference pair | 同一 prompt 下，一个 chosen 与一个 rejected 回答 |
| Reward Model（RM） | 将 prompt-response 映射成标量分数的模型 |
| Bradley–Terry | 用 `sigmoid(r_chosen-r_rejected)` 描述成对偏好概率 |
| policy | 当前被优化、会产生回答的模型 |
| reference policy | 冻结的参照模型，用于限制相对偏移 |
| behavior/old policy | 真正生成 rollout、记录 old log-prob 的策略版本 |
| on-policy | rollout 与用于首轮更新的策略分布一致 |
| advantage | 一个动作/序列相对基线“比预期好多少” |
| importance ratio | `exp(logπ_new-logπ_old)`，衡量新旧策略概率变化 |
| clipping | 限制单次更新从旧策略偏离得过多 |
| DPO | 直接从偏好 pair 优化策略的离线方法，不是在线 RL |
| PPO | 使用 old policy、advantage 与裁剪代理目标的 actor-critic 方法族 |
| GRPO | 同一 prompt 多回答，在组内构造相对 advantage；无需独立 critic |
| staleness | rollout 生成后，训练 policy 已变化导致数据“过期” |
| reward hacking | 模型利用 reward/评测漏洞取得高分而未完成真实目标 |

## 18 站学习路线

| 站 | 主题 | 先读 | 代码追踪 | 必做验收 |
|---:|---|---|---|---|
| 1 | 偏好数据是什么 | 讲义一 1–2 | `schema.py` 的 `PreferenceRecord` | 手写一条合法 pair |
| 2 | 身份、哈希与防泄漏 | 讲义一 3 | `prompt_fingerprint`、`assert_split_disjoint` | 解释为什么先切分后造 rejected |
| 3 | verifier 与攻击样本 | 讲义一 4 | `verify_response`、`adversarial_responses` | 构造“正确前缀+垃圾”攻击 |
| 4 | Reward Model | 讲义一 5–7 | `reward.py` | 手算 3 个 margin 的 BT loss |
| 5 | DPO 的四个 log-prob | 讲义二 1–3 | `response_sequence_log_probs` | 标出 prompt/PAD mask |
| 6 | DPO 推导与 beta | 讲义二 4–5 | `dpo_loss` | 手算一条 DPO logit |
| 7 | reference 身份 | 讲义二 6 | `assert_reference_frozen`、`hf_common.py` | 区分 policy/ref/Base/Adapter |
| 8 | DPO 框架对照 | 讲义二 7 | `test_stage5_tiny_alignment.py` | 解释为何要对齐 TRL 固定 batch |
| 9 | DPO 正式实验 | 讲义二 8 | `hf_dpo.py` 与 v2 report | 同时报告正、负结果 |
| 10 | 从期望回报到 REINFORCE | 讲义三 1–3 | `policy_gradient.py` | 枚举 categorical 精确梯度 |
| 11 | baseline 与方差 | 讲义三 4 | `run_reinforce_variance_study` | 解释“不改期望但可改方差” |
| 12 | PPO ratio 与正负 clip | 讲义三 5–7 | `ppo.py` | 手算四个裁剪分支 |
| 13 | rollout 版本契约 | 讲义四 1–2 | `RolloutRecord`、`assert_initial_on_policy` | 解释 Dropout v1 失败 |
| 14 | GRPO group advantage | 讲义四 3–5 | `grpo.py` | 手算一组奖励并处理零方差 |
| 15 | 真实 1-step GRPO | 讲义四 6 | `hf_grpo.py` 与 v2 report | 画 rollout→update 链 |
| 16 | DrGRPO 与 DAPO | 讲义四 7–8 | `frontier.py` 前半 | 每种方法只说一个主修复点 |
| 17 | GSPO 与 VESPO | 讲义四 9–10 | `frontier.py` 中段 | 比较 token/sequence/stale 权重 |
| 18 | Kimi K3、蒸馏与结论边界 | 讲义五 | `frontier.py` 后半及实验记录 | 完成综合口述验收 |

## 文件阅读顺序

只按以下顺序打开，不按目录字母顺序通读：

1. `docs/lessons/stage05_preference_data_and_reward_models.md`
2. `src/forgellm/alignment/schema.py`
3. `src/forgellm/alignment/preference_data.py`
4. `src/forgellm/alignment/reward.py`
5. `docs/lessons/stage05_dpo.md`
6. `src/forgellm/alignment/dpo.py`
7. `src/forgellm/alignment/hf_common.py`
8. `src/forgellm/alignment/hf_dpo.py`
9. `docs/lessons/stage05_policy_gradient_and_ppo.md`
10. `src/forgellm/alignment/policy_gradient.py`
11. `src/forgellm/alignment/ppo.py`
12. `docs/lessons/stage05_grpo_and_modern_variants.md`
13. `src/forgellm/alignment/grpo.py`
14. `src/forgellm/alignment/hf_grpo.py`
15. `src/forgellm/alignment/frontier.py`
16. `docs/lessons/stage05_frontier_distillation_and_failure_analysis.md`
17. `docs/experiments/2026-07-28_stage05_implementation.md`
18. `artifacts/stage05/method_lab/report_v2.json`、DPO v2 与 GRPO v2 报告

## 固定复现命令

先运行不烧卡的核心测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_alignment_*.py -q
```

重跑 CPU 方法实验时必须写新文件：

```powershell
.\.venv\Scripts\python.exe scripts\stage5_method_lab.py `
  --train data\processed\stage5_preference_constraints_v1\train.jsonl `
  --output artifacts\stage05_student\method_lab.json
```

查看已有正式证据：

```powershell
Get-Content -Raw -Encoding UTF8 artifacts\stage05\qwen_dpo_bounded_v2\report.json
Get-Content -Raw -Encoding UTF8 artifacts\stage05\qwen_grpo_one_step_v2\report.json
```

DPO 正式运行和 GRPO rollout 无需重复。若确实想追调用链，只能使用 smoke 配置和全新输出目录。

## 五个最小改动练习

1. 交换一条 pair 的 chosen/rejected，证明 margin 和梯度方向同时反转。
2. 故意把一个 prompt token 计入 DPO response mask，观察 log-prob 与长度相关性如何变化；完成后恢复。
3. 对同一个 PPO ratio 分别给正、负 advantage，解释 `min` 为什么产生不同裁剪方向。
4. 构造四个完全相同 reward 的 GRPO group，验证更新信号必须为零。
5. 临时令 old policy 使用 Dropout、new policy 使用 eval，确认 on-policy 门会拒绝；完成后恢复。

## 学习者完成判定

以下全部完成后，G5-L 才能由你确认关闭：

- 按顺序完成 18 站与即时问答；
- 能从 pair 推导 Bradley–Terry 与 DPO，并指出它们优化的对象不同；
- 能手算 REINFORCE、PPO clip 与 GRPO advantage；
- 能准确区分 Base、SFT policy、reference、old/behavior policy、new policy；
- 能解释 DrGRPO、DAPO、GSPO、VESPO 和 Kimi K3 MOPD 各自试图解决什么；
- 能用 v2 证据说明 DPO objective 成功而生成行为失败，不作越界结论。
