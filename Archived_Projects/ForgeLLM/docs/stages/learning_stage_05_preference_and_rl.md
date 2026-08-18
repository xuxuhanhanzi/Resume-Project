# Stage 5：偏好优化与在线 RL 实验计划（已实施 v2）

> 制定日期：2026-07-28（Asia/Singapore）  
> 状态：方案已获批准；G5-A～G5-E 自动化、讲义与 G5-L 学习者验收均已完成；能力收益未证明
> 周期：3 周，按每周 60 小时上限规划  
> 阶段门：G5-A / G5-B / G5-C / G5-D / G5-E / G5-L  
> 前置状态：Stage 4 自动化完成；正式 Adapter 的 SFT loss 改善，但 strict generation 0/16，适合作为偏好/RL 的真实问题起点

## 1. 阶段目标

Stage 5 不以复现某个超大推理模型为目标，也不靠长时间 rollout 追求榜单。它要让学习者真正掌握：

1. 人类/规则偏好如何变成 pair、reward 和可审计数据；
2. Reward Model、DPO 分别在优化什么，reference policy 为何存在；
3. Policy Gradient、advantage、importance ratio、KL、clipping 的数学和 Python 实现；
4. PPO 与 GRPO 的相同点和差异；
5. DAPO、Dr. GRPO、GSPO 等方法分别修复哪些已观察问题；
6. rollout policy、training policy 和 logged log-prob 的版本如何保持一致；
7. 如何识别 reward hacking、长度偏差、模式坍缩、重复和 Base 能力回退；
8. 在 8GB GPU 上用微型/离线 objective 实验掌握算法，并只做极小的真实 0.6B on-policy smoke。

Stage 4 已经证明“teacher-forced loss 下降”不保证生成行为通过。Stage 5 的核心正是引入比较或环境反馈，但新的 reward 也可能被利用；因此“加入 RL”不等于自动得到更好模型。

## 2. 一手材料新鲜度审计

截至计划冻结日，已公开且纳入核心学习的材料：

- [DPO 原论文](https://arxiv.org/abs/2305.18290)：通过 reward/policy 的闭式关系把带 KL 的偏好 RL 目标改写为二分类式 loss；
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：clipped surrogate 是理解后续 group policy optimization 的基础；
- [DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)：R1-Zero 展示无 SFT 大规模 RL，同时报告可读性和语言混合问题；R1 使用 cold start 与多阶段训练；
- [DAPO](https://arxiv.org/abs/2503.14476)：公开 decoupled clip、dynamic sampling、token-level policy gradient 与 overlong reward shaping；
- [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783)：分析 GRPO 的长度偏差并提出更无偏的归一化；
- [Qwen GSPO 官方说明](https://qwenlm.github.io/blog/gspo/)：把 importance ratio、clipping 和优化提升到 sequence level，目标是改善长程和 MoE RL 稳定性；
- [VESPO](https://arxiv.org/abs/2602.10693)：面向 stale/off-policy rollout 的 sequence-level soft reweighting；属于 2026 新方法，进入扩展对照；
- [On-Policy Delta Distillation](https://arxiv.org/abs/2607.15161)：用 teacher 相对其 Base 的 delta signal 提供 dense supervision；发布时间很近，只进入阅读/张量 reference，不做效果复现。

实施期 F0 增量审计确认 Kimi K3 已发布官方仓库与技术报告，因此加入 Stage 5：学习其 SFT cold start、3 领域×3 reasoning effort 的 9 个 RL 专家、partial rollout/staleness、Agentic GRM verbosity control、MOPD 与部署感知 QAT，并只实现 MOPD 的最小张量 reference。DeepSeek V4 仍没有找到可核验的 DeepSeek 官方技术报告，继续标为“待核验”，不得猜测其 RL 算法。中途新论文不改变已经冻结的主实验，只进入 radar。

## 3. 明确不做

- 不训练 7B/32B Reward Model 或推理模型；
- 不执行 RLHF 大规模人工标注；
- 不购买云 GPU，除非用户另行批准具体费用和时长；
- 不做 beta、KL、clip、group size、reward weight 的网格搜索；
- 不用训练 reward 作为唯一验收指标；
- 不让同一 verifier 同时定义 train reward 和全部 test 成功标准；
- 不把 offline DPO 称作 on-policy RL；
- 不把 GRPO 的“无 critic”误写成“无 baseline/advantage”；
- 不因新方法名称更新而实现所有 2026 变体；
- 不把大模型论文结果归因到本项目的微型 reference。

## 4. 双轨实验设计

| 轨道 | 对象 | 学什么 | 可做结论 |
|---|---|---|---|
| A：算法轨 | categorical bandit + Stage 3 5.36M Decoder | RM、DPO、REINFORCE、PPO、GRPO/DAPO/DrGRPO/GSPO objective 和梯度 | 数学、shape、更新与偏差机制正确 |
| B：框架轨 | Stage 4 Qwen3-0.6B Base + frozen SFT Adapter | PEFT DPO、真实 rollout/log-prob、1-step GRPO smoke、Artifact/显存 | 本机框架链路可运行 |
| C：张量对照 | 固定 synthetic logits/rewards/lengths | 方法之间只改变一个 objective 项 | 归一化、clip、长度和 staleness 差异 |

0.6B 轨道不是正式质量排名。主要学习证据来自手算、reference、测试和受控张量实验；真实 rollout 只证明接口和版本契约。

## 5. 数据与环境设计

### 5.1 `preference_constraints_v1`

项目原创、确定性生成，计划 384/64/64 个 prompt。每条包含：

```text
prompt messages
chosen response
rejected response
preference source / rule
chosen/rejected verifier fields
generator version
pair fingerprint
```

任务覆盖精确短答、JSON schema、数值运算、集合排序和简单字符串变换。rejected 至少包含：答案错、格式错、额外解释、截断、重复、长度投机六类。split 按任务实例内容哈希，在生成 rejected 前冻结，避免同一问题变体跨 split。

### 5.2 Reward 与 test 分离

- train reward：只用确定性 verifier 的训练任务族；
- held-out task：新 ID 和新输入，不复用答案；
- adversarial test：专门测试“只输出正确前缀后追加垃圾”、超长、空白欺骗、JSON 包裹和重复；
- retention：沿用 Stage 4 固定集合；
- reward unit tests：对每条规则准备正、负、边界样本。

### 5.3 Rollout record

每条 rollout 必须记录：prompt ID、policy revision、Adapter hash、generation config、seed、token IDs、old log-probs、reward components、verifier version、response length、timestamp。若 training policy 与 rollout policy 的版本关系不明确，数据不得进入 update。

## 6. 三周安排

### Week 1：偏好数据、Reward Model 与 DPO

- Bradley–Terry：`P(chosen>rejected)=sigmoid(r_c-r_r)`；
- scalar reward head、pairwise loss、reward margin、pair accuracy；
- reward shift 不可识别性与 scale 漂移；
- DPO 从 KL-regularized reward objective 到 log-ratio loss 的推导；
- policy/reference 的 chosen/rejected log-prob 与 response-only mask；
- beta、长度求和/平均和 reference-free 误区；
- 手写 DPO 与 TRL 单 batch 数值/梯度对照；
- Qwen Adapter 的有界 DPO smoke。

### Week 2：Policy Gradient、PPO 与 GRPO

- MDP、trajectory、return、score-function estimator；
- REINFORCE 与 baseline 为什么不引入期望偏差；
- advantage 标准化的边界；
- on-policy/off-policy、importance ratio 与 policy staleness；
- PPO clipped surrogate、value loss、entropy、KL；
- LLM 中 prompt、response token、sequence reward 和 credit assignment；
- GRPO 的组内相对 advantage、无独立 critic 的资源收益与零方差组；
- categorical bandit 和 tiny verifier environment 的真实更新；
- Qwen 0.6B group rollout + 1 optimizer-step smoke。

### Week 3：长度/稳定性方法、失败注入与综合验收

- Dr. GRPO：question/group 与 token/length normalization 偏差；
- DAPO：clip-higher、dynamic sampling、token-level loss、overlong shaping；
- GSPO：sequence likelihood ratio、length-normalized log ratio、sequence clipping；
- VESPO：stale policy 下的 soft sequence weight，只做 frozen tensor study；
- OPD/OPD²：dense teacher signal 与 reward 的区别，只做公式/小张量 reference；
- reward hacking、长度投机、重复、KL runaway、entropy collapse 故障注入；
- 固定前后评测、Artifact、讲义和 G5 口述验收。

## 7. 工作包与完成条件

### G5-A：Preference Schema 与 Reward Model

实现：

- strict `PreferenceRecord` 和 `RolloutRecord`；
- chosen/rejected 不得相同，prompt 内容 hash 去重，split 防泄漏；
- response-only tokenize/mask；
- scalar reward head 和 Bradley–Terry loss；
- pair accuracy、margin、长度相关性、reward 分布；
- adversarial verifier tests。

通过条件：

- 手算 4 对 pair loss 与实现对齐；
- swap chosen/rejected 后 margin 符号和 loss 方向正确；
- tiny RM 能过拟合 16 pairs；
- reward 与长度相关性单独报告；
- 不用 RM 训练准确率声明生成质量。

### G5-B：DPO

实现：

- policy/reference 的 response log-prob sum 与 token counts；
- chosen/rejected 四条 log-prob 的 DPO logits；
- fixed beta `0.1`；
- reference policy 冻结和 Adapter disable/enable 边界；
- 手写 loss 与 TRL 1.8.0 固定 batch 对照；
- Qwen LoRA Adapter-only 保存与 before/after 评测。

通过条件：

- loss、logits、gradient 与独立公式在登记容差内；
- prompt/padding 不计入 response log-prob；
- reference 无梯度、版本 hash 固定；
- chosen/rejected 交换后优化方向反转；
- held-out preference accuracy、严格任务、长度、重复、retention 全部报告。

### G5-C：Policy Gradient 与 PPO

实现：

- categorical policy 的 exact expected return 与 Monte Carlo REINFORCE；
- moving/value baseline 的偏差/方差实验；
- PPO ratio、clip、value、entropy/KL reference；
- old-policy snapshot 和多 epoch update 的 staleness 测试；
- tiny verifier environment。

通过条件：

- 大样本 Monte Carlo gradient 对齐 exact gradient；
- baseline 降方差且不改变期望方向；
- positive/negative advantage 的 clip 分支手算正确；
- old/new policy 混用测试必须失败；
- 真实 toy environment 的平均 reward 在 3 seeds 上改善，但不要求每 seed 单调。

### G5-D：GRPO 与真实 rollout

实现：

- 每 prompt 组内 reward mean/std 与 advantage；
- 零方差组的显式策略；
- token-level ratio、clipped objective、可选 KL；
- rollout/token/log-prob/version Artifact；
- Qwen 0.6B、group size 4、最多 4 prompts、32 new tokens 的 1-step smoke。

通过条件：

- 固定 logits/rewards 的手算 objective/gradient 对齐；
- group permutation 不改变结果；
- policy staleness 被检测；
- Base/reference 无梯度；
- on-policy smoke 完成生成→reward→log-prob→update 全链，不要求能力提升。

### G5-E：DAPO、Dr. GRPO、GSPO 与新方法 radar

实现单变量 frozen-tensor studies：

1. GRPO vs Dr. GRPO：只改 normalization，构造相同 reward、不同长度；
2. GRPO vs DAPO clip-higher：只改 upper clip；
3. 普通 sampling vs dynamic sampling：测全对/全错零信号组利用率；
4. sequence truncation vs overlong shaping：测边界梯度和 reward 跳变；
5. token ratio vs GSPO sequence ratio：测单 token outlier、长度和 precision mismatch；
6. GSPO vs VESPO：只在 stale ratio 张量上比较 hard clip 与 soft weight；
7. reward PG vs OPD delta：小 categorical teacher/student，不加载外部大教师。

通过条件：每项有旧问题、公式、代码、边界测试、最小实验和“不解决什么”；新方法不要求胜出，不把张量结果写成大模型复现。

### G5-L：学习者门

自动化完成后生成至少五份完整讲义与 18 站唯一入口。学习者需：

- 手推 Bradley–Terry 与 DPO；
- 手算一次 PPO clip 和一组 GRPO advantages；
- 解释 policy/reference/old policy/rollout policy 四个身份；
- 画出 SFT→DPO→on-policy rollout/update 数据流；
- 分析一次 reward hacking 和长度偏差；
- 对 DAPO/DrGRPO/GSPO 分别回答“修复什么、代价什么”；
- 对所有实验作不越界口述。

## 8. 单变量实验矩阵

| ID | 问题 | Baseline | 唯一主变量 | 预算 | 成功定义 |
|---|---|---|---|---:|---|
| E0 | pair schema 是否确定 | 同一输入 | 重建 | CPU | JSON/hash byte-exact |
| E1 | BT loss 是否正确 | 手算 | PyTorch | CPU | loss/gradient 对齐 |
| E2 | RM 链路可学习 | 初始 tiny RM | 训练 100 steps | 3 seeds，≤10 min | overfit gate |
| E3 | DPO 公式是否正确 | 手写 | TRL | CPU/GPU | logits/loss/grad 对齐 |
| E4 | reference 是否必要 | frozen ref | 错误更新 ref | 单测 | 错误路径 fail-fast |
| E5 | DPO 是否改变行为 | Stage 4 Adapter | DPO update | ≤20k pair tokens/20 min | 全指标报告 |
| E6 | baseline 如何降方差 | REINFORCE | 加 baseline | 3 seeds CPU | 方差下降/方向不变 |
| E7 | PPO clip 做什么 | unclipped PG | clip | frozen tensors | 分支 100% 手算对齐 |
| E8 | toy PPO 是否学习 | 初始 policy | updates | 3 seeds ≤15 min | mean reward 改善 |
| E9 | GRPO group 语义 | 手算 | 实现 | frozen tensors | permutation/zero-std 门 |
| E10 | 真实 on-policy 链 | Qwen Adapter | 1 GRPO step | ≤4×4 rollouts/15 min | 全链通过 |
| E11 | 长度偏差 | GRPO | Dr. GRPO norm | synthetic lengths | 梯度差可解释 |
| E12 | DAPO 的四项机制 | GRPO | 每次仅一项 | frozen tensors | 各自独立报告 |
| E13 | token/sequence ratio | GRPO | GSPO | synthetic staleness | outlier/长度差可解释 |
| E14 | 新方法雷达 | GSPO/OPD | VESPO/OPD² 小 reference | CPU | 只证明公式代码 |
| E15 | 故障注入 | 正常 reward | exploit/长答/重复 | CPU/GPU | verifier/metric 抓住退化 |

## 9. 冻结配置建议

为了避免搜索，实施时默认：

- DPO beta `0.1`，单配置，不扫描；
- DPO response log-prob 使用 token sum，同时报告 token count 和长度相关性；
- PPO/GRPO clip epsilon `0.2`；
- GRPO group size `4`；
- KL beta 首个 reference 固定 `0.01`，beta=0 只作为独立单变量；
- greedy 用于量化 task evaluation；rollout 用固定 temperature `0.8`、top-p `0.95`；
- max new tokens `32`；
- Qwen 轨只训练 Adapter，Base/reference 全冻结；
- low-cost stochastic experiments 用 seeds 41/42/43；Qwen smoke 单 seed 20260728；
- 不 packing preference/rollout baseline；
- 所有长度、reward components、KL、entropy、clip fraction 和 invalid rate 必须落盘。

这些值是教学起点，不是“最佳 RL 配方”。F0 若发现框架 API 语义不兼容，只允许为正确性调整一次，并登记原因。

## 10. 资源与停止条件

| 运行 | 上限 | 说明 |
|---|---:|---|
| 单元/张量实验 | 每项 ≤5 min | 数学正确性优先 |
| tiny RM/DPO/PG/PPO/GRPO | 每项 ≤15 min，3 seeds | 算法轨 |
| Qwen DPO smoke | ≤20k pair target tokens / 50 steps / 20 min | 先到即停 |
| Qwen GRPO smoke | 4 prompts×4 rollouts×32 tokens，1 update / 15 min | 只验 on-policy 链 |
| Stage 5 新增磁盘 | ≤6 GiB | 不复制完整 Base/reference 权重 |
| 外部成本 | 0 USD 默认 | 任何云端运行单独审批 |

OOM 时先保留失败证据，只允许减 micro batch、启用 checkpoint 或缩小 smoke prompt 数；不得静默换模型、缩 context 后仍称同实验。时间上限包含 rollout 和 update，不允许只计 optimizer 时间。

## 11. 指标与失败门

| 维度 | 指标 |
|---|---|
| Preference/RM | pair accuracy、margin、reward mean/std、长度相关性 |
| DPO | chosen/rejected log-ratio、loss、implicit reward accuracy、KL proxy |
| RL | mean reward、pass rate、advantage/std、entropy、KL、clip fraction、importance ratio |
| 行为 | strict verifier、exact match、invalid/empty、长度分布、token/character repetition |
| 保留 | Stage 4 retention loss 与固定通用 prompt |
| 系统 | rollout tokens/s、update tokens/s、显存、stale ratio、Artifact size |

立即停止并记录：非有限 log-prob/ratio/gradient；reference 或 Base 出现 grad；rollout hash/版本不匹配；reward 全常数却继续更新；response 长度持续撞上限；字符重复恶化超过预登记阈值；KL 或 entropy 越界；verifier 可被简单包装/追加垃圾欺骗。

## 12. 预计产物

```text
src/forgellm/alignment/
  schema.py reward.py reward_model.py dpo.py policy_gradient.py
  ppo.py grpo.py dapo.py gspo.py evaluation.py
configs/alignment/
scripts/stage5_*.py
tests/unit/test_alignment_*.py
tests/integration/test_stage5_*.py
data/processed/stage5_preference_constraints_v1/
artifacts/stage05/
docs/lessons/stage05_learning_order.md
docs/lessons/stage05_*.md
docs/experiments/2026-*_stage05_implementation.md
```

## 13. Stage 5 总完成定义

G5 自动化完成要求：数据/版本可追溯；RM/DPO/PG/PPO/GRPO reference 与手算/框架对齐；至少一个 toy 环境真实提升；Qwen DPO 和 1-step on-policy 链可运行；DAPO/DrGRPO/GSPO 单变量实验完成；失败/回退和 reward hacking 被报告；全仓质量门通过。以上 G5-A～G5-E 已于 2026-07-28 完成，证据见 `docs/experiments/2026-07-28_stage05_implementation.md`。

G5-L 必须由学习者确认，自动化不能代替。2026-07-28 学习者已确认完成推导、代码追踪、最小改动与口述验收，因此 G5-A～G5-L 全部关闭。当前只能声明“掌握并实现了偏好/RL 方法和小规模受控实验”；DPO 的 pair 指标改善但 strict generation 未改善，不能声明复现 DeepSeek-R1、DAPO、GSPO、训练出生产推理模型或证明能力收益。

## 14. 实施结果补记

- 数据固定为 384/64/64；320/320 个 verifier 攻击被严格拒绝；
- tiny RM、baseline variance、toy PPO/GRPO 与现代方法张量门通过；
- DPO v2 完成 50 steps / 4,214 response tokens，pair accuracy 0→1，但 strict generation 保持 0/16，SFT loss 与字符重复退化；
- GRPO v2 完成真实 4 prompts × 4 rollouts × 32 tokens 上限的一步链路，首步 KL=0、clip fraction=0；
- v1 的 Dropout policy-mode 错配原样保留，v2 关闭 Dropout并增加自动 on-policy 门；
- 五份完整讲义与 `docs/lessons/stage05_learning_order.md` 的 18 站唯一入口已生成；
- Stage 6 讨论稿位于 `docs/stages/learning_stage_06_evaluation_and_acceptance.md`。

## 15. 实施前已确认的三个决策（归档）

1. 已接受 `preference_constraints_v1` 以项目原创 verifier 任务为主，不引入大规模人工偏好集；理由是本阶段目标为算法学习和 reward 可审计。
2. 已接受 Qwen 在线 RL 只做 1 update smoke，把多步学习证据放在 tiny policy；理由是控制 rollout 预算。
3. 已接受 VESPO、OPD² 等新方法只做张量/reference radar，不进入主训练；理由是先稳定掌握 DPO/PPO/GRPO/DAPO/GSPO 主线。
