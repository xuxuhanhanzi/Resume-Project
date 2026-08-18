# 2026-07-28 Stage 5 偏好优化与在线 RL 实施记录

## 1. 结论先行

G5-A～G5-E 自动化门已完成：可审计偏好/rollout Schema、确定性偏好数据与 verifier、Bradley–Terry RM、response-only DPO、REINFORCE/PPO/GRPO、现代方法单变量 reference、Qwen Adapter DPO bounded run 和真实 4×4 rollout 的 1-step GRPO 全部落地。五份完整讲义和 18 站唯一入口已生成；G5-L 仍需学习者验收。

方法正确性门全部通过，但正式 DPO 的行为结果是负面的：held-out pair 偏好准确率达到 1.0，strict generation 仍为 0/16，同时 SFT validation loss 与字符重复退化。这证明代理目标改善不等于能力改善。DPO/GRPO v1 还暴露了 Dropout 模式导致的身份不一致；v1 原样保留，关闭 Dropout、加入 on-policy fail-fast 后的 v2 才是主证据。

## 2. 冻结环境与身份

| 项目 | 冻结值 |
|---|---|
| Base | `Qwen/Qwen3-0.6B-Base` |
| revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| 初始 SFT Adapter | Stage 4 BF16 LoRA v2 |
| 初始 Adapter SHA-256 | `d301bea2...e9ff` |
| GPU | NVIDIA GeForce RTX 4070 Laptop，8188 MiB |
| torch / transformers | 2.6.0+cu124 / 5.14.1 |
| peft / trl | 0.19.1 / 1.8.0 |
| 外部费用 | 0 USD |

训练只更新 10,092,544 个 LoRA 参数；606,142,464 总参数中的 Base 均冻结。Stage 5 加载时显式将全部 `nn.Dropout.p` 设为 0，保证 policy/reference/old/new 的可比较性。

## 3. 数据与 verifier

`stage5_preference_constraints_v1` 为项目原创确定性教学数据：5 个任务族、6 类 rejected，任务身份哈希排序后先切分，再生成错误回答。

| split | 记录数 | SHA-256 |
|---|---:|---|
| train | 384 | `c2326f8d...1545` |
| validation | 64 | `d563bdde...267d` |
| test | 64 | `1e818038...315d` |

三个 split 的 prompt fingerprint 无交叉。64 条 validation 上构造 320 个正确前缀/包装/重复/空格攻击，strict total reward 拒绝 320/320。该数据只覆盖模板化确定性约束，不代表真人开放偏好。

## 4. 实现范围

### G5-A：Schema、数据和 RM

- 严格 `PreferenceRecord`、`RolloutRecord` 与 canonical SHA-256；
- 字段全集、pair 不同、prompt 身份与 split 防泄漏；
- strict verifier 与分离的审计 reward components；
- UTF-8 byte GRU scalar RM、Bradley–Terry loss、pair accuracy、margin 和长度相关性。

### G5-B：DPO

- shifted response-only sequence log-prob sum；
- standard reverse-KL sigmoid DPO，固定 beta=0.1；
- frozen initial-SFT reference log-prob Artifact；
- 手写实现与 TRL 1.8.0 `DPOTrainer` 固定 batch 的 loss/gradient 对照；
- Qwen Base 冻结、Adapter-only bounded update 和前后同口径评测。

### G5-C：Policy Gradient 与 PPO

- categorical exact expected return 和 score-function 枚举；
- REINFORCE、detached action-independent baseline、entropy、categorical KL；
- PPO positive/negative advantage clip、clipped value loss 与 policy revision 门；
- 3-seed tiny PPO 实际更新。

### G5-D：GRPO

- 组内 mean/std advantage 与零方差组显式策略；
- token ratio、PPO-style clip、可选 KL；
- versioned rollout JSONL；
- old/new 首步 log-prob 最大差门；
- Qwen 4 prompts × 4 responses 的真实采样、reward、log-prob 和一步更新。

### G5-E：现代方法最小 reference

- DrGRPO：center-only advantage 与固定 token normalizer；
- DAPO：Clip-Higher、Dynamic Sampling、overlong shaping；
- GSPO：length-normalized geometric-mean sequence ratio；
- VESPO：真实 sequence IS 与 sign-specific Gamma soft kernel；
- Kimi K3 风格 OPD/MOPD：stop-gradient clipped dense delta 与多教师选择。

这些仅为 frozen-tensor/小环境机制证据，不是前沿大模型效果复现。

## 5. CPU 方法实验

`method_lab/report_v2.json` 的全部门通过：

- tiny RM 三个 seed 的 pair accuracy 均为 1.0，final loss 约 `3e-7`～`2e-6`；
- REINFORCE 方差最优 baseline 将梯度样本平均方差从约 0.326–0.335 降到 0.246–0.252；
- toy PPO 三个 seed 的 expected reward 从 0.2050 提升至约 1.491；
- toy GRPO 从 0.2250 提升至 1.1116–1.1257，平均改善 0.8937；
- DAPO、DrGRPO、GSPO、VESPO、MOPD 的单变量张量输出均写入报告。

`method_lab/report.json` v1 使用 expected-reward baseline，却错误预期它必然降低 score-gradient 方差，因此 gate 失败。v2 改用方差最优常数 baseline；v1 保留为理论假设被实验推翻的记录。

## 6. DPO v1 偏差与 v2 正式结果

v1 完成 50 steps，但 reference/evaluation 使用 eval，训练 policy 使用 train；Stage 4 LoRA dropout=0.05 使同权重不等分布，不满足冻结 reference 的严格比较语义。它不作为主结果，报告 SHA-256 为 `158b0896...41da`。

v2 将所有 Dropout 设为 0，并用 smoke 验证首 batch loss 精确为 `log 2=0.693147`。正式运行：

- 50 optimizer steps / 200 micro-steps；
- 200 pairs、4,214 response tokens；
- 129.26 秒、32.60 pair-response tokens/s；
- peak allocated 1,638,436,352 bytes；
- stop=`max_steps`；Base gradients absent；
- final Adapter SHA-256 `01f46228...e019`。

| 指标 | Before | After | 解释 |
|---|---:|---:|---|
| preference accuracy | 0.0 | 1.0 | 离线 pair 目标被拟合 |
| mean DPO margin | 0.0 | 74.6852 | 相对 reference 偏移很大 |
| strict generation | 0/16 | 0/16 | 行为未改善 |
| Stage 4 assistant val loss | 1.15002 | 1.35844 | 保留分布退化 |
| preference generation char 8-gram | 0.05814 | 0.55276 | 重复退化 |

报告 SHA-256：`dbf34452...ba0`。结论仅适用于该小数据、单 beta、单学习率、短预算运行；不作 DPO 算法总体有效/无效判断。

## 7. GRPO v1 偏差与 v2 正式 smoke

v1 的 old log-prob 在 eval 计算，new log-prob 在 train 计算，造成首步 approximate KL `3.83e-5`、clip fraction `0.00591`。虽然数值小，身份契约已失败。修正后新增自动门：有效 token 的 old/new log-prob 最大绝对差必须不超过 `1e-5`。

v2：

- 4 prompts × group size 4 = 16 rollouts；
- 508 response tokens；4 usable groups、0 zero-variance groups；
- reward mean `-0.03843`、population std `0.02512`；
- initial approximate KL `0.0`、clip fraction `0.0`；
- gradient norm `1.77480`；Base gradients absent；
- 1 optimizer step，12.97 秒；
- peak allocated 4,011,605,504 bytes；
- Adapter SHA-256 `3e0d21ba...01f`。

report SHA-256 `915b2df3...d75`；rollout JSONL SHA-256 `4f1e2ca4...cf8`。该 smoke 没有 before/after 能力结论，只证明全链和版本契约。

## 8. 关键命令

```powershell
.\.venv\Scripts\python.exe scripts\prepare_stage5_preference_data.py `
  --output-dir data\processed\stage5_preference_constraints_v1
.\.venv\Scripts\python.exe scripts\stage5_method_lab.py `
  --train data\processed\stage5_preference_constraints_v1\train.jsonl `
  --output artifacts\stage05\method_lab\report_v2.json
.\.venv\Scripts\python.exe scripts\stage5_dpo_train.py `
  --config configs\alignment\stage5_qwen_bounded.toml `
  --output-dir artifacts\stage05\qwen_dpo_bounded_v2
.\.venv\Scripts\python.exe scripts\stage5_grpo_smoke.py `
  --config configs\alignment\stage5_qwen_bounded.toml `
  --output-dir artifacts\stage05\qwen_grpo_one_step_v2
```

已有目录不可覆盖；学习者复现必须换到 `artifacts/stage05_student/` 下的新路径。

## 9. Artifact 清单与可追溯性

| Artifact | SHA-256 | 身份 |
|---|---|---|
| method lab v1 | `4509c996...6baf` | 失败假设记录 |
| method lab v2 | `d349dae1...5605` | CPU 主证据 |
| DPO v1 report | `158b0896...41da` | Dropout 偏差 |
| DPO v2 report | `dbf34452...ba0` | 正式 DPO 证据 |
| GRPO v1 report | `237b9d1d...a763` | mode mismatch 偏差 |
| GRPO v2 report | `915b2df3...d75` | 正式 1-step 链路 |

## 10. 质量门与剩余工作

新增 alignment 单元与 integration 测试覆盖 schema、公式、梯度、零方差、版本错配、config、tiny RM/PPO/GRPO 和 TRL 固定 batch 对照。设置 `FORGELLM_TEST_EXTENSION=1` 后执行 `scripts/dev.py check`：148 个 Python 文件通过 Ruff format、Ruff lint 与 strict mypy；pytest `226/226 passed`。两条 warning 仍来自 Windows C++ 编译器版本探测编码和 setuptools 私有接口，不是失败。

自动化 G5-A/B/C/D/E 已关闭；五份讲义和 18 站入口已完成。G5-L 只能由用户完成学习和口述验收后关闭。Stage 6 将综合评价 Stage 3 Base、Stage 4 SFT Adapter 与 Stage 5 DPO Adapter，不继续用训练代理指标替代最终验收。
