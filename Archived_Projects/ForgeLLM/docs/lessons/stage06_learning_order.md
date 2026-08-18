# Stage 6 唯一学习入口：综合评测与最终验收 18 站

Stage 6 的目标不是再训练一次模型，而是学会回答一个更困难的问题：**现有证据究竟允许我们说什么？** 自动化实现与正式本地评测已经保存到 `artifacts/stage06/`；学习时不要修改冻结测试集、不要在看到测试答案后调整生成参数，也不要覆盖现有 Artifact。

所有讲解均采用完整讲义模式。每道思考题的答案紧跟题目，不存在独立答案文件。请严格按下表前进，不要先通读全部源码。

## 先记住五个对象

- `Q0`：Qwen3-0.6B Base；
- `Q1`：Stage 4 SFT Adapter；
- `Q2`：Stage 5 DPO Adapter；
- `Q3`：Stage 5 一步 GRPO Adapter，只审计流程，不参与能力排名；
- `M3`：Stage 3 自研 DecoderLM，只在自己的 Tokenizer 与语言建模套件内评价。

## 18 站顺序

| 站 | 学习任务 | 讲义/代码入口 | 完成证据 |
|---|---|---|---|
| 1 | 区分训练指标、代理指标、最终行为 | `stage06_evaluation_identity_and_contamination.md` §1–2 | 能解释“loss 下降不等于验收通过” |
| 2 | 手写一次模型身份清单 | 同讲义 §3；`evaluation/schema.py` | 写出 Base、revision、Tokenizer、Adapter 四要素 |
| 3 | 追踪运行指纹 | `schema.py::EvaluationManifest` | 能说明改一项生成参数为何得到新实验 |
| 4 | 学习冻结数据与污染 | 同讲义 §4–7；`contamination.py` | 手算 NFKC 后的精确匹配 |
| 5 | 检查 Stage 6 的 64 例构成 | `cases.py`；`preparation_report.json` | 解释 16+16+32，而非只报“64” |
| 6 | 学习 exact/constraint/extra/repetition | `stage06_metrics_and_statistics.md` §1–4 | 对一条输出手算所有行为指标 |
| 7 | 学习 NLL、loss、PPL、BPB | 同讲义 §5–7；`language_modeling.py` | 解释为何不能平均各 batch 的 PPL |
| 8 | 学习 Tokenizer 公平性 | 同讲义 §8 | 能拒绝 M3 与 Qwen 的 PPL 直接排名 |
| 9 | 手算 Wilson 区间 | 同讲义 §9；`statistics.py` | 给出 0/16 也不是“真实率精确为 0” |
| 10 | 学习成对 Bootstrap | 同讲义 §10 | 解释为何必须保持同一 case 配对抽样 |
| 11 | 区分规则、模型 Judge、人工盲评 | `stage06_judges_robustness_and_blind_review.md` §1–4 | 能列出各自可评价和不可评价的内容 |
| 12 | 做位置交换与一致性审计 | 同讲义 §5；`judge.py` | 解释 position bias 与 Cohen’s κ |
| 13 | 完成 30 对盲评 | `artifacts/stage06/quality_v5/blind_review/blind_review.html` | 未看私钥完成评分；保存个人结果 |
| 14 | 学习等义鲁棒性及其边界 | 同讲义 §6–8 | 区分“原题成功”与“变换前后一致” |
| 15 | 学习 TTFT、E2E、decode tok/s | `stage06_system_benchmark_and_acceptance.md` §1–5 | 能从两个时延算近似 decode tok/s |
| 16 | 手算 GQA KV-cache | 同讲义 §6；`systems.py` | 写出 K/V、层、batch、序列、KV heads 六因子 |
| 17 | 阅读多维门禁和失败结论 | 同讲义 §7–10；`final_v1/report.json` | 不把多个指标压成单一总分 |
| 18 | 完成 G6-L 学习者验收 | 本文件“最终验收” | 口述、代码追踪、盲评和结论边界全部通过 |

## 推荐节奏

- 第 1 天：站 1–5，建立身份与数据边界；
- 第 2 天：站 6–10，完成指标和统计手算；
- 第 3 天：站 11–14，完成盲评与 Judge 审计；
- 第 4 天：站 15–17，阅读系统结果和最终卡；
- 第 5 天：站 18，完成学习者验收。

## 可以运行的命令

```powershell
.venv\Scripts\python.exe -m pytest -q tests\unit\test_evaluation_schema_config.py tests\unit\test_evaluation_metrics.py tests\unit\test_evaluation_judge_systems_report.py
```

正式评测已有 Artifact 时不要重跑。若你确实要复现，必须把 `--output` 指向 `artifacts/stage06_student/` 下的新目录；冻结案例文件也不要覆盖。

## G6-L 最终学习者验收

你需要同时满足以下要求：

1. 不看代码，画出 `模型身份 → 冻结案例 → 原始输出 → 确定性指标 → 统计区间 → 多维门禁 → 证据声明`；
2. 解释 M3 与 Q0–Q2 为什么不允许直接比较 PPL，以及 BPB解决了什么、仍未解决什么；
3. 对一个 5/16 成功率手算点估计，并解释 Wilson 区间为何比只报告 31.25% 更诚实；
4. 解释 Q2 的 pair accuracy 改善为何不能自动推出生成行为改善；
5. 在没有查看 `private_key.jsonl` 的前提下完成 30 对盲评；
6. 追踪 `EvaluationManifest.fingerprint()` 与 `model_acceptance()`；
7. 给出一段带证据路径和结论边界的最终陈述；
8. 明确说出 Q3 只验收一次 on-policy 管线，不验收能力提升。

<details>
<summary>思考题：为什么 Stage 6 的学习者门不能由单元测试自动关闭？</summary>

单元测试只能证明公式、Schema 和程序边界按预期工作。它不能证明学习者能够发现过度结论、独立解释置信区间、在盲评中遵守隔离，或面对互相冲突的指标做出诚实判断。因此 G6-A～G6-E 可自动关闭，G6-L 必须由学习者完成。
</details>
