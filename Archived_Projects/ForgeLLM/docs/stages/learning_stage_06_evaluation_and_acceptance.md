# Stage 6：综合评测与最终验收实验计划（实施版 v2）

> 制定日期：2026-07-28（Asia/Singapore）  
> 状态：G6-A～G6-E 自动化实现与正式评测完成；G6-L 待学习者完成；Q0/Q1/Q2 模型行为均 rejected
> 周期：2 周，按每周 60 小时上限规划  
> 阶段门：G6-A / G6-B / G6-C / G6-D / G6-E / G6-L  
> 前置状态：Stage 4、Stage 5 学习者验收完成；Stage 6 已启动

## 1. 阶段目标

Stage 6 不再增加训练预算，目标是建立一套能对 Stage 3 预训练模型、Stage 4 SFT Adapter、Stage 5 DPO Adapter 作出诚实结论的最终验收系统。核心能力是：

1. 把训练代理指标与最终行为指标分开；
2. 为正确性、指令约束、偏好、保留、重复、效率分别定义指标；
3. 处理 tokenizer 不同导致 PPL 不可横比的问题；
4. 用冻结 test、污染检查、bootstrap interval 和 paired comparison 控制结论；
5. 校准 rule-based 与 model/judge 类评价，并保留原始逐题记录；
6. 测量 prefill/decode、batching、KV cache、显存和 wall-clock；
7. 产出最终 Model Card、Evaluation Card、claim-evidence matrix 和可复现验收命令。

本阶段不要求把当前 0.6B 模型变成可用产品。最终允许的结论可能是“训练链路完整，但严格行为未达标”；这同样是成功的工程验收。

## 2. 被验收对象与身份冻结

| ID | 对象 | 用途 | 是否可直接横比 |
|---|---|---|---|
| M3 | Stage 3 5.36M / 320-vocab checkpoint | 预训练闭环教学模型 | 只在自身 tokenizer 下评 loss/BPB |
| Q0 | Qwen3-0.6B Base 固定 revision | 后训练基线 | 与 Q1/Q2 同 tokenizer 可 paired 比较 |
| Q1 | Stage 4 SFT Adapter v2 | SFT 后模型 | 与 Q0/Q2 同协议比较 |
| Q2 | Stage 5 DPO Adapter v2 | 偏好后模型 | 与 Q0/Q1 同协议比较 |
| Q3 | Stage 5 GRPO one-step Adapter v2 | 管线 smoke | 仅做完整性检查，不作能力排名 |

所有评测记录 Base revision、Adapter SHA、tokenizer hash、generation config、eval dataset hash、代码 commit 和环境版本。任何一项变化都生成新 run ID。

## 3. 最新模型技术对本阶段的启发

### 3.1 已核验的官方材料

- DeepSeek-R1：区分 cold start、多阶段 RL、可读性与语言混合问题；启发本阶段把正确性和可读性拆开。
- DAPO/DrGRPO/GSPO/VESPO：启发报告长度分布、零信号组、ratio/ESS/staleness，而不是只报 reward。
- Kimi K3 官方报告：区分领域、reasoning effort、agentic verbosity 与部署量化；启发本阶段设计 effort/cost Pareto、agent trajectory schema 和 deployment-aware 指标。

### 3.2 待核验名称

截至冻结日，没有找到 DeepSeek 官方发布的 V4 技术报告，因此 registry 中继续标为 pending。若 Stage 6 开始前出现官方材料，只做一次 F0 增量审计：加入“问题—机制—证据—本地可实现层级”表，不改变已冻结核心实验。

### 3.3 明确不覆盖

当前本地模型不是多模态模型，也没有真实浏览器/tool-agent 环境；K3 的视觉和 agentic benchmark 不进入硬验收。只实现可迁移的 evaluation contract：reasoning-effort token budget、trajectory 字段、verbosity/cost 分离。不得用文本 smoke 声称复现 K3。

## 4. 评测数据分层

### T0：训练内诊断，不参与最终结论

现有 train/validation loss、DPO pair margin、GRPO shaped reward。用途是定位优化过程，不进入最终“能力通过”判定。

### T1：冻结项目 test

- Stage 3 TinyStories test 固定子集：M3 的 NLL/BPB 与生成退化；
- Stage 4 correctness test：精确/格式约束；
- Stage 4 SmolTalk test：assistant NLL 与样例行为；
- Stage 5 preference test：pair log-prob 与 strict generation；
- 新建但训练前冻结的 robustness transformations：空格、大小写、键顺序、数字范围与 prompt 表述变化。

### T2：污染哨兵

对每个 eval prompt/answer 计算 canonical exact hash，并在所有 Stage 3/4/5 train 文本中扫描；再做规范化 n-gram overlap 和 MinHash/相似候选检查。exact overlap 必须为 0；近重复只标记和分层报告，不自动删除证据。

### T3：人工小样本审计

固定 30 条、双盲乱序展示 Q0/Q1/Q2 输出。评分维度为 correctness、instruction following、repetition、clarity、harmfulness；允许 tie。若只有一个人评分，明确“不估计 inter-rater reliability”；推荐由用户作为第二评分者，计算 Cohen's kappa/一致率。

<details>
<summary>思考题：为什么 test 数据也需要哈希，而不仅是 train？</summary>

答案：结果必须绑定到精确题目集合。若 test 后来被编辑，旧指标就无法解释；哈希使数据变更强制产生新评测身份。
</details>

## 5. 指标体系

### 5.1 语言建模与 tokenizer 公平性

- 同 tokenizer 的 Q0/Q1/Q2：报告 token NLL/PPL，可 paired；
- 不同 tokenizer 的 M3 与 Qwen：不横比 PPL，改报 byte-normalized NLL/BPB；
- 同时报告 bytes、target tokens、被截断条数和空样本拒绝数。

```text
BPB = total negative log-likelihood / (total UTF-8 bytes × ln 2)
```

### 5.2 确定性行为

- exact match；
- all-constraints-pass；
- 每个约束组件；
- valid JSON / schema；
- correct-prefix-but-extra；
- character 8-gram 与 token repetition；
- response tokens、stop reason、截断率。

### 5.3 偏好与 reward

- pair accuracy、DPO margin、margin-length correlation；
- chosen/rejected token counts；
- strict generation success，绝不由 pair accuracy 替代；
- 若使用 judge：位置交换一致率、tie 率、与 strict verifier 的混淆矩阵。

### 5.4 统计不确定性

- 比例指标使用 Wilson interval；
- 连续指标使用 paired bootstrap 95% interval；
- Q0/Q1/Q2 用同一 prompt、同一 generation protocol 做 paired difference；
- 同时报告样本量和原始逐题 JSONL，不只报均值；
- 多指标不做“挑最好的一项”结论，预先冻结 primary/secondary metrics。

### 5.5 系统与成本

- cold load 与 warm load 分开；
- prefill latency、time-to-first-token、decode tokens/s；
- batch 1/4、prompt 32/128/256 tokens、decode 32/64 tokens；
- peak allocated/reserved CUDA memory；
- KV cache bytes 的理论值与实测增量；
- Adapter 文件大小、总 wall-clock、外部费用。

<details>
<summary>思考题：为什么吞吐与延迟不能只报一个？</summary>

答案：batching 可提高总 tokens/s，却增加单请求等待；prefill 与 decode 的瓶颈也不同。部署选择需要至少 TTFT、decode rate 和 batch throughput 三个维度。
</details>

## 6. Week 1：评测契约、正确性与统计

### G6-A：Evaluation Schema 与冻结数据

实现：

- `EvaluationCase`、`GenerationRecord`、`MetricRecord`、`EvaluationManifest`；
- 模型/Adapter/tokenizer/data/generation 身份哈希；
- 原始输出与派生指标分文件，派生指标可从原始输出重算；
- train/validation/test 污染扫描；
- eval test 一旦冻结，代码不得读取答案来选 generation 参数。

通过条件：同输入重建 manifest byte-exact；模型或 generation config 改一项，run fingerprint 必变；污染 fixture 可被抓住；未知字段 fail-fast。

### G6-B：确定性与语言模型指标

实现：

- exact/constraint/JSON/repetition/length/stop 指标；
- token NLL、PPL、byte-normalized NLL/BPB；
- Q0/Q1/Q2 同一 prompt 的 paired evaluator；
- M3 独立 tokenizer 路径；
- Wilson 与 paired bootstrap interval。

通过条件：手算 fixture 对齐；字符无空格重复可被抓住；不同 tokenizer PPL 的报告器必须拒绝直接排名；固定 seed 的 bootstrap 可重现。

### G6-C：Judge 校准与人工审计包

默认不调用付费外部 API。先实现 deterministic rubric 和可导出的盲评 JSONL/HTML；可选本地 judge 只作为 secondary metric。

通过条件：输出顺序 A/B 交换时 strict 指标不变；judge 报告位置一致率；人工评分保留匿名输出、rubric 与 raw label；单评分者不报告 kappa。

## 7. Week 2：系统、鲁棒性与最终验收

### G6-D：效率、显存与 KV cache

实现固定矩阵 benchmark，预热后计时，CUDA 前后 synchronize；分别测 Base、SFT、DPO Adapter。Adapter 不 merge，避免改变 Artifact；可另做一次临时 merge 数值等价测试，不保存覆盖正式模型。

通过条件：每格至少 5 次，报告 median/p10/p90；OOM 被记录为容量边界而非吞吐 0；理论 KV 元素数与实测趋势一致；结果绑定 GPU/torch/precision。

### G6-E：综合报告与 acceptance matrix

每个模型形成五维雷达式数据表，但不把不同量纲压成一个总分：

| 维度 | primary gate | secondary evidence |
|---|---|---|
| correctness | strict exact/constraint | component pass、prefix-extra |
| preference | held-out pair accuracy | margin 与长度相关 |
| retention | paired BPB/NLL 不显著恶化 | TinyStories/SmolTalk 切片 |
| stability | repetition/截断不过门限 | 长度、乱码、stop |
| efficiency | 8GB 内完成固定矩阵 | TTFT、decode、peak memory |

最终 Model Card 必须列出：适用范围、未通过门、数据局限、已知异常输出、复现命令、所有关键 SHA、外部费用和不可支持的 claim。

### G6-L：学习者最终答辩

学习者需：

- 在不查看 `private_key.jsonl` 的前提下完成 30 对盲评，并保存个人评分 Artifact；
- 从预训练 BPB 讲到 SFT loss、DPO pair metric 与 generation metric；
- 解释为什么不同 tokenizer 的 PPL 不横比；
- 手算一个 Wilson interval 与一次 paired difference；
- 诊断“pair accuracy=1、strict=0、repetition 上升”；
- 画出 eval 数据冻结、模型推理、raw output、metric、report 的单向数据流；
- 给每条最终 claim 指向一个 Artifact，并指出至少一个未证明事项。

## 8. 单变量实验矩阵

| ID | 问题 | 对照 | 唯一主变量 | 预算 | 成功定义 |
|---|---|---|---|---:|---|
| E0 | manifest 是否确定 | 同一输入 | 重建 | CPU | byte-exact/hash exact |
| E1 | 污染检测是否可靠 | clean fixture | 注入 exact/near duplicate | CPU | 命中预期样本 |
| E2 | metric 是否正确 | 手算输出 | 实现 | CPU | 全部数值对齐 |
| E3 | tokenizer 公平门 | 同 tokenizer | 改 tokenizer ID | CPU | PPL 排名 fail-fast |
| E4 | bootstrap 是否稳定 | 固定 paired scores | seed | CPU | 同 seed exact |
| E5 | SFT 的净变化 | Q0 | Q1 | 96 correctness + 固定保留集 | paired 全指标 |
| E6 | DPO 的净变化 | Q1 | Q2 | 同 E5 | 同协议 paired 全指标 |
| E7 | GRPO smoke 是否越界 | Q1 | Q3 | 只做 4×4 原题 | 仅完整性，不排名 |
| E8 | prompt 改写鲁棒性 | 原 prompt | 冻结 paraphrase | 32 pairs | 报性能下降区间 |
| E9 | 格式攻击 | clean prompt | whitespace/wrapper | 32 pairs | verifier 不被绕过 |
| E10 | reasoning effort | 32-token decode | 64-token decode | 16 prompts | 正确性—成本 Pareto |
| E11 | judge 位置偏差 | A/B | B/A | 30 pairs | 位置一致率显式报告 |
| E12 | Adapter 系统成本 | Q0 | Q1/Q2 | 固定 benchmark matrix | latency/memory paired |
| E13 | batch 效应 | batch 1 | batch 4 | prompt/decode 固定 | 延迟/吞吐同时报告 |
| E14 | KV cache 标度 | context 32 | 128/256 | batch/model 固定 | 理论与实测趋势一致 |
| E15 | 最终 claim 审计 | 报告 claims | evidence lookup | CPU | 每条 claim 有证据/边界 |

## 9. 资源上限与停止规则

- 不训练新模型，不做超参数 sweep；
- 本地 RTX 4070 Laptop 8GB，外部费用固定 0 USD；
- 每个质量评测 run 最多 30 分钟；每个系统矩阵最多 20 分钟；
- generation 默认 greedy；stochastic robustness 只对预先指定小集做 3 seeds；
- max new tokens 固定 32/64 两档，不以 test 结果调整；
- OOM、NaN、身份 hash 不匹配、答案泄漏、输出目录已存在时立即停止；
- 任何新增前沿论文不触发重新训练，只进入 registry/radar。

## 10. 建议的文件产物

```text
src/forgellm/evaluation/
  schema.py
  contamination.py
  language_modeling.py
  behavioral.py
  statistics.py
  judge.py
  systems.py
  report.py
scripts/
  prepare_stage6_evaluation.py
  stage6_run_quality_eval.py
  stage6_run_system_bench.py
  stage6_build_report.py
configs/evaluation/stage6_final.toml
docs/lessons/stage06_*.md
docs/cards/evaluation_card.md
artifacts/stage06/<run-id>/
```

## 11. 需要在实施前确认的三项决策

推荐默认方案如下，用户若无修改即可据此实施：

1. 人工审计：学习者必须完成 30 对盲评才能关闭 G6-L；若没有第二位独立评分者，只报告单评分者结果且不计算 kappa。
2. Judge：不使用付费 API；rule-based 为 primary，本地 model judge 仅作可选 secondary。
3. 验收哲学：不设综合总分；任何模型必须同时展示改善、退化和置信区间，严格行为门未通过时最终状态写“training pipeline accepted, model behavior not accepted”。

## 12. 完成定义

G6 自动化完成需要 E0–E15 的代码、测试、冻结配置、原始逐题 Artifact、综合报告和卡片全部落地；全仓质量门通过；失败运行不覆盖。G6-L 只有在学习者完成 30 对盲评、统计手算、代码追踪和最终答辩后关闭。Stage 6 不以“某个 Adapter 获得最高单指标”作为完成标准。

## 13. 实施冻结说明

用户已批准三项默认决策并要求开始 Stage 6。实施使用 64 个冻结案例、Q0/Q1/Q2 greedy 32-token 主质量协议、Q3 pipeline-only 审计、M3 独立语言建模套件、2,000 次成对 Bootstrap、30 对 Q1/Q2 盲评，以及 Q0/Q1/Q2 的固定系统矩阵。E10 另在 Q1 的 16 个 correctness 案例上冻结 32/64 token budget，只报告质量—耗时变化，不把更长输出自动称为更深推理。

第一次 quality v1 暴露多轮历史 assistant turn 被删除的问题；修复后的案例为 `stage6_evaluation_v2`。后续又发现 expected-response BPB 分子含 ChatML `<|im_end|>` 而 byte 分母不含该特殊 token；v3 启动审计又发现 `commit+dirty` 未唯一标识未提交源码。正式质量报告因此升级为 v4：只对回答文本 token 计算 BPB 分子，并把全部 Stage 6 可执行源码与配置的聚合 SHA-256 写入 `code_revision`。所有失败/被替代目录保留，版本间不拼接。

自动化实现覆盖：严格 Schema/Manifest、模型与数据哈希、污染审计、原始生成、行为/NLL/PPL/BPB、Wilson/paired bootstrap、盲评包、位置一致性、TTFT/吞吐/显存/KV、声明证据审计和多维门禁。G6-L 仍只由学习者完成。
