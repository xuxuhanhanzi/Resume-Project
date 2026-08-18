# P8C 数据分析领域扩展评测与失败驱动改进

- 阶段: P8C
- 日期: 2026-08-09
- 数据集: InfiAgent-DABench dev split, revision `b455d578e30fee513abd79936cbdf7a6de026cb5`
- 评测规模: **35 题**(此前仅 10 题 smoke)
- 模型: `qwen2.5:7b` @ `845dbda0ea48ed74...`, temperature=0, 网络 deny
- 执行环境: Docker `repopilot-dabench:py311-v1` @ `sha256:fc82eedb08c1...`
- 状态: 完成

## 1. 目标

此前 DABench 只有 10 题 smoke 结果(90%),样本过小,不能作为领域指标。P8C 要做三件事:

1. 把评测规模扩大到 35 题,拿到**可引用的领域主指标**
2. 修复此前 val35 运行卡在 5/35 的流程缺陷
3. 基于真实失败做**失败驱动改进**并复测,验证改进是否有效

## 2. 流程缺陷与修复

扩大规模后暴露出两个此前被 10 题 smoke 掩盖的工程缺陷。

### 2.1 单题异常杀死整个运行(卡在 5/35)

`scripts/run_dabench_agent_smoke.py` 的主循环直接 `await runner.run_one(...)`,无异常隔离。
第 6 题 `dabench-dev-0024` 抛异常后整个 35 题运行直接终止,且已完成的 5 题结果未落盘。

**修复**: 每题包 `try/except`,失败记录 `error` 字段并继续;每题结束后**增量写** `results.json`,
崩溃也不丢进度;新增 `[i/N] task done acc_so_far=...` 进度输出。

### 2.2 manifest 与冻结快照不一致

`dabench_validation35_v1.json` 的 35 个 task_id 中有 2 个(`dabench-dev-0066`、`dabench-dev-0070`)
引用 `beauty and the labor market.csv`,而该表不在冻结的 11 表快照内。adapter 正确拒绝执行
(`DABench table is not frozen in this snapshot`),但因为没有异常隔离,整个运行直接失败。

**修复**: `scripts/fix_dabench_manifest.py` 确定性地把 2 个非法任务替换为编号最小的合法任务,
生成 `dabench_validation35_v2.json`(35 题、11 表、全部通过 sha256 校验)。

> 注意: 我们**没有**为了凑数而放宽表哈希校验。冻结校验是数据完整性保证,应当保留;
> 错的是选题清单,不是校验器。

## 3. 评测配置

| 配置 | run-id | 说明 |
|------|--------|------|
| **base** | `20260809_p8c_dabench_val35_run3` | 修复流程后的基线 |
| **D1** | `20260809_p8c_dabench_val35_run4_d1` | 方法规定式 numeric guardrail |
| **D2** | `20260809_p8c_dabench_val35_run5_d2` | 目标规定式 guardrail + 数据损失校验 |

三次运行使用完全相同的 manifest(v2)、模型、温度、镜像与网络策略,唯一变量是 prompt guardrail。

数据集分层: easy 13 / medium 12 / hard 10。

## 4. 基线结果(base)

**主指标: 准确率 25/35 = 71.4%**

| 难度 | 正确/总数 | 准确率 |
|------|-----------|--------|
| easy | 11/13 | 84.6% |
| medium | 8/12 | 66.7% |
| hard | 6/10 | 60.0% |
| **总计** | **25/35** | **71.4%** |

运行开销: 平均 1.23 轮迭代 / 2.23 次工具调用 / 9.5 秒每题,35 题总计 5.6 分钟。

### 4.1 与 10 题 smoke 的对比 —— 小样本严重高估

| 评测规模 | 准确率 |
|----------|--------|
| 10 题 smoke(此前 3 次重复) | 90.0% |
| **35 题 val35** | **71.4%** |

**10 题 smoke 高估了约 19 个百分点。** 这直接验证了项目约定"不把 smoke 结果外推为
完整数据集性能"的必要性。后续所有 DABench 数值一律以 35 题为准。

## 5. 失败归因

10 个失败中,**5 个 prediction 完全为空**(agent 两次尝试都没产出任何答案标签),
这是最严重也最可能被修复的一类。逐个查看容器 stderr,归因高度一致:

```
TypeError: '>' not supported between instances of 'str' and 'int'
Unable to parse string "...northeastsouthwest..." to numeric
```

追到生成的代码(`dabench-dev-0056/solution_attempt_1.py`):

```python
df['No. of deaths'] = df['No. of deaths'].str.replace(',', '').astype('Int64', errors='ignore')
df = df[df['No. of deaths'].notna() & (df['No. of deaths'] > 0)]   # <-- 在这里崩溃
```

**根因**: agent 其实已经正确执行了 `inspect_table` 拿到 DTYPES,也意识到该列是文本,
但用了 `astype('Int64', errors='ignore')`。pandas 该参数在转换失败时**静默返回原始字符串**
而不抛错,于是下一行的数值比较崩溃,任务无输出。

这是一个可用提示词修复的、明确的 API 误用模式 —— 因此值得做定向改进。

## 6. 改进 D1: 方法规定式 guardrail

在代码生成 prompt 中注入固定清洗配方:强制 `pd.to_numeric(..., errors='coerce')`,
明令禁止 `astype(..., errors='ignore')`;同时在 `diagnostic_hint` 中为 dtype 类错误
增加针对性的重试反馈。用 `DockerDataAnalysisConfig.dtype_guardrail` 开关控制以支持消融。

### 6.1 结果

| 指标 | base | D1 | 变化 |
|------|------|----|----|
| 准确率 | 71.4% | 71.4% | **0** |
| 空答案数 | 5 | **2** | **-60%** |
| 修复的题 | — | 3 (`0006`,`0034`,`0077`) | |
| 弄坏的题 | — | 3 (`0007`,`0023`,`0057`) | |

**D1 完全达成了它的设计目标(消灭空答案),但对准确率零贡献**,因为修好 3 题的同时弄坏了 3 题。

### 6.2 副作用机制

以 `dabench-dev-0057`(easy,base 正确 → D1 错误)为例:

| 配置 | 清洗代码 | 结果 |
|------|----------|------|
| base | `df[col].str.extract(r'(\d+)').astype('Int64')` | 相关系数 **0.97** ✓ |
| D1 | `pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')` | 相关系数 **0.52** ✗ |

该列的值带有额外标注,`str.extract` 能提取出数字,而被强制的 `to_numeric(errors='coerce')`
直接把它们变成 NaN,丢掉大量样本点,相关系数因此严重偏低。

**教训**: guardrail 规定了唯一方法,反而抑制了模型针对具体数据格式的自适应清洗能力。
把"避免崩溃"写成"必须用某个 API"是过度约束。

## 7. 改进 D2: 目标规定式 guardrail

根据 D1 的教训重写: 只声明**不变量**与**验证义务**,不指定实现方法。

- 声明目标: 任何被当作数值使用的列在比较/排序/聚合前必须是数值 dtype
- 保留唯一禁令: 禁止 `astype(..., errors='ignore')`(静默失败 API)
- **交还方法选择权**: str.replace / str.extract / to_numeric 由模型按观察到的数据格式自选
- **新增数据损失校验**: 若转换让超过 20% 的非空值变成 NaN,说明方法选错,应改用更宽容的提取

### 7.1 三配置总览

| 配置 | 准确率 | 空答案 | 重试次数 | easy | medium | hard |
|------|--------|--------|----------|------|--------|------|
| base | 71.4% | 5 | 8 | 11/13 | 8/12 | 6/10 |
| D1 | 71.4% | 2 | 7 | 10/13 | **10/12** | 5/10 |
| D2 | 71.4% | 2 | 8 | 10/13 | 8/12 | **7/10** |

D2 成功修回了被 D1 弄坏的 `0007`、`0023`(验证了"目标导向优于方法导向"的假设),
同时保住了空答案的改善(5 → 2),但又弄坏了 `0075`、`0124`,总分仍是 71.4%。

## 8. 核心发现

### 8.1 总分稳定掩盖了大量逐题翻转

三个配置的总准确率**完全相同**(均 25/35),但逐题结果差异巨大:

| 类别 | 题数 | 占比 |
|------|------|------|
| 三配置**恒对** | 20 | 57.1% |
| 三配置间**翻转** | 10 | 28.6% |
| 三配置**恒错** | 5 | 14.3% |

**若只看总分,会错误地得出"D1/D2 完全没有效果"的结论;逐题 diff 才揭示出
28.6% 的题目处于不稳定边界,且 D1/D2 确实各自改变了其中一批。**

方法论结论: **单次 run 的总分不足以判断一个改动的好坏,必须做逐题 diff,
并对边界题做多次重复。** 这与 P8A 在 FRAMES 上观察到的高波动(R2 hybrid 三次重复
20%/30%/10%)是同一个现象在不同领域的表现。

### 8.2 修复崩溃 ≠ 提升准确率

D1/D2 都可靠地把空答案从 5 降到 2,但准确率零变化。失败**并没有消失,只是换了形态**:
从"执行崩溃、无输出"变成"有输出、但数值错误"。

例如 `dabench-dev-0056`: base 输出为空 → D2 输出 `@max_deaths_country[South Africa]`(仍错)。
从工程可用性看这是进步(有结构化输出可供下游消费),从评测指标看则毫无收益。

**这提示: prompt 层面的健壮性改进主要提升的是"输出完整率"而非"答案正确率"。
如果只用准确率单一指标,这类改进会被完全埋没。** 建议 DABench 报告固定同时给出
准确率与空答案率两个指标。

### 8.3 稳定失败核心集

5 题在三个配置下都失败,是真正的能力瓶颈而非提示词问题:

| 任务 | 难度 |
|------|------|
| `dabench-dev-0028` | hard |
| `dabench-dev-0055` | easy |
| `dabench-dev-0056` | easy |
| `dabench-dev-0062` | medium |
| `dabench-dev-0109` | hard |

值得注意的是其中 2 题是 easy —— 说明官方难度标注与本 agent 的实际难度并不一致。
这 5 题是后续改进(更强模型 / 多轮自检 / 多样本投票)最该盯的目标集。

## 9. 结论

- **DABench 领域主指标确定为 71.4%(25/35)**,取代此前不可外推的 10 题 smoke 90%。
- 修复了两个真实的评测流程缺陷(单题异常杀全程、manifest 与冻结快照不一致),
  评测链路现在可以跑完整 35 题且崩溃不丢进度。
- 失败驱动改进 D1/D2 **对准确率无收益**,这是本阶段最重要的诚实结论,不做任何粉饰。
- 但 D1/D2 **把空答案率从 14.3% 降到 5.7%**,并暴露出两个有价值的方法论发现:
  过度规定方法会抑制模型自适应;总分稳定会掩盖逐题翻转。

## 10. 局限

- 每配置只跑 1 次,考虑到 28.6% 的翻转率,±3pp 内的差异都不应被解读为真实效果。
- 35 题只覆盖 11 张表(dev 全集为 257 题 / 52 表),表分布有偏(titanic.csv 占比偏高)。
- 全部结论仅对 `qwen2.5:7b` + 本 agent 框架成立,不可外推到其他模型。

## 11. 产物

- `evaluation/benchmarks/manifests/dabench_validation35_v2.json` — 修正后的 35 题 manifest
- `scripts/fix_dabench_manifest.py` — manifest 确定性修复脚本
- `scripts/run_dabench_agent_smoke.py` — 增加单题容错 + 增量落盘 + 进度输出
- `src/repopilot/benchmarks/executors/data_analysis.py` — D1/D2 guardrail(`dtype_guardrail` 开关)
- `artifacts/benchmarks/20260809_p8c_dabench_val35_run3/` — base 完整结果
- `artifacts/benchmarks/20260809_p8c_dabench_val35_run4_d1/` — D1 完整结果
- `artifacts/benchmarks/20260809_p8c_dabench_val35_run5_d2/` — D2 完整结果
