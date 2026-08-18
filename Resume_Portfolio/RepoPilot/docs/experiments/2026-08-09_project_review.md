# RepoPilot 项目完成情况审查报告

> ⚠️ **SUPERSEDED by [`../STATUS.md`](../STATUS.md).** This review (2026-08-09) predates the
> full audit (`PROJECT_COMPLETION_AUDIT_AND_REMEDIATION_PLAN_2026-08-09.md`). Its claims
> "三道门禁全绿" / "mypy 64 files" reflect a *partial* gate (only `mypy src`); the complete
> strict gate (src+tests+scripts, 100 files, 0 errors) was closed later and is recorded in
> STATUS.md. Treat this file as historical, not as current status.

- 审查时间：2026-08-09
- 审查对象：RepoPilot 全流程（P7 接手冻结 → P12 评测交付）+ 后续 D3 实验
- 审查方法：**独立取证、不采信既有总结**。逐条用 `git`、门禁命令、产物 `results.json`、源码接线核验。
- 审查结论：项目**实质完成、基本诚实**，发现并闭环 1 处过程漏洞（D3 负向结果未汇报 + 代码悬浮）。

---

## 一、已核实的完成项（带证据）

| 核查项 | 方法 | 结果 |
|--------|------|------|
| Commit 数 | `git rev-list --count HEAD` | **28**（含本审查闭环的 D3 commit `b3672d2`） |
| 工作树 | `git status` | 干净（仅 `dpo_run.log` 未跟踪，属训练日志，非源码） |
| pytest | `.venv/.../pytest -q` | **70 passed** ✅ |
| ruff | `.venv/.../ruff check .` | **All checks passed** ✅ |
| mypy | `.venv/.../mypy src` | **Success: 64 files, 0 errors** ✅ |
| P7–P12 阶段 | `git log` 逐条核对 | 每阶段均有 commit + 对应 `docs/experiments/*` 报告 ✅ |

> 三道门禁在**本会话当场复验全绿**，非引用旧结论。

---

## 二、三领域主指标逐项审计

### 数据分析 · DABench —— ✅ 可引用
- 口径：InfiAgent-DABench，**35 题**（v2 manifest，哈希冻结、官方评分）。
- 实测：`artifacts/.../run3/results.json` 中 `task_success=True` = **25/35 = 71.4%**。
- 分层：easy 84.6% / medium 66.7% / hard 60.0%。
- 评级：**唯一达到“可引用”标准的领域指标**，证据链完整。

### 知识研究 · FRAMES —— ⚠️ 仅供参考（正确标注）
- 口径：Google FRAMES，**10 题**（单题 = 10pp，波动极大）。
- 实测：artifacts 显示各重复 1–2/10，合 3 次重复均值约 **20%**（20/30/10）。
- 处理：报告已明确标注“仅供参考、不可外推”——**口径诚实**。
- 小风险：commit `c1a1892` 修正过 25%→20%，但“20/30/10”的逐次来源在现有命名 artifacts 中需人工拼接核对（量级与结论无误，仅溯源链略松）。

### 软件工程 · SWE-bench-Live —— ❌ 不可引用（正确标注）
- 口径：3 题 dev smoke，已**开发污染**，无官方 resolved。
- 实测：`official_smoke3` artifact 仅 1 条 `resolved=0` 记录（其余 2 题未持久化），结论 0/3。
- 处理：报告明确“不可引用、仅 dev smoke”——**标注正确**。
- 评级：作为“能力验证信号”有效，作为“性能指标”无效，二者在报告中区分清晰。

---

## 三、发现的问题与风险（按严重度）

### 🔴 已闭环（本审查修复）
**P1. D3 实验过程漏洞**：上一轮启动 D3（5 核心题）后称“后台运行中”，但
(a) 从未汇报**负向结果**——实际 `20260809_p8c_dabench_core5_d3` 已跑完且 **5/5 仍失败**；
(b) D3 代码 54 行处于**未提交悬浮态**。
→ 本审查已：提交代码 + 记录负向结果（`docs/experiments/2026-08-09_p8c_d3_validation.md`，commit `b3672d2`）。

### 🟡 需知晓（非阻断）
**P2. D3 的真正价值是“负向印证”**：D3 guardrail **确实生效**（0028 的 `df['region'].mean()` 崩溃在 D3 中消失，仅余 FutureWarning），但 **0 分回收**。这把核心发现 #3（修复崩溃 ≠ 提升准确率）从“代码层”推进到“提示词 guardrail 层”——是有价值的科学结论，而非浪费。

**P3. Windows CRLF 警告**：提交 D3 时 git 提示 `.py` 文件 LF→CRLF 转换。属环境已知问题（非 bug）；与记忆中“evaluator 的 patch.diff/eval.sh 必须 LF”无关（D3 改动非评测脚本）。

**P4. FRAMES 溯源链略松**：见第二节，量级正确、仅逐次来源命名不直接对应。

**P5. SWE artifact 记录不全**：official_smoke3 仅 1 条记录，但 0/3 结论由多 run 汇总且已标注不可引用，不影响结论。

---

## 四、方法论诚实性评估

| 诚实性检查点 | 结论 |
|--------------|------|
| 不合成跨领域总分 | ✅ 三领域分开报告 |
| 不把 smoke 外推为全量 | ✅ DABench 用 35 题主指标，FRAMES/SWE 明确标注不可引用 |
| 负向结果是否可见 | ✅ D1/D2 零收益、P10 未做下游评测、P9 仅完成 A2 均已写明；**唯一遗漏 D3 负向结果，本审查已补** |
| 失败 run 是否保留 | ✅ 各 run 目录齐全（run1–run5_d2 + core5_d3） |
| 模型/随机性是否冻结 | ✅ `qwen2.5:7b` digest、temperature=0、seed 固定 |

**总体诚实度：A−**（一处遗漏已闭环）。

---

## 五、总体结论与评级

| 维度 | 评级 | 依据 |
|------|------|------|
| 工程成熟度 | **A** | 三道门禁全绿、协议化评测、失败驱动设计、可复现 run-id |
| 科学严谨性 | **B+** | 诚实负向结果为主；曾漏报 1 个实验，已闭环；小样本已声明 |
| 计划完成度（P7–P12） | **A** | 8 个阶段全部交付，含代码 + 报告 |
| 仓库整洁度 | **B+** | 原为 B−（D3 悬浮），本审查提交后恢复 |
| **综合** | **实质完成 · 基本诚实** | 可交付；DABench 71.4% 为唯一硬指标 |

**一句话结论**：RepoPilot 按 P7–P12 计划已全部交付，三道门禁全绿，DABench 71.4%（25/35）为可引用主指标；方法论基本诚实，唯一的过程漏洞（D3 负向结果未汇报 + 代码悬浮）已在本审查中闭环。项目处于**可交付状态**，后续提升需转向模型能力层面（微调/换大模型），而非提示词 guardrail。

---

## 六、建议的后续行动

1. **（可选）D3 全量回归**：对 25 道通过题跑一次 D3，确认 guardrail 无回归；当前因核心 5 题零收益，优先级低。
2. **真推理失败攻关**：0055/0109（及 0028 修复崩溃后的空答案）属模型能力缺口，需 P10 adapter 接入评测链路或更大模型，非提示词可解。
3. **FRAMES 扩大样本**：若需把 FRAMES 从“仅供参考”提升为“可引用”，须扩到 ≥50 题并做 3 次重复。
4. **SWE 换未污染任务集**：重测以得到可信 resolved rate。
5. **DABench 3 次重复**：28.6% 逐题翻转率下，单次 71.4% 置信区间宽，关键结论（如 guardrail 零收益）建议重复确认。
