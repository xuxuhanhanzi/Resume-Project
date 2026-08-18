# RepoPilot 项目最终总结（P7–P12 交付）

> ⚠️ **SUPERSEDED** by [`docs/STATUS.md`](../STATUS.md) (2026-08-09) — the authoritative
> current-state document. This summary is kept for historical narrative only; where
> it conflicts with STATUS.md, STATUS.md wins.
>
> 日期：2026-08-09
> 范围：P7 接手冻结 → P12 评测交付
> 门禁状态（原始记录）：pytest 70 passed / ruff all passed / mypy 64 files 0 errors
> **更正**：当前 mypy 已扩展到 src+tests+scripts 共 100 文件、0 错误（见 STATUS.md / R0）。

RepoPilot 是一个**本地优先、反馈驱动、可治理的 Agent 运行时与评测骨架**，
覆盖知识研究、数据分析、软件工程三个领域。它不是生产平台，而是一个
**能对 Agent 改动做出可信因果判断的实验台**。

---

## 1. 三领域主指标（唯一权威口径）

三个领域的主指标定义不同，**不合成跨领域总分**。

| 领域 | 数据集 | 评测规模 | 主指标 | 结果 |
|------|--------|----------|--------|------|
| 知识研究 | Google FRAMES | 10 题开发集 | 答案准确率 | **20.0%**（R2 hybrid，3 次重复均值） |
| 数据分析 | InfiAgent-DABench | **35 题 validation** | 问题准确率 | **71.4%（25/35）** |
| 软件工程 | SWE-bench-Live | 3 题 dev smoke | 官方 resolved rate | **0/3**（未过门槛） |

### 结果可信度分级

| 领域 | 可引用性 | 说明 |
|------|----------|------|
| DABench | **可引用** | 35 题、11 表、哈希冻结、官方评分口径 |
| FRAMES | **仅供参考** | 10 题，单题即 10pp，波动大（同配置 20/30/10） |
| SWE-bench-Live | **不可引用** | 3 题且已被开发过程污染，只能作 dev smoke |

---

## 2. 各阶段完成情况

| 阶段 | 内容 | 状态 |
|------|------|------|
| P7 | 接手验证与状态冻结 | ✅ 完成 |
| P8A | FRAMES 检索改进（6 配置矩阵） | ✅ 完成 |
| P8B | SWE 失败驱动改进（代码） | ✅ 完成 |
| P8C | DABench 扩展评测 + 失败驱动改进 | ✅ 完成 |
| P9 | 机制消融（A2 无检索） | ✅ 完成（A3/A6 前置不满足，已说明） |
| P10 | QLoRA 微调（SFT + DPO） | ✅ 完成（未做下游评测，已声明） |
| P11 | Multi-Agent Harness | ✅ 完成（骨架 + 单测） |
| P12 | 评测协议与交付 | ✅ 完成 |

---

## 3. 完整实验矩阵

### 3.1 FRAMES 检索消融（10 题）

| 配置 | 准确率 | 结论 |
|------|--------|------|
| B1 BM25（基线，3 次） | 16.7% | 词法检索基线 |
| R1 Dense 单独 | 0.0% | 通用 embedding 完全失效 |
| **R2 Hybrid（BM25+Dense RRF，3 次）** | **20.0%** | 名义最佳，但方差极大（20/30/10） |
| R3 Hybrid + Reranker | 20.0% | reranker 无额外收益 |
| R4 Hybrid + Query Rewrite | 0.0% | LLM 改写显著有害 |
| A2 无检索（首段 baseline） | 20.0% | **与最佳检索配置持平** |

### 3.2 DABench 扩展评测与 guardrail 消融（35 题）

| 配置 | 准确率 | 空答案 | easy | medium | hard |
|------|--------|--------|------|--------|------|
| base | 71.4% | 5 | 11/13 | 8/12 | 6/10 |
| D1 方法规定式 guardrail | 71.4% | **2** | 10/13 | 10/12 | 5/10 |
| D2 目标规定式 guardrail | 71.4% | **2** | 10/13 | 8/12 | 7/10 |

### 3.3 P10 微调

| 阶段 | 结果 |
|------|------|
| SFT（9 样本，QLoRA r=16，4-bit NF4） | 12.7 秒，adapter 8.7 MB |
| DPO（8 对，手写训练循环） | 24 分钟，adapter 17.5 MB，loss 0.60→0.21，偏好间隔 0.19→1.45 |

---

## 4. 核心研究发现

### 4.1 小样本评测系统性高估性能

DABench 在 10 题 smoke 上是 **90.0%**，扩到 35 题后是 **71.4%** —— 小样本高估约 **19 个百分点**。
这不是噪声，而是选题偏差（smoke 题偏简单）。

**结论：任何 smoke 结果都不得外推为数据集性能。** 这条已写入项目硬约定。

### 4.2 总分稳定会掩盖大量逐题翻转

DABench base / D1 / D2 三个配置总准确率**完全相同**（均 25/35），但逐题看：

| 类别 | 题数 | 占比 |
|------|------|------|
| 三配置恒对 | 20 | 57.1% |
| 三配置间翻转 | 10 | **28.6%** |
| 三配置恒错 | 5 | 14.3% |

**若只看总分会得出"改动完全无效"的错误结论。**
必须做逐题 diff 才能看清改动真正改变了什么。这与 FRAMES 上 R2 三次重复
20%/30%/10% 是同一现象的不同表现：**单次 run 的总分不足以判断改动好坏。**

### 4.3 修复崩溃 ≠ 提升准确率

D1/D2 把 DABench 空答案从 5 降到 2（-60%），准确率却零变化。
失败没有消失，只是从"执行崩溃、无输出"变成"有输出、但数值错误"。

**建议：健壮性类改进必须同时报告"输出完整率"，否则会被准确率单一指标完全埋没。**

### 4.4 过度规定方法会抑制模型自适应

D1 强制 `pd.to_numeric(errors='coerce')`，把 base 中原本用 `str.extract` 正确处理的
带标注数值列全部变成 NaN，导致 `dabench-dev-0057` 的相关系数从 0.97 掉到 0.52。

D2 改为只声明不变量（必须是数值 dtype）+ 唯一禁令（禁用静默失败 API）+ 验证义务
（NaN 比例 >20% 说明方法错），把方法选择权交还模型，修回了 D1 弄坏的题。

**结论：guardrail 应当规定目标与验证方式，而不是规定实现方法。**

### 4.5 oracle 设定下检索改进价值有限

FRAMES 中"无检索"（A2，直接取文档首段）达到 20.0%，与最佳检索配置 R2 持平。
说明在 oracle document 设定下，瓶颈不在检索排序而在**证据合成与多跳推理**。
FRAMES 的后续改进应投向推理链，而非继续调检索。

---

## 5. 工程缺陷修复（扩大规模后暴露）

| 缺陷 | 影响 | 修复 |
|------|------|------|
| DABench runner 无单题异常隔离 | 第 6 题异常杀死整个 35 题运行，已完成结果全丢 | 每题 try/except + 增量落盘 + 进度输出 |
| manifest 与冻结快照不一致 | 2 个任务引用未冻结表，运行直接失败 | 确定性修复脚本生成 v2（35 题全部通过哈希校验） |
| TRL 1.9.2 与 torch 2.5.1 不兼容 | `DPOTrainer` 硬依赖不存在的 `FSDPModule`，无法 import | 放弃 TRL，手写 DPO 训练循环 |
| DPO 首次运行静默 OOM | epoch1 后进程被杀，无 traceback | 每步 `empty_cache()` + 顶层异常落盘 + 后台运行 |

这些缺陷都是**只有把评测规模从 10 题扩到 35 题才会暴露**的，本身就是 P8C 的价值之一。

---

## 6. 诚实声明（不粉饰的部分）

1. **FRAMES 准确率仍然很低（20%）**，且检索改进的 +3.3pp 在 10 题规模上不显著。
2. **SWE-bench-Live 是 0/3**，未通过任何门槛，且 3 题已被开发污染，不能作为能力证据。
3. **DABench 的 D1/D2 改进对准确率零收益**，只改善了输出完整率。
4. **P10 微调未做下游评测**，因此**不宣称微调带来任何指标提升**，只宣称训练链路跑通、
   DPO 偏好信号正确。adapter 接入评测链路需要 merge→GGUF→Ollama 导入，是基础设施工作量。
5. **P9 消融只完成 A2**，A3（无记忆）/A6（无路由）因前置机制未接入而未做。
6. 每个 DABench 配置只跑 1 次，考虑到 28.6% 的翻转率，**±3pp 内的差异都不应被解读为真实效果**。

---

## 7. 后续路线（按性价比排序）

| 优先级 | 工作 | 理由 |
|--------|------|------|
| 高 | DABench 稳定失败核心集（5 题）专项攻关 | 已定位，是真实能力瓶颈而非提示词问题 |
| 高 | 各配置做 3 次重复 | 28.6% 翻转率下，单次结果不可信 |
| 中 | FRAMES 转向推理链改进（放弃继续调检索） | A2 已证明检索不是瓶颈 |
| 中 | SWE-bench-Live 换未污染任务集重测 | 当前 3 题无法作为证据 |
| 低 | P10 adapter merge → GGUF → 接入评测 | 基础设施工作量大，科研收益不确定 |

---

## 8. 关键产物索引

**实验报告**

- `docs/experiments/2026-08-08_p8a_retrieval_ablation.md` — FRAMES 6 配置检索消融
- `docs/experiments/2026-08-09_p8c_dabench_val35.md` — DABench 35 题 + D1/D2 消融
- `docs/experiments/2026-08-08_p10_finetuning.md` — QLoRA SFT + DPO
- `docs/experiments/2026-08-08_p9_ablation_initial.md` — A2 无检索消融
- `docs/experiments/2026-08-08_p12_evaluation_protocol.md` — 三级评测协议

**核心代码**

- `src/repopilot/retrieval/` — BM25 / Dense / Hybrid / Reranker / FirstChunk
- `src/repopilot/benchmarks/executors/data_analysis.py` — Docker 隔离执行 + guardrail
- `src/repopilot/orchestration/harness.py` — StateGraph 多智能体骨架
- `scripts/print_progress.py` — 分领域结果汇总（**唯一权威口径入口**）
- `scripts/fix_dabench_manifest.py` — manifest 确定性修复

**冻结结果**

- `artifacts/benchmarks/20260809_p8c_dabench_val35_run3/` — DABench 主指标
- `artifacts/benchmarks/20260808_p8a_r2_hybrid_run{1,2,3}/` — FRAMES 最佳配置
- `artifacts/training/p10_qlora/` `artifacts/training/p10_dpo/` — 微调 adapter

---

## 9. 复现方式

```bash
# 查看所有分领域结果（权威口径）
python scripts/print_progress.py

# 复现 DABench 主指标（约 6 分钟）
python scripts/run_dabench_agent_smoke.py \
  --run-id <new-run-id> --model qwen2.5:7b \
  --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e \
  --manifest evaluation/benchmarks/manifests/dabench_validation35_v2.json

# 关闭 D2 guardrail 复现 base（DockerDataAnalysisConfig(dtype_guardrail=False)）
```

模型固定 `qwen2.5:7b` @ `845dbda0ea48...`，temperature=0，网络 deny，
Docker 镜像 `repopilot-dabench:py311-v1` @ `sha256:fc82eedb08c1...`。
