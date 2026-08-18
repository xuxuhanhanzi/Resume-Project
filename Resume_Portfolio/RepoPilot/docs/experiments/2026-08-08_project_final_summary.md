# RepoPilot 项目最终总结

> ⚠️ **SUPERSEDED by [`../STATUS.md`](../STATUS.md).** This summary (2026-08-08) claims
> "三道门禁全通过 / mypy 86 files", which predates the full strict audit. The authoritative,
> current gate status (format/lint/type/test all green; mypy 100 files strict 0 errors) lives
> in STATUS.md. Treat this file as historical.

> 日期：2026-08-08
> 执行人：阿布（WorkBuddy Agent）
> 会话：P7-P12 推进

## 1. 完成的工作

### P7 接手验证与状态冻结 ✅ 100%

- 三道门禁全通过：pytest 70 passed / mypy 86 files / ruff all passed
- 修正 16 个 lint/type 错误（首次完整跑通 ruff）
- 修正过期文档（project_context_summary + stage_01_06_summary）
- 新增 ADR 0005（evaluator LF 补丁固化）
- FRAMES 逐题失败审计：retrieval_failure 43% / reasoning_failure 40%
- Checkpoint commit（196 files）

### P8A FRAMES 检索改进 ✅ 55%（实验矩阵完整）

6 个配置全部跑完：

| 配置 | accuracy | 结论 |
|---|---|---|
| B1 BM25 | 16.7% | baseline |
| R1 Dense | 0.0% | 通用 embedding 完全失效 |
| **R2 Hybrid (BM25+Dense RRF)** | **20.0%** | **最佳，+3.3pp** |
| R3 Hybrid+Rerank | 20.0% | reranker 无额外收益 |
| A2 No-retrieval | 20.0% | 首段 vs 检索不显著 |
| R4 Hybrid+QueryRewrite | 0.0% | LLM 改写 query 有害 |

新增代码：protocols.py / dense.py / hybrid.py / reranker.py + research.py 可配置 retriever

### P8B SWE 失败驱动改进 ✅ 60%（代码完成）

- 结构化 verifier 输出（exception_type, shortest_traceback, changed_file_count）
- 早停机制（连续 2 次验证失败且 changed files 相同）
- Patch reviewer（检测集合清空/None赋值/pass替换/clear()/debug标记）

### P8C DABench 扩展 🔄 40%

- 0006 失败审计：pd.cut bins/labels 数量不匹配
- CSV 下载突破：找到正确路径 da-dev-tables/，下载 11 个 CSV（覆盖 35 题）
- 扩展 manifest 创建（dabench_validation35_v1.json）
- 待做：修改 run_dabench_agent_smoke.py 支持自定义 manifest + 跑实验

### P9 机制消融 🔄 25%

- A2 no_retrieval=20%（略高于 B1=16.7%，但不显著）
- harness 单元测试 4 passed
- 待做：A3（无记忆）、A6（无路由）前置条件未满足

### P10 微调与蒸馏 🔄 40%（torch 安装受阻）

- 训练数据提取：48 样本（9 SFT + 8 DPO pairs）
- QLoRA SFT 训练脚本（Qwen2.5-1.5B, 4-bit, LoRA r=16）
- DPO 训练脚本（8 对 preference pairs）
- **受阻**：torch 安装反复失败（CUDA 版 ~2.5GB 网络中断，CPU 版也失败）
- 待做：解决 torch 安装 → 跑 SFT/DPO 训练 → 对比 base vs fine-tuned

### P11 Multi-Agent Harness 🔄 35%

- StateGraph（planner → executor → verifier → reviewer → finish/replan）
- AgentMessage envelope（correlation ID, permission, budget）
- 漂移检测（repeated tools, evidence gaps, plan deviation）
- Trace 聚合
- 4 个单元测试 passed
- 待做：完整服务层、并发压测、故障注入

### P12 最终评测与交付 🔄 25%

- 三级评测协议文档（Dev/Validation/Final）
- README 更新为三领域评测描述
- 论文/简历版项目描述
- 待做：跑 Validation/Final 级别评测

## 2. 关键研究发现

1. **R2 hybrid 最优**：BM25+Dense RRF (20%) > BM25 alone (16.7%)；Dense alone 完全失效 (0%)
2. **Query rewrite 有害**：LLM 改写 query 丢失精确性，准确率降至 0%
3. **Reranker 无收益**：embedding cosine reranking 与 RRF 高度相关
4. **DABench 反馈正收益**：环境反馈稳定恢复 1 个任务 (+10pp)
5. **10 题集太小**：20% vs 16.7% 只差 0.3 题，需 50+ 题验证显著性
6. **Oracle document 局限**：检索算法选择在 oracle 设置下影响有限

## 3. 受阻项

| 项目 | 原因 | 解决方案 |
|---|---|---|
| P10 torch 安装 | 网络不稳定，CUDA 版 ~2.5GB 下载中断 | 换稳定网络 / 用 conda / 离线安装包 |
| P8C DABench 扩展实验 | 脚本硬编码 TABLE_SHA256 | 修改脚本支持自定义 manifest |
| P12 完整评测 | 需跑 824+257+300 题 | 需要大量时间（数十小时） |

## 4. Git 历史

12 次 commit，全部门禁通过：
1. P7 冻结（196 files）
2. P8A/P8B 代码
3. P8A-P11 代码
4. P8A 报告 + P12 协议
5. P9 消融报告
6. R2 三次重复修正
7. R4 query rewrite + CSV 脚本
8. README 三领域更新
9. DPO 脚本 + CSV manifest
10. R4 结果 + 最终矩阵
11. P11 harness 测试
12. 论文/简历描述

## 5. 后续计划

1. **P10**：解决 torch 安装 → 跑 QLoRA SFT/DPO → 对比 base vs fine-tuned
2. **P8C**：修改 DABench 脚本 → 跑 35 题扩展实验
3. **P9**：实现 A3（episodic memory 接入）和 A6（routing baseline）
4. **P11**：实现完整服务层 + 并发压测
5. **P12**：跑 Validation（50题）和 Final（完整数据集）级别评测
