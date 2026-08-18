# RepoPilot — 论文/简历版项目描述

## 项目概述

RepoPilot 是一个本地优先、反馈驱动、可治理的 Agent Runtime Lab + 评测骨架。
它以最小 Python 实现（~5000 行源码）覆盖 Agent 全链路：规划、检索、工具调用、
执行反馈、验证、记忆、治理，并在知识研究、数据分析、软件工程三个领域进行
可复现评测。

## 技术栈

- **语言/运行时**: Python 3.12, asyncio, dataclasses, pathlib
- **本地模型**: Qwen2.5-7B (Q4_K_M) via Ollama OpenAI-compatible API
- **评测数据集**: Google FRAMES (知识研究), InfiAgent-DABench (数据分析), SWE-bench-Live (软件工程)
- **检索**: BM25 (词法), nomic-embed-text (语义), RRF 混合, embedding reranker
- **安全**: Docker sandbox (network none, non-root, read-only root, dropped caps)
- **训练**: PyTorch + PEFT + TRL (QLoRA 4-bit, LoRA adapter)
- **门禁**: pytest (70+ tests), mypy strict, ruff
- **硬件**: NVIDIA RTX 4070 Laptop (8GB VRAM), Windows 11

## 三领域评测结果

| 领域 | 数据集 | 主指标 | Baseline | 最佳改进 |
|---|---|---|---|---|
| 知识研究 | FRAMES (10题 dev) | 答案准确率 | B1=16.7% | R2 hybrid=20.0% (+3.3pp) |
| 数据分析 | DABench (35题 val) | 问题准确率 | 25/35 | 25/35 (71.4%) ✅ 可引用 |
| 软件工程 | SWE-bench-Live (3题 smoke) | resolved rate | 0/3 | 0/3 (待改进) |

## 关键技术贡献

1. **Provider-neutral Agent Runtime**: ReAct 循环 + 硬预算 + 确定性验证器 + checkpoint/journal/trace
2. **检索消融矩阵**: BM25 vs Dense vs Hybrid(RRF) vs Rerank vs QueryRewrite vs No-retrieval
3. **结构化失败反馈**: verifier 输出 exception_type + shortest_traceback + changed_file_count
4. **早停机制**: 连续 2 次验证失败且 changed files 相同则终止
5. **Patch reviewer**: 检测集合清空/None赋值/pass替换/clear()/debug标记
6. **Multi-Agent harness**: 显式状态图 + 消息 envelope + 漂移检测 + trace 聚合
7. **QLoRA 训练管线**: 数据提取 → SFT/DPO 脚本 → adapter 保存 → 评测对比

## 关键研究发现

1. **Hybrid 检索最优**: BM25+Dense RRF (20%) > BM25 alone (16.7%); Dense alone 完全失效 (0%)
2. **Query rewrite 有害**: LLM 改写 query 丢失精确性，准确率降至 0%
3. **Reranker 无收益**: embedding cosine reranking 与 RRF 高度相关，不引入新信号
4. **DABench 反馈正收益**: 环境反馈稳定恢复 1 个任务 (+10pp)，代价是耗时 +36%
5. **10 题集太小**: 20% vs 16.7% 只差 0.3 题，需 50+ 题验证显著性

## 代码规模

- 源码: ~5000 行 (src/repopilot/)
- 测试: 70+ tests (tests/)
- 脚本: 15+ (scripts/)
- 文档: 20+ (docs/)
- Git commits: 35 (P7-P12 推进)
