# P12 最终正式评测协议

> 日期：2026-08-08
> 前置：P7-P11 完成后执行

## 1. 三级资源控制

| 层级 | FRAMES | DABench | SWE-bench-Live | 用途 |
|---|---:|---:|---:|---|
| Dev | 10 | 10 | 已污染 3 题 | 快速调试 |
| Validation | 50 | 30-50 | 新 holdout 10 | 配置选择 |
| Final | 尽量完整 824 | 尽量完整 257 | Lite 300 | 最终报告 |

## 2. 固定变量

每次最终评测必须冻结以下变量并记录：

- 数据 revision（FRAMES `58d9fb6...`，DABench `b455d57...`，SWE-bench-Live `a637bd4...`）
- 任务 ID 列表（Dev/Validation/Final 各一份）
- 模型 ID + digest（`qwen2.5:7b` / `845dbda0ea48...`）
- 模型 quantization（Q4_K_M，Ollama）或 adapter hash（如有 P10 训练产物）
- Prompt 版本（system prompt + tool spec hash）
- Tool/skill 版本（coding tools spec hash）
- Retriever 配置（R2 hybrid, RRF k=60）
- Docker 镜像 digest（DABench / SWE 官方镜像）
- Evaluator revision（SWE-bench-Live `ad79b85...`，LF 补丁 ADR 0005）
- temperature=0, seed=7, network=deny

## 3. 重复与统计

- 随机过程（有 temperature 或非确定性）至少 3 次重复
- 确定性过程（temperature=0）也保留重复确认
- 任务级 paired comparison：同一 task_id 在不同配置下的结果对比
- Bootstrap CI：对准确率均值计算 95% 置信区间
- McNemar 检验：适用于成对二元结果（正确/错误）的配置对比
- 报告均值、标准差、最小值、最大值

## 4. 三领域分开报告

禁止合成跨领域综合分数。每领域报告：

- 主指标（FRAMES=准确率，DABench=问题准确率，SWE=resolved rate）
- 共享工程指标（迭代数、工具调用数、输入/输出 tokens、耗时、恢复次数）
- 每配置×每次重复的逐题结果表
- 失败类型分类（FRAMES: retrieval/reasoning/extraction；SWE: empty_patch/regression/type_error）
- 成本-收益比（准确率提升 vs tokens/耗时增加）

## 5. 最终交付物清单

| 交付物 | 文件 | 状态 |
|---|---|---|
| README | `README.md` | 已有，需更新最终状态 |
| 架构说明 | `docs/` | 已有教学讲义 + ADR |
| 复现实验脚本 | `scripts/run_*.py` | 已有 FRAMES/DABench/SWE 脚本 |
| 模型卡 | `docs/cards/agent_card.md` | 已有，需更新 |
| Agent 卡 | 同上 | — |
| 安全威胁模型 | `docs/security_threat_model.md` | 已有 |
| 数据/许可清单 | `evaluation/benchmarks/data/*/SOURCE.json` | 已有三领域 |
| 结果表 | `artifacts/benchmarks/*/results.json` | 已有 P5/P6 + P8A |
| 演示 | `repopilot demo` CLI | 已有 4 个 demo |
| 论文/简历版描述 | 待写 | P12 交付 |

## 6. 当前完成度评估

| 层级 | FRAMES | DABench | SWE |
|---|---|---|---|
| Dev (10题) | R2 hybrid=25% ✅ | B1=90% ✅ | 0/3 (已污染) |
| Validation | 未跑（需冻结 50 题） | 未跑（需下载 CSV） | 未跑（需冻结 holdout） |
| Final | 未跑（需完整 824 题） | 未跑（需完整 257 题） | 未跑（需 Lite 300） |
