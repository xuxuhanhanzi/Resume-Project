# 实验记录：S0 evidence foundation

## 1. 目标

实现 R1--R4 共用的不可变 manifest、receipt、配对统计与本地 smoke；核验公开数据、外部
评测源与当前机器是否满足正式评测前置。

## 2. 环境

- 平台：Windows 11，PowerShell
- Python：3.12.3（项目 .venv）
- Docker Server：29.6.1
- 代码 commit（开始时）：d3e195e453bef9e72f58ecc9a1384497f233394c（工作树原本已 dirty）
- LongMemEval source：9e0b455f4ef0e2ab8f2e582289761153549043fc，MIT
- CodeRAG-Bench source：f9e100ca9ed94b8f1983b356ae81966e30210cf4
- BFCL source：6ea57973c7a6097fd7c5915698c54c17c5b1b6c8，Apache-2.0
- LongMemEval-S 文件：500 records，SHA-256
  d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442

## 3. 实验变量

- 主变量：无；仅建立证据协议与离线模块。
- 固定项：数据文件、外部 checkout、模型均不参与结果选择。
- 对比对象：无。

## 4. 命令

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_evidence_protocol.py tests/unit/test_adaptive_retrieval.py tests/unit/test_provenance_memory.py tests/unit/test_pev_and_tool_contracts.py tests/unit/test_longmemeval_evaluation.py tests/unit/test_kernel_tool_contracts.py tests/unit/test_qrel_retrieval_evaluation.py

.\.venv\Scripts\python.exe -m ruff check src/repopilot/evidence src/repopilot/retrieval/adaptive.py src/repopilot/memory/provenance.py src/repopilot/evaluation/longmemeval.py src/repopilot/orchestration/pev.py src/repopilot/tools/contracts.py
~~~

## 5. 输出路径

- 代码：src/repopilot/evidence、src/repopilot/retrieval/adaptive.py、
  src/repopilot/memory/provenance.py、src/repopilot/evaluation/longmemeval.py、
  src/repopilot/orchestration/pev.py、src/repopilot/tools/contracts.py
- 配置：configs/evals/r1_*.yaml、r2_*.yaml、r3_*.yaml、r4_*.yaml
- 原始诊断：
  artifacts/dataset_receipts/longmemeval_s_group_leakage_diagnostic_2026-08-23.json

## 6. 结果

| 指标 | 数值 |
|---|---:|
| 新增离线/契约回归测试 | 11 / 11 passed |
| 新增文件的 Ruff 检查 | passed |
| LongMemEval-S 本地记录数 | 500 |
| 满足严格 session-group 的 20/40/40 split | 否（单一 500 题 connected component） |
| SWE-bench Verified 正式 Docker 运行 | 未运行（本机可用空间不足） |

## 7. 失败与异常

- 现象：首次严格 split manifest 将 500 个 LongMemEval-S 实例全部放入同一组。
- 判断：官方数据复用了 filler session；“任一共享 session 即同组”的预注册规则使全部实例连通。
- 处理：适配器现在在写 manifest 前拒绝空 development/validation/final split；诊断保留，不生成伪
  holdout。

## 8. 结论

S0 的实现和 smoke 已完成，但它没有产生模型能力结果。R4 的预注册拆分规则需要由计划作者明确
修订后才能继续；R1/R2 正式数据/资源前置也尚未满足，不能用现有历史 smoke 替代。

## 9. 下一步

1. 明确 R4 是否改为按问题来源/用户身份等非 filler session 的泄漏组，或改用支持独立 split 的外部数据；
2. 在资源满足的隔离 Linux Docker 环境 materialize SWE-bench Verified manifest；
3. 下载并冻结 CodeRAG-Bench RepoEval repository corpus，再执行 R1-A 至 R1-D；
4. 使用 BFCL 官方 offline harness 对 R3 产生的模型输出执行 AST/executable 评测。
