# P7 接手验证与状态冻结

> 日期：2026-08-08
> 执行人：阿布（WorkBuddy Agent）
> 前置文档：`docs/PROJECT_HANDOFF_2026-08-07.md`

## 1. 环境预检

| 检查项 | 结果 |
|---|---|
| Python | E:\anaconda3\python.exe 3.12.3（符合 >=3.11,<3.13） |
| Ollama | 可用，qwen2.5:7b digest 845dbda0ea48...（与冻结模型一致） |
| Docker | client/server 29.6.1，daemon 可用 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB（P10 本地 QLoRA 可行） |
| 根 .venv | 已用 Python 3.12.3 新建，依赖安装成功 |

可用模型：qwen2.5:7b（正式）、qwen2.5:3b、repopilot-qwen3:4b、qwen3:4b。

## 2. 三道门禁

| 门禁 | 结果 |
|---|---|
| pytest | **66 passed** in 11.36s（用 `--basetemp=D:/Temp/rp_pytest` 避开 `.venv` 路径过滤） |
| mypy | **Success: no issues found in 86 source files**（--no-incremental） |
| ruff | **All checks passed!** |

### 2.1 修复项

交接文档记录基线为"66 passed；mypy 对 61 个源文件通过；ruff 未执行"。本次首次完整跑通三道门禁，修复了以下问题：

1. **mypy unused-ignore**（3处）：`task.py:9`、`skills/registry.py:9`、`tests/integration/test_micro_benchmark.py:9` 的 `import yaml # type: ignore[import-untyped]` 因已安装 types-PyYAML 而变为 unused，删除注释。
2. **mypy import-not-found**（1处）：`scripts/prepare_swebench_live_smoke.py:72` 的 pyarrow ignore code 应为 `import-not-found` 而非 `import-untyped`。
3. **ruff E501**（11处）：scripts/ 和 src/ 中 11 行超过 100 字符，已全部断行。
4. **ruff I001**（1处）：`scripts/summarize_p5_p6_repeats.py` imports 未排序，已 `--fix`。
5. **ruff SIM102**（1处）：`src/repopilot/tools/coding.py:70` 嵌套 if 已合并。

### 2.2 发现的潜在 bug（未修，记录待跟进）

`_iter_files` 的 `_IGNORED_PARTS` 包含 `.venv`。若 pytest basetemp 路径含 `.venv`（如 `--basetemp=.venv/pytest_tmp`），所有文件被过滤导致 3 个测试假失败。不影响生产使用，但应在文档中注明 basetemp 不可含 `.venv`。

## 3. Manifest 校验与聚合复现

### 3.1 Manifest 校验

全部 8 个 manifest 文件（JSON/YAML/TOML）加载通过：
- `evaluation/benchmarks/manifests/*.json`（5个）
- `evaluation/benchmarks/experiment_matrix.yaml`
- `configs/**/*.toml`（3个）

### 3.2 聚合脚本复现

`scripts/summarize_p5_p6_repeats.py` 重跑成功，输出 6 groups、3 repetitions、10 tasks/run，结果写入 `artifacts/benchmarks/20260807_p5_p6_formal_repeats/results.json`。

## 4. Git 状态

- 根仓库：main 分支，基线 commit `e9dd59c`，82 个变更文件（11 modified + 71 untracked）。
- evaluator 子仓库：`external/SWE-bench-Live-python-only`，commit `ad79b85`，分支 python-only，1 处 LF 补丁改动。
- **未清理任何工作树改动**。

## 5. 过期文档修正

### 5.1 `docs/project_context_summary.md`

- "尚未执行任何不可信代码" → 更新为三领域实验已完成的具体结果。
- "Docker CLI 已发现但当前环境没有可用 daemon" → 更新为 daemon 29.6.1 已验证。
- "本地 Qwen 的具体 checkpoint 尚未冻结" → 更新为已冻结 qwen2.5:7b。
- P0 goal 更新为三领域研究问题。

### 5.2 `docs/implementation/stage_01_06_summary.md`

- Stage 2："A real Qwen checkpoint is intentionally not claimed" → 更新为已冻结。
- Stage 5："no verified daemon, so no hostile code was executed" → 更新为 daemon 已验证、SWE smoke 已评测。
- Stage 6："Formal local Qwen results require..." → 更新为三次重复已完成。

## 6. Evaluator LF 补丁固化

新增 ADR：`docs/adr/0005-windows-lf-evaluator-patch.md`。

补丁内容：`swebench/harness/run_evaluation.py` 两处 `write_text` 调用强制 `newline="\n"`，避免 Windows CRLF 污染 patch.diff 和 eval.sh。这是 evaluator 子仓库唯一的本地改动，不可删除。

## 7. FRAMES 逐题失败审计

审计脚本：`scripts/audit_frames_failures.py`（只读，不改算法）。
详细数据：`artifacts/benchmarks/20260808_p7_frames_failure_audit.json`。

### 7.1 前置事实

- 审计范围：FRAMES smoke10（frames-test-0000~0009），3 次重复 = 30 次运行。
- B1 平均准确率 16.7%（5/30 正确），与正式重复一致。

### 7.2 逐题结果

| Task | Gold | Run1 | Run2 | Run3 | 失败类型 |
|---|---|---|---|---|---|
| 0000 | Jane Ballou | reasoning | reasoning | reasoning | 推理错误（母亲/娘家姓链搞错） |
| 0001 | 37th | retrieval | retrieval | retrieval | 检索失败（建筑排名表缺失） |
| 0002 | 87 | retrieval | retrieval | retrieval | 检索失败（数值所需信息缺失） |
| 0003 | France | reasoning | reasoning | reasoning | 推理错误（答 Argentina） |
| 0004 | Jens Kidman | **correct** | **correct** | retrieval | 2/3 正确 |
| 0005 | 506000 | retrieval | retrieval | retrieval | 检索失败（数值缺失） |
| 0006 | Mendelevium... | retrieval | retrieval | retrieval | 检索失败（答 Indium） |
| 0007 | 2 | **correct** | **correct** | **correct** | 全对 |
| 0008 | 4 | reasoning | reasoning | reasoning | 推理错误（答 0/2） |
| 0009 | Battle of Hastings | reasoning | reasoning | reasoning | 推理错误（答 Thirteen Years' War） |

### 7.3 失败类型分布

| 类型 | 次数 | 占比 |
|---|---|---|
| correct | 5 | 16.7% |
| retrieval_failure | 13 | 43.3% |
| reasoning_failure | 12 | 40.0% |

### 7.4 P8A 决策依据

- 检索失败（43%）和推理失败（40%）几乎各占一半，不能只改其一。
- 检索失败集中在：多跳信息未召回、数值/表格信息缺失。
- 推理失败集中在：关系链错误（mother vs maiden name）、时间/数值计算错误、答案抽取错误。
- **P8A 第一个单变量实验**：先做 R1（dense retrieval 替换 BM25），因为 retrieval_failure 占比最高且 dense 可能覆盖 BM25 遗漏的语义相关片段。

## 8. 通过门槛

| 门槛 | 要求 | 实际 | 状态 |
|---|---|---|---|
| pytest | 66 passed 或更多 | 66 passed | ✓ |
| mypy | 全通过 | 86 source files, 0 errors | ✓ |
| ruff | 全通过 | All checks passed | ✓ |
| manifest 可加载 | 全部通过 | 8/8 | ✓ |
| 聚合可复现 | results.json 可重跑 | 6 groups, 3 reps | ✓ |
| git 保留/忽略策略 | 明确 | .gitignore 已更新 | ✓ |

## 9. 下一步

P7 完成，进入 P8A（FRAMES 检索与证据合成改进）。第一个实验：R1 dense retrieval 替换 BM25，在相同 10 题开发集上三次重复，门槛为准确率稳定超过 16.7%。
