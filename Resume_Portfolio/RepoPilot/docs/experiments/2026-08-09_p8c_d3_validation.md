# P8C 补遗：D3 解析健壮性 guardrail 验证（负向结果）

- 日期：2026-08-09
- 触发：P8C 报告把 5 题稳定失败核心集（0028/0055/0056/0062/0109）笼统归为“能力瓶颈”。深挖 run3 代码与 stderr 后，重新分类为 3 题“可修复解析崩溃”+ 2 题“真推理错误”，并据此实现 D3 guardrail。
- 运行：`20260809_p8c_dabench_core5_d3`（仅 5 核心题，`--parse-robustness`）。

## 设计（加法式，默认关闭）

`DockerDataAnalysisConfig.parse_robustness_guardrail: bool = False`，三类崩溃的具体不变量：
1. 只对数值列做聚合，禁止对 object/category 列调用 `.mean()`。
2. 含括号/区间的字符串用 `str.extract(r'(\d+(?:\.\d+)?)')` 取标量，而非 `astype(float)`。
3. 禁用 `astype(..., errors='ignore')`；混合列优先 `str.extract`。

接线：executor `data_analysis.py` L239 注入；runner `run_dabench_agent_smoke.py` `--parse-robustness` 开关。ruff + mypy 通过。

## 结果：5/5 仍失败（0 分回收）

| 题 | 难度 | run3 失败 | D3 结果 | 行为变化 |
|----|------|-----------|---------|----------|
| 0028 | hard | 崩溃 `df['region'].mean()` TypeError | success=False, pred='' | **崩溃已避免**（仅 FutureWarning），但无有效输出 |
| 0056 | easy | 崩溃 dtype | success=False | 非空但答案错 |
| 0062 | medium | 崩溃 `astype(float)` 区间串 | success=False | 非空但答案错 |
| 0055 | easy | 答错 | success=False | 仍错 |
| 0109 | hard | 答错 | success=False | 仍错 |

## 结论

1. **guardrail 确实生效**：0028 的 `mean(region)` 崩溃在 D3 中消失（执行 error=None），证明提示词注入改变了 agent 行为。
2. **但 0 分回收**：避免崩溃 ≠ 产出正确答案。0028 不再崩，却给出空答案；0056/0062 不再崩，却算错。
3. **再次印证核心发现 #3**——“修复崩溃 ≠ 提升准确率”——且证明在 *提示词 guardrail* 层面同样成立：已知崩溃模式可经指令规避，但缺乏底层推理能力的模型不会被提示词凭空补上。
4. **原“3 题可修复”假设部分成立（崩溃层）、整体不成立（得分层）**。P8C 报告对 5 题的“能力瓶颈”定性在 0028/0056/0062 上偏粗，但 D3 证明即使修掉崩溃也无净收益。

## 后续含义

- 0055/0109（及 0028 修复崩溃后的空答案）是**模型能力问题**，需 SFT/DPO 或更大模型，非提示词可解。
- 未跑全 35 题 D3：因核心 5 题零收益，全量回归性价比低；若需证明 D3 对 25 道通过题无回归，可后续补跑。
