# RepoPilot 简历证据报告

更新时间：2026-08-23。此报告只记录已验证的事实；它不是性能宣称。

| 实验 | 当前状态 | 已验证证据 | 不能写入简历的内容 |
|---|---|---|---|
| P0 证据协议 | 已实现并 smoke | immutable manifest/receipt、Wilson/paired bootstrap、11 个回归测试 | 任何能力提升 |
| R1 代码检索 | CodeRAG final retrieval 已完成；SWE 未运行 | 5-repo RepoEval frozen final：BM25 NDCG@10 0.903961、Recall@10 0.606940；所有 raw output/receipt 已保存 | Pass@1、SWE resolved 差值，或“hybrid 优于 BM25” |
| R2 控制环 | E 盘官方评测器、严格 pre-call budget、收据绑定 scratch reset 与 generation smoke 已通过；development matrix 已冻结，正式消融未完成 | PEV 状态机、风险 gate、一次有界 replan、SWE-bench Verified v2 manifest、六臂/三 seed development config、官方 `--gold` 1/1 evaluator smoke、direct/PEV bounded receipts、一次成功的 reset-to-public-base receipt | 错误接受率、resolved 或成本改善 |
| R3 工具契约 | B0/B1 local smoke 完成；E 盘 BFCL official evaluator 环境已通过 CLI 预检；A/正式比较未运行 | schema/path/stage/recovery gate 已可选接入 AgentKernel；168 个 BFCL 单轮开发样本的 contract-validity / irrelevance raw outputs 已保存；Python 3.10 isolated BFCL `generate/evaluate` 可导入 | BFCL AST/executable 分数或 A 相对 B0/B1 的提升 |
| R4 长期记忆 | 预处理被严格 split 阻断 | TPM schema、provenance validator、content hash cache、LongMemEval retrieval adapter | LongMemEval QA、Recall@5 或 abstention F1 |

## Formal execution blockers recorded on 2026-08-23

- R2 已将官方 SWE-bench evaluator、数据、workspaces 与 run outputs 放在 E:；
  2026-08-24 预检时 E: 有 148.20 GiB 空闲，并已完成官方 `--gold` 1/1 和
  一个 actual direct generation smoke、一个由 shared-token ledger 标为不可比较的
  PEV generation smoke。Docker image VHDX 仍位于 C:，且正式
  R2-A/B/C 的全量配对配置、重复、工作树与镜像缓存远超过一个 smoke；没有把
  部分 Docker 结果写成正式分数。
- R4's strict rule groups every item sharing any `haystack_session_id`.  The
  downloaded official 500-record LongMemEval-S-cleaned input forms one
  connected leakage component, so non-empty 20/40/40 splits are impossible.
  The diagnostic is preserved under `artifacts/dataset_receipts/`; no R4 score
  is reported.
- R3's BFCL smoke is intentionally narrower than official BFCL: the upstream
  AST/state evaluator is now installed in a compatible E: Python 3.10 isolated
  environment, but the local `qwen2.5:1.5b` is absent from that upstream
  checkout's registered model protocol and BFCL has no valid non-gold stage
  mapping for R3-A.

当前唯一安全简历表述是：

> 实现并建立面向本地 Coding Agent 的证据驱动评测协议，包括可消融的代码检索、验证控制环、阶段化工具
> 契约和带来源的长期记忆；在冻结的 CodeRAG-Bench RepoEval 5-repo holdout 上，BM25 retrieval 获得
> NDCG@10 0.903961；其余性能结论仅在对应的公开数据、官方评测器和 final holdout 通过后报告。

要把表述替换成计划文件中的“提升/优于”模板，仍需满足原计划第 4.3 节的 final holdout、可比预算、
trace/patch/Docker 复查及配对置信区间条件。
