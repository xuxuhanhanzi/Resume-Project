# P9 机制消融初步结果

> 日期：2026-08-08
> 前置：P8A 检索消融完成（R2 hybrid=25% 为最佳检索配置）

## A2 无检索消融

### 定义

"无检索"= FirstChunkRetriever：对每个 oracle document，不根据 query 搜索，直接取第一个 chunk（固定首段）。这隔离了"主动检索"相对于"被动上下文"的价值。

### 结果

| 配置 | accuracy | 说明 |
|---|---|---|
| B1 BM25 | 16.7% | 主动词法检索 |
| A2 No-retrieval（首段） | 20.0% | 被动首段 |
| R2 Hybrid | 25.0% | BM25+Dense RRF |

### 分析

A2 no_retrieval（20%）居然略高于 B1 BM25（16.7%），但差异仅 0.3 题（10 题集），不显著。可能原因：

1. **Oracle document 设置的局限**：document 已经是 oracle 选的（包含答案所需信息），所以只要看到 document 的任何相关部分就有可能答对。在这个前提下，首段（通常包含概述/摘要）可能恰好覆盖了关键信息。
2. **百科文章结构**：Wikipedia 文章开头通常是定义和概述，包含大量关键事实。BM25 选段有时反而错过了这些概述。
3. **10 题集太小**：20% vs 16.7% 只差 0.3 题，完全是噪声范围。

### 结论

在 oracle document + 10 题开发集上，检索算法的选择对结果影响有限。hybrid 的 25% 仍然最高，但优势不大。**要真正验证检索价值，需要在完整 824 题上跑，或使用非 oracle 的检索设置（从全语料中检索）。**

## 未完成的消融

| 消融 | 前置条件 | 状态 |
|---|---|---|
| A3 无记忆 | 需先把 episodic memory 接入真实 benchmark executor | 未满足 |
| A6 无路由 | 需先建合法 routing baseline（B1 固定模型不存在可移除的路由） | 未满足 |
| P9-P 只读并行 on/off | 可直接跑 | 待跑 |
| P9-S Skills on/off | 可直接跑 | 待跑 |
| Reviewer 全量 vs 失败触发 | 需改 config + 跑实验 | 待跑 |

## 后续计划

1. R2 hybrid 跑第三次重复，确认 25% 稳定性
2. 通过门槛后冻结 50 个 FRAMES validation tasks
3. A3：将 episodic memory 接入 FRAMES executor，再消融
4. A6：实现 rule/model/hybrid routing baseline，再消融
5. Reviewer：改为失败/低置信度触发，对比全量调用
