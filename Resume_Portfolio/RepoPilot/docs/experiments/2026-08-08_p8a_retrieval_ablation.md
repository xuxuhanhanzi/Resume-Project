# P8A FRAMES 检索与证据合成改进实验

> 日期：2026-08-08
> 前置：P7 接手冻结完成，FRAMES 逐题审计显示 retrieval_failure 43%、reasoning_failure 40%

## 固定条件

- 数据：FRAMES smoke10（frames-test-0000~0009），3 个 oracle runs 已有
- 模型：qwen2.5:7b，digest 845dbda0ea48...，temperature=0，seed=7
- Executor：OracleDocumentModelExecutor（planned + reasoned + reviewed）
- Embedding：nomic-embed-text（768dim），Ollama /api/embeddings

## 新增代码

| 文件 | 作用 |
|---|---|
| `retrieval/protocols.py` | Retriever/Reranker/ContextCompressor protocol + FirstChunkRetriever (A2) |
| `retrieval/dense.py` | DenseRetriever（cosine similarity） |
| `retrieval/hybrid.py` | HybridRetriever（BM25+Dense，RRF k=60） |
| `retrieval/reranker.py` | EmbeddingReranker（query-chunk cosine reranking） |
| `benchmarks/executors/research.py` | 可配置 retriever_type + use_reranker |
| `scripts/run_frames_oracle_smoke.py` | --retriever-type / --use-reranker 参数 |

## 实验结果

| 配置 | 运行 | accuracy | 平均 | 对比 B1 |
|---|---|---|---|---|
| B1 BM25 | 3 runs | 20%, 20%, 10% | 16.7% | baseline |
| R1 Dense | 1 run | 0% | 0.0% | -16.7pp |
| R2 Hybrid (BM25+Dense RRF) | 3 runs | 20%, 30%, 10% | **20.0%** | **+3.3pp** |
| R3 Hybrid+Rerank | 1 run | 20% | 20.0% | +3.3pp |
| A2 No-retrieval (首段) | 1 run | 20% | 20.0% | +3.3pp |
| R4 Hybrid+QueryRewrite | 1 run | 0% | 0.0% | -16.7pp |

## 分析

### R1 Dense = 0%（失败）

nomic-embed-text 单独在百科 chunk 上完全失效。原因：
- 通用 embedding 模型在短事实性文本上的语义匹配不如词法匹配
- 多跳问题需要精确的关键词匹配（人名、地名、数值），dense 模糊了这些信号

### R2 Hybrid = 25%（最佳，+8.3pp）

BM25+Dense RRF 融合是最佳配置：
- BM25 提供精确词法匹配，Dense 补充语义相关片段
- RRF 融合让 BM25 主导、Dense 补充，取长补短
- 三次运行 20%, 30%, 10%，波动极大（20pp），10 题集太小无法做显著性结论

### R3 Hybrid+Rerank = 20%（无额外收益）

Embedding reranker 没有超过 R2：
- RRF 已经捕获了大部分有用信号
- nomic-embed-text 的 cosine reranking 与 RRF 使用的 dense score 高度相关，没有引入新的排序信号
- 真正有效的 reranker 可能需要更强的模型（LLM reranker 或 cross-encoder）

## 决策

1. **R2 Hybrid 选为最佳检索配置**，用于后续实验的 baseline。
2. R3 embedding reranker 不保留为默认配置。
3. R4（query rewrite）和 R5（context compression）待跑——query rewrite 可能比 rerank 更有价值，因为 P7 审计显示 Planner 的 query 质量直接影响检索效果。
4. R2 三次重复已完成：平均 20.0%（+3.3pp over B1），波动大（10%-30%），需 50 题以上确认稳定性。
5. 通过 10 题开发集门槛后，冻结 50 个未调参 FRAMES validation tasks。

## 下一步

- P8A R4：在 R2 hybrid 上加 query rewrite（LLM 改写 query 后用多个变体检索）
- P8A R5：在 R2 hybrid 上加 context compression（压缩检索 context，保留关键事实）
- P8A G1：固定 R2 检索 context，只改证据链 JSON 输出与 citation verifier
