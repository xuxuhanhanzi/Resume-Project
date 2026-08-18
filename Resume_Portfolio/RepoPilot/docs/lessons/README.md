# RepoPilot Stage 1–6 教学讲义

这套讲义面向第一次系统学习 LLM Agent 的读者。阅读时不要求先理解所有名词，也不建议一次性读完。
请从“唯一学习入口”开始，按照阶段运行 Demo、阅读源码、完成练习。

## 推荐顺序

1. [唯一学习入口与 12 天学习安排](stage01_06_learning_order.md)
2. [Agent 术语表](agent_glossary.md)
3. [Stage 1：最小 Agent Runtime](stage01_minimal_agent_runtime.md)
4. [Stage 2：本地模型、工具调用与安全边界](stage02_local_model_and_tools.md)
5. [Stage 3：Context、RAG、Memory 与 Skills](stage03_context_rag_memory_skills.md)
6. [Stage 4：MCP、规划、并行与 Reviewer](stage04_mcp_planning_parallel_reviewer.md)
7. [Stage 5：持久化恢复、权限审批与沙箱](stage05_recovery_permission_sandbox.md)
8. [Stage 6：评测、Benchmark、指标与消融](stage06_evaluation_and_benchmark.md)

## 每个阶段怎样学

```text
先读“本阶段解决什么问题”
        ↓
运行讲义中的最小实验
        ↓
对照讲义阅读少量核心源码
        ↓
完成练习，先不要看折叠答案
        ↓
用阶段检查表确认自己是否真的理解
```

如果一个词看不懂，先查[术语表](agent_glossary.md)；如果代码看不懂，回到
[学习入口中的 Python 预备知识](stage01_06_learning_order.md#2-学习前的最低-python-知识)。

## 讲义与实现证据的区别

- `docs/lessons/` 负责解释“为什么、是什么、怎样观察”；
- `docs/implementation/` 负责记录“项目实际实现了什么、验证了什么、还没有验证什么”；
- `src/` 是实际 Runtime 源码；
- `tests/` 与 `evaluation/` 是可重复执行的证据。

讲义不会把尚未完成的真实本地模型或 Docker 实验写成已完成结果。运行任何命令时，都应以你本机输出
为准。
