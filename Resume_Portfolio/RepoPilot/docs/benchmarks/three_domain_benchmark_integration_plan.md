# RepoPilot 三领域权威基准接入计划

> 日期：2026-08-07  
> 状态：P1 已完成；P2 真实本地模型 smoke 进行中；P3–P4 协议已接入但仍受 Docker 门禁限制  
> 目标：用三个公开权威基准验证一个“本地优先、自适应、可治理的任务执行 Agent”，而不是把系统限定为软件修复 Agent。

## 1. 本阶段结论

采用三个领域、三个独立主指标，不构造含义模糊的综合总分：

| 场景 | 数据集 | 首要回答的问题 | 领域主指标 |
|---|---|---|---|
| 知识研究 | Google FRAMES | Agent 能否检索、组合多来源证据并给出正确答案 | 答案正确率 |
| 数据分析 | InfiAgent-DABench | Agent 能否理解表格任务、编写并执行分析代码、输出闭式答案 | 官方问题/子问题正确率 |
| 软件工程 | SWE-bench-Live | Agent 能否在真实仓库中定位问题、修改代码并通过回归测试 | Resolved rate |

三者共享同一套运行记录与治理指标：迭代数、工具调用数、输入/输出 token、耗时、恢复次数、无效工具调用率、预算耗尽率和安全拦截次数。

接入顺序固定为 **FRAMES → DABench → SWE-bench-Live**。该顺序只代表工程风险，不代表研究重要性：FRAMES 最适合验证通用协议，DABench 增加代码执行与文件资产，SWE-bench-Live 最后引入完整仓库、容器镜像和回归测试。

## 2. 当前代码审计

现有 `PublicTaskSpec`、`EvaluatorTaskSpec` 和 `DeterministicVerifier` 均围绕 Python 仓库修复设计：任务必须包含 workspace、allowed paths、visible tests 和 test command，且语言被限制为 Python。`EvaluationRecord` 也主要记录 hidden tests 与 changed files。

因此不能把三个数据集硬塞进现有 dataclass。迁移采用兼容式分层：

1. 保留当前代码修复任务和微型基准，使已有测试继续通过。
2. 新增领域无关的 benchmark contracts 与运行输出协议。
3. 由 SWE-bench 适配器把通用任务转换为现有代码任务；FRAMES 和 DABench 不依赖代码专用字段。
4. 至少两个非软件工程适配器跑通后，再讨论项目重命名或删除旧接口。

## 3. 数据集可行性审计

### 3.1 Google FRAMES

- 官方来源：[Hugging Face 数据集](https://huggingface.co/datasets/google/frames-benchmark)、[论文](https://arxiv.org/abs/2409.12941)
- 许可：Apache-2.0。
- 规模：824 个多跳问答任务，提供标准答案、相关 Wikipedia 页面 URL 和推理类型标签。
- 优点：无需代码容器；答案和证据页面可分别评估；适合最先验证统一协议。
- 风险：URL 指向的页面会变化。正式实验不得直接依赖实时 Wikipedia 内容。
- 冻结方案：记录数据集 revision；第一阶段做 oracle-document smoke test，使用官方给出的相关页面；第二阶段建立带时间戳和哈希的离线语料快照，评估真实检索。

### 3.2 InfiAgent-DABench

- 官方来源：[GitHub](https://github.com/InfiAgent/InfiAgent)、[Hugging Face 数据集](https://huggingface.co/datasets/infiagent/DABench)、[论文](https://arxiv.org/abs/2401.05507)
- 许可：Apache-2.0。
- 公开验证集：257 个问题、52 个 CSV 文件、461 个子问题，包含 easy/medium/hard 难度；闭式答案允许无 LLM 裁判评分。
- 优点：可验证规划、Python 代码生成、执行反馈、结果格式化和错误恢复。
- 当前阻塞：Hugging Face 自动生成/加载会因 CSV schema 不一致失败。不得把 `datasets.load_dataset()` 作为唯一获取路径。
- 冻结方案：使用仓库或 Hub snapshot 按 revision 下载原始文件，在 manifest 中显式记录“问题 → CSV → 标准答案/格式要求”的对应关系和 SHA-256。
- 安全要求：不复用基准仓库中宽松的 subprocess 执行方式；所有生成代码必须进入 RepoPilot 的隔离环境，默认断网、限制时间/内存、只挂载当前任务资产。

### 3.3 SWE-bench-Live

- 官方来源：[项目网站](https://swe-bench-live.github.io/)、[GitHub](https://github.com/microsoft/SWE-bench-Live)、[Hugging Face 数据集](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live)
- 基准代码与元数据采用 MIT 许可；每个目标仓库和镜像仍需单独保留第三方许可信息。
- 优点：真实 GitHub 问题、补丁、FAIL_TO_PASS/PASS_TO_PASS 测试和官方 Docker evaluator，可直接衡量问题解决率。
- 风险：数据持续更新、镜像体积远大于元数据、需要可用 Docker daemon，并且当前主分支与早期 Python-only 评测路径存在版本差异。
- 冻结方案：首轮只选择 Python 任务，固定数据集 revision、官方 evaluator commit、实例镜像 digest 和任务 ID；优先在 Linux/WSL/AutoDL 环境运行，不把 Windows 本地可运行性作为默认假设。

## 4. 统一接口设计

不设计一个充满可选字段的“大一统 Task”。任务描述、执行环境和评分器彼此分离，尤其要保证标准答案与隐藏测试永远不进入 Agent 上下文。

```python
@dataclass(frozen=True)
class BenchmarkTask:
    benchmark_id: str
    dataset_revision: str
    task_id: str
    domain: Literal["research", "data_analysis", "software_engineering"]
    instruction: str
    assets: tuple[AssetRef, ...]
    environment_id: str
    allowed_tools: tuple[str, ...]
    budget: RunBudget
    network_policy: Literal["deny", "allowlist"]
    public_metadata: Mapping[str, JSONValue]

@dataclass(frozen=True)
class EvaluatorCase:
    benchmark_id: str
    dataset_revision: str
    task_id: str
    evaluator_id: str
    oracle_ref: SecretRef
    evaluator_config: Mapping[str, JSONValue]

@dataclass(frozen=True)
class TaskOutput:
    final_answer: str | None
    structured_payload: Mapping[str, JSONValue]
    artifacts: tuple[ArtifactRef, ...]
    evidence: tuple[EvidenceRef, ...]
    changed_files: tuple[str, ...]

@dataclass(frozen=True)
class EvaluationOutcome:
    task_success: bool
    primary_metric_name: str
    primary_metric_value: float
    domain_metrics: Mapping[str, float]
    failure_type: str | None
    evaluator_trace_ref: str
```

```python
class BenchmarkAdapter(Protocol):
    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]: ...
    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask: ...
    def evaluator_case(self, task_id: str) -> EvaluatorCase: ...

class TaskEnvironment(Protocol):
    def start(self, prepared: PreparedTask) -> EnvironmentHandle: ...
    def reset(self, handle: EnvironmentHandle) -> None: ...
    def close(self, handle: EnvironmentHandle) -> None: ...

class TaskEvaluator(Protocol):
    def evaluate(self, output: TaskOutput, case: EvaluatorCase,
                 handle: EnvironmentHandle) -> EvaluationOutcome: ...
```

数据流固定如下：

`冻结数据快照 → Adapter → BenchmarkTask → Agent/Environment → TaskOutput → 私有 EvaluatorCase → EvaluationOutcome → 报告`

## 5. 数据冻结与可复现协议

每次正式运行必须生成一个不可变 manifest，至少记录：

- benchmark 名称、split、dataset revision、evaluator commit；
- task ID 列表和选择规则；
- 数据文件、语料、仓库 commit、容器镜像 digest 的 SHA-256；
- 模型名称与 revision、推理参数、prompt/skill/tool 配置版本；
- 环境、依赖锁、随机种子、预算与网络策略；
- 每个任务的原始输出、trace、artifact、评分结果和失败分类。

manifest 只保存公开任务信息；oracle、gold patch、隐藏测试和标准答案保存在 evaluator-only 区域，运行时不可被 Agent 工具读取。

## 6. 分领域评分协议

### FRAMES

- 主指标：官方答案正确率。
- 诊断指标：相关页面 Recall@k、证据覆盖率、无效引用率、按推理类型分层的正确率。
- oracle-document 与 retrieval 两种设置必须分开报告，不能横向混算。

### DABench

- 主指标：官方问题正确率；同时报告子问题正确率。
- 诊断指标：代码执行成功率、答案格式错误率、分析结果与最终答案不一致率、sandbox/超时失败率，并按难度分层。

### SWE-bench-Live

- 主指标：官方 resolved rate，必须同时满足目标修复测试与回归测试要求。
- 诊断指标：环境构建成功率、patch apply 成功率、FAIL_TO_PASS 通过率、PASS_TO_PASS 保持率、无效补丁率。
- 官方 evaluator 输出是最终判据，RepoPilot 内部 verifier 只用于过程反馈，不能替代官方评分。

## 7. 实施阶段与门禁

| 阶段 | 工作内容 | 通过门禁 |
|---|---|---|
| P0 协议冻结 | 本文、目录和 manifest schema 评审 | 三个适配器无需改接口即可表达；gold/public 边界明确 |
| P1 公共骨架（已完成） | contracts、registry、runner、通用 metrics；兼容现有代码任务 | 现有测试通过；伪造的三个领域 fixture 均可完成 load→run→evaluate |
| P2 FRAMES | revision 固定、oracle-document 适配器、答案与检索评分 | 3 个协议 smoke + 10 个 Agent smoke 可复现 |
| P3 DABench | snapshot 获取、问题/CSV 映射、隔离 Python 环境、闭式评分 | easy/medium/hard 各至少 1 个协议 smoke；10 个 Agent smoke 可复现 |
| P4 SWE-bench-Live | Python 子集 manifest、官方镜像/evaluator 包装、资源预检 | 1 个 evaluator smoke、3 个 Agent smoke；结果与官方 evaluator 一致 |
| P5 固定基线 | 三领域固定模型、固定预算的预注册任务集 | 每项任务都有 trace、domain metrics、shared metrics 和失败类型 |
| P6 机制实验 | 反馈控制、记忆、检索、模型路由等逐项消融 | 每次只改变一个主变量；固定同一任务清单和评分器版本 |

正式样本量不在工程 smoke 前武断确定。先记录 10-task smoke 的成功率、耗时、token 和失败分布，再冻结正式任务 ID 与资源预算。smoke 结果只能用于排错，不能作为论文主结果。

## 8. 首批实现目录

```text
src/repopilot/benchmarks/
  contracts.py
  registry.py
  runner.py
  adapters/
    frames.py
    dabench.py
    swebench_live.py
evaluation/benchmarks/
  registry.yaml
  manifests/
docs/benchmarks/
  three_domain_benchmark_integration_plan.md
```

现有 `task.py`、`verification/verifier.py` 和微型 benchmark 暂不删除。P1 只增加兼容层和通用记录；在 FRAMES、DABench 均跑通之前，不进行破坏性重构。

## 9. 下一步工作包

立即进入 **P1 公共骨架**，范围严格限定为：

1. 添加上述 contracts、adapter registry 与 evaluator 边界。
2. 定义 JSON Schema/数据类校验和版本化 manifest。
3. 用三个不包含真实数据的最小 fixture 做契约测试。
4. 将现有 Python 修复任务包成兼容 adapter，确保现有功能与测试不回退。

P1 不下载完整数据集、不拉取 Docker 镜像、不运行模型。只有公共骨架通过测试后，才进入 FRAMES 的真实数据接入。

当前执行矩阵已冻结在 `evaluation/benchmarks/experiment_matrix.yaml`。协议层已完成三个官方数据集的 revision、哈希、公开/私有拆分与 scorer smoke；P5/P6 不得绕过其中列出的模型、离线语料、完整表格、Docker daemon 和官方实例镜像门禁。

## 10. 本计划完成定义

- 数据源、许可、评分器、访问方式和已知风险均有记录。
- 公共任务、私有评分数据、执行环境和输出协议已经解耦。
- 三领域分别保留权威主指标，共享工程指标但不制造综合总分。
- 实施顺序、门禁、smoke 规模、失败处理与可复现记录均可直接执行。
- 当前仍未验证的事项被明确保留到相应门禁，不宣称数据集或 Docker 已在本机跑通。
