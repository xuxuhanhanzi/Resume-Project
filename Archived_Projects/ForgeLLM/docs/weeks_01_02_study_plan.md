# ForgeLLM 第一、二周知识学习计划（14 天）

## 1. 学习目标

这两周不是继续增加功能，而是把已经完成的 Day 1–10 转化为你自己的能力。学习完成后，你应当能够：

1. 从零解释 ForgeLLM 前两周的工程结构；
2. 理解每个模块解决的问题，而不是只会运行命令；
3. 沿着 CLI 追踪到配置、核心逻辑、输出和测试；
4. 解释数据从 JSONL 输入到 Manifest 的完整生命周期；
5. 修改一个小规则、补一条测试并预测哪些证据会变化；
6. 清楚说明当前已经证明什么、尚未证明什么。

建议按每周 6 个高强度学习日 + 1 个综合复盘日执行。每天约 7–9 小时，总投入约 50–60 小时/周。

## 2. 先建立整体知识框架

前两周工程可以压缩为两条链。

### 工程证据链

```text
资源约束与阶段范围
→ 独立 Python 环境与依赖
→ CLI 与配置校验
→ Run ID、Git/环境快照、日志
→ format/lint/typecheck/test
→ 陌生环境复现与 CI
→ 可以审计的工程结论
```

### 数据证据链

```text
原始 JSONL
→ Schema 校验
→ Unicode 与换行规范化
→ 质量过滤
→ Document ID / 内容精确去重
→ 内容哈希确定性切分
→ 稳定写出 shard
→ report + manifest + 文件哈希
```

学习时始终问五个问题：

1. 这个模块的输入是什么？
2. 输出是什么？
3. 必须保持的不变量是什么？
4. 失败时如何表现？
5. 哪个测试或 Artifact 能证明它正确？

## 3. 每日固定学习方法

每天使用同一结构：

| 环节 | 建议时间 | 做什么 |
|---|---:|---|
| 知识概要 | 1–1.5h | 阅读本计划概要，并按关键词补充搜索 |
| 代码追踪 | 2h | 从入口逐函数跟踪，不追求逐行背诵 |
| 小实验 | 2h | 改临时输入/配置，先预测再运行 |
| 测试理解 | 1–1.5h | 找到正常、边界、失败和确定性测试 |
| 笔记与口述 | 1h | 画图、回答自测、做 5–10 分钟讲解 |
| 缓冲 | 0.5–1h | 处理不理解的概念，不提前进入下一天 |

建议在 `C:\tmp` 下复制输入和配置做实验，不直接改正式 Smoke 配置或已记录 Artifact。当前工作区尚未提交，学习过程中尤其不要随意覆盖正式文件。

# 第一周：理解工程基础设施

## Day 1：项目范围、资源约束与证据链

### 知识概要

- **P0/P1/P2：** P0 是作品成立所需的最小主线；P1 是有条件升级；P2 是长期研究扩展。
- **门禁：** 不是“日期到了就进入下一阶段”，而是证据满足后才能进入。
- **资源约束：** 时间、显存、预算和存储会改变实验规模，但不应取消正确性验证。
- **ADR：** 用于记录为什么选择一种工程方案，以及何时重新评审。

### 工程原理

个人项目最常见的问题不是不会写模型，而是范围不断扩大，最后没有可发布证据。ForgeLLM 先冻结本月 P0，再把 Docker、云 GPU、分布式训练等放到条件门禁之后。

```text
目标岗位
→ 需要证明的能力
→ 可交付项目切片
→ 资源预算
→ 阶段任务与退出条件
```

### 对照文件

- `docs/project_resource_card.md`
- `docs/stage_01_engineering_plan.md`
- `docs/adr/0001-development-platform-and-runtime.md`
- `docs/month_01_progress.md`

### 动手任务

1. 用一页纸写出 P0/P1/P2 各三个例子；
2. 解释为什么本地 8 GB GPU 不阻塞数据与 Tokenizer，却会约束后训练；
3. 不看 ADR，自己比较 Windows 原生、复用 SCI、独立环境 + Linux CI 三种方案。

### 自测

- 什么是阶段退出条件？
- 为什么“做了 Docker”不一定比“CPU Smoke 可复现”更重要？
- 什么情况下应该复审 ADR-0001？

### 搜索关键词

`software architecture decision record ADR`、`project scope P0 P1 P2`、`ML experiment stage gate`、`GPU VRAM training constraints`

## Day 2：Python 环境、包与依赖系统

### 知识概要

- **虚拟环境：** 隔离 Python 解释器环境中的依赖，避免项目互相污染。
- **模块与包：** `.py` 文件通常是模块；含可导入结构的目录是包。
- **src layout：** 把可安装代码放在 `src/`，降低误从仓库根目录导入的风险。
- **pyproject.toml：** 描述构建后端、项目元数据、依赖、CLI 和工具配置。
- **editable install：** 开发时包指向源码；修改源码后无需反复重新复制安装。
- **依赖层次：** build、runtime、development 依赖解决不同问题。

### 工程构成

```text
pyproject.toml
├─ build-system：如何构建
├─ project：包名、Python 范围、依赖
├─ project.scripts：CLI 名称 → Python 函数
└─ tool.*：pytest/mypy/ruff 配置

src/forgellm/
├─ __init__.py：公开包入口
├─ _version.py：单一版本源
├─ __main__.py：python -m forgellm
└─ cli.py：console script 入口
```

### 对照文件

- `pyproject.toml`
- `requirements-dev.lock`
- `src/forgellm/__init__.py`
- `src/forgellm/__main__.py`
- `src/forgellm/_version.py`
- `src/forgellm/cli.py`

### 动手任务

1. 画出 `forgellm doctor` 从命令到 `entrypoint()` 的调用链；
2. 分别解释 `python -m forgellm` 与 `forgellm.exe` 如何到达同一个核心入口；
3. 运行 `python -m pip show forgellm`，观察 editable 安装位置；
4. 列出 requirements lock 中每个直接依赖的用途。

### 自测

- 为什么不直接在 base 或 SCI 环境开发？
- `pyproject.toml` 中动态版本是如何读取 `_version.py` 的？
- editable install 能证明生产 wheel 一定正确吗？

### 搜索关键词

`Python Packaging User Guide pyproject.toml`、`Python src layout`、`editable install`、`console scripts entry points`、`virtual environment dependency isolation`

## Day 3：CLI、配置 Schema 与 fail-fast

### 知识概要

- **CLI 是边界层：** 负责解析字符串参数、显示错误和选择用例，不承载核心业务算法。
- **配置是实验输入：** 一个正式 Run 的参数必须可保存、校验和哈希。
- **Schema 校验：** 检查必填字段、未知字段、类型、取值范围和跨字段约束。
- **fail-fast：** 无效配置应在昂贵计算开始前终止。
- **解析后配置：** 保存程序实际使用的值，而不是只保存用户原始输入。

### 工程原理

```text
CLI 字符串参数
→ Path/整数等基本解析
→ TOML 读取
→ RunConfig/DataConfig 严格校验
→ 不可变 dataclass
→ 核心函数
```

CLI 与核心函数分离，使测试可以直接传入 Python 对象，不必每次启动子进程。

### 对照文件

- `src/forgellm/cli.py`
- `src/forgellm/config.py`
- `src/forgellm/data/config.py`
- `configs/runtime/smoke.toml`
- `configs/data/smoke.toml`

### 动手任务

1. 在临时配置中增加一个未知字段，预测错误后运行；
2. 把 `train_bps + validation_bps + test_bps` 改成不等于 10000，观察失败位置；
3. 解释为什么 `bool` 必须从整数配置中显式排除；
4. 追踪一次 `data-pipeline` 从 CLI 到 `run_data_pipeline()` 的参数传递。

### 自测

- 为什么拒绝未知字段通常比静默忽略更安全？
- 原始配置和 resolved config 有何区别？
- 为什么 DataConfig 使用 frozen dataclass？

### 搜索关键词

`configuration schema validation fail fast`、`Python dataclass frozen slots`、`TOML configuration`、`CLI boundary clean architecture`

## Day 4：哈希、Run ID、Git 快照与结构化日志

### 知识概要

- **SHA-256：** 把任意字节映射成固定长度摘要，用于变化检测与内容寻址。
- **哈希不等于真实性：** 它能说明内容是否相同，不能证明数据合法、无偏或高质量。
- **Run ID：** 人类可定位的运行身份；通常组合时间、阶段、名称和配置短哈希。
- **Git 快照：** commit 标识已提交代码，dirty 状态表示还有未提交差异。
- **结构化日志：** 每行一个 JSON Object，便于机器解析、筛选和统计。
- **脱敏：** 凭据字段进入日志前必须由程序控制，而不是依赖使用者记得删除。

### 工程证据链

```text
config.resolved.json + config SHA
environment.json
git.json
run.json
events.jsonl
        ↓
     Run ID
        ↓
结果表中的数字可反向定位
```

### 对照文件

- `src/forgellm/runtime.py`
- `src/forgellm/structured_logging.py`
- `tests/unit/test_runtime.py`
- `tests/unit/test_structured_logging.py`

### 动手任务

1. 运行两次 `init-run`，比较 Run ID、配置哈希和时间；
2. 查看 `git.json`，解释 commit、dirty、diff_sha256；
3. 构造临时日志字段 `api_key`、`prompt_tokens`、`tokenizer`，预测哪些会脱敏；
4. 解释 `_write_json()` 为什么先写临时文件再 replace。

### 自测

- 同一配置两次运行，哪些字段应相同，哪些应不同？
- dirty workspace 为什么会降低实验可信度？
- 为什么不能简单把所有包含 `token` 的字段都脱敏？

### 搜索关键词

`SHA-256 content addressing`、`ML experiment provenance`、`structured logging JSONL`、`secret redaction logging`、`atomic file write replace`

## Day 5：质量工具与测试分层

### 知识概要

| 工具/测试 | 主要回答的问题 |
|---|---|
| Formatter | 代码格式是否统一？ |
| Linter | 是否存在可疑、错误或不规范写法？ |
| Type checker | 静态类型契约是否一致？ |
| Unit test | 一个小模块的行为是否正确？ |
| Integration test | 多模块组合后是否正确？ |
| Smoke test | 用户入口的最小闭环是否能跑？ |

测试通过只说明“已定义的测试条件通过”，不说明没有未知 Bug，也不说明产品有价值。

### 工程构成

```text
python scripts/dev.py check
├─ ruff format --check
├─ ruff check
├─ mypy
└─ pytest
   ├─ tests/unit
   ├─ tests/integration
   └─ tests/smoke
```

统一入口的关键价值是：本地、CI 和陌生环境执行同一组规则。

### 对照文件

- `scripts/dev.py`
- `Makefile`
- `CONTRIBUTING.md`
- `tests/unit/`
- `tests/integration/`
- `tests/smoke/`

### 动手任务

1. 分别运行 `unit`、`integration`、`smoke`、`check`；
2. 为一个已有测试标注 Arrange、Act、Assert；
3. 找出配置测试中的正常、边界和失败案例；
4. 解释 subprocess 测试为什么需要 timeout、capture_output 和 returncode。

### 自测

- 哪类问题只能在 integration 或 smoke 中暴露？
- 为什么 `scripts/dev.py` 遇到第一个失败就停止？
- mypy 通过后为什么运行时仍可能失败？

### 搜索关键词

`test pyramid unit integration smoke`、`pytest fixtures tmp_path`、`static type checking mypy`、`ruff formatter linter`、`process exit code CI`

## Day 6：陌生环境复现、CI 与跨平台边界

### 知识概要

- **陌生环境复现：** 用一个未安装项目的环境按照文档从头执行。
- **CI：** 每次 push/PR 自动运行固定质量门禁。
- **Matrix：** 同一代码在多个 Python 版本上测试。
- **最小权限：** CI 默认只获得任务需要的权限。
- **缓存：** 只加速下载；清空缓存后仍必须正确。
- **跨平台：** Windows 本地通过不等于 Linux、容器或云 GPU 已通过。

### 对照文件

- `.github/workflows/ci.yml`
- `requirements-dev.lock`
- `docs/experiments/2026-07-11_stage01_day05_reproduction.md`
- `docs/adr/0001-development-platform-and-runtime.md`

### 动手任务

1. 逐行解释 CI workflow 的 trigger、permissions、job、matrix、steps；
2. 比较 `.venv` 与 `.venv-repro` 的用途；
3. 解释为什么项目没有 Git 远端时，不能宣称 CI 已通过；
4. 写出未来 Linux GPU Smoke 还必须记录的五类信息。

### 自测

- `ubuntu-latest` CI 通过能证明本地 CUDA 训练正确吗？
- 为什么不在 CI 中直接跑昂贵 GPU 训练？
- 依赖 lock 与环境快照各解决什么问题？

### 搜索关键词

`continuous integration GitHub Actions matrix`、`CI least privilege`、`dependency cache correctness`、`reproducible Python environment`

## Day 7：第一周综合复盘

### 上午：画图

不看代码，画出：

```text
用户命令
→ argparse
→ 配置读取与校验
→ 核心用例
→ Artifact
→ 测试与 CI
```

在每条箭头旁写出数据类型和可能失败的异常。

### 下午：复现

1. 按 README 运行 doctor；
2. 初始化一个临时 Run；
3. 找到五个元数据文件并解释作用；
4. 运行 unit/integration/smoke/check；
5. 用 10 分钟讲解整个工程证据链。

### 第一周验收

- [ ] 能解释包、CLI、配置、运行元数据和测试之间的关系；
- [ ] 能独立运行所有质量命令；
- [ ] 能解释 G0 尚缺的外部 Linux CI 证据；
- [ ] 能说明哈希、测试和 CI 各自不能证明什么；
- [ ] 能回答 Day 1–6 的自测问题。

# 第二周：理解数据工程闭环

## Day 8：JSONL、Schema、Manifest 与数据血缘

### 知识概要

- **JSONL：** 每行一个独立 JSON，适合流式读取、局部失败和分片。
- **Schema：** 定义每条记录允许的字段、类型和约束。
- **Data lineage：** 数据从来源、原始文件、处理配置到输出的可追踪关系。
- **Manifest：** 一个数据版本的索引，记录来源、哈希、配置、输出和统计。
- **Report：** 面向分析的结果摘要；Manifest 更关注身份和追踪。

### 工程构成

```text
sample_documents.jsonl
→ CandidateDocument
→ OutputDocument / Rejection
→ train/validation/test/rejects
→ report.json
→ manifest.json
```

### 对照文件

- `docs/data_pipeline_design.md`
- `src/forgellm/data/schema.py`
- `src/forgellm/data/config.py`
- `tests/fixtures/data/sample_documents.jsonl`
- Day 10 `manifest.json`

### 动手任务

1. 手工检查夹具的 9 条记录，预测保留/拒绝；
2. 区分 manifest、report、rejects、shard 的职责；
3. 如果输入内容不变但文件名改变，预测 Manifest 哪些字段变化；
4. 解释为什么 Reject 不保存完整原始文本。

### 自测

- JSONL 相比一个巨大 JSON Array 有什么工程优势？
- Schema version 为什么重要？
- 数据来源名称和许可证标签为什么不能在训练结束后再补？

### 搜索关键词

`JSON Lines streaming data`、`data schema validation`、`data lineage manifest`、`dataset versioning content hash`

## Day 9：Unicode、UTF-8 与文本规范化

### 知识概要

- **Unicode code point：** 抽象字符编号。
- **UTF-8：** code point 到 byte 序列的编码方式。
- **组合字符：** 一个视觉字符可能由基础字符 + combining mark 组成。
- **NFC：** 尽可能组合为规范形式；通常比 NFKC 更保守。
- **NFKC：** 还会折叠兼容字符，可能改变语义或格式。
- **换行：** Windows 常见 CRLF，Linux 常见 LF。

规范化顺序非常重要：

```text
原始 bytes
→ UTF-8 decode
→ 换行统一
→ Unicode normalize
→ 行尾/首尾空白策略
→ 计算内容哈希
```

如果先哈希后规范化，视觉上相同的文本可能无法去重。

### 对照代码

- `normalize_text()`
- `_record_ref()`
- `test_normalize_text_handles_unicode_newlines_and_trailing_space()`
- 测试夹具中的 `Cafe\u0301` 和 CRLF 文本

### 动手任务

1. 使用 Python 的 `len()`、`.encode('utf-8')`、`unicodedata.normalize()` 比较组合与预组合形式；
2. 比较同一文本在 CRLF/LF 下的原始文件哈希；
3. 再比较经过流水线后的内容哈希；
4. 写出为什么第一次确定性测试失败、如何修复。

### 自测

- character count、code point count、UTF-8 byte count 是否总相同？
- 为什么不能未经评测直接使用 NFKC？
- `_record_ref()` 为什么排除 CR/LF？

### 搜索关键词

`Unicode normalization NFC NFKC`、`combining characters UTF-8`、`CRLF LF cross platform`、`text normalization tokenizer`

## Day 10：过滤、精确去重与确定性 Tie-break

### 知识概要

- **质量过滤：** 根据可解释规则拒绝数据；每条规则需要原因统计。
- **Exact dedup：** 规范化文本完全相同才判重复。
- **Near dedup：** 文本高度相似也判重复，常用 MinHash/LSH；当前尚未实现。
- **Tie-break：** 多个候选都可保留时，用稳定规则决定保留者。
- **顺序无关：** 输入行重排不应改变保留集合和输出哈希。

### 当前处理顺序

```text
Schema 有效
→ normalize
→ empty/length/control filter
→ 相同 ID 去重
→ 相同内容 SHA-256 去重
→ split
```

先规范化后去重，能消除换行、组合字符等表示差异。先按 ID 再按内容去重，避免一个 ID 对应多个内容。

### 对照代码

- `_filter_candidates()`
- `_deduplicate()`
- `Rejection`
- `test_retained_outputs_do_not_depend_on_input_order()`

### 动手任务

1. 人工构造相同 ID 不同文本，预测保留项；
2. 构造不同 ID 相同规范化文本，预测 `duplicate_exact`；
3. 把输入行逆序，解释为什么输出仍一致；
4. 比较 exact dedup 与 near dedup 的误杀/漏检风险。

### 自测

- 为什么不能直接用 document ID 作为内容去重依据？
- 为什么保留字典序最小 ID 是确定性规则，但不一定是质量最优规则？
- 正式数据中还需要哪些过滤器？

### 搜索关键词

`exact deduplication SHA256`、`MinHash LSH near duplicate text`、`deterministic tie breaking`、`LLM data quality filtering`

## Day 11：确定性切分与稳定 Writer

### 知识概要

- **随机 shuffle split：** 依赖完整集合、顺序和随机状态；数据新增可能导致重排。
- **hash split：** 每条记录独立映射到固定 bucket；新增其他记录不会影响旧记录。
- **basis points：** 用整数 0–10000 表达比例，避免浮点边界问题。
- **稳定序列化：** 固定记录排序、JSON key 顺序、分隔符、编码和换行。
- **覆盖保护：** 输出目录必须不存在，避免新运行静默混入旧 Artifact。

### 当前公式

```text
split_hash = SHA256(f"{seed}:{content_sha256}")
bucket = 前 8 bytes 转整数 mod 10000

0 ... train_bps-1                 → train
train_bps ... +validation_bps-1   → validation
其余                                 → test
```

### 对照代码

- `assign_split()`
- `_jsonl_bytes()`
- `_write_json()`
- `run_data_pipeline()` 中 Writer 部分

### 动手任务

1. 为一个固定 content hash 手算流程结构，不要求手算 SHA-256；
2. 改临时 seed，观察 split 和配置哈希变化；
3. 解释为何 Smoke seed 从 1337 改为 6；
4. 解释为什么 Python 内置 `hash()` 不适合跨进程数据切分。

### 自测

- hash split 是否保证小数据集严格满足 80/10/10？
- 为什么 4 条夹具可能全部进入 train？
- 哪些序列化细节会让语义相同的输出产生不同文件哈希？

### 搜索关键词

`deterministic train validation test hash split`、`Python hash randomization`、`canonical JSON deterministic serialization`、`basis points integer percentage`

## Day 12：流水线架构、错误处理与隐私边界

### 工程分层

```text
接口层
  cli.py
      ↓
配置与契约层
  data/config.py + schema.py
      ↓
领域逻辑层
  parse → normalize → filter → dedup → split
      ↓
基础设施层
  filesystem + JSONL writer + hashing
      ↓
证据层
  report + manifest + experiment record
```

理想依赖方向是外层调用内层，核心数据规则不依赖 argparse 或 GitHub Actions。

### 错误分类

- **配置错误：** 启动前失败；
- **单条数据错误：** 记录拒绝原因，流水线继续；
- **系统错误：** 输入不存在、输出冲突、磁盘异常，整次运行失败；
- **安全/隐私问题：** 不应把原始敏感文本写进错误日志或 Reject。

### 当前边界

已经实现：Schema、基本过滤、精确去重、确定性切分。  
尚未实现：PII、凭据、许可证扫描、污染检测、近似去重、大规模并行和原子目录提交。

### 动手任务

1. 为每个错误类型找一个对应测试；
2. 解释为什么无效 JSON 是 record rejection，而输出目录已存在是 run failure；
3. 从 CLI 开始完整追踪一次错误如何转成退出码 2；
4. 列出正式数据阶段必须补充的五类风险。

### 自测

- 为什么不是所有异常都应该跳过继续？
- Reject 保存 record_ref 而不保存原文的代价是什么？
- 当前输出写入过程中断可能留下什么状态？

### 搜索关键词

`data pipeline architecture reader filter writer`、`error taxonomy fail fast fail safe`、`PII secret scanning dataset`、`atomic dataset write`

## Day 13：实验记录、失败分析与 Claim 边界

### 知识概要

- **Smoke Test：** 证明接口和最小链路能跑，不证明规模、性能或生产稳定性。
- **实验记录：** 目标、环境、变量、命令、输出、结果、失败、结论、下一步。
- **单一主变量：** 尽量让一次实验只改变一个关键因素，才能解释因果。
- **失败实验：** 如果失败改变了设计决策，就是有效证据。
- **Claim boundary：** 项目描述必须严格小于或等于已有证据。

### 三个核心失败案例

1. format-check 发现格式漂移：说明统一质量入口有效；
2. CRLF/LF 导致 record_ref 不一致：说明 byte 表示会影响哈希；
3. seed 1337 让 4 条记录全部进入 train：说明概率比例不保证极小样本覆盖。

### 当前允许的表述

> 在 9 条自建 JSONL 夹具上实现并测试确定性数据流水线，覆盖 Schema、NFC、基础过滤、SHA-256 精确去重、哈希切分、Manifest 和报告；相同配置两次运行的 6 个输出文件哈希一致。

当前不允许写：工业级、大规模、高质量训练数据、完成 PII/污染治理、显著提升模型效果。

### 对照文件

- `docs/experiments/2026-07-11_stage01_day04_quality.md`
- `docs/experiments/2026-07-11_stage01_day05_reproduction.md`
- `docs/experiments/2026-07-11_data_day10_smoke.md`

### 动手任务

1. 用“现象→假设→验证→修复→回归测试”讲解三个失败；
2. 找出 Day 10 记录中的主变量和固定项；
3. 写一条不过度声明的简历草稿；
4. 写三条当前禁止使用的夸大表述并解释原因。

### 自测

- 33/33 测试通过能否证明流水线适合 TB 级数据？
- 6/6 文件哈希一致能否证明数据内容高质量？
- 为什么保留失败比只展示成功更有工程价值？

### 搜索关键词

`reproducible ML experiments`、`controlled experiment single variable`、`smoke test limitations`、`scientific claim evidence engineering`

## Day 14：完整复现、代码讲解与学习验收

### 第一部分：完整运行

在新的临时输出目录执行：

1. `forgellm doctor`；
2. `python scripts/dev.py check`；
3. `forgellm data-pipeline ...`；
4. 打开 train/validation/test/rejects/report/manifest；
5. 检查输入、配置和输出哈希。

### 第二部分：闭卷架构讲解

用 15 分钟讲清：

1. 为什么先做工程基线；
2. CLI、配置和核心函数如何解耦；
3. 运行证据链如何构成；
4. 一个文档如何经过完整数据流水线；
5. 测试如何覆盖正常、边界、失败和确定性；
6. 当前证据边界和下一阶段缺口。

### 第三部分：小型修改考核

从以下任务选一个，在临时分支或备份后完成：

- 新增一种可解释拒绝原因及测试；
- 给 DataConfig 新增一个布尔策略并完成校验测试；
- 为 CLI 新增只读的 `show-manifest` 子命令；
- 为 report 增加一个不泄漏原文的统计指标。

要求先写：需求、输入输出、不变量、测试，再修改实现。

### 最终验收清单

- [ ] 能独立画出工程证据链和数据证据链；
- [ ] 能解释 10 个核心文件的职责；
- [ ] 能运行并解释 33 个测试覆盖的层次；
- [ ] 能预测修改配置、输入、seed 后哪些哈希变化；
- [ ] 能解释 NFC、精确去重和 hash split；
- [ ] 能复盘三个真实失败及修复；
- [ ] 能完成一个小修改并增加测试；
- [ ] 能给出不过度声明的项目介绍；
- [ ] 能说明 G0 尚缺外部 Linux CI；
- [ ] 能列出 Day 11 正式数据审计的前置问题。

全部完成后，再进入 Day 11–15 正式数据源审计。

## 4. 最少必须掌握的 12 个核心概念

如果时间不足，优先掌握以下框架，再自行搜索细节：

1. 虚拟环境与依赖隔离；
2. `pyproject.toml`、src layout、CLI entry point；
3. 配置 Schema 与 fail-fast；
4. SHA-256、Run ID 与 Git provenance；
5. JSONL 结构化日志与 Secret 脱敏；
6. formatter/linter/type checker/test 的区别；
7. unit/integration/smoke 与 CI；
8. JSONL、Schema、Manifest、data lineage；
9. Unicode、UTF-8、NFC 与换行；
10. 过滤、精确去重和确定性 Tie-break；
11. hash split 与稳定序列化；
12. Smoke、实验记录与 Claim 边界。

## 5. 学习笔记模板

每天只需填写以下结构，避免大量抄录：

```markdown
# Day N：主题

## 一句话解释

## 输入 / 输出

## 三个核心不变量

## 对应代码与测试

## 今天运行的命令

## 一个失败或边界

## 我仍不理解的问题

## 5 分钟口述结论
```

判断是否掌握的标准不是“看过”，而是能不看资料完成：解释、定位代码、运行、修改、测试、说明边界。
