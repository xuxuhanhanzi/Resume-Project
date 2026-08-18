# 第一个月逐日完成与学习日志

本日志把“项目已经替你构建的内容”和“你随后需要亲自掌握的知识”分开记录。完成状态只代表代码和验证已完成，不代表学习已经掌握。

## Day 1：资源、范围与平台决策

**项目完成：**

- 填写每周 60h+、本地 8 GB GPU、外部资源 ≤20 USD/月和岗位优先级；
- 实测 Windows、Python、Conda、Git、CUDA、GPU、Docker/WSL 状态；
- 建立 Stage 01 计划与 ADR-0001；
- 决定首月 Windows 开发 + Linux CI，后续 Linux GPU/容器按门禁引入。

**证据：** `docs/project_resource_card.md`、`docs/environment_audit_2026-07-11.md`、`docs/stage_01_engineering_plan.md`、`docs/adr/0001-development-platform-and-runtime.md`。

**你要学习：**

- P0/P1/P2 与为什么高级功能不能阻塞主线；
- 本地 GPU 显存、系统 CUDA Toolkit、PyTorch CUDA runtime 的区别；
- 为什么项目环境不能复用其他项目的 `SCI` 环境；
- ADR 的背景、候选、决策、代价和复审条件。

**自测：** 为什么 8 GB 显存不会阻塞首月，却会影响 0.5B–1.5B 后训练？

## Day 2：Python 包、依赖与 CLI

**项目完成：**

- 建立 `src/forgellm` 包和单一版本源；
- 完成 `pyproject.toml`、editable install、依赖锁和独立 `.venv`；
- 实现 `forgellm --help` 与 `forgellm doctor`；
- 建立首批 unit/smoke 测试。

**证据：** `pyproject.toml`、`requirements-dev.lock`、`src/forgellm/cli.py`、Day 2 实验记录；当日 4/4 测试通过。

**你要学习：**

- `src` layout、模块、包、console script；
- editable install 与普通 wheel 安装的差别；
- build dependency、runtime dependency、development dependency；
- 为什么直接依赖要锁定，而本地 `.venv` 不提交 Git。

**自测：** 从输入 `forgellm doctor` 到执行 `entrypoint()`，Python 经过了哪些入口？

## Day 3：配置、哈希、Run ID 与日志

**项目完成：**

- 用标准库 TOML 实现严格配置校验；
- 实现解析配置 SHA-256、Run ID、Git 快照、环境快照；
- 实现 JSONL 结构化日志和 Secret 脱敏；
- 防止日志字段伪造 event/timestamp，并避免误脱敏 tokenizer/token 计数。

**证据：** `src/forgellm/config.py`、`runtime.py`、`structured_logging.py`；Run ID `20260711T140723Z_stage01_cpu-smoke_21c69700`；17/17 测试通过。

**你要学习：**

- 配置解析、Schema 校验和 fail-fast；
- SHA-256 的用途与它不能证明什么；
- Git commit、dirty diff 与可复现结论的关系；
- JSONL、结构化日志、Secret/Token 脱敏边界。

**自测：** 为什么同一配置哈希可以相同，但两次 Run ID 仍应不同？

## Day 4：统一质量门禁与测试分层

**项目完成：**

- 建立 `python scripts/dev.py check` 权威入口；
- 串联 format-check、lint、mypy strict 和 pytest；
- unit、integration、smoke 可以独立运行；
- Makefile 改为短代理，未知命令有稳定退出码。

**证据：** `scripts/dev.py`、`CONTRIBUTING.md`、Day 4 实验记录；19/19 测试通过。

**你要学习：**

- formatter、linter、type checker、test runner 各自发现什么问题；
- unit/integration/smoke 的成本与边界；
- 退出码如何驱动 CI；
- 为什么所有测试都通过仍不等于功能满足用户需求。

**自测：** 一个函数 shape 正确但业务规则错误，最可能由哪一层测试发现？

## Day 5：陌生环境复现与 CI

**项目完成：**

- 在第二个 `.venv-repro` 从零安装；
- 使用锁定依赖再次运行完整质量门禁；
- 建立 Ubuntu/Python 3.11–3.12 GitHub Actions；
- README 提供安装、检查和 CPU Smoke 入口。

**证据：** `.github/workflows/ci.yml`、README、Day 5 实验记录；新环境 19/19 测试通过。

**你要学习：**

- “我的环境能跑”与“陌生环境可复现”的区别；
- CI matrix、最小权限、超时、依赖缓存；
- 为什么缓存不能成为正确性前提；
- 为什么存在 CI YAML 不等于 CI 已经通过。

**自测：** 当前 G0 为什么只能称为本地候选，而不是已发布 Release？

## Day 6：数据 Schema 与 Manifest

**项目完成：**

- 定义严格 JSONL `{id, text}` 输入；
- 定义保留文档、拒绝项和输出 Schema；
- Manifest 记录来源、许可证标签、输入/配置/输出哈希、计数和字节数；
- 输出目录禁止覆盖已有结果。

**证据：** `docs/data_pipeline_design.md`、`src/forgellm/data/schema.py`、`config.py`。

**你要学习：**

- JSONL 为什么适合流式数据；
- Schema version、data lineage、manifest；
- 文件哈希、配置哈希和数据版本；
- 为什么 Manifest 不应记录不可移植的绝对用户路径。

**自测：** 如果只修改过滤阈值但输入文件不变，哪些哈希应变化？

## Day 7：Reader 与 Unicode 规范化

**项目完成：**

- 实现 UTF-8 JSONL Reader；
- 无效 JSON/Schema 进入拒绝统计而不泄漏原文；
- 实现 CRLF/CR→LF、NFC、行尾空白和文档首尾空白处理；
- 覆盖中英文、组合字符、换行和控制字符夹具。

**证据：** `src/forgellm/data/pipeline.py` 的 parse/normalize 流程及相应测试。

**你要学习：**

- Unicode code point、UTF-8 byte、组合字符；
- NFC 与 NFKC 的差别及语义风险；
- Windows CRLF 与 Linux LF；
- 文本规范化为什么必须先冻结规则再训练 Tokenizer。

**自测：** 为什么 `Cafe\u0301` 和 `Café` 看起来一样却可能有不同 byte/hash？

## Day 8：质量过滤与精确去重

**项目完成：**

- 实现空文本、过短、过长和禁止控制字符过滤；
- 每个拒绝项有固定 reason；
- 先按 Document ID 去重，再按规范化内容 SHA-256 精确去重；
- 处理顺序不依赖输入行顺序。

**证据：** `_filter_candidates()`、`_deduplicate()`、`rejects.jsonl` 和重排输入测试。

**你要学习：**

- exact dedup 与 MinHash/LSH 近似去重的区别；
- 为什么去重必须在规范化之后；
- 为什么只统计“删了多少”不够，还要记录删除原因；
- 决定保留哪个重复记录时为什么需要确定性 Tie-break。

**自测：** 两个不同 ID 的文本经过 NFC 后完全相同，应保留哪一条，为什么？

## Day 9：确定性切分与 Writer

**项目完成：**

- 用内容哈希 + seed 映射 10000 个 bucket；
- 实现 80/10/10 basis points 配置校验；
- 稳定写出 train/validation/test/rejects；
- 输出 JSON key、记录顺序、UTF-8 和换行固定；
- 输入顺序变化时输出 shard 哈希保持一致。

**证据：** `assign_split()`、Writer、Manifest 输出哈希和确定性集成测试。

**你要学习：**

- 随机 shuffle split 与 hash split 的差别；
- seed、bucket、basis points；
- 为什么新增文档不应让旧文档重新分组；
- 稳定排序和确定性序列化。

**自测：** 为什么不能用 Python 内置 `hash(text)` 作为跨环境数据切分依据？

## Day 10：CPU Smoke、报告与结论边界

**项目完成：**

- 在 9 条自建夹具上跑通完整 CLI；
- 4 条保留、5 条拒绝，train/validation/test 为 2/1/1；
- 生成 6 个输出文件、固定哈希和统计报告；
- 使用相同输入与配置独立重复运行，6/6 输出文件哈希完全一致；
- 质量门禁 33/33 测试通过；
- 保留 CRLF 哈希失败和 seed 覆盖不足两次真实失败记录。

**证据：** `docs/experiments/2026-07-11_data_day10_smoke.md`，正式本地产物 `win_data_fixture_v2_20260711`。

**你要学习：**

- 如何读 Manifest、report、rejects 和 shard；
- 结果分母、拒绝率、split counts 与输出哈希；
- Smoke Test、正式数据实验和规模性能测试的边界；
- 为什么失败实验可以成为有效工程证据。

**自测：** 当前证据允许写进简历的最强表述是什么？哪些表述仍然属于过度声明？

## 前半月学习验收

在进入 Day 11 前，建议你完成以下动作：

1. 不看代码，画出 CLI → 配置 → 数据流水线 → Manifest 的调用链；
2. 手工解释一个文档从原始 JSONL 到 split shard 的每一步；
3. 修改一个临时配置阈值，预测哪些输出和哈希会变化，再运行验证；
4. 人工构造一个无效 Schema、一个 Unicode 组合字符和一个重复文本；
5. 用 5 分钟说明两次失败：CRLF record hash、seed 导致三路覆盖不足；
6. 回答每一天的自测问题，并把不确定项记录下来。

完成标准：你能够解释、修改、测试、运行和说明边界，而不只是阅读代码。