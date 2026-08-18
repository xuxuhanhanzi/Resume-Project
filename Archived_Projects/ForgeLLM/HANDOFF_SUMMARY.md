# ForgeLLM 项目交接与工作总结

> 用途：将本文档交给下一位工作人员后，对项目背景、当前代码、已完成证据、学习进度和下一步任务形成完整认识。
>
> 更新时间：2026-07-28（Asia/Singapore）
>
> 当前实际项目路径：`D:\Users\27475\Desktop\Resume_Project\ForgeLLM`
>

## 1. 用户目标与约束

用户希望以 LLM 算法/训练岗位为优先方向，长期成长为 AI 全栈工程师。项目目标不是只写一个模型脚本，而是建立可解释、可复现、可测试、可继续扩展的 LLM 工程与训练项目。

已知资源与约束：

- 每周可投入 60 小时以上，计划按每周 60 小时基线制定。
- 本地 NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB 显存，已通过实际训练验证。
- D 盘 2026-07-27 实测约 71.6 GiB 可用。
- 必要时可租用在线 GPU。
- 外部阅读/服务预算不超过每月 20 美元。
- 不应在资源和停止条件未固定前启动付费长训练。

## 2. 项目定位与范围

ForgeLLM 是一个用于学习和展示以下能力的长期项目：

```text
数据获取与清洗
→ Schema / Manifest / 去重 / 切分
→ Tokenizer
→ 小模型训练与后训练
→ 评估与回归
→ 推理服务
→ 监控、部署与工程化
```

首月刻意冻结范围，先完成工程和数据基础，不提前引入大型训练框架或生产基础设施。

### P0：必须完成的最小闭环

- 可安装 Python 包和独立虚拟环境
- CLI 入口与配置校验
- 数据 Schema、JSONL Reader、清洗、过滤、精确去重、确定性切分、Manifest
- 统一格式、lint、类型检查、单元/集成/Smoke 测试
- CI 配置、运行元数据、实验记录和基础文档

### P1：条件允许后的升级

- 本地 8GB GPU Smoke Run
- 小模型训练
- LoRA / QLoRA
- Linux 环境复现
- 基础推理服务

### P2：研究/生产扩展

- FSDP / DeepSpeed、多 GPU 训练
- vLLM、压测和服务化
- DPO / GRPO 等后训练
- Kubernetes、监控、自动扩缩容

## 3. 当前 Git 和 GitHub 状态

本地 ForgeLLM 仓库状态：

- 当前分支：`agent/month01-engineering-study`
- 工作区：dirty；包含用户已有计划/注册表和 2026-07-24 模型优先路线重构文档，接手时必须先检查具体 diff
- 本地最新提交：`983a64ab15ce160e232210d30b63912b082b8a22`
- 提交信息：`build: complete month one engineering and data baseline`
- 初始提交：`f934b00 chore: initialize ForgeLLM project scaffold`
- 远程：`https://github.com/xuxuhanhanzi/ForgeLLM.git`

GitHub 上传通过已认证的 GitHub connector 完成（本机没有 `gh` CLI，因此没有使用本地 `git push`）。

- GitHub 工作分支：`agent/month01-engineering-study`
- GitHub 远程提交：`4a5ff10dfad2b6697d7400f4e34b7d74dbb6c098`
- Draft PR：[xuxuhanhanzi/ForgeLLM Pull Request #1](https://github.com/xuxuhanhanzi/ForgeLLM/pull/1)
- PR 标题：`Build ForgeLLM month-one engineering and data baseline`
- PR 目标：`main`
- PR 状态：open、draft、尚未合并
- 上传内容：108 个项目文件

本地提交与 GitHub 提交 SHA 不同，是因为 GitHub 分支先有 bootstrap README 提交，再基于该提交创建远程提交；内容目标一致。接手时不要强行重写历史或删除分支。

## 4. 已完成的工程工作

### 工程结构

项目使用 `src` layout：

```text
ForgeLLM/
├── pyproject.toml
├── requirements-dev.lock
├── Makefile
├── scripts/dev.py
├── configs/
├── data_pipeline/
├── evaluation/
├── inference/
├── models/
├── post_training/
├── serving/
├── training/
├── src/forgellm/
├── tests/
└── docs/
```

### Python 包和 CLI

- `src/forgellm/__init__.py`：公开 `__version__`。
- `src/forgellm/_version.py`：版本单一来源，当前为 `0.1.0.dev0`。
- `src/forgellm/__main__.py`：支持 `python -m forgellm`。
- `src/forgellm/cli.py`：统一 CLI；已有 `doctor`、`init-run`、`data-pipeline`、`tokenizer-train`、`tokenizer-evaluate`、`tokenizer-compare` 子命令。
- `src/forgellm/config.py`：运行配置加载、校验和解析。
- `src/forgellm/runtime.py`：Run ID、配置/环境/Git 元数据和 Artifact 初始化。
- `src/forgellm/structured_logging.py`：结构化 JSONL 日志与敏感字段处理。

### Stage 1 Tokenizer

- `src/forgellm/tokenization/config.py`：严格 Tokenizer 配置与指纹；
- `src/forgellm/tokenization/corpus.py`：严格 JSONL 语料读取、重复 ID 与内容哈希检查；
- `src/forgellm/tokenization/bpe.py`：纯 Python 确定性 byte-level BPE、encode/decode、保存/加载和结构校验；
- `src/forgellm/tokenization/evaluation.py`：raw-byte baseline、round-trip、bytes/token、chars/token、fertility、unknown 和 subset；
- `src/forgellm/tokenization/hf_reference.py`：可选 Hugging Face ByteLevel+BPE 参考适配；
- `src/forgellm/tokenization/workflow.py`：训练 Manifest、评测与三路对照 Artifact；
- `src/forgellm/tokenization/pretokenization.py`：无损 none/whitespace/Unicode-class 边界；
- `src/forgellm/tokenization/advanced_bpe.py`：经典边界 BPE、BPE-dropout、Picky merge/remove 与 SuperBPE 两阶段课程；
- `src/forgellm/tokenization/unigram.py`：手写 byte Unigram、forward-backward EM、Viterbi 与 sampling；
- `src/forgellm/tokenization/special_tokens.py`：控制 Token 显式许可与 byte offsets；
- `src/forgellm/tokenization/entropy_patching.py`：BLT 思路的 bigram entropy patch 教学实现；
- `src/forgellm/tokenization/method_lab.py` 与 `scripts/tokenizer_method_lab.py`：统一 G1-B 方法实验；
- `configs/tokenizer/bpe_v1.toml`：词表 320、最小 pair 频率 2；
- `requirements-tokenizer.lock`：当前联合依赖使用 `tokenizers==0.22.2`，以兼容 Transformers 5.14.1；Stage 1 原始证据使用 0.23.1。

### 数据流水线

- `src/forgellm/data/schema.py`：数据记录和 Manifest 结构。
- `src/forgellm/data/config.py`：数据配置和比例校验。
- `src/forgellm/data/pipeline.py`：JSONL 读取、规范化、质量过滤、精确去重、确定性切分、输出和报告。
- 示例输入：`tests/fixtures/data/sample_documents.jsonl`。
- 数据设计说明：`docs/data_pipeline_design.md`。

### 质量工具和测试

- `scripts/dev.py` 是权威跨平台开发入口。
- `Makefile` 只提供快捷别名。
- `pyproject.toml` 统一配置 Ruff、mypy、pytest。
- CI：`.github/workflows/ci.yml`，目标为 Ubuntu + Python 3.11/3.12。
- 测试分层：`tests/unit`、`tests/integration`、`tests/smoke`。

## 5. 已有验证证据

最近一次完整验证命令（2026-07-27）：

```powershell
.\.venv\Scripts\python.exe scripts\dev.py check
```

结果：

- Ruff format：通过
- Ruff lint：通过
- mypy strict：通过
- Stage 1 基线 pytest：100/100 通过
- Stage 2 启用原生扩展门后的全仓测试：156/156 passed
- Stage 3 完成后的最新全仓质量门：Ruff、mypy strict 通过，175/175 passed，2 个非阻塞工具链 warning
- Stage 4 完成后的当前全仓质量门：125 个文件通过 Ruff format、Ruff lint 与 mypy strict，208/208 passed，2 个非阻塞 Windows C++ 工具链 warning
- Stage 5 完成时质量门：148 个 Python 文件通过 Ruff format、Ruff lint 与 strict mypy，226/226 passed，2 个非阻塞 Windows C++ 工具链 warning
- Stage 6 最终质量门：173 个 Python 文件通过格式与 Ruff，mypy 检查 113 个源文件无问题，244 passed、3 个 C++ 工具链测试按配置跳过；Stage 6 专项 21/21 通过
- 数据 Smoke：9 条输入，4 条保留，5 条拒绝；三路 split 为 2/1/1
- Tokenizer correctness candidate：320 词表、60 merge、模型 fingerprint `82ccbedc...05223`；
- Tokenizer test：ForgeLLM 152 Token vs raw byte 182 Token，round-trip 100%，unknown 0%；
- Hugging Face `tokenizers==0.23.1` 对照：149 Token，round-trip 100%，unknown 0%；
- G1-B 方法实验：六条 tokenizer 路径 round-trip 100%；Picky 56 merge/12 remove；SuperBPE 16 个 phase-2 merge；BPE-dropout/Unigram sampling 分别得到 20/6 种合法切分；
- Windows 新环境复现：通过

尚未完成或尚未证明的内容：

- GitHub Actions 的真实 Linux CI 运行结果尚未确认
- G0 `v0.1-engineering` 正式验收尚未勾选
- Stage 1 学习与测试已由用户确认完成
- G2-Core、G2-Arch、G2-Systems 自动化工程门全部完成
- Stage 2 学习与测试已由用户确认完成
- Stage 3 正式教学数据、训练、Checkpoint/Resume、Muon/MTP 和 1M-token 有界运行已完成
- Stage 3 学习者 16 站验收已由用户确认完成
- Stage 4 自动化 G4-A～G4-E 与 G4-L 学习者验收均已完成
- Stage 5 G5-A～G5-L 已完成，用户已确认学习者验收
- Stage 6 G6-A～G6-E、五份完整讲义、18 站入口、质量/系统正式评测和最终报告已完成；G6-L 待学习者完成
- Stage 6 三个模型行为均未接受：Q0/Q1/Q2 strict 均 0/32；推理服务尚未开始
- Docker、WSL、Linux GPU 环境不是当前 P0，不能作为首月完成前置条件

## 6. 关键文档索引

### 计划和状态

- [README.md](README.md)：项目入口和快速开始。
- [docs/model_first_21_week_learning_plan.md](docs/model_first_21_week_learning_plan.md)：当前唯一主计划。
- [docs/implementation_plan_clarity_audit_2026-08-09.md](docs/implementation_plan_clarity_audit_2026-08-09.md)：Stage 0–6 目标、步骤、门槛、失败分支与状态一致性审查。
- [docs/stage_plan_template.md](docs/stage_plan_template.md)：后续阶段统一计划模板。
- [docs/month_01_progress.md](docs/month_01_progress.md)：当前进度、门禁证据和风险。
- [docs/lessons/stage00_model_first_transition.md](docs/lessons/stage00_model_first_transition.md)：Stage 0 详细讲义。
- [docs/lessons/stage01_learning_order.md](docs/lessons/stage01_learning_order.md)：Stage 1 新手唯一学习入口，规定章节、文件、函数、练习和通过标准。
- [docs/lessons/stage01_byte_level_bpe_tokenizer.md](docs/lessons/stage01_byte_level_bpe_tokenizer.md)：Stage 1 零基础完整讲义，思考题后紧跟答案；按学习导航分段阅读。
- [docs/lessons/stage01_tokenizer_method_lab.md](docs/lessons/stage01_tokenizer_method_lab.md)：G1-B 现代 Tokenizer 方法完整讲义。
- [docs/stages/g1b_tokenizer_method_lab_plan.md](docs/stages/g1b_tokenizer_method_lab_plan.md)：G1-B 实验矩阵、实现边界与学习者验收。
- [docs/stages/learning_stage_01_tokenizer.md](docs/stages/learning_stage_01_tokenizer.md)：Stage 1 双门禁计划。
- [docs/lessons/stage02_learning_order.md](docs/lessons/stage02_learning_order.md)：Stage 2 新手唯一学习入口。
- [docs/lessons/stage02_transformer_core.md](docs/lessons/stage02_transformer_core.md)：Stable Decoder 完整讲义。
- [docs/lessons/stage02_frontier_architectures.md](docs/lessons/stage02_frontier_architectures.md)：现代架构方法完整讲义。
- [docs/lessons/stage02_model_systems.md](docs/lessons/stage02_model_systems.md)：模型系统与自定义算子完整讲义。
- [docs/stages/learning_stage_02_transformer.md](docs/stages/learning_stage_02_transformer.md)：Stage 2 三子门计划与完成定义。
- [docs/lessons/stage03_learning_order.md](docs/lessons/stage03_learning_order.md)：Stage 3 新手唯一学习入口与 16 站顺序。
- [docs/lessons/stage03_pretraining_data_and_objective.md](docs/lessons/stage03_pretraining_data_and_objective.md)：数据、packing、目标和指标讲义。
- [docs/lessons/stage03_optimization_and_precision.md](docs/lessons/stage03_optimization_and_precision.md)：优化、累积和混合精度讲义。
- [docs/lessons/stage03_checkpoint_and_recovery.md](docs/lessons/stage03_checkpoint_and_recovery.md)：全状态恢复与故障诊断讲义。
- [docs/lessons/stage03_modern_methods_and_experiments.md](docs/lessons/stage03_modern_methods_and_experiments.md)：Muon、MTP 与有界实验讲义。
- [docs/stages/learning_stage_03_pretraining.md](docs/stages/learning_stage_03_pretraining.md)：G3 子门和学习者完成定义。
- [docs/stages/learning_stage_04_supervised_posttraining.md](docs/stages/learning_stage_04_supervised_posttraining.md)：Stage 4 两周 SFT/LoRA/QLoRA 实现计划、实验矩阵、资源上限与 G4 子门。
- [docs/lessons/stage05_learning_order.md](docs/lessons/stage05_learning_order.md)：Stage 5 新手唯一学习入口与 18 站顺序。
- [docs/stages/learning_stage_05_preference_and_rl.md](docs/stages/learning_stage_05_preference_and_rl.md)：Stage 5 已实施方案、实验矩阵与 G5 子门。
- [docs/experiments/2026-07-28_stage05_implementation.md](docs/experiments/2026-07-28_stage05_implementation.md)：Stage 5 正式结果、失败修正和 Artifact 哈希。
- [docs/stages/learning_stage_06_evaluation_and_acceptance.md](docs/stages/learning_stage_06_evaluation_and_acceptance.md)：Stage 6 已实施计划、冻结协议与版本修正。
- [docs/lessons/stage06_learning_order.md](docs/lessons/stage06_learning_order.md)：Stage 6 当前唯一学习入口与 18 站顺序。
- [docs/experiments/2026-07-28_stage06_implementation.md](docs/experiments/2026-07-28_stage06_implementation.md)：Stage 6 v1–v6 修正、质量/系统结果、E0–E15 与 Artifact 哈希。
- [docs/cards/evaluation_card.md](docs/cards/evaluation_card.md)：无综合总分的最终评测卡。
- [docs/tokenizer/tokenizer_spec_v1.md](docs/tokenizer/tokenizer_spec_v1.md)：byte BPE 行为与 Artifact 规格。
- [docs/reference/minimum_engineering_support_card.md](docs/reference/minimum_engineering_support_card.md)：按需工程支撑执行卡。
- [docs/learning_path.md](docs/learning_path.md)：模型优先学习路径导航。

### 架构、资源和证据

- [docs/project_resource_card.md](docs/project_resource_card.md)：时间、硬件、预算与资源决策。
- [docs/environment_audit_2026-07-11.md](docs/environment_audit_2026-07-11.md)：Windows/Python/Git/CUDA/Docker/WSL 环境盘点。
- [docs/stages/learning_stage_00_model_first_transition.md](docs/stages/learning_stage_00_model_first_transition.md)：Stage 0 目标、假设、门槛与完成定义。
- [docs/adr/0001-development-platform-and-runtime.md](docs/adr/0001-development-platform-and-runtime.md)：开发平台和运行时决策。
- [docs/data_pipeline_design.md](docs/data_pipeline_design.md)：数据闭环设计。

### 实验记录

`docs/experiments/` 中记录 Day 2–Day 10 的实现和验证，包括：

- `2026-07-11_stage01_day02.md`
- `2026-07-11_stage01_day03.md`
- `2026-07-11_stage01_day04_quality.md`
- `2026-07-11_stage01_day05_reproduction.md`
- `2026-07-11_data_day10_smoke.md`
- `2026-07-25_stage01_tokenizer_candidate.md`

## 7. 学习进度与路线修订

### Day 1–3：用户已完成

- Day 1：范围、资源、阶段门和证据边界；
- Day 2：Python 环境、包、依赖与 CLI；
- Day 3：CLI、TOML、配置 Schema 与 fail-fast；
- 2026-07-24 用户确认完成 Day 3 学习与测试；
- Day 3 对应配置/CLI 测试曾验证 15/15 通过。

### Stage 0：已完成

用户发现旧方案的一般工程/数据知识占比过高，影响 Tokenizer、模型、预训练和后训练主线。2026-07-24 已完成路线切换：

- 新建 `docs/model_first_21_week_learning_plan.md`，作为唯一当前主计划；
- Day 4–14 旧课程降级为按需查阅，不再是前置门槛；
- 新建 `docs/reference/minimum_engineering_support_card.md`；
- Prompt 使用 core/support 教学分级；
- 完整质量门通过：Ruff、mypy、pytest 33/33；
- 详细讲义：`docs/lessons/stage00_model_first_transition.md`；
- 执行记录：`docs/experiments/2026-07-24_stage00_model_first_transition.md`。

### Stage 1：G1-A 已完成，G1-B 教学实现完成

2026-07-25 已完成：

- 纯 Python byte-level BPE 的训练、保存、加载、encode 与 decode；
- 固定 PAD/BOS/EOS/UNK = 0/1/2/3 和 256 byte 基础词表；
- 明确 pair tie-break、非重叠替换和模型 canonical SHA-256；
- raw UTF-8 byte 与 Hugging Face ByteLevel+BPE 同口径对照；
- 训练 Manifest、test 评测、reference comparison 和候选 Artifact；
- 英文、中文、Emoji、组合字符、空串、空白、损坏模型、文档顺序确定性测试；
- Ruff、mypy strict 和 pytest 68/68 通过；
- 完整讲义含 16 道思考题，每题后紧跟可折叠答案。

2026-07-27 用户明确目标是掌握算法原理与 Python 实现，不追求生产级 Tokenizer，也不继续优化词表规模。原 64 MiB Wikipedia/4096–8192 词表计划已删除并替换为 `docs/stages/g1b_tokenizer_method_lab_plan.md`。

新增并验证：

- 三种 pre-tokenization、BPE-dropout；
- 手写 Unigram forward-backward EM、Viterbi 与 sampling；
- Picky BPE 教学版 merge/remove 事件流；
- SuperBPE 教学版两阶段 curriculum；
- 特殊 Token policy、byte offset 和 bigram entropy patch；
- 统一六路算法/子集报告、完整讲义与 32 个新增测试；全仓 100/100 通过。

用户随后确认已完成 Stage 1 学习与测试。测试夹具仍不能替代真实模型质量证据。

### Stage 2：已完成

2026-07-27 已形成：

- `ModelConfig`、RMSNorm、RoPE、SwiGLU、MHA/GQA/QK-Norm、Causal Mask；
- Pre-Norm Decoder、权重共享、shifted CE、greedy/top-k/top-p、KV Cache；
- Tiny Overfit、保存加载、因果性、manual/SDPA 数值与梯度对照；
- MLA、MoBA、CSA/HCA-lite、Gated DeltaNet/Hybrid、Sparse MoE、Hash/Sinkhorn、mHC-lite、Attention Residuals、MTP 与 Muon 的缩小 reference；
- online-softmax、INT8 weight-only、`torch.compile(fullgraph=True)` 和真实双进程 CPU/Gloo DDP；
- `silu_mul` Python reference、C++/CUDA 源码、Dispatcher、FakeTensor 与 Autograd 注册；
- 三份完整讲义、14 站学习顺序、Stage 计划和实验记录。

固定 GPU 实验使用 RTX 4070 Laptop GPU：5,361,856 参数 Decoder 的 forward/backward 通过，峰值已分配显存约 60.53 MiB；manual/SDPA 最大绝对误差约 `1.49e-6`；MLA 教学配置的缓存元素数为普通 MHA 的 0.125；INT8 reference 存储比例约 0.254。

Visual Studio C++ 工作负载已安装；`silu_mul` 使用 MSVC 19.42/CUDA 12.4 完成真实 CPU/CUDA 编译、链接、forward、gradcheck、opcheck、空 Tensor 和 benchmark。G2 自动化工程门和学习者验收均已关闭。Windows 无可工作 Triton，所以只证明 Dynamo 单图捕获，不声明 Inductor 性能。

### Stage 3：已完成

- 冻结 TinyStories 19.4MB 教学子集：21,990 文档，CDLA-Sharing-1.0，来源/哈希/切分可追溯；
- 新增 `src/forgellm/training/`：packing、cursor、cache、AdamW/Muon、scheduler、mixed precision、metrics、checkpoint 和 trainer；
- CPU/FP32 连续与中断恢复的 loss、模型、optimizer、scheduler 和 next batch 全部 exact；
- 100-step 5.36M/BF16 qualification 通过；
- AdamW/Muon 与 single/MTP 独立三种子短实验完成；
- 正式 run 第 125 步中断，由新进程恢复到第 247 步和 1,003,808 target tokens；
- 最终 validation loss 1.6245、PPL 5.0759、BPB 2.0543，日志 step 1–247 连续；
- 唯一学习入口为 `docs/lessons/stage03_learning_order.md`。

2026-07-28 用户明确确认 Stage 3 学习者验收完成，G3 自动化和学习者门均已关闭。

### Stage 4：自动化与学习者验收完成

- 用户确认 Stage 3 验收并批准 Stage 4 方案后启动；
- 冻结 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；
- 冻结 96 条原创 correctness（64/16/16）和 SmolTalk `smol-constraints@5feaf2f...`（2048/256/256）；
- 实现严格对话 Schema、分段 ChatML、assistant-only shifted CE、右 padding 与真实 token 分母累积；
- 实现手写 LoRA 的 no-op、冻结、注入、Adapter 保存加载和 merge/unmerge；
- PEFT BF16 LoRA 与 bitsandbytes NF4 QLoRA 在本机完成真实 forward/backward；
- tiny qualification：loss 5.4090→0.00459、accuracy 1.0；
- 同协议 20-step：BF16/QLoRA 均 14,864 tokens；QLoRA validation loss 1.6801→1.2970，峰值只低约 9.5 MiB、吞吐慢约 25%；
- 合规正式 BF16 LoRA v2：3% token warmup+cosine，129 steps、100,293 assistant tokens、375.0 秒、peak allocated 1.81 GiB；
- 正式结果：assistant loss 1.5660→1.1500，correctness TF loss 3.1000→1.8033，但 strict generation 0/16，retention loss 1.7770→1.8098；
- 原 word trigram 漏检无空格重复；补充 character 8-gram audit 为 0.2080→0.4446，确认行为退化；
- 恒定学习率的 bounded v1 被审计为计划偏差并保留；v2 才是主证据；
- 四份完整讲义、16 站唯一入口、数据/模型卡和实验记录均已生成。

Stage 4 自动化 G4-A～G4-E 完成；用户已明确确认 G4-L 学习者验收。

### Stage 5：自动化与学习者验收完成

- 偏好数据固定为 384/64/64，prompt split 无泄漏，320/320 个 verifier 攻击被拒绝；
- 实现 Bradley–Terry RM、response-only DPO、exact/REINFORCE、PPO、GRPO；
- 实现 DrGRPO、DAPO、GSPO、VESPO 与 Kimi K3 MOPD 的 frozen-tensor reference；
- method lab v2 中 tiny RM、baseline variance、toy PPO/GRPO 全部门通过；
- DPO v2：50 steps、4,214 response tokens；pair accuracy 0→1，但 strict 0/16，SFT validation loss 与字符重复退化；
- GRPO v2：4 prompts × 4 rollouts、508 response tokens、一步更新；首步 KL=0、clip fraction=0，Base 无梯度；
- v1 的 Dropout mode mismatch 和 baseline 错误假设均原样保留；v2 增加 Dropout 关闭与首步 on-policy 自动门；
- 五份完整讲义、18 站唯一入口、数据/模型卡和实验记录已生成。

G5-A～G5-L 已关闭，用户已确认完成学习者验收。

### Stage 6：自动化完成，学习者验收待完成

- 64 个冻结案例：16 correctness、16 preference、32 robustness；
- 2,880 条结构化训练记录污染审计：0 exact、16 个同模板 near-match；
- v4 质量：Q0/Q1/Q2 共 192 条 raw generation，strict 均 0/32、截断率均 1.0；
- Q2 preference accuracy=1.0、mean margin=93.9996，但 char-8 repetition=0.4426；
- M3 只在独立套件报告 loss/PPL/BPB=1.6245/5.0759/2.0543；
- Q3 只验收 4×4 rollout 与一步 on-policy 管线；
- 30 对含 Prompt 的 Q1/Q2 盲评包已生成，人工评分待完成；
- Q0/Q1/Q2 的 18 格系统矩阵完成；Q1 32→64 token 耗时约翻倍、strict 仍 0；
- `final_v6` 自动化门通过；Q0/Q1/Q2 模型行为均未接受；
- 五份完整讲义和 18 站唯一入口已生成。

G6-A～G6-E 已关闭；G6-L 等待用户完成。

## 8. 当前模型优先路线

```text
Stage 1：数据最低闭环与 Tokenizer（2 周）
→ Stage 2：Transformer、现代架构与模型系统（7 周）
→ Stage 3：预训练闭环（3 周）
→ Stage 4：监督式后训练基础——SFT/LoRA/QLoRA（2 周）
→ Stage 5：偏好优化与在线 RL——RM/DPO/PG/PPO/GRPO（3 周）
→ Stage 6：综合评测与最终验收（2 周）
→ Buffer：补弱与作品集（2 周）
```

核心模块使用完整讲义、实现、正确性测试和阶段实验；配置、CLI、Git、CI、日志与普通数据处理默认只使用执行卡并按需补学。

## 9. 当前证据边界

已经证明：

- 工程与数据夹具代码存在；
- 当前 Windows 环境质量门通过；
- Day 1–3 学习由用户确认完成；
- Stage 0 路线切换完成。
- Stage 1 G1-A Tokenizer 算法与工程候选完成。
- Stage 1 G1-B 方法代码、自动测试、统一实验与完整讲义完成。
- G2-Core 与 G2-Arch 的教学 reference、测试、实验与讲义完成。
- G2-Systems 的 SDPA/online-softmax、图捕获、INT8、CPU/Gloo DDP 与原生 C++/CUDA 算子通过。
- Stage 2 学习者验收由用户确认完成。
- Stage 3 数据、训练、全状态恢复、现代方法短实验、有界正式运行和学习者验收通过。
- Stage 4 数据/模板/SFT/LoRA/QLoRA/固定评测和有界正式运行通过；四份讲义与 16 站入口完成。
- Stage 4 学习者验收由用户确认完成。
- Stage 5 偏好数据/RM/DPO/PG/PPO/GRPO 与现代方法 reference 已实现；DPO/GRPO v2 Artifact 可追溯。
- Stage 5 学习者验收由用户确认完成。
- Stage 6 评测 Schema、污染、质量/统计、盲评、系统矩阵、证据报告与讲义已实现；自动门通过。

尚未证明：

- Picky/SuperBPE 在真实语言模型上的下游收益；
- 前沿 reference 对官方大模型效果的复现；
- 更大数据、长训练或生产模型质量；
- DPO 对严格生成行为的改善（Stage 6 为 0/32）或 GRPO 的能力收益；
- Stage 6 学习者验收；Q0/Q1/Q2 的模型行为已明确未通过，不是待验证的成功；
- Linux CI 当前结果。

## 10. 接手人员的第一步

不要重新搭建项目，也不要删除现有文件。首先执行：

```powershell
Set-Location D:\Users\27475\Desktop\Resume_Project\ForgeLLM
git status --short
git branch --show-current
git log --oneline -3
.\.venv\Scripts\python.exe -m forgellm doctor
.\.venv\Scripts\python.exe scripts\dev.py check
```

然后依次阅读：

1. `docs/model_first_21_week_learning_plan.md`；
2. `docs/month_01_progress.md`；
3. `docs/lessons/stage06_learning_order.md`；
4. 按导航分段阅读五份 Stage 6 讲义，不并行通读源码；
5. `docs/experiments/2026-07-28_stage06_implementation.md`；
6. `docs/cards/evaluation_card.md`。

当前下一步是让学习者从 Stage 6 第 1 站开始，完成 30 对盲评、统计手算、代码追踪与最终答辩。不要重复 Stage 3 1M-token、Stage 4 100k-token、Stage 5 DPO/GRPO 或 Stage 6 正式 run；如复现实验必须使用 `artifacts/stage06_student/` 下的新输出路径。
