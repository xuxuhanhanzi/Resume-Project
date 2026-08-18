# ForgeLLM

端到端 LLM 训练、评测与推理系统项目。

当前已完成工程与数据夹具基线、Stage 0–5，以及 Stage 6 的自动化实现、正式质量/系统评测和最终报告。当前唯一学习入口是 Stage 6 的 18 站导航。Q0/Q1/Q2 在冻结 64 例上均为 strict 0/32，说明训练链完整不等于模型行为达标；G6-L 等待学习者完成，服务与平台工程整体后移。

## 项目边界

- 自研并测试最小 BPE、Decoder-only Transformer、Causal Mask、KV Cache、SFT Label Mask 与 DPO Loss；
- 复用 FSDP2、FlashAttention、vLLM、MLflow 与容器基础设施；
- 所有结论都需关联配置、数据版本、Git Commit、运行记录与 Artifact；
- Kubernetes、TP/PP/CP 与第二推理后端均为条件升级，不阻塞当前模型学习主线；GRPO 的最小真实链路已在 Stage 5 完成。

## 目录导航

- `data_pipeline/`：数据治理；
- `src/forgellm/tokenization/`、`src/forgellm/model/`、`src/forgellm/training/`：核心训练链路；
- `src/forgellm/post_training/`、`evaluation/`、`inference/`：后训练、评测与推理；
- `serving/`、`deployment/`、`observability/`：服务与运维；
- `docs/`：资源卡、实验记录、卡片与 ADR；
- `artifacts/`：本地运行产物，不提交版本库。

详细计划、资源卡、阶段计划和实验记录均已放入 `docs/`；任何新增或复现实验仍需先完成对应资源、数据和基线门禁，并使用新的输出目录。

## 当前执行入口

- **当前唯一主计划：`docs/model_first_21_week_learning_plan.md`**
- 2026-08-09 全阶段计划清晰度审查：`docs/implementation_plan_clarity_audit_2026-08-09.md`
- Stage 6 唯一学习入口：`docs/lessons/stage06_learning_order.md`
- Stage 6 综合评测实施计划：`docs/stages/learning_stage_06_evaluation_and_acceptance.md`
- Stage 6 完整讲义：`docs/lessons/stage06_*.md`
- Stage 6 正式质量证据：`artifacts/stage06/quality_v5/`
- Stage 6 实验记录：`docs/experiments/2026-07-28_stage06_implementation.md`
- Stage 6 最终 Evaluation Card：`docs/cards/evaluation_card.md`
- 第一个月进度台账：`docs/month_01_progress.md`
- 前沿模型与技术滚动注册表：`docs/frontier_model_technology_registry.md`
- 项目与知识学习路径：`docs/learning_path.md`
- 最低工程支撑执行卡：`docs/reference/minimum_engineering_support_card.md`
- 当前环境盘点：`docs/environment_audit_2026-07-11.md`

Day 1–10 工程实现和 Stage 0–5 的自动化与学习者验收均已完成。Stage 6 的实现、测试、冻结数据、正式质量运行和完整讲义已落地；G6-L 仍等待用户按 `docs/lessons/stage06_learning_order.md` 完成。已有正式 run 无需重跑。正式付费运行和大数据下载仍需通过对应资源门禁。

## 最小开发入口

项目要求 Python 3.11 或 3.12。请在独立环境中执行：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-model.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -e . -r requirements-dev.lock
.venv/Scripts/python.exe -m pip install -r requirements-post-training.lock --index-url https://pypi.org/simple
.venv/Scripts/forgellm.exe --help
.venv/Scripts/forgellm.exe doctor
.venv/Scripts/python.exe -m pytest
```

以上是 Windows PowerShell 写法；Linux/macOS 将 `Scripts` 替换为 `bin`。初始化一个带配置、环境与 Git 证据的 CPU Run：

```bash
.venv/Scripts/forgellm.exe init-run --config configs/runtime/smoke.toml --artifacts-dir artifacts --repo-root .
```

运行产物默认写入 Git 忽略的 `artifacts/<stage>/<run_id>/`。

## 质量检查

权威入口在所有平台保持一致：

```bash
python scripts/dev.py check
```

也可以分别执行 `format-check`、`lint`、`typecheck`、`unit`、`integration`、`smoke` 或 `test`。GitHub Actions 配置会在 Ubuntu 的 Python 3.11 和 3.12 上执行同一入口；外部 CI 结果只有在仓库配置远端并推送后才能形成正式证据。

## 数据流水线 Smoke

```powershell
.venv/Scripts/forgellm.exe data-pipeline --config configs/data/smoke.toml --input tests/fixtures/data/sample_documents.jsonl --output-dir artifacts/data_pipeline/local-smoke --source-name forgellm-test-fixture --source-license project-test-fixture
```

该命令只使用自建测试夹具，用于验证接口和确定性行为，不能代替正式数据审计。历史工程证据保留在 `docs/experiments/`，当前学习顺序只由模型优先路线和对应 Stage 学习入口定义。

## Tokenizer 候选

### Stage 1 学习入口

第一次学习时，请只从 `docs/lessons/stage01_learning_order.md` 开始。该导航给出逐站章节、文件、函数、练习和通过标准；不要把下方规格、源码和实验记录当作并列的入门材料。

纯 Python 主路径不需要额外运行时依赖；成熟实现对照是可选项：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-tokenizer.lock
.venv/Scripts/python.exe -m forgellm tokenizer-train --config configs/tokenizer/bpe_v1.toml --input tests/fixtures/tokenizer/train.jsonl --output-dir artifacts/my_tokenizer/model --source-name forgellm-test-fixture --source-license project-test-fixture
.venv/Scripts/python.exe -m forgellm tokenizer-evaluate --model artifacts/my_tokenizer/model/tokenizer.json --input tests/fixtures/tokenizer/test.jsonl --output artifacts/my_tokenizer/evaluation.json
```

完成 G1-A 后，统一运行 G1-B 方法实验：

```powershell
.venv/Scripts/python.exe scripts/tokenizer_method_lab.py --train tests/fixtures/tokenizer/train.jsonl --evaluation tests/fixtures/tokenizer/method_lab_evaluation.jsonl --output artifacts/stage01_tokenizer_method_lab/student_report.json
```

学习顺序由 `docs/lessons/stage01_learning_order.md` 唯一定义。G1-A 讲义之后进入 `docs/lessons/stage01_tokenizer_method_lab.md`。当前结果证明项目夹具上的算法机制与可逆性，不代表正式语料质量或模型下游收益。

## Stage 2 模型实验室

Stage 2 请只从 `docs/lessons/stage02_learning_order.md` 开始。该导航按“词汇与 Tensor→原子模块→Attention→Decoder→生成/缓存→现代架构→系统→自定义算子”的顺序给出讲义章节、源码函数、练习和通过标准。

固定实验：

```powershell
.\.venv\Scripts\python.exe scripts\stage2_model_lab.py
.\.venv\Scripts\python.exe scripts\stage2_ddp_smoke.py
.\.venv\Scripts\python.exe scripts\build_silu_mul.py
```

三项均已在本机通过。自定义算子使用 MSVC 19.42 与 CUDA 12.4 完成 CPU/CUDA 编译、forward、gradcheck、opcheck、空 Tensor 和固定 benchmark。实验结果证明微型 reference 的数值和结构性质，不代表 DeepSeek、Kimi 或其他官方大模型的训练效果复现。

## Stage 3 预训练实验室

第一次学习请只从 `docs/lessons/stage03_learning_order.md` 开始。该导航按“数据与目标→优化与精度→checkpoint 与恢复→Muon/MTP→有界运行”的顺序给出 16 个站点。

核心复现入口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_training_*.py tests\integration\test_stage3_training.py

.\.venv\Scripts\python.exe scripts\stage3_resume_equivalence.py `
  --config configs\training\stage3_smoke.toml `
  --tokenizer artifacts\stage01_tokenizer_candidate\model\tokenizer.json `
  --train data\processed\stage3_tinystories\train.jsonl `
  --validation data\processed\stage3_tinystories\validation.jsonl `
  --output-dir artifacts\stage03_student\resume
```

正式证据使用 5.36M 模型和 1,003,808 target tokens，在第 125 步主动中断并由新进程恢复；最终 validation loss 1.6245、PPL 5.0759、BPB 2.0543。该结果只证明小规模预训练闭环，不代表可用语言模型质量。

## Stage 4 监督式后训练实验室

第一次学习只从 `docs/lessons/stage04_learning_order.md` 开始。四份讲义依次覆盖数据/模板/loss、SFT 训练与评测、LoRA/PEFT、QLoRA 与正式结果。

自动测试入口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_post_training_*.py tests\integration\test_stage4_tiny_sft.py
```

正式 BF16 LoRA 已处理 100,293 assistant tokens：held-out loss 改善，但 strict generation 仍为 0/16，retention 和字符级重复退化。结果见 `docs/experiments/2026-07-28_stage04_implementation.md`；不要为学习重复正式 run。

## Stage 5 偏好优化与在线 RL 实验室

第一次学习只从 `docs/lessons/stage05_learning_order.md` 开始。五份讲义按“偏好数据/RM→DPO→PG/PPO→GRPO/现代变体→Kimi K3/蒸馏/失败分析”组织，所有思考题后紧跟答案。

不烧卡的核心入口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_alignment_*.py -q
.\.venv\Scripts\python.exe scripts\stage5_method_lab.py `
  --train data\processed\stage5_preference_constraints_v1\train.jsonl `
  --output artifacts\stage05_student\method_lab.json
```

DPO v2 将 held-out pair accuracy 从 0 提高到 1，但 strict generation 保持 0/16，且验证 loss 与字符重复退化；GRPO v2 只证明 4×4 rollout→reward→log-prob→一步更新链路和首步 on-policy 契约。结果见 `docs/experiments/2026-07-28_stage05_implementation.md`，不得概括为“模型已对齐”。

## Stage 6 综合评测与最终验收

第一次学习只从 `docs/lessons/stage06_learning_order.md` 开始。五份完整讲义依次覆盖评测身份/污染、行为与统计、Judge/盲评/鲁棒性、系统与多维门禁、前沿协议与失败分析；所有思考题后紧跟可折叠答案。

不烧卡的代码门：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_evaluation_schema_config.py tests\unit\test_evaluation_metrics.py tests\unit\test_evaluation_judge_systems_report.py
```

正式 v4 使用 64 个冻结案例、192 条原始生成、Q0/Q1/Q2 同一 Qwen Tokenizer 与 greedy 32-token 协议。Q1/Q2 的 expected-response NLL 与 pair ranking 改善，但三者 strict 均为 0/32、截断率均为 1.0；Q2 字符重复升至 0.4426。Q3 仅审计一步 on-policy 管线，M3 仅在自身 Tokenizer/套件报告 PPL/BPB。不得把这些结果概括为“后训练提升了模型能力”。
