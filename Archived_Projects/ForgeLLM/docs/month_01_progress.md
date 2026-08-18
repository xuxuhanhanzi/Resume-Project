# 第一个月进度台账

更新规则：只在产物存在且验证命令通过后勾选；“写了代码”不等于完成。每次更新在变更日志中写日期、证据和下一步。

## 当前状态

| 项目 | 状态 |
|---|---|
| 当前阶段 | Stage 6 自动化、正式质量/系统评测和最终报告完成；学习者按 18 站验收 |
| 当前门禁 | G4/G5 全部关闭；G6-A～G6-E 完成，G6-L 待学习者确认 |
| 最近验证 | 2026-07-28：173 个 Python 文件格式/静态检查通过，mypy 检查 113 个源文件，244 passed、3 个 C++ 工具链测试按配置跳过；Stage 6 专项 21 passed |
| 当前阻塞 | 无工程阻塞；只剩 Stage 6 人工盲评与学习者答辩 |
| 下一步 | 从 `docs/lessons/stage06_learning_order.md` 第 1 站开始，不重复正式运行 |

状态只使用：`未开始`、`进行中`、`受阻`、`已完成`。

## 模型优先 21 周路线

- [x] Stage 0：路线切换、支撑执行卡、质量门和详细讲义
- [x] Stage 1：数据最低闭环与 Tokenizer（用户已确认学习与测试完成）
- [x] Stage 2：Transformer、现代架构与模型系统（用户已确认学习完成）
- [x] Stage 3：预训练闭环（G3；用户已确认学习者验收）
- [x] Stage 4：监督式后训练基础——SFT/LoRA/QLoRA（用户已确认 G4-L）
- [x] Stage 5：偏好优化与在线 RL——RM/DPO/PG/PPO/GRPO（用户已确认 G5-L）
- [ ] Stage 6：综合评测与最终验收（G6-A～E 完成；G6-L 待完成）
- [ ] Buffer：补弱与作品集

当前唯一主计划：`docs/model_first_21_week_learning_plan.md`。以下 W1–W4 保留为历史实现台账，不再定义学习顺序。

## W1：工程基线

- [x] Day 1：资源卡、环境盘点、阶段计划和首个 ADR
- [x] Day 2：可安装包、最小 CLI 和开发依赖
- [x] Day 3：配置校验、日志、Run ID 和运行元数据
- [x] Day 4：测试分层和统一质量命令
- [x] Day 5：CI、README Quickstart 和新环境复现
- [ ] G0：`v0.1-engineering` 验收通过

## W2：数据流水线最小闭环

- [x] Day 6：数据 Schema 与 Manifest
- [x] Day 7：单格式 Reader 与 Unicode 规范化
- [x] Day 8：质量 Filter、拒绝原因与精确去重
- [x] Day 9：确定性切分与 Writer
- [x] Day 10：CPU Smoke、统计报告和实验记录

## W3：正式数据审计

- [ ] Day 11：候选数据源审计与 ADR
- [ ] Day 12：获取、哈希与原始质量审计
- [ ] Day 13：固定样本上的过滤规则校准
- [ ] Day 14：冻结数据 v1、切分与统计
- [ ] Day 15：Data Card 与从原始输入重建审计

## W4：BPE 与 Tokenizer

- [x] Day 16：BPE 规格、手算样例和测试
- [x] Day 17：手写最小 BPE 训练、保存与加载
- [x] Day 18：编码、解码、属性和边界测试
- [x] Day 19：工程版 Tokenizer 与固定评测
- [ ] Day 20：G1 审计、README 与发布候选
- [ ] G1：`v0.2-data-tokenizer` 验收通过

## 门禁证据

### G0 工程可复现

| 证据 | 路径/命令 | 状态 |
|---|---|---|
| 资源卡 | `docs/project_resource_card.md` | 已完成；约 71.6 GiB 工作盘余量已确认 |
| 路线切换与阶段证据 | `docs/stages/learning_stage_00_model_first_transition.md`、`docs/experiments/2026-07-24_stage00_model_first_transition.md` | 已完成 |
| 环境/依赖锁定 | `docs/environment_audit_2026-07-11.md`、`requirements-dev.lock`、`.venv` | 已完成（Windows 基线） |
| lint/typecheck/test/smoke | 设置 `FORGELLM_TEST_EXTENSION=1` 后运行 `python scripts/dev.py check`；175 passed | 已完成，0 skipped；2 个非阻塞工具链 warning |
| CI 运行 | `.github/workflows/ci.yml` | 配置完成；无远端，外部运行待补 |
| 新环境复现记录 | `docs/experiments/2026-07-11_stage01_day05_reproduction.md` | 已完成（Windows） |

### G1 数据与 Tokenizer 可审计

| 证据 | 路径/命令 | 状态 |
|---|---|---|
| 数据来源与许可证 | 项目原创测试夹具及来源标记；G1-B 不下载外部语料 | 已完成（教学边界） |
| 数据 manifest 与哈希 | correctness train/test、训练 Manifest 与方法实验输入 SHA-256 | 已完成（教学边界） |
| 经典 BPE 测试 | `tests/unit/test_bpe_tokenizer.py` 等 | G1-A 已完成 |
| 现代方法实现 | pre-tokenization、Unigram、Picky/SuperBPE、dropout、特殊 Token、entropy patch | G1-B 自动化完成 |
| Tokenizer 评测报告 | raw、两种 classic BPE、Picky、Super、Unigram 与 subset 指标 | 已完成 |
| 完整实验记录 | 2026-07-25 G1-A 与 2026-07-27 G1-B 两份记录 | 已完成 |
| 学习者验收 | 用户已确认完成 Stage 1 学习与测试 | 已完成 |

### G2 Transformer、现代架构与模型系统

| 证据 | 路径/命令 | 状态 |
|---|---|---|
| Stable Decoder | `src/forgellm/model/` 与核心模型测试 | G2-Core 自动化完成 |
| 现代架构 reference | `frontier_attention.py`、`frontier_layers.py`、`optim.py` | G2-Arch 自动化完成 |
| 固定模型实验 | `scripts/stage2_model_lab.py` 与 `artifacts/stage02/stage2_model_lab_report.json` | 已完成 |
| SDPA/online-softmax/compile/INT8 | `systems.py` 与对应单元测试 | 已完成；Inductor 性能未验证 |
| 两进程 DDP | `scripts/stage2_ddp_smoke.py` | CPU/Gloo 已通过 |
| 自定义算子 | `scripts/build_silu_mul.py` 与 `artifacts/stage02/silu_mul_extension_report.json` | CPU/CUDA 编译、梯度、opcheck、边界与 benchmark 通过 |
| 完整实验记录 | `docs/experiments/2026-07-27_stage02_implementation.md` | 已完成 |
| 学习者验收 | 用户已确认完成 Stage 2 学习与测试 | 已完成 |

### G3 预训练闭环

| 证据 | 路径/命令 | 状态 |
|---|---|---|
| 正式教学语料 | `docs/cards/data_card.md`、本地 manifest | 21,990 文档、CDLA-Sharing-1.0、哈希冻结 |
| 数据/训练实现 | `src/forgellm/training/` | packing、cache、optimizer、precision、validation 完成 |
| Checkpoint/Resume | `stage3_resume_equivalence.py`、`checkpoint.py` | CPU 全状态 exact；CUDA 真中断恢复 |
| Qualification | `artifacts/stage03/qualification/report.json` | 100 steps、loss 下降、无非有限值 |
| 现代方法实验 | `artifacts/stage03/method_lab/report.json` | AdamW/Muon/MTP，三种子独立实验 |
| 有界正式运行 | `artifacts/stage03/bounded_1m/report.json` | 1,003,808 tokens、step 1–247 连续 |
| 完整实验记录 | `docs/experiments/2026-07-27_stage03_implementation.md` | 已完成 |
| 学习者验收 | `docs/lessons/stage03_learning_order.md` 16 站 | 用户于 2026-07-28 确认完成 |

### G4 监督式后训练基础

| 计划证据 | 路径/决策 | 状态 |
|---|---|---|
| 两周详细计划 | `docs/stages/learning_stage_04_supervised_posttraining.md` | 已执行；自动化门完成 |
| 路线边界 | Stage 4=SFT/LoRA/QLoRA；Stage 5=偏好/RL | 已冻结；后训练总周期仍为 5 周 |
| 双轨模型 | 5.36M 自研 Decoder + `Qwen/Qwen3-0.6B-Base` | revision `da87bfb...c0cd` 已冻结并实跑 |
| 数据候选 | 96 条原创 correctness + SmolTalk `smol-constraints` 确定性小子集 | 64/16/16 与 2048/256/256 已冻结 |
| 资源边界 | 本地优先、外部 0 USD 默认、正式短跑 ≤100k assistant tokens/500 steps/45 min | 已冻结 |
| 自动化门 | G4-A 数据模板、G4-B SFT、G4-C LoRA、G4-D QLoRA、G4-E 评测短跑 | 已完成 |
| 学习者门 | `docs/lessons/stage04_learning_order.md` 16 站 | 用户已确认完成 |

### G5 偏好优化与在线 RL

| 证据 | 路径/结果 | 状态 |
|---|---|---|
| 偏好数据与 verifier | `data/processed/stage5_preference_constraints_v1`；384/64/64，攻击 320/320 拒绝 | 已完成 |
| RM/PG/PPO/GRPO 方法实验 | `artifacts/stage05/method_lab/report_v2.json` | 全部自动门通过 |
| DPO 正式运行 | `artifacts/stage05/qwen_dpo_bounded_v2/report.json` | pair acc 0→1；strict 0/16，退化已报告 |
| GRPO 真实一步链 | `artifacts/stage05/qwen_grpo_one_step_v2/report.json` | 4×4 rollouts；initial KL/clip=0 |
| 前沿 reference | DrGRPO、DAPO、GSPO、VESPO、Kimi K3 MOPD | frozen-tensor 机制门完成 |
| 完整实验记录 | `docs/experiments/2026-07-28_stage05_implementation.md` | 已完成 |
| 学习者门 | `docs/lessons/stage05_learning_order.md` 18 站 | 用户已确认完成 |

### G6 综合评测与最终验收

| 证据 | 路径/结果 | 状态 |
|---|---|---|
| 冻结案例与污染 | `data/processed/stage6_evaluation_v2`；64 例，0 exact/16 near | 已完成 |
| 质量评测 | `artifacts/stage06/quality_v5/report.json`；192 raw generations | 已完成 |
| Q0/Q1/Q2 行为 | strict 均 0/32；Q2 pair=1、char-8 repetition=0.4426 | 未达模型行为门，已诚实报告 |
| Q3/M3 边界 | Q3 pipeline-only；M3 独立 PPL/BPB | 已完成 |
| 盲评包 | `quality_v5/blind_review`；30 对含 Prompt | 已生成；待学习者评分 |
| 系统矩阵 | `artifacts/stage06/systems_v5/report.json`；18/18 格 + E10 | 已完成 |
| 最终报告 | `artifacts/stage06/final_v6/report.json`；无综合总分 | 自动门通过 |
| 完整讲义/入口 | `docs/lessons/stage06_learning_order.md` + 五份讲义 | 已完成 |
| 学习者门 | 18 站、30 对盲评、统计/边界答辩 | 待用户完成 |

## 风险与决策队列

| ID | 风险/决策 | 影响 | 处理方式 | 状态 |
|---|---|---|---|---|
| R-001 | 每周时间未知 | 计划节奏无法校准 | 已登记 60h+/周，按 60h 基线执行 | 关闭 |
| R-002 | GPU/预算未知 | 禁止规划长训练 | 已登记本地 8GB 与外部资源 ≤20 USD/月；按门禁启用云端 | 关闭 |
| R-003 | 正式数据源未选 | 曾阻塞预训练数据结论 | Stage 3 已冻结 TinyStories 19.4MB 教学子集、许可证、哈希和切分 | 关闭 |
| R-004 | 目标领域未定 | 无法声称正式 Tokenizer 质量 | 当前只声称算法机制；不做领域质量结论 | 降级 |
| R-005 | Docker/WSL Linux 环境缺失 | 后续服务与 Linux GPU 工具不可本地复现 | 首月采用 Windows + Linux CI；阶段前复审 | 开放 |
| R-006 | 工作盘可用容量未知 | 曾有数据下载风险 | 2026-07-27 实测约 71.6 GiB 可用；Stage 3 仅使用小型语料 | 关闭 |
| R-007 | 模型/论文引用快速过时 | 计划继续把旧版本称为当前前沿 | Stage 启动前执行 F0；官方源核验后冻结版本；阶段中途新版本进入下轮 | 开放 |
| R-008 | 支撑工程再次扩张为主线 | Tokenizer、模型和训练被持续推迟 | core/support 教学分级；支撑任务默认 30–120 分钟时间盒 | 开放 |
| R-009 | Windows 缺少 MSVC C++ 工具链 | `silu_mul` 曾不能编译 | 已安装 MSVC 19.42/Windows SDK，CPU/CUDA extension 验收通过 | 关闭 |
| R-010 | Windows 官方环境无可工作 Triton | 不能验证 Inductor kernel 与性能 | 当前只声明 Dynamo 单图；未来在受控 Linux 环境补测 | 开放 |
| R-011 | Stage 4 模型/数据只有候选、未冻结 revision | 无法审计 SFT 结果 | Qwen/SmolTalk revision、许可证、split 与 hash 已冻结 | 关闭 |
| R-012 | 本机 QLoRA 实际兼容性尚未验证 | 可能无法完成真实 4-bit backward | bitsandbytes 0.49.2 Windows NF4 load/backward/20-step 实验通过 | 关闭 |
| R-013 | 空格词 trigram 漏检无空格重复 | 把生成退化误报为改善 | 保留原指标并新增 character 8-gram；Stage 5 DPO 固定使用并捕获退化 | 关闭 |
| R-014 | DPO pair 指标满分但 strict generation 仍失败 | 把代理目标误报为模型能力 | Stage 6 证实 pair=1、strict=0/32、重复恶化；最终卡拒绝能力结论 | 关闭 |

## 变更日志

| 日期 | 变更 | 证据/结果 | 下一步 |
|---|---|---|---|
| 2026-07-11 | 建立第一个月执行计划与进度台账 | 计划文件已加入仓库 | 填写资源卡，开始 Day 1 |
| 2026-07-11 | 完成 Day 1 资源与环境盘点 | 资源卡、环境审计、Stage 01 计划、ADR-0001 | 开始 Day 2 独立环境与最小 CLI |
| 2026-07-11 | 完成 Day 2 包与 CLI 基线 | ruff、mypy、pytest 4/4、安装后 CLI 通过；详见实验记录 | 开始 Day 3 配置、日志与 Run ID |
| 2026-07-11 | 完成 Day 3 运行证据基线 | 首个 Run ID、配置哈希、Git/环境快照、JSONL 日志；17/17 测试 | 开始 Day 4 统一命令与测试分层 |
| 2026-07-11 | 完成 Day 4–5 工程闭环 | 统一质量命令、19/19 测试、新环境复现、CI 定义 | G0 外部 CI 待远端；进入数据闭环 |
| 2026-07-11 | 完成 Day 6–10 数据夹具闭环 | 33/33 测试；9 输入、4 保留、5 拒绝、三路 split；Manifest/报告 | 用户学习 Day 1–10，之后进入 Day 11 |
| 2026-07-13 | 建立第一、二周两周学习计划 | 14 天知识概要、代码追踪、实验、自测与验收 | 完成学习后进入 Day 11 |
| 2026-07-22 | 制定项目月 01 后半月至项目月 03 的详细实现计划 | 50 个有效开发日；覆盖 G0–G4、逐日任务、门禁、测试、实验和风险 | 完成学习复盘与启动检查后进入 Day 11 |
| 2026-07-22 | 建立前沿模型与技术滚动注册表 | DeepSeek 主参考 V3→V4；同步审计 Qwen3.6、Gemma 4、OLMo 3/Hybrid、Kimi K2.5/Linear；加入 F0 新鲜度门禁 | Day 16/21/41 和项目月 04 启动前增量复审 |
| 2026-07-24 | 用户确认完成 Day 1–3 学习与测试 | Day 3 讲义、15 项对应测试；用户完成状态确认 | 重审学习时间分配 |
| 2026-07-24 | 完成 Stage 0 模型优先路线切换 | 新 21 周计划、支撑执行卡、详细讲义；Ruff/mypy/pytest 33/33 通过 | 建立 Stage 1 Tokenizer 计划与核心讲义 |
| 2026-07-25 | 完成 Stage 1 G1-A Tokenizer 候选 | 纯 Python byte BPE、固定特殊 ID、CLI、Manifest、raw/HF 对照、完整讲义；Ruff/mypy/pytest 68/68 | 冻结正式语料与 Data Card，关闭 G1-B；准备 Stage 2 |
| 2026-07-25 | 重构 Stage 1 新手学习入口 | 新增逐站文件/函数顺序；讲义增加两层词汇表；教学 Prompt v0.4 禁止用相关文件列表替代学习路线 | 从学习导航第 1 站开始 |
| 2026-07-25 | 删除过时学习与实施计划 | 删除 5 个已被模型优先路线取代的文档并清理失效引用；保留实验、ADR、讲义和通用模板 | 当前文件只按学习入口与证据职责使用 |
| 2026-07-26 | 建立 G1-B 正式语料与 Tokenizer 完成计划 | 冻结单变量实验矩阵、7 个执行 Phase、学习顺序、25–30 小时预算、产物与 Stop/Go 门禁；尚未下载数据 | 确认默认语料方案和 20 GB 可用空间，进入 Phase 1 来源/许可证审计 |
| 2026-07-27 | 按学习目标重构并完成 G1-B 教学实现 | 取消大语料和词表搜索；新增 pre-tokenization、BPE-dropout、手写 Unigram、Picky/SuperBPE、特殊 Token、offset、entropy patch；100/100 测试通过 | 学习者按第 9～14 站完成讲义、代码追踪与口述验收 |
| 2026-07-27 | 启动并实现 Stage 2 三条主线 | Stable Decoder、前沿架构 reference、系统实验、三份完整讲义；全仓 153 passed/3 skipped；5.36M GPU Smoke 和双进程 DDP 通过 | 安装 MSVC，完成原生扩展；学习者从 Stage 2 第 1 站开始验收 |
| 2026-07-27 | 解除 Stage 2 原生算子阻塞 | MSVC 19.42/CUDA 12.4 编译链接通过；CPU/CUDA 误差 0、gradcheck/opcheck/空 Tensor/benchmark 通过；全仓 156/156 | 学习者从第 1 站开始，完成 14 站验收后关闭 G2 |
| 2026-07-27 | 用户确认完成 Stage 2 学习与测试 | G2-Core/G2-Arch/G2-Systems 与学习者验收均关闭 | 启动 Stage 3 预训练闭环 |
| 2026-07-27 | 完成 Stage 3 自动化实现与有界实验 | 正式数据冻结；CPU exact-resume；Muon/MTP 三种子；5.36M BF16 在第 125 步中断恢复并达到 1,003,808 tokens；全仓 175 passed | 学习者从 Stage 3 16 站入口开始验收 |
| 2026-07-28 | 冻结 Stage 4 监督式后训练实现计划 | 后训练拆为 Stage 4 两周 SFT/LoRA/QLoRA 与 Stage 5 三周偏好/RL；双轨模型、数据候选、9 项实验、资源停止条件和 G4 子门已定义 | Stage 3 学习者验收后启动 Stage 4 Day 1 F0/资源审计 |
| 2026-07-28 | 用户确认 Stage 3 学习者验收并完成 Stage 4 自动化实现 | 严格模板/loss、手写 LoRA、PEFT/NF4 QLoRA；100,293-token 正式 run；loss 改善但 strict 0/16、retention/字符重复退化；完整讲义与实验记录 | 学习者按 Stage 4 16 站验收；讨论确认 Stage 5 计划 |
| 2026-07-28 | 用户确认 Stage 4 验收并批准、完成 Stage 5 自动化实现 | RM/DPO/PG/PPO/GRPO、现代方法张量门、DPO v2、GRPO v2、五份讲义与 18 站入口；保留两类失败 | 学习者完成 G5-L；讨论确认 Stage 6 综合验收计划 |
| 2026-07-28 | 用户确认 Stage 5 验收并完成 Stage 6 自动化实现 | 64 例/192 输出、污染与统计、30 对盲评包已生成（人工评分待完成）、18 格系统矩阵、E0–E15、五份讲义；自动门通过，三个模型行为均未接受 | 学习者按 Stage 6 18 站完成 G6-L |
