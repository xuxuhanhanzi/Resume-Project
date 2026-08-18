# 大模型算法、训练与软件开发方向：完整项目学习与求职转型计划

> 适用对象：希望从计算机视觉、信号处理、端侧部署方向，系统转向大模型算法、大模型训练、推理部署、Agent 与 AI 软件开发岗位的学习者。  
> 当前背景：具备信号处理、机器视觉、Transformer、FlashAttention、模型压缩、量化和端侧部署经验；在保证工程主线可复现的前提下，系统吸收 2025–2026 新论文与最新开放模型的可验证模块。  
> 原始版本：2026-07-11  
> 审阅增强版：2026-07-11  
> 前沿版本修订：2026-07-22；以 `ForgeLLM/docs/frontier_model_technology_registry.md` 为滚动版本入口  
> 修订原则：保留原计划的知识覆盖面，将执行方式改为“主线优先、证据驱动、阶段门禁、条件升级”。
> 执行状态修订：2026-08-09；本文仅作为长期能力地图和历史规划，不是 ForgeLLM 当前实施计划。当前唯一执行路线为 `ForgeLLM/docs/model_first_21_week_learning_plan.md`。

---

# 零、审阅结论与执行裁剪

## 0.1 总体判断

原计划在知识覆盖上已经非常全面，覆盖数据、Tokenizer、模型、训练、后训练、评测、推理、服务、部署、监控与 Agent，适合作为长期能力地图；但它还不是一份足够稳健的个人执行计划。主要问题不是缺少技术名词，而是同时承担的目标过多，且部分里程碑只有“做了什么”，没有把资源约束、对照基线、量化门槛、失败处理和求职证据绑定起来。

本增强版将计划拆成三层：

- **P0 必做主线**：必须形成可运行、可复现、可评测的作品；
- **P1 条件升级**：仅在 P0 达标且能产生额外证据时开展；
- **P2 研究型扩展**：用于长期学习，不阻塞求职版本发布。

## 0.2 最重要的五项修正

1. **ForgeLLM 与 RepoPilot 解耦。** RepoPilot 必须先用能力足够的外部或开源指令模型建立系统基线；ForgeLLM 自训练小模型只作为受控对比对象，不能成为 RepoPilot MVP 的前置依赖。
2. **从“技术栈清单”改为“证据链”。** 每个阶段必须包含假设、基线、主变量、固定项、指标、成功门槛、失败处理、命令、日志与报告。
3. **从“大而全”改为“可发布切片”。** 每 2–4 周形成一个可展示 Release；Kubernetes、GRPO、TP/PP、KServe/Ray Serve、TensorRT-LLM 等均改为条件触发项。
4. **RepoPilot 补齐安全与评测。** 容器不是完整沙箱；必须增加威胁模型、默认断网、最小权限、路径与符号链接防逃逸、Secret 隔离、命令策略、资源配额、隐藏测试与基准污染控制。
5. **增加前沿版本治理。** DeepSeek 当前主参考从 V3 更新到 V4，并同步滚动审计 Qwen、Gemma、OLMo、Kimi 等家族；旧版本保留为演化对照，新方法只能通过官方来源核验和单变量资源门禁后进入实现。

## 0.3 推荐的项目定位

```text
ForgeLLM：训练与推理系统能力证明
  P0：从零小模型 + 开源模型后训练 + 评测 + 推理服务
  P1：FSDP2/量化/可观测性
  P2：TP/PP/CP、RL、Kubernetes 生产化

RepoPilot：可执行软件工程 Agent 能力证明
  P0：单仓库、单语言、受限工具、自动测试闭环
  P1：混合检索、多模型路由、任务恢复、并发执行
  P2：多语言、SWE-bench 大规模评测、多租户平台
```

## 0.4 阅读方式

正文和附录 D–L 继续保留为长期能力地图、历史决策和模板参考；不得直接据此启动 ForgeLLM 阶段，也不得用本文的旧阶段编号更新当前进度。ForgeLLM 真正执行时，以 `ForgeLLM/docs/model_first_21_week_learning_plan.md` 的 Stage 0–6、对应 `ForgeLLM/docs/stages/` 子计划和 `ForgeLLM/HANDOFF_SUMMARY.md` 为准；前沿技术版本以 `ForgeLLM/docs/frontier_model_technology_registry.md` 为准。若本文与上述执行文档冲突，以执行文档为准并按最保守状态解释。

---

# 目录

0. [审阅结论与执行裁剪](#零审阅结论与执行裁剪)
1. [项目规划背景](#一项目规划背景)
2. [现有简历与目标岗位的匹配分析](#二现有简历与目标岗位的匹配分析)
3. [总体转型目标](#三总体转型目标)
4. [最终项目组合](#四最终项目组合)
5. [GitHub 高水平项目参考体系](#五github-高水平项目参考体系)
6. [项目设计原则](#六项目设计原则)
7. [主项目 ForgeLLM 总体设计](#七主项目-forgellm-总体设计)
8. [ForgeLLM 阶段 0：工程基础设施](#八forgellm-阶段-0工程基础设施)
9. [ForgeLLM 阶段 1：大模型数据工程](#九forgellm-阶段-1大模型数据工程)
10. [ForgeLLM 阶段 2：Tokenizer 从零构建](#十forgellm-阶段-2tokenizer-从零构建)
11. [ForgeLLM 阶段 3：Decoder-only Transformer](#十一forgellm-阶段-3decoder-only-transformer)
12. [ForgeLLM 阶段 4：预训练系统](#十二forgellm-阶段-4预训练系统)
13. [ForgeLLM 阶段 5：继续预训练与领域适配](#十三forgellm-阶段-5继续预训练与领域适配)
14. [ForgeLLM 阶段 6：SFT、LoRA 与 QLoRA](#十四forgellm-阶段-6sftlora-与-qlora)
15. [ForgeLLM 阶段 7：偏好优化与强化学习](#十五forgellm-阶段-7偏好优化与强化学习)
16. [ForgeLLM 阶段 8：模型评测系统](#十六forgellm-阶段-8模型评测系统)
17. [ForgeLLM 阶段 9：量化与模型导出](#十七forgellm-阶段-9量化与模型导出)
18. [ForgeLLM 阶段 10：推理引擎与性能优化](#十八forgellm-阶段-10推理引擎与性能优化)
19. [ForgeLLM 阶段 11：API 与后端服务](#十九forgellm-阶段-11api-与后端服务)
20. [ForgeLLM 阶段 12：容器化和生产部署](#二十forgellm-阶段-12容器化和生产部署)
21. [ForgeLLM 阶段 13：监控与 LLMOps](#二十一forgellm-阶段-13监控与-llmops)
22. [应用项目 RepoPilot](#二十二应用项目-repopilot)
23. [两个项目的连接关系](#二十三两个项目的连接关系)
24. [查漏补缺后的升级清单](#二十四查漏补缺后的升级清单)
25. [推荐技术栈](#二十五推荐技术栈)
26. [项目里程碑与验收标准](#二十六项目里程碑与验收标准)
27. [建议实施顺序](#二十七建议实施顺序)
28. [最终简历应如何变化](#二十八最终简历应如何变化)
29. [岗位覆盖范围](#二十九岗位覆盖范围)
30. [最终结论](#三十最终结论)
31. [参考 GitHub 项目](#三十一参考-github-项目)
32. [附录 D：范围优先级与停止清单](#附录-d范围优先级与停止清单)
33. [附录 E：资源预算与条件触发](#附录-e资源预算与条件触发)
34. [附录 F：阶段门禁与量化验收](#附录-f阶段门禁与量化验收)
35. [附录 G：RepoPilot 执行版设计](#附录-grepopilot-执行版设计)
36. [附录 H：安全威胁模型与供应链](#附录-h安全威胁模型与供应链)
37. [附录 I：实验与证据管理](#附录-i实验与证据管理)
38. [附录 J：24 周执行路线](#附录-j24-周执行路线)
39. [附录 K：求职交付与简历证据](#附录-k求职交付与简历证据)
40. [附录 L：最终 Definition of Done](#附录-l最终-definition-of-done)

---

# 一、项目规划背景

当前简历中的主要项目集中在：

- UWB 长序列信号建模；
- CNN-BERT 混合架构；
- FlashAttention；
- YOLO 系列模型；
- 轻量化骨干网络；
- 剪枝、蒸馏与 INT8 量化；
- NPU 和端侧部署。

这些经历能够证明已经具备：

- PyTorch 深度学习开发能力；
- Transformer 和注意力机制基础；
- 计算量、显存和速度优化意识；
- 模型压缩和硬件部署经验；
- 计算机视觉与信号处理背景。

但期望寻找的岗位已经转向：

1. 大模型算法工程师；
2. 大模型训练工程师；
3. LLM Research Engineer；
4. 大模型推理优化工程师；
5. Agent 算法工程师；
6. AI 应用开发工程师；
7. 大模型软件开发工程师；
8. Applied AI Engineer；
9. LLM Systems Engineer。

因此，当前问题不是已有项目没有价值，而是项目呈现出的职业标签仍然偏向：

> 计算机视觉算法 + 端侧模型部署工程师。

接下来的目标是完成一次能力迁移：

> 从视觉模型和端侧部署，迁移为具备大模型构建、训练、后训练、评测、推理、部署、Agent 和软件工程能力的 LLM Engineer。

---

# 二、现有简历与目标岗位的匹配分析

## 2.1 已有优势

### 2.1.1 数学、信号处理和机器学习基础较好

通信工程、信号处理与机器学习背景，对以下内容具有天然优势：

- 线性代数；
- 概率统计；
- 信号与系统；
- 时序建模；
- 优化算法；
- 机器学习；
- 深度学习；
- 模型实验分析。

相比只做应用调用的候选人，这种背景更适合进一步学习：

- Transformer 内部原理；
- 大模型训练动力学；
- 优化器；
- 数据分布；
- 分布式训练；
- 模型性能分析。

### 2.1.2 已经接触过 Transformer 效率问题

原有 UWB 项目已经涉及：

- 长序列输入；
- CNN 与 Transformer 结合；
- FlashAttention；
- 显存优化；
- 速度优化；
- 数据增强；
- 鲁棒性。

这些经验可迁移到：

- 大语言模型长序列训练；
- Attention 内存复杂度；
- FlashAttention；
- Activation Checkpointing；
- KV Cache；
- 推理性能分析。

### 2.1.3 具备模型压缩与部署经验

已有项目还涉及：

- 模型剪枝；
- 知识蒸馏；
- INT8 量化；
- NPU 部署；
- 模型体积优化；
- 推理帧率分析。

这些经验对以下岗位有明显帮助：

- LLM Inference Engineer；
- 大模型量化与部署工程师；
- ML Systems Engineer；
- 边缘大模型工程师；
- 多模态模型部署工程师。

## 2.2 当前主要断层

### 2.2.1 有 Transformer 经验，但没有语言模型训练闭环

CNN-BERT 时序模型不等于具备大语言模型训练经验。

招聘者仍无法确认是否掌握：

- Tokenizer；
- Causal Language Modeling；
- Decoder-only Transformer；
- 自回归生成；
- 预训练数据治理；
- Continued Pre-training；
- SFT；
- DPO；
- GRPO；
- Reward Model；
- LLM Evaluation。

### 2.2.2 有模型优化，但缺少大规模训练系统经验

当前经历没有证明掌握：

- DistributedDataParallel；
- FSDP；
- ZeRO；
- Tensor Parallel；
- Pipeline Parallel；
- 混合精度训练；
- 梯度累积；
- Activation Checkpointing；
- 分布式 Checkpoint；
- 故障恢复；
- GPU 吞吐量和扩展效率分析；
- 通信开销分析。

### 2.2.3 缺少大模型应用软件工程证明

技能栏仅强调 Python 与 MATLAB，不足以支撑现代 LLM 软件开发岗位。

需要通过真实项目证明：

- Linux；
- Git；
- Docker；
- FastAPI；
- PostgreSQL；
- Redis；
- 异步任务；
- 单元测试；
- CI/CD；
- 日志系统；
- Tracing；
- Prometheus；
- Kubernetes；
- API 设计；
- 模型服务管理。

### 2.2.4 现有项目缺少完整工程证据

高质量项目需要提供：

- GitHub 仓库；
- 项目时间；
- README；
- 架构图；
- 数据说明；
- 实验配置；
- 可复现命令；
- 测试；
- Demo；
- 性能报告；
- 技术文档。

---

# 三、总体转型目标

最终要建立的能力链为：

```text
原始数据
→ 数据治理
→ Tokenizer
→ 模型架构
→ 预训练
→ 继续预训练
→ SFT
→ 偏好优化
→ 强化学习
→ 模型评测
→ 量化
→ 推理优化
→ API 服务
→ 容器部署
→ Kubernetes
→ 监控
→ Agent 应用
```

项目不能只做到：

- 调用 Hugging Face 模型；
- 使用一个微调框架；
- 写一个 RAG Demo；
- 接一个大模型 API；
- 实现一个简单聊天网页。

项目必须能够证明：

1. 理解底层原理；
2. 能够完成真实训练；
3. 能够使用成熟框架；
4. 理解工业系统设计；
5. 能够做自动化评测；
6. 能够部署上线；
7. 能够维护和监控系统；
8. 能够把模型用于复杂软件应用。

---

# 四、最终项目组合

在“工程 P0 稳定 + 前沿模块单变量验证”的约束下，最合适的组合是两个相互关联的核心项目。

## 4.1 主项目：ForgeLLM

### 项目名称

**ForgeLLM：端到端大语言模型构建、训练与部署平台**

### 覆盖流程

```text
原始数据
→ 数据清洗与治理
→ Tokenizer 训练
→ Decoder-only Transformer 构建
→ 从零预训练
→ 领域继续预训练
→ SFT
→ LoRA / QLoRA
→ DPO
→ Reward Model
→ GRPO
→ 自动评测
→ 量化
→ 推理服务
→ API 网关
→ Docker
→ Kubernetes
→ 监控与回滚
```

### 项目主要作用

ForgeLLM 用于证明：

- 会构建语言模型；
- 会训练语言模型；
- 会做后训练；
- 会做模型评测；
- 会做分布式训练；
- 会做推理优化；
- 会搭建模型服务；
- 会完成上线部署；
- 会完成模型运维。

## 4.2 应用项目：RepoPilot

### 项目名称

**RepoPilot：可执行、可验证、可评测的软件开发 Agent**

### 覆盖流程

```text
GitHub 仓库
→ 代码解析
→ 任务规划
→ 代码检索
→ 工具调用
→ 文件修改
→ 沙箱执行
→ 测试反馈
→ 错误修复
→ 自动评测
→ 在线服务
```

### 项目主要作用

RepoPilot 用于证明：

- 会构建 RAG；
- 会构建 Agent；
- 会进行工具调用；
- 会设计 Agent Loop；
- 会执行和修改代码；
- 会构建沙箱；
- 会开发后端系统；
- 会设计数据库与 API；
- 会对 Agent 进行自动评测；
- 会将模型能力转化为实际产品能力。

## 4.3 两个项目的关系

```text
ForgeLLM
负责数据、模型、训练、评测、推理和部署
        ↓
OpenAI-compatible API
        ↓
RepoPilot
负责代码理解、Agent、工具、沙箱和应用系统
```

这不是两个平行 Demo，而是一套上下游系统。

---

# 五、GitHub 高水平项目参考体系

不能选择单一项目进行照抄。

推荐使用三层参考法：

1. 教学级项目：理解原理与完整流程；
2. 工程级项目：学习合理代码组织；
3. 工业级项目：理解大规模系统设计。

## 5.1 从零构建 LLM

### nanochat

重点参考：

- Tokenizer；
- 小模型预训练；
- 指令微调；
- 推理；
- 训练流程组织；
- 最小完整 ChatGPT 链路。

定位：

> 教学主参考。

### LitGPT

重点参考：

- 现代 Decoder-only 模型结构；
- LoRA；
- QLoRA；
- FlashAttention；
- FSDP；
- 多模型兼容；
- 配置管理；
- 训练脚本工程化。

定位：

> 教学与工程之间的桥梁。

### llm.c

重点参考：

- C 语言实现；
- CUDA Kernel；
- 张量存储；
- 数值一致性；
- 算子性能；
- CPU/GPU 执行；
- 底层推理和训练。

定位：

> 底层系统进阶参考。

## 5.2 大规模训练

### TorchTitan

重点参考：

- PyTorch 原生训练；
- FSDP2；
- Tensor Parallel；
- Pipeline Parallel；
- Context Parallel；
- Activation Checkpointing；
- 分布式 Checkpoint；
- 可组合并行。

定位：

> 主要工业训练参考。

### Megatron-LM

重点参考：

- Tensor Parallel；
- Pipeline Parallel；
- Data Parallel；
- Expert Parallel；
- Context Parallel；
- 大规模 GPU 集群训练；
- 通信与计算重叠；
- 超大模型训练架构。

定位：

> 超大规模训练系统参考。

## 5.3 数据工程

### DataTrove

重点参考：

- 数据 Reader；
- Filter；
- Writer；
- Pipeline；
- 去重；
- 文本清洗；
- 可扩展数据处理；
- 数据统计。

### NeMo Curator

重点参考：

- 大规模数据清洗；
- GPU 加速数据处理；
- 质量过滤；
- PII 处理；
- 去重；
- 数据治理。

## 5.4 后训练

### TRL

重点参考：

- SFT；
- DPO；
- GRPO；
- Reward Model；
- Preference Optimization；
- Trainer 设计；
- 数据格式。

### LLaMA-Factory

重点参考：

- 模型兼容；
- 统一配置；
- 多种微调方法；
- LoRA/QLoRA；
- Web UI；
- 模型导出；
- 训练任务管理。

### Axolotl

重点参考：

- YAML 配置；
- 数据集配置；
- 多模型训练；
- 分布式训练；
- 高效微调；
- 工程组织。

### verl

重点参考：

- 强化学习训练系统；
- rollout；
- actor；
- critic；
- reward；
- 推理和训练解耦；
- GPU 资源调度；
- 分布式 RL。

## 5.5 模型评测

### lm-evaluation-harness

重点参考：

- 标准 Benchmark；
- 多任务评测；
- 模型后端统一；
- few-shot；
- 结果统计；
- 可复现评测。

### LightEval

重点参考：

- 自定义任务；
- 样本级结果；
- 多推理后端；
- 分布式评测；
- 结果分析。

## 5.6 推理服务

### vLLM

重点参考：

- PagedAttention；
- Continuous Batching；
- Prefix Caching；
- KV Cache；
- Tensor Parallel；
- 量化模型；
- OpenAI-compatible API；
- 高吞吐推理。

### SGLang

重点参考：

- 高性能生成；
- Prefix Sharing；
- Structured Generation；
- Agent 推理；
- Runtime 调度。

### TensorRT-LLM

重点参考：

- NVIDIA GPU 推理；
- Kernel Fusion；
- 量化；
- 多 GPU；
- 高性能 Engine；
- 推理编译优化。

## 5.7 分布式服务和部署

### Ray

重点参考：

- Ray Data；
- Ray Train；
- Ray Tune；
- Ray Serve；
- 分布式任务；
- 资源调度；
- 服务扩缩容。

### KServe

重点参考：

- Kubernetes 模型服务；
- InferenceService；
- 自动扩缩容；
- 模型版本管理；
- 标准化推理入口。

## 5.8 LLMOps 和监控

### MLflow

重点参考：

- 实验跟踪；
- 参数与指标；
- 模型注册；
- Artifact；
- 模型版本；
- LLM Evaluation；
- Tracing。

### Prometheus + Grafana

重点参考：

- 请求指标；
- 服务延迟；
- GPU 指标；
- 错误率；
- Tokens/s；
- 告警；
- 可视化 Dashboard。

### Langfuse 或 Phoenix

重点参考：

- LLM Trace；
- Prompt 版本；
- Agent 步骤；
- 工具调用；
- Token 成本；
- 延迟；
- 评测；
- 调试。

---

# 六、项目设计原则

## 6.1 每个模块学习三层

### 第一层：手写最小实现

目标：

- 理解原理；
- 知道输入输出；
- 知道梯度如何传播；
- 能做单元测试；
- 能做数值验证。

例如注意力：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
attention = torch.softmax(scores, dim=-1)
output = attention @ v
```

### 第二层：使用成熟框架完成真实任务

例如：

- PyTorch SDPA；
- FlashAttention；
- FSDP；
- TRL；
- vLLM；
- MLflow。

目标：

- 提高开发效率；
- 完成真实规模实验；
- 学习框架 API；
- 学习常见工程模式。

### 第三层：阅读工业级项目并对比

例如：

- TorchTitan 为什么使用可组合并行；
- Megatron 为什么需要 TP、PP、DP；
- vLLM 为什么使用分页式 KV Cache；
- verl 如何分离 rollout 与 training；
- DataTrove 如何组织大规模数据 Pipeline。

## 6.2 核心逻辑自己实现，重型基础设施合理复用

应自己实现：

- 最小 BPE；
- Transformer；
- Attention；
- Causal Mask；
- KV Cache；
- SFT Label Mask；
- DPO 核心 Loss；
- 最小评测；
- 最小推理循环。

应合理复用：

- FSDP；
- FlashAttention；
- vLLM；
- Kubernetes；
- Prometheus；
- PostgreSQL；
- Redis；
- MLflow。

## 6.3 不追求框架堆叠

项目不应变成：

> 调用十几个开源框架拼装而成。

而应做到：

> 核心算法能够解释，核心系统能够维护，复杂基础设施知道为什么使用。

## 6.4 所有阶段必须可测试、可复现、可度量

每个模块都必须具备：

- 配置文件；
- 测试；
- 日志；
- Benchmark；
- 复现命令；
- 输出报告；
- 明确验收标准。

---

# 七、主项目 ForgeLLM 总体设计

## 7.1 推荐目录结构

```text
forgellm/
├── configs/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── post_training/
│   ├── evaluation/
│   └── serving/
├── data_pipeline/
│   ├── readers/
│   ├── cleaners/
│   ├── filters/
│   ├── dedup/
│   ├── contamination/
│   ├── mixtures/
│   └── reports/
├── tokenizer/
│   ├── bpe_from_scratch.py
│   ├── train_tokenizer.py
│   ├── evaluate_tokenizer.py
│   └── chat_template.py
├── models/
│   ├── embedding.py
│   ├── rope.py
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   ├── transformer.py
│   ├── generation.py
│   └── kv_cache.py
├── training/
│   ├── train_single_gpu.py
│   ├── train_ddp.py
│   ├── train_fsdp.py
│   ├── optimizers.py
│   ├── schedulers.py
│   ├── checkpoint.py
│   └── profiler.py
├── post_training/
│   ├── sft/
│   ├── lora/
│   ├── dpo/
│   ├── reward_model/
│   └── grpo/
├── evaluation/
│   ├── perplexity/
│   ├── benchmarks/
│   ├── custom_tasks/
│   ├── regression/
│   └── reports/
├── inference/
│   ├── sampling.py
│   ├── generation.py
│   ├── benchmark.py
│   └── quantization/
├── serving/
│   ├── gateway/
│   ├── authentication/
│   ├── rate_limit/
│   ├── routing/
│   └── streaming/
├── deployment/
│   ├── docker/
│   ├── compose/
│   ├── kubernetes/
│   └── kserve/
├── observability/
│   ├── mlflow/
│   ├── prometheus/
│   ├── grafana/
│   └── tracing/
├── benchmarks/
├── tests/
├── scripts/
├── docs/
├── pyproject.toml
├── Makefile
└── README.md
```

---

# 八、ForgeLLM 阶段 0：工程基础设施

## 8.1 目标

在编写模型前建立标准工程结构，避免后期出现：

- 单个脚本越来越长；
- 配置写死；
- 训练无法复现；
- 不知道模型由哪个 Commit 产生；
- 没有测试；
- 模块相互耦合；
- 更换模型或数据需要大量修改。

## 8.2 必做内容

### Git

- main 分支；
- feature 分支；
- Pull Request；
- Commit 规范；
- Tag；
- Release；
- Issue；
- Project Board。

### Python 工程

- `pyproject.toml`；
- 包结构；
- 类型标注；
- Ruff；
- mypy 可选；
- pytest；
- pre-commit；
- 日志；
- 异常处理。

### 配置系统

支持：

- YAML；
- dataclass；
- Pydantic；
- Hydra 可选。

### 自动化命令

```bash
make install
make lint
make test
make train CONFIG=configs/training/pretrain_20m.yaml
make evaluate CHECKPOINT=...
make serve MODEL=...
```

### CI

GitHub Actions 至少包含：

- lint；
- unit test；
- import test；
- 小模型 smoke test；
- Docker build test。

## 8.3 验收标准

- 新环境可一条命令安装；
- 所有测试通过；
- 配置参数可覆盖；
- 日志清晰；
- 训练结果包含 Git Commit；
- Docker 开发环境可运行。

---

# 九、ForgeLLM 阶段 1：大模型数据工程

## 9.1 数据加载

支持：

- TXT；
- JSON；
- JSONL；
- Parquet；
- Hugging Face Dataset；
- 网页文本；
- 代码；
- 多轮对话。

统一格式：

```json
{
  "text": "document content",
  "source": "dataset_name",
  "license": "license_name",
  "language": "zh",
  "timestamp": "2026-07-11",
  "quality_score": 0.87,
  "document_id": "sha256..."
}
```

## 9.2 数据清洗

实现：

- 空文本过滤；
- 超短文本过滤；
- 超长异常文本过滤；
- HTML 清理；
- Unicode 规范化；
- 乱码检测；
- 重复标点；
- 特殊字符比例；
- 广告和导航内容过滤；
- 语言识别；
- 代码与自然语言分类；
- PII 检测与脱敏。

## 9.3 数据质量过滤

可使用：

- 规则分数；
- 文本长度；
- 标点比例；
- 字母和数字比例；
- 词汇多样性；
- 重复 n-gram；
- 困惑度过滤；
- 小分类器；
- LLM 质量打分。

## 9.4 数据去重

### 精确去重

- SHA256；
- 文档完全一致；
- 标准化文本后比较。

### SimHash

适合：

- 近似文本；
- 网页模板；
- 局部修改文本。

### MinHash + LSH

适合：

- 大规模近似去重；
- n-gram 集合相似度；
- 可扩展候选搜索。

必须分析：

- 去重阈值；
- 删除比例；
- 误删风险；
- 文档级和段落级差异；
- 去重对最终训练效果的影响。

## 9.5 数据污染检测

需要检查：

```text
训练语料 ∩ 评测数据
```

方法：

- 精确字符串；
- n-gram overlap；
- MinHash；
- Embedding 相似度；
- 答案片段搜索。

输出：

- 污染样本；
- 相似度；
- 来源；
- 是否删除；
- 最终评测集净化状态。

## 9.6 数据混合

支持配置：

```yaml
mixture:
  general_text: 0.40
  chinese_text: 0.20
  code: 0.20
  math: 0.15
  instruction: 0.05
```

记录：

- 理论采样比例；
- 实际采样比例；
- Token 数量；
- 每个数据源 Loss；
- 数据源之间的竞争关系。

## 9.7 数据版本

每次训练记录：

- 数据版本号；
- 文件哈希；
- 清洗配置；
- 去重配置；
- Tokenizer 版本；
- 总 Token 数；
- 许可证；
- 生成日期。

## 9.8 数据报告

输出：

```text
原始文档数
清洗后文档数
过滤比例
精确去重比例
近似去重比例
语言分布
长度分布
质量分布
来源分布
Token 数量
PII 处理数量
许可证分布
污染检测结果
```

---

# 十、ForgeLLM 阶段 2：Tokenizer 从零构建

## 10.1 手写最小 BPE

流程：

```text
原始文本
→ 字节或字符序列
→ 统计相邻 Pair
→ 合并最高频 Pair
→ 更新词表
→ 重复
→ 获得子词词表
```

需要实现：

- 训练；
- Encode；
- Decode；
- Merge Table；
- Vocabulary 保存；
- 特殊 Token。

## 10.2 正式 Tokenizer

使用成熟库完成：

- Byte-level BPE；
- Unicode normalization；
- Byte fallback；
- BOS；
- EOS；
- PAD；
- UNK；
- Chat Template；
- 中英文混合；
- 代码；
- 数学符号。

## 10.3 Tokenizer 评测

指标：

- Compression Ratio；
- Token Fertility；
- 每个中文字符 Token 数；
- 每个英文单词 Token 数；
- 代码文本 Token 数；
- 数学公式 Token 数；
- Encode-Decode 一致性；
- 未知字符比例；
- 词表大小；
- Embedding 参数量。

对比：

- 自训练 Tokenizer；
- GPT 类 Tokenizer；
- LLaMA 类 Tokenizer；
- 不同词表大小。

## 10.4 验收标准

- 能从零训练 BPE；
- 能正确 Encode/Decode；
- 有单元测试；
- 生成评测报告；
- 能处理中文、英文、代码和特殊字符；
- 能通过 Chat Template 构建训练输入。

---

# 十一、ForgeLLM 阶段 3：Decoder-only Transformer

## 11.1 模型规模

建议分两步：

### 调试模型

- 5M–20M 参数；
- 快速测试；
- 小数据过拟合；
- CPU 或单 GPU 可运行。

### 正式小模型

- 100M–300M 参数；
- 用于完整预训练流程；
- 用于性能分析；
- 用于分布式训练。

## 11.2 必须实现的模块

- Token Embedding；
- RoPE；
- RMSNorm；
- Causal Self-Attention；
- Multi-Head Attention；
- Grouped Query Attention；
- SwiGLU；
- Residual Connection；
- Transformer Block；
- LM Head；
- Weight Tying；
- Cross Entropy Loss；
- KV Cache；
- 自回归生成。

## 11.3 模型数据流

```text
Token IDs
→ Token Embedding
→ RoPE Position Information
→ Transformer Blocks
    ├─ RMSNorm
    ├─ Causal Attention
    ├─ Residual
    ├─ RMSNorm
    ├─ SwiGLU
    └─ Residual
→ Final Norm
→ LM Head
→ Logits
→ Next-token Loss
```

## 11.4 正确性测试

### Shape Test

检查：

- Batch；
- Sequence；
- Hidden；
- Head；
- Vocabulary。

### Causal Mask Test

修改未来 Token，不应影响过去位置输出。

### Gradient Test

确认：

- Embedding 有梯度；
- Attention 有梯度；
- MLP 有梯度；
- 梯度非 NaN；
- 梯度规模合理。

### Numerical Alignment

与 PyTorch SDPA 或参考实现比较：

- Forward；
- Loss；
- Gradient；
- 容差。

### Overfit Test

在几十条数据上：

- Loss 持续下降；
- 能记住训练样本；
- 能生成训练内容。

## 11.5 KV Cache

实现：

```text
第 1 步：完整 Prompt 得到历史 K/V
第 2 步：只输入新 Token
第 3 步：复用历史 K/V
```

比较：

- 无 Cache 延迟；
- 有 Cache 延迟；
- 不同序列长度；
- 不同 Batch；
- Cache 显存。

---

# 十二、ForgeLLM 阶段 4：预训练系统

## 12.1 Level 1：单 GPU 正确训练

实现：

- BF16；
- FP32 Master Weight 可选；
- AdamW；
- Weight Decay；
- Learning Rate Warmup；
- Cosine Decay；
- Gradient Clipping；
- Gradient Accumulation；
- Validation；
- Checkpoint；
- Resume；
- Logging；
- 随机种子。

记录：

- Train Loss；
- Validation Loss；
- Learning Rate；
- Gradient Norm；
- Tokens/s；
- GPU Memory；
- Step Time。

## 12.2 Level 2：训练性能优化

加入：

- PyTorch SDPA；
- FlashAttention；
- Activation Checkpointing；
- `torch.compile`；
- Fused Optimizer；
- Data Prefetch；
- Pinned Memory；
- Sequence Packing。

对比：

| 配置 | 显存 | Tokens/s | Step Time | 验证 Loss |
|---|---:|---:|---:|---:|
| 标准 Attention | | | | |
| SDPA | | | | |
| FlashAttention | | | | |
| + Checkpointing | | | | |
| + torch.compile | | | | |

## 12.3 Level 3：数据并行

顺序：

```text
Single GPU
→ DDP
→ FSDP2
```

学习问题：

- 参数存放在哪里？
- 梯度何时同步？
- All-Reduce 是什么？
- Reduce-Scatter 是什么？
- All-Gather 是什么？
- FSDP 如何分片参数？
- Optimizer State 如何分片？
- 为什么通信会降低扩展效率？

指标：

- 单卡吞吐；
- 多卡吞吐；
- Scaling Efficiency；
- 单卡峰值显存；
- 通信时间；
- Checkpoint 时间。

## 12.4 Level 4：模型并行

在模型无法放入单卡时学习：

- Tensor Parallel；
- Pipeline Parallel；
- Context Parallel；
- Sequence Parallel；
- Expert Parallel 可选。

不建议第一版直接实现完整 Megatron。

应先：

1. 理解原理；
2. 使用 TorchTitan；
3. 阅读 Megatron；
4. 做小规模对比实验。

## 12.5 Checkpoint

需要保存：

- 模型参数；
- Optimizer；
- Scheduler；
- 当前 Step；
- Epoch；
- Random State；
- Data Loader 位置；
- 配置；
- 数据版本；
- Git Commit。

验收：

- 中断训练；
- 从 Checkpoint 恢复；
- Loss 曲线连续；
- 数据顺序正确；
- 分布式训练可恢复。

---

# 十三、ForgeLLM 阶段 5：继续预训练与领域适配

## 13.1 目标

从零预训练用于理解完整流程；求职项目还需要展示实际开源模型领域适配能力。

选择：

- 0.5B；
- 1B；
- 1.5B；
- 3B；

范围内的开源基础模型。

## 13.2 流程

```text
Base Model
→ 通用能力评测
→ 领域语料处理
→ Continued Pre-training
→ 通用能力复测
→ 领域能力评测
```

## 13.3 重点问题

- 学习率如何设置？
- 通用和领域数据如何混合？
- 是否发生灾难性遗忘？
- 训练 Token 数多少合适？
- 领域 Perplexity 是否降低？
- 通用 Benchmark 是否下降？
- 是否需要 Replay General Data？
- 是否需要冻结部分参数？

## 13.4 验收标准

- 完成领域数据集；
- 有训练前后对比；
- 有通用和领域 Benchmark；
- 有遗忘分析；
- 有数据配比实验；
- 有完整 Model Card。

---

# 十四、ForgeLLM 阶段 6：SFT、LoRA 与 QLoRA

## 14.1 SFT 数据格式

支持：

- 单轮指令；
- 多轮对话；
- System Prompt；
- Tool Message；
- Structured Output。

示例：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "解释反向传播。"},
    {"role": "assistant", "content": "反向传播是……"}
  ]
}
```

## 14.2 Label Mask

训练时：

```text
System Token：-100
User Token：-100
Assistant Token：真实 Token ID
```

目的：

- 只学习 Assistant 回答；
- 避免对 Prompt 计算 Loss；
- 支持多轮对话。

## 14.3 Packing

将多个短样本拼接，减少 Padding：

```text
Sample A + EOS + Sample B + EOS + Sample C
```

需要处理：

- Attention Mask；
- Position ID；
- Label 边界；
- 不同样本之间隔离；
- Loss Mask。

## 14.4 Full Fine-tuning、LoRA 和 QLoRA

对比：

| 方法 | 可训练参数 | 显存 | 训练速度 | 文件大小 | 效果 |
|---|---:|---:|---:|---:|---:|
| Full FT | | | | | |
| LoRA | | | | | |
| QLoRA | | | | | |

需要理解：

- LoRA 低秩更新；
- Rank；
- Alpha；
- Target Modules；
- Adapter Merge；
- 4-bit 量化；
- Double Quantization；
- NF4。

## 14.5 框架对照

先实现最小 SFT，再使用：

- TRL；
- LLaMA-Factory；
- Axolotl；

复现实验。

最终项目描述应是：

> 自主实现支持 Chat Template、Packing、Label Mask、LoRA 和分布式训练的 SFT Pipeline，并使用成熟框架进行数值和效果对照。

---

# 十五、ForgeLLM 阶段 7：偏好优化与强化学习

## 15.1 DPO

数据：

```json
{
  "prompt": "用户问题",
  "chosen": "更优回答",
  "rejected": "较差回答"
}
```

需要实现：

- Policy Model；
- Reference Model；
- Chosen Log Probability；
- Rejected Log Probability；
- DPO Loss；
- Beta；
- Preference Accuracy。

需要理解：

- 为什么不直接训练 Reward Model？
- Reference Model 有什么作用？
- Beta 控制什么？
- 长回答偏置如何影响结果？
- 数据质量如何影响 DPO？

## 15.2 Reward Model

训练目标：

```text
reward(chosen) > reward(rejected)
```

评测：

- Pairwise Accuracy；
- Reward Margin；
- 长度偏置；
- 格式偏置；
- 风格偏置；
- Out-of-distribution 表现。

## 15.3 GRPO

适用于可验证任务：

- 数学；
- 代码；
- JSON；
- SQL；
- 单元测试；
- 规则任务。

流程：

```text
Prompt
→ 生成多个回答
→ 计算规则奖励
→ 组内相对优势
→ 更新模型
```

前期不建议做开放式聊天 RL。

## 15.4 强化学习系统进阶

后续参考 verl 学习：

- rollout engine；
- training engine；
- actor；
- critic；
- reward；
- worker；
- resource placement；
- distributed orchestration。

---

# 十六、ForgeLLM 阶段 8：模型评测系统

## 16.1 基础评测

- Validation Loss；
- Perplexity；
- 按数据领域 Perplexity；
- 长文本 Loss；
- 数据规模与 Loss 曲线；
- 不同 Checkpoint 对比。

## 16.2 能力评测

包括：

- 常识；
- 阅读理解；
- 中文；
- 英文；
- 数学；
- 代码；
- 指令遵循；
- 结构化输出；
- 多轮对话；
- 长文本。

## 16.3 标准 Benchmark

接入：

- lm-evaluation-harness；
- 或 LightEval。

要求：

- 配置化运行；
- 保存样本级输出；
- 保存总体指标；
- 保存错误样本；
- 记录模型和数据版本。

## 16.4 自定义 Benchmark

建议构建：

- 代码解释任务；
- Bug 定位任务；
- JSON 格式任务；
- 多轮约束任务；
- 中文技术问答；
- 工具调用任务。

## 16.5 LLM-as-a-Judge

可用于：

- 表达质量；
- 相关性；
- 完整性；
- 风格；
- 帮助程度。

但需要注意：

- Judge 偏置；
- 位置偏置；
- 长度偏置；
- 自我偏好；
- 不同 Judge 一致性。

## 16.6 回归测试

每次模型更新检查：

```text
数学能力是否下降
代码能力是否下降
中文能力是否下降
格式遵循是否下降
平均输出长度是否异常
安全拒答是否异常
幻觉是否增加
延迟是否增加
显存是否增加
```

部署前必须通过评测门槛。

---

# 十七、ForgeLLM 阶段 9：量化与模型导出

## 17.1 精度格式

学习：

- FP32；
- FP16；
- BF16；
- FP8；
- INT8；
- INT4。

## 17.2 量化方法

- Dynamic Quantization；
- Static Quantization；
- Weight-only；
- GPTQ；
- AWQ；
- SmoothQuant 基本思想；
- KV Cache Quantization 可选。

## 17.3 模型导出

支持：

- Hugging Face；
- Safetensors；
- Adapter；
- Merge Adapter；
- GGUF 可选；
- TensorRT-LLM Engine 可选。

## 17.4 对比指标

| 模型版本 | 文件大小 | 显存 | TTFT | Tokens/s | Benchmark |
|---|---:|---:|---:|---:|---:|
| BF16 | | | | | |
| INT8 | | | | | |
| INT4 | | | | | |

## 17.5 与已有能力的连接

原有 INT8、NPU、模型压缩经验可以迁移为：

- LLM Quantization；
- Weight-only Quantization；
- Edge LLM；
- On-device LLM；
- 推理精度与速度权衡。

---

# 十八、ForgeLLM 阶段 10：推理引擎与性能优化

## 18.1 自定义生成模块

实现：

- Greedy；
- Temperature；
- Top-k；
- Top-p；
- Repetition Penalty；
- Stop Tokens；
- Batch Generation；
- Streaming；
- KV Cache。

## 18.2 推理指标

- TTFT；
- TPOT；
- End-to-end Latency；
- Throughput；
- p50；
- p95；
- p99；
- GPU Memory；
- Concurrent Requests；
- Error Rate；
- Queue Time。

## 18.3 vLLM 服务

学习：

- PagedAttention；
- Continuous Batching；
- Prefix Caching；
- Chunked Prefill；
- Tensor Parallel；
- OpenAI API；
- Streaming；
- LoRA Serving；
- Quantized Serving。

## 18.4 横向对比

后续可比较：

- Hugging Face Generate；
- vLLM；
- SGLang；
- TensorRT-LLM。

要求使用：

- 同模型；
- 同硬件；
- 同输入长度；
- 同输出长度；
- 同并发；
- 同精度。

---

# 十九、ForgeLLM 阶段 11：API 与后端服务

## 19.1 架构

```text
Client
↓
FastAPI Gateway
↓
Authentication / Rate Limit / Routing / Logging
↓
vLLM OpenAI-compatible Server
↓
Model
```

## 19.2 必做功能

- API Key；
- 用户认证；
- 限流；
- SSE Streaming；
- 请求取消；
- 超时；
- 重试；
- 输入长度限制；
- 输出长度限制；
- 请求 ID；
- 错误码；
- Token 统计；
- 模型路由；
- Adapter 路由；
- Structured Output；
- Health Check。

## 19.3 数据库

### PostgreSQL

保存：

- 用户；
- API Key；
- 请求元数据；
- 模型版本；
- Adapter；
- 评测任务；
- 部署记录。

### Redis

用于：

- Rate Limit；
- Cache；
- Session；
- Job Status；
- Distributed Lock；
- Queue。

## 19.4 测试

- Unit Test；
- API Test；
- Integration Test；
- Load Test；
- Failure Injection；
- Timeout Test；
- Streaming Test。

---

# 二十、ForgeLLM 阶段 12：容器化和生产部署

## 20.1 Docker Compose

本地服务：

```text
FastAPI
vLLM
PostgreSQL
Redis
MLflow
Prometheus
Grafana
Tracing Service
```

一条命令启动：

```bash
docker compose up
```

## 20.2 Kubernetes

实现：

- Namespace；
- Deployment；
- Service；
- Ingress；
- ConfigMap；
- Secret；
- Persistent Volume；
- GPU Request；
- Node Selector；
- Readiness Probe；
- Liveness Probe；
- HPA。

## 20.3 KServe 或 Ray Serve

### KServe

适合学习：

- Kubernetes 原生模型服务；
- InferenceService；
- 自动扩缩容；
- 模型版本；
- 标准部署。

### Ray Serve

适合学习：

- Python 原生服务编排；
- 多模型；
- Pipeline；
- 资源调度；
- Replica；
- Scaling。

第一版不必二者同时使用。

## 20.4 上线流程

```text
训练完成
→ 离线评测
→ 模型注册
→ 测试环境部署
→ Smoke Test
→ 灰度发布
→ 线上监控
→ 全量发布
→ 异常则回滚
```

## 20.5 灰度与回滚

支持：

- Model A；
- Model B；
- 按用户比例路由；
- 按 API Key 路由；
- 按请求类型路由；
- 快速切换模型版本；
- 保留旧模型；
- 自动回滚条件。

---

# 二十一、ForgeLLM 阶段 13：监控与 LLMOps

## 21.1 MLflow

记录：

- 超参数；
- 数据版本；
- Git Commit；
- Loss；
- Benchmark；
- Checkpoint；
- Artifact；
- 模型版本；
- 量化版本；
- 部署状态。

## 21.2 Prometheus

服务指标：

- Request Count；
- Error Rate；
- Latency；
- TTFT；
- TPOT；
- Tokens/s；
- GPU Utilization；
- GPU Memory；
- Queue Length；
- Active Requests。

## 21.3 Grafana

Dashboard：

- 实时请求；
- 延迟分位数；
- GPU；
- 错误；
- 吞吐量；
- 模型版本；
- 用户流量；
- 队列。

## 21.4 LLM Trace

使用 MLflow Tracing、Langfuse 或 Phoenix 之一。

记录：

```text
User Input
Prompt Version
Model Version
Retrieved Context
Tool Calls
Token Usage
Latency
Errors
Final Output
Evaluation Score
```

## 21.5 告警

例如：

- p95 延迟超阈值；
- 错误率上升；
- GPU OOM；
- 队列过长；
- 生成 Token 异常；
- 空输出增加；
- 某模型版本质量下降。

---

# 二十二、应用项目 RepoPilot

## 22.1 项目目标

输入：

- GitHub 仓库；
- Issue；
- Bug 描述；
- 功能需求；
- 测试要求。

Agent 自动：

```text
理解任务
→ 分析仓库
→ 定位相关文件
→ 制定计划
→ 修改代码
→ 运行测试
→ 分析错误
→ 重新修改
→ 输出 Diff 和结果
```

## 22.2 推荐目录

```text
repopilot/
├── repo_ingestion/
├── parsers/
├── symbol_index/
├── retrieval/
├── planner/
├── executor/
├── verifier/
├── memory/
├── sandbox/
├── tools/
├── evaluation/
├── api/
├── frontend/
├── tests/
└── docs/
```

## 22.3 仓库理解

解析：

- 文件树；
- README；
- pyproject；
- requirements；
- package.json；
- Dockerfile；
- CI；
- 配置文件；
- 测试；
- Git History。

## 22.4 代码检索

不能只做向量检索。

需要组合：

- 文件名；
- grep；
- BM25；
- Embedding；
- AST；
- Symbol Definition；
- Symbol Reference；
- Import Graph；
- Git History；
- Reranker。

## 22.5 Agent Loop

```text
User Task
↓
Planner
↓
Retriever
↓
Tool Executor
↓
Sandbox
↓
Verifier
↓
成功：输出
失败：重新规划
```

## 22.6 工具

- `list_directory`；
- `read_file`；
- `search_code`；
- `find_symbol`；
- `inspect_git_history`；
- `edit_file`；
- `run_tests`；
- `run_command`；
- `git_diff`；
- `inspect_logs`。

## 22.7 沙箱

限制：

- 可访问目录；
- CPU；
- 内存；
- GPU；
- 执行时间；
- 网络；
- 命令；
- 输出长度；
- Secret；
- 文件大小；
- 子进程数量。

## 22.8 Verifier

判断：

- 测试是否通过；
- 是否满足 Issue；
- 是否修改了无关文件；
- 是否破坏现有功能；
- 是否有语法错误；
- 是否符合代码风格；
- 是否需要重新规划。

## 22.9 后端

使用：

- FastAPI；
- PostgreSQL；
- Redis；
- WebSocket/SSE；
- 异步任务；
- Docker；
- 身份认证；
- 日志；
- Trace。

## 22.10 自动评测

构建任务集：

- Bug Fix；
- Feature Implementation；
- Test Completion；
- Refactor；
- Config Change；
- Documentation；
- Dependency Update。

指标：

- Task Success Rate；
- Test Pass Rate；
- First-attempt Success；
- Average Iterations；
- Tool Success Rate；
- Recovery Rate；
- Token Cost；
- Latency；
- Invalid Edit Ratio；
- Unrelated File Change Ratio。

---

# 二十三、两个项目的连接关系

## 23.1 模型层

ForgeLLM 产生：

- Base Model；
- Continued Pre-trained Model；
- SFT Model；
- DPO Model；
- Quantized Model。

RepoPilot 可以对比：

- 不同训练阶段；
- 不同模型规模；
- 不同量化；
- 不同上下文长度；
- 不同推理后端。

## 23.2 服务层

ForgeLLM 提供：

```text
/v1/chat/completions
/v1/completions
/v1/models
/health
/metrics
```

RepoPilot 通过统一 API 调用。

## 23.3 评测层

可比较：

| 模型 | Bug Fix | Test Pass | Tool Use | Token Cost | Latency |
|---|---:|---:|---:|---:|---:|
| Base | | | | | |
| SFT | | | | | |
| DPO | | | | | |
| INT4 | | | | | |

## 23.4 部署层

两个项目共用：

- Docker；
- Kubernetes；
- PostgreSQL；
- Redis；
- MLflow；
- Prometheus；
- Grafana；
- Tracing。

---

# 二十四、查漏补缺后的升级清单

| 原计划不足 | 升级内容 |
|---|---|
| Tokenizer 只简单提及 | 手写 BPE、正式 Tokenizer、压缩率和多语言评测 |
| 数据只做基础清洗 | 增加许可证、来源追踪、PII、去重、污染、版本 |
| 模型只要求跑通 | 增加数值对齐、梯度、Mask、过拟合测试 |
| 分布式训练笼统 | 明确 Single GPU → DDP → FSDP2 → TP/PP |
| Checkpoint 只保存参数 | 增加 Optimizer、Scheduler、RNG、Data Position |
| 后训练只关注 Trainer | 增加 Chat Template、Packing、Label Mask |
| DPO 只调用框架 | 自己实现核心 Loss 和 Log Probability |
| 评测只看总分 | 增加样本级输出、污染、回归、错误分析 |
| 推理只看 Tokens/s | 增加 TTFT、TPOT、p95、并发、错误率 |
| API 过于简单 | 增加认证、限流、流式、取消、路由、错误码 |
| Docker 后直接上线 | 增加健康检查、灰度、回滚、自动扩缩容 |
| 缺少 LLMOps | 增加实验跟踪、模型注册、Tracing、告警 |
| 缺少安全 | 增加 Secret、沙箱、资源限制、网络限制 |
| 缺少持续集成 | 增加 Unit、Integration、E2E、Performance Test |
| Agent 只做流程展示 | 增加可自动验证任务集和 Benchmark |
| RAG 只用向量搜索 | 增加 BM25、AST、Symbol、Git、Reranker |

---

# 二十五、推荐技术栈

## 25.1 主技术栈

```text
语言：
Python
C++/CUDA 后期可选
SQL

模型训练：
PyTorch
自定义 Transformer
FSDP2
FlashAttention

数据：
自定义 Pipeline
DataTrove 作为后续参考或接入

后训练：
自定义 SFT / DPO
TRL 用于验证

评测：
lm-evaluation-harness 或 LightEval

推理：
vLLM

API：
FastAPI

数据库：
PostgreSQL
Redis

实验管理：
MLflow

监控：
Prometheus
Grafana

Tracing：
MLflow Tracing / Langfuse / Phoenix 三选一

部署：
Docker Compose
Kubernetes
KServe 或 Ray Serve

工程：
pytest
Ruff
pre-commit
GitHub Actions
```

## 25.2 作为参考而非全部依赖

- nanochat；
- LitGPT；
- llm.c；
- TorchTitan；
- Megatron-LM；
- NeMo Curator；
- LLaMA-Factory；
- Axolotl；
- verl；
- SGLang；
- TensorRT-LLM；
- Ray；
- KServe；
- Phoenix；
- Langfuse。

---

# 二十六、项目里程碑与验收标准

> 本章 M0–M8 是长期能力地图，不是当前 Stage 0–6 的执行门禁。100M–300M、FSDP、多卡、推理服务和 Kubernetes 等内容均为后续条件升级项；当前完成状态与验收阈值只在 `ForgeLLM/docs/model_first_21_week_learning_plan.md` 中维护。

## M0：工程框架

产物：

- 项目目录；
- pyproject；
- 配置系统；
- pytest；
- Ruff；
- GitHub Actions；
- Docker 开发环境；
- Makefile。

验收：

- `make test` 通过；
- `make lint` 通过；
- Docker 可运行；
- Smoke Test 通过。

## M1：数据与 Tokenizer

产物：

- 数据 Reader；
- 清洗；
- 质量过滤；
- 精确去重；
- MinHash 去重；
- 数据报告；
- BPE；
- 正式 Tokenizer；
- Tokenizer 报告。

验收：

- Encode/Decode 正确；
- 去重结果可解释；
- 数据版本可复现；
- 污染检查完成。

## M2：最小语言模型

产物：

- 5M–20M Transformer；
- Attention；
- RoPE；
- RMSNorm；
- SwiGLU；
- KV Cache；
- Generation。

验收：

- Shape Test；
- Causal Test；
- Gradient Test；
- Numerical Alignment；
- 小样本过拟合。

## M3：从零预训练

产物：

- 100M–300M 模型；
- 单卡训练；
- DDP；
- FSDP；
- Checkpoint；
- FlashAttention 对比；
- 性能报告。

验收：

- Loss 正常下降；
- 可中断恢复；
- 多卡训练成功；
- 有吞吐与显存报告。

## M4：继续预训练与 SFT

产物：

- 领域数据；
- Continued Pre-training；
- SFT；
- LoRA；
- QLoRA；
- Model Card。

验收：

- 领域能力提升；
- 通用能力下降可控；
- SFT 格式遵循率提高。

## M5：DPO 与评测

产物：

- Preference 数据；
- 自定义 DPO；
- TRL 对照；
- Benchmark；
- 回归测试。

验收：

- Preference Accuracy 提升；
- 自动评测可重复；
- 错误分析完整。

## M6：推理服务

产物：

- 自定义生成；
- vLLM；
- FastAPI；
- Streaming；
- Auth；
- Rate Limit；
- PostgreSQL；
- Redis。

验收：

- API 正常；
- 并发测试；
- TTFT/TPOT 报告；
- 错误和超时可处理。

## M7：生产部署

产物：

- Docker Compose；
- Kubernetes；
- MLflow；
- Prometheus；
- Grafana；
- 灰度；
- 回滚。

验收：

- 一键部署；
- 健康检查；
- 指标可视化；
- 模型版本可切换；
- 故障可回滚。

## M8：RepoPilot

产物：

- 仓库解析；
- 混合检索；
- Planner；
- Executor；
- Verifier；
- 沙箱；
- Agent Benchmark。

验收：

- 能完成真实小型代码任务；
- 测试可自动验证；
- 有任务成功率；
- 有成本和延迟报告。

---

# 二十七、建议实施顺序

> 本章保留原始 6–9 个月路线用于历史比较，不再作为当前排期。不得用这里的“完成”清单覆盖 ForgeLLM 当前阶段状态。

## 第一阶段：基础建设

建议内容：

- 工程模板；
- PyTorch Transformer；
- Tokenizer；
- 数据处理；
- 单元测试；
- Linux；
- Docker；
- Git。

## 第二阶段：ForgeLLM MVP

完成：

- 数据 Pipeline；
- Tokenizer；
- 20M 模型；
- 小规模预训练；
- 生成；
- Checkpoint；
- 基础评测。

## 第三阶段：训练系统升级

完成：

- 100M–300M；
- FlashAttention；
- DDP；
- FSDP；
- Profiling；
- 性能报告。

## 第四阶段：后训练

完成：

- Continued Pre-training；
- SFT；
- LoRA；
- QLoRA；
- DPO；
- Benchmark。

## 第五阶段：推理与上线

完成：

- vLLM；
- FastAPI；
- PostgreSQL；
- Redis；
- Docker；
- Kubernetes；
- MLflow；
- Prometheus。

## 第六阶段：RepoPilot

完成：

- 代码索引；
- Agent Loop；
- 工具；
- 沙箱；
- 自动评测；
- 与 ForgeLLM 打通。

## 建议周期

根据学习时间，可划分为约 6–9 个月：

```text
第 1 个月：
工程、数据、Tokenizer、最小 Transformer

第 2 个月：
预训练、Checkpoint、评测

第 3 个月：
DDP、FSDP、FlashAttention、性能分析

第 4 个月：
继续预训练、SFT、LoRA、QLoRA

第 5 个月：
DPO、Reward Model、GRPO 入门、评测系统

第 6 个月：
vLLM、FastAPI、数据库、Docker

第 7 个月：
Kubernetes、监控、灰度、回滚

第 8–9 个月：
RepoPilot、Agent、沙箱、Benchmark
```

不必等待全部完成后再求职。

当 M3 或 M4 完成后，就可以开始在简历中加入 ForgeLLM，并投递部分岗位。

---

# 二十八、最终简历应如何变化

## 28.1 推荐结构

```text
姓名与联系方式
GitHub | LinkedIn | 技术主页

教育背景

核心项目
1. ForgeLLM：端到端大语言模型训练与部署平台
2. RepoPilot：可执行代码 Agent
3. UWB CNN-Transformer 项目
4. YOLO 端侧部署项目

开源贡献
技术博客
技术技能
奖项
```

## 28.2 ForgeLLM 项目描述方向

未来可写成：

> 从零构建 Decoder-only Transformer 与 BPE Tokenizer，开发覆盖数据清洗、近似去重、污染检测、预训练、SFT、DPO、自动评测、量化和 vLLM 服务的端到端 LLM Pipeline。

> 实现单 GPU、DDP 与 FSDP2 训练，集成 FlashAttention、Activation Checkpointing 和分布式 Checkpoint，对峰值显存、Tokens/s 和扩展效率进行系统 Benchmark。

> 使用 FastAPI、PostgreSQL、Redis、Docker 与 Kubernetes 构建 OpenAI-compatible 服务，接入 MLflow、Prometheus 和 Grafana，实现模型注册、灰度发布、监控与回滚。

## 28.3 RepoPilot 项目描述方向

未来可写成：

> 构建面向 GitHub 仓库的软件开发 Agent，结合 BM25、Embedding、AST、Symbol Graph 和 Reranker 完成代码检索，采用 Planner–Executor–Verifier Loop 自动修改代码并执行测试。

> 设计受限容器沙箱与任务 Benchmark，评估任务成功率、测试通过率、错误恢复率、平均迭代次数、Token 成本和端到端延迟。

## 28.4 技能栏升级

```text
编程与工程：
Python、SQL、Linux、Git、Docker、FastAPI、PostgreSQL、Redis、pytest

大模型：
PyTorch、Transformers、Tokenizer、Causal LM、SFT、LoRA、QLoRA、DPO、GRPO

训练系统：
DDP、FSDP、FlashAttention、Activation Checkpointing、Distributed Checkpoint

推理部署：
vLLM、Quantization、OpenAI-compatible API、Kubernetes、KServe/Ray Serve

评测与运维：
lm-evaluation-harness、MLflow、Prometheus、Grafana、LLM Tracing

Agent：
RAG、Hybrid Retrieval、Tool Calling、Agent Loop、Sandbox、Agent Evaluation
```

每个技能都必须在项目中真实使用。

---

# 二十九、岗位覆盖范围

## 29.1 ForgeLLM 对应岗位

- 大模型算法工程师；
- 大模型训练工程师；
- LLM Research Engineer；
- 大模型推理优化工程师；
- LLM Systems Engineer；
- Machine Learning Engineer；
- AI Infrastructure Engineer；
- Model Serving Engineer。

## 29.2 RepoPilot 对应岗位

- Agent 算法工程师；
- AI 应用工程师；
- 大模型软件开发工程师；
- Applied AI Engineer；
- AI Backend Engineer；
- Coding Agent Engineer；
- RAG Engineer。

## 29.3 原有经验带来的差异化

原有视觉、信号处理、FlashAttention、量化和端侧部署经验，可以形成：

- 不只会调 API；
- 理解模型内部计算；
- 重视性能；
- 能处理硬件限制；
- 能做模型压缩；
- 能从算法走到部署；
- 具备多模态和边缘大模型发展空间。

---

# 三十、最终结论

最终项目组合应为：

```text
主项目：ForgeLLM
负责大模型从数据到生产部署的完整生命周期

应用项目：RepoPilot
负责将模型用于真实代码任务和 Agent 软件系统
```

ForgeLLM 承担约 70% 的学习任务，RepoPilot 用于综合验证模型、推理、Agent 和软件工程能力。

ForgeLLM 的最终目标不是：

> 训练一个小语言模型。

而是：

> 构建一个支持数据治理、Tokenizer、Decoder-only Transformer、预训练、继续预训练、SFT、DPO、评测、量化、推理、服务、部署和监控的端到端大模型平台。

RepoPilot 的最终目标不是：

> 构建一个会聊天的代码助手。

而是：

> 构建一个能够理解代码仓库、制定计划、调用工具、修改文件、运行测试、根据反馈重新规划，并通过自动任务集进行评测的软件开发 Agent。

完成两个项目后，能力标签将从：

```text
计算机视觉 + 信号处理 + 端侧部署
```

扩展为：

```text
大模型算法
+ 大模型训练
+ 分布式系统
+ 推理优化
+ 软件开发
+ Agent
+ 部署与 LLMOps
```

---

# 三十一、参考 GitHub 项目

以下项目用于学习和架构参考，不建议全部直接作为依赖。

## 从零构建与教学

- nanochat  
  https://github.com/karpathy/nanochat

- nanoGPT  
  https://github.com/karpathy/nanoGPT

- LitGPT  
  https://github.com/Lightning-AI/litgpt

- llm.c  
  https://github.com/karpathy/llm.c

## 大规模训练

- TorchTitan  
  https://github.com/pytorch/torchtitan

- Megatron-LM  
  https://github.com/NVIDIA/Megatron-LM

## 数据工程

- DataTrove  
  https://github.com/huggingface/datatrove

- NeMo Curator  
  https://github.com/NVIDIA-NeMo/Curator

## 后训练

- TRL  
  https://github.com/huggingface/trl

- LLaMA-Factory  
  https://github.com/hiyouga/LLaMA-Factory

- Axolotl  
  https://github.com/axolotl-ai-cloud/axolotl

- verl  
  https://github.com/volcengine/verl

## 评测

- lm-evaluation-harness  
  https://github.com/EleutherAI/lm-evaluation-harness

- LightEval  
  https://github.com/huggingface/lighteval

## 推理

- vLLM  
  https://github.com/vllm-project/vllm

- SGLang  
  https://github.com/sgl-project/sglang

- TensorRT-LLM  
  https://github.com/NVIDIA/TensorRT-LLM

## 分布式系统与部署

- Ray  
  https://github.com/ray-project/ray

- KServe  
  https://github.com/kserve/kserve

## LLMOps 与监控

- MLflow  
  https://github.com/mlflow/mlflow

- Prometheus  
  https://github.com/prometheus/prometheus

- Grafana  
  https://github.com/grafana/grafana

- Langfuse  
  https://github.com/langfuse/langfuse

- Phoenix  
  https://github.com/Arize-ai/phoenix

---

# 附录 A：项目完成质量检查表

## 数据

- [ ] 支持多种数据格式
- [ ] 数据来源和许可证可追踪
- [ ] 完成规则清洗
- [ ] 完成质量过滤
- [ ] 完成精确去重
- [ ] 完成近似去重
- [ ] 完成污染检测
- [ ] 完成 PII 处理
- [ ] 生成数据报告
- [ ] 支持数据版本

## Tokenizer

- [ ] 手写 BPE
- [ ] 正式训练 Tokenizer
- [ ] 支持中英文
- [ ] 支持代码
- [ ] 支持特殊 Token
- [ ] 支持 Chat Template
- [ ] 完成压缩率评测
- [ ] 完成 Encode/Decode 测试

## 模型

- [ ] Token Embedding
- [ ] RoPE
- [ ] RMSNorm
- [ ] Causal Attention
- [ ] GQA
- [ ] SwiGLU
- [ ] Weight Tying
- [ ] KV Cache
- [ ] Generation
- [ ] Shape Test
- [ ] Causal Test
- [ ] Gradient Test
- [ ] Numerical Alignment
- [ ] Overfit Test

## 训练

- [ ] BF16
- [ ] Gradient Accumulation
- [ ] Gradient Clipping
- [ ] Warmup
- [ ] Cosine Scheduler
- [ ] Checkpoint
- [ ] Resume
- [ ] FlashAttention
- [ ] Activation Checkpointing
- [ ] DDP
- [ ] FSDP
- [ ] Profiling
- [ ] Scaling Report

## 后训练

- [ ] Continued Pre-training
- [ ] SFT
- [ ] Label Mask
- [ ] Packing
- [ ] LoRA
- [ ] QLoRA
- [ ] DPO
- [ ] Reward Model
- [ ] GRPO 可选
- [ ] 框架对照

## 评测

- [ ] Perplexity
- [ ] 标准 Benchmark
- [ ] 自定义 Benchmark
- [ ] 样本级结果
- [ ] 错误分析
- [ ] 回归测试
- [ ] 污染检查
- [ ] 质量门槛

## 推理

- [ ] Greedy
- [ ] Temperature
- [ ] Top-k
- [ ] Top-p
- [ ] Streaming
- [ ] KV Cache
- [ ] vLLM
- [ ] TTFT
- [ ] TPOT
- [ ] p95/p99
- [ ] 并发测试
- [ ] 量化测试

## 服务

- [ ] FastAPI
- [ ] OpenAI-compatible API
- [ ] API Key
- [ ] Auth
- [ ] Rate Limit
- [ ] Retry
- [ ] Timeout
- [ ] Cancellation
- [ ] Structured Output
- [ ] PostgreSQL
- [ ] Redis

## 部署与运维

- [ ] Docker
- [ ] Docker Compose
- [ ] Kubernetes
- [ ] Health Check
- [ ] GPU Scheduling
- [ ] Auto Scaling
- [ ] Model Registry
- [ ] Gray Release
- [ ] Rollback
- [ ] Prometheus
- [ ] Grafana
- [ ] Tracing
- [ ] Alert

## RepoPilot

- [ ] 仓库解析
- [ ] BM25
- [ ] Embedding
- [ ] AST
- [ ] Symbol Graph
- [ ] Planner
- [ ] Executor
- [ ] Verifier
- [ ] Tool Calling
- [ ] Sandbox
- [ ] Test Execution
- [ ] Agent Benchmark
- [ ] 成功率报告
- [ ] 成本和延迟报告

---

# 附录 B：学习过程中应始终回答的问题

对于每一个模块，都应回答：

1. 这个模块解决什么问题？
2. 输入和输出分别是什么？
3. 最小实现是什么？
4. 工业实现是什么？
5. 有哪些常见方法？
6. 各自原理是什么？
7. 各自优点和缺点是什么？
8. 在项目代码中如何组织？
9. 如何测试正确性？
10. 如何评测性能？
11. 常见失败现象是什么？
12. 如何排查问题？
13. 参考项目如何实现？
14. 我们为什么选择当前方案？
15. 简历中应如何准确描述？

---

# 附录 C：项目开发纪律

1. 不写无法复现的实验结论。
2. 不写缺少硬件和配置条件的性能数字。
3. 不把“调用已有库”描述为“实现底层算子”。
4. 不为了展示技术栈而堆叠框架。
5. 不在模型未通过回归评测时部署。
6. 不在没有测试的情况下进行大型重构。
7. 所有实验记录数据版本、代码版本和配置。
8. 所有性能对比使用相同硬件、模型和请求条件。
9. 所有项目结论尽量有表格、曲线和错误案例。
10. README 必须让其他人可以复现最小结果。

---

# 附录 D：范围优先级与停止清单

## D.1 ForgeLLM 优先级

| 能力块 | P0 必做 | P1 条件升级 | P2 长期扩展 |
|---|---|---|---|
| 数据 | 单一可追踪语料、清洗、精确去重、污染检查、Data Card | MinHash/LSH、质量分类器、混合采样消融 | 大规模分布式数据处理、GPU Curator |
| Tokenizer | 手写最小 BPE、正式 Tokenizer、压缩率与一致性测试 | 词表大小消融、代码/中英专项分析 | 多模态或超大词表研究 |
| 从零模型 | 5M–20M 正确性模型和完整训练闭环 | 100M–300M 性能实验 | 追求计算最优的大规模预训练 |
| 开源模型适配 | 0.5B–1.5B 模型的 SFT/LoRA，含前后评测 | Continued Pre-training、QLoRA、DPO | Reward Model、GRPO、分布式 RL |
| 分布式训练 | Single GPU；有多卡时做 DDP 或 FSDP2 至少一种 | DDP 与 FSDP2 对照、DCP 恢复、扩展效率 | TP/PP/CP/EP 组合并行 |
| 推理 | 自定义 KV Cache 与生成；vLLM；统一 API | INT4/INT8、前缀缓存、并发压测 | SGLang/TensorRT-LLM 横向竞赛 |
| 服务 | FastAPI 网关、认证、限流、流式、超时、日志 | 模型路由、灰度、OpenTelemetry | 多租户计费与复杂调度 |
| 部署 | Docker Compose 可复现部署 | Kubernetes，仅当有真实 GPU 集群或部署证据 | KServe/Ray Serve 二选一，不同时做 |

## D.2 RepoPilot 优先级

| 能力块 | P0 必做 | P1 条件升级 | P2 长期扩展 |
|---|---|---|---|
| 任务范围 | Python 单仓库 Bug Fix/Test Completion | Python Feature/Refactor，多仓库并发 | 多语言与大型 monorepo |
| 模型 | 强模型基线 + 可替换 OpenAI-compatible 接口 | 本地开源模型、模型路由、ForgeLLM 对照 | 针对 Agent 的 RL 训练 |
| 检索 | 文件树、ripgrep、BM25、符号定义/引用 | Embedding、Reranker、Git History | 学习型检索策略 |
| Agent | 单 Planner–Executor–Verifier 闭环 | 检查点恢复、并发子任务、预算调度 | 多 Agent 组织系统 |
| 沙箱 | 临时工作区、默认断网、资源限制、命令策略 | gVisor/微虚拟机、镜像池 | 多租户生产执行平台 |
| 评测 | 自建小任务集 + 隐藏测试 + 简单 Agent 基线 | SWE-bench Lite/Verified 子集 | 全量排行榜规模评测 |

## D.3 明确停止做的内容

以下行为会显著稀释项目证据，应写入项目纪律：

- 不在 P0 未发布前同时接入 LLaMA-Factory、Axolotl、TRL 三套训练框架；保留“一套主实现 + 一套对照”。
- 不同时部署 KServe 和 Ray Serve；先写技术决策记录，再选一个。
- 不为简历展示而同时比较 vLLM、SGLang、TensorRT-LLM；只有在提出明确性能假设时才增加第二后端。
- 不把 20M 自训练模型包装成“可用代码 Agent 模型”；它用于证明训练闭环和系统正确性。
- 不把“启动 Kubernetes YAML”当作生产经验；没有负载、故障和回滚证据时，Docker Compose 更可信。
- 不在 DPO/SFT 尚无稳定基线时进入 Reward Model 或 GRPO。
- 不把公开测试当作唯一 Verifier；RepoPilot 必须保留 Agent 不可见的隐藏测试。
- 不在真实算力预算未知时承诺训练 100M–300M 模型到计算最优。

---

# 附录 E：资源预算与条件触发

## E.1 启动前资源卡

项目开始前填写并提交 `docs/project_resource_card.md`：

```yaml
time:
  hours_per_week: null
  target_weeks: null
compute:
  gpu_type: null
  gpu_count: null
  vram_per_gpu_gb: null
  local_or_cloud: null
budget:
  max_training_cost: null
  max_evaluation_cost: null
storage:
  local_gb: null
  object_storage_gb: null
constraints:
  network: null
  data_license: null
  target_job_priority: null
```

资源卡为空时，只允许启动 CPU/单卡 Smoke Test，不允许发起长训练或大规模 Agent 评测。

## E.2 三档实施方案

| 档位 | 典型资源 | ForgeLLM 上限 | RepoPilot 上限 |
|---|---|---|---|
| A：单卡节制版 | 1×16–24GB GPU | 5M–20M 从零模型；0.5B–1.5B LoRA/QLoRA；单机推理 | 自建 20–50 个 Python 任务，串行评测 |
| B：多卡工程版 | 2–8 张 GPU | 100M–300M 受限预训练；DDP/FSDP2；DCP；量化服务 | SWE-bench 子集；并发任务；本地开源模型对比 |
| C：集群研究版 | 稳定多节点集群 | TP/PP/CP；更长上下文；分布式 RL | 大规模基准、调度与多租户隔离 |

档位 A 完全可以形成合格求职项目。档位 B/C 不是“更高级的勾选项”，只有产生吞吐、扩展效率、故障恢复或成本曲线时才值得投入。

## E.3 训练预算方法

每次正式训练前记录：

- 参数量 `N`；
- 训练 Token 数 `D`；
- 序列长度与全局 Batch；
- 预计训练步数；
- 单步耗时与峰值显存（来自 50–200 步 Smoke Test）；
- 预计总 GPU 小时与费用；
- Checkpoint、评测和失败重跑预留，建议至少保留 20% 预算。

可使用 `约 6ND` 作为 Decoder-only 稠密模型训练 FLOPs 的粗略规划估计，但最终预算必须以本机 Smoke Test 外推为准。预算偏差超过 25% 时暂停长跑，先定位数据、通信或算子瓶颈。

## E.4 条件触发规则

- **100M–300M 从零预训练**：只有 20M 模型完成数值对齐、恢复测试、评测接口和成本外推后才能启动。
- **FSDP2**：模型、优化器状态或目标 Batch 在单卡不可承受，或确有扩展效率实验需求时启动。
- **TP/PP/CP**：只有模型或上下文无法用 FSDP2 放入目标硬件，并且有至少 4 张 GPU 时考虑。
- **DPO**：SFT 基线稳定、偏好数据通过一致性审计，并且离线能力回归已固定时启动。
- **GRPO**：存在确定性 Verifier、Rollout 预算和清晰的 SFT/DPO 基线时启动；否则保持 P2。
- **Kubernetes**：Docker Compose 已稳定，且需要展示滚动发布、GPU 调度或自动扩缩容中的至少一项时启动。
- **第二推理后端**：只有当前后端的性能瓶颈已通过 Profiling 定位，且新后端对应明确假设时引入。

---

# 附录 F：阶段门禁与量化验收

## F.1 通用阶段模板

每个 Stage 必须先创建计划，再允许执行：

```markdown
# Stage：<name>

## 目标与用户价值
## 可证伪假设
## 官方/公认基线
## 当前工程基线
## 唯一主变量
## 固定项
## 数据与代码版本
## 指标及统计方法
## 成功门槛
## 预算与停止条件
## 失败处理
## 预期产物
```

## F.2 ForgeLLM 阶段门禁

| Gate | 必须产物 | 通过条件 | 未通过时的处理 |
|---|---|---|---|
| G0 工程 | 安装、Lint、单测、CPU Smoke、容器 | 新环境按 README 在 30 分钟内跑通最小例子；CI 全绿 | 暂停模型开发，修复入口和依赖 |
| G1 数据/Tokenizer | Data Card、清洗报告、污染报告、Tokenizer 报告 | 来源/许可证/哈希齐全；Encode–Decode 属性测试通过；评测集无已知泄漏 | 缩小数据源，先建立可审计版本 |
| G2 模型正确性 | 参考实现对齐、梯度/因果性/过拟合测试 | FP32 参考路径在约定容差内；未来 Token 不影响过去；Tiny Set 可稳定过拟合 | 禁止长训练，逐层二分定位 |
| G3 训练闭环 | Loss 曲线、Checkpoint、Resume、资源报告 | 固定种子恢复后曲线连续；数据位置无重复/跳过；无 NaN/OOM | 降低规模并补充故障复现 |
| G4 性能/分布式 | Profile、单卡与多卡表、DCP 恢复 | 正确性不回退；2 卡扩展效率目标 ≥70%，未达到也须给出可验证根因 | 不继续堆并行策略，先处理瓶颈 |
| G5 CPT/SFT | 前后能力表、遗忘分析、Model Card | 领域指标有明确提升；通用核心指标相对下降建议不超过 3%；格式遵循率建议 ≥95% | 调整数据混合、学习率或停止训练 |
| G6 DPO 可选 | 数据审计、DPO/TRL 对齐、偏好与回归表 | 自实现 Loss 与参考实现对齐；偏好指标改善且核心能力不越过回归门槛 | 保留为负结果，不进入 RL |
| G7 推理服务 | API Contract、负载报告、量化对比 | 流式、取消、限流、超时和错误码测试通过；所有性能数字含硬件与请求条件 | 缩小服务范围，优先保证正确性 |
| G8 发布 | Release、复现脚本、Demo、技术报告 | 陌生环境可复现最小结果；所有简历数字可追溯到 Artifact | 不写入简历，继续补证据 |

说明：3% 和 70% 是个人项目的初始工程门槛，可在资源卡中修改，但必须在实验前固定，不能看到结果后再调整。

## F.3 核心实验矩阵

### 训练正确性与性能

| Run | 主变量 | 固定项 | 正确性 | 峰值显存 | Tokens/s | Step time | 结论 |
|---|---|---|---|---:|---:|---:|---|
| reference | 标准 Attention | 模型/数据/Batch/硬件 |  |  |  |  | 工程基线 |
| sdpa | SDPA | 同上 |  |  |  |  |  |
| flash | FlashAttention | 同上 |  |  |  |  |  |
| flash_ac | + Activation Checkpointing | 同上 |  |  |  |  |  |

### 模型阶段对比

| 模型版本 | 领域指标 | 通用指标 | 格式遵循 | RepoPilot Resolve | TTFT | 成本 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| 开源 Base |  |  |  |  |  |  |  |
| CPT |  |  |  |  |  |  |  |
| SFT |  |  |  |  |  |  |  |
| DPO |  |  |  |  |  |  |  |
| INT4 |  |  |  |  |  |  |  |

## F.4 统计纪律

- 核心模型结论尽量使用至少 3 个随机种子；算力不足时明确标记为单次工程结果。
- Agent 成功率必须同时报告分母、绝对成功数和区间估计，不能只写百分比。
- 性能测试至少预热，重复多轮，报告中位数与 p95/p99，并固定输入/输出长度、并发和精度。
- 任何“提升”必须与预先指定的基线比较；只与自己更差的临时版本比较不构成有力证据。
- 失败实验保留记录。若失败改变后续决策，它就是有效工程证据。

---

# 附录 G：RepoPilot 执行版设计

## G.1 先纠正两个项目的依赖关系

RepoPilot 使用统一 `ModelProvider` 接口，至少支持三条模型通道：

```text
Lane A：能力足够的外部/托管模型
  用途：建立 Agent 系统上限与主基线

Lane B：本地开源指令模型 + vLLM
  用途：验证私有部署、成本、延迟和量化

Lane C：ForgeLLM Base/SFT/DPO 模型
  用途：研究训练阶段对代码 Agent 能力的影响
```

任何一条模型通道故障，都不能阻塞 RepoPilot 的索引、沙箱、工具和评测开发。比较模型时固定 Agent Scaffold、Prompt、工具、预算与 Benchmark 版本。

## G.2 P0 任务契约

每个任务必须转成不可变 Task Spec：

```yaml
task_id: repo__issue__001
repo_url: null
base_commit: null
problem_statement: null
language: python
allowed_paths: []
forbidden_paths: []
visible_tests: []
hidden_tests: []
setup_command: null
test_command: null
network_policy: deny
time_budget_seconds: null
token_budget: null
max_iterations: null
max_changed_files: null
success_rule: fail_to_pass_and_pass_to_pass
```

Agent 看不到 `hidden_tests`、Gold Patch 与最终标签。任务环境必须固定基础 Commit、依赖版本与镜像摘要。

## G.3 状态机而非无限循环

```text
CREATED
→ INGESTED
→ PLANNED
→ EDITING
→ VERIFYING
├─ PASSED → COMPLETED
├─ RETRYABLE → REPLANNING → EDITING
├─ BUDGET_EXCEEDED → FAILED
└─ SECURITY_BLOCKED → FAILED
```

每次状态转换记录：输入摘要、模型版本、Prompt 版本、工具参数、输出、耗时、Token、文件 Diff、退出码和错误类型。任务必须支持幂等重试、断点恢复和显式取消。

## G.4 工具契约

工具必须使用结构化 Schema，禁止让模型直接拼接任意 Shell 字符串作为默认路径。P0 工具建议：

- `list_files(root, depth, include, exclude)`；
- `read_file(path, start_line, end_line)`；
- `search_text(pattern, paths, max_results)`；
- `find_symbol(name, language)`；
- `apply_patch(patch)`；
- `run_test(test_ids, timeout)`；
- `git_diff()`；
- `inspect_failure(run_id)`。

`run_command` 保留为受策略控制的逃生工具，而不是默认工具；命令须经过 Token 化解析、可执行文件 Allowlist、参数检查、工作目录检查和超时限制。

## G.5 检索先单独评测

不要等端到端 Agent 完成后才判断检索质量。先建立定位数据集，评测：

- File Recall@k；
- Symbol Recall@k；
- MRR；
- 首次找到相关文件的工具调用次数；
- 检索上下文 Token 数；
- Oracle Retrieval 与实际 Retrieval 的差距。

推荐消融顺序：

```text
文件树 + ripgrep
→ + BM25
→ + AST/Symbol
→ + Embedding
→ + Reranker
```

只有新增组件在固定任务集上改善定位质量或以相近质量降低成本时才保留。

## G.6 Benchmark 分层

1. **开发集**：10–20 个极小任务，允许查看全部测试，用于调试工具。
2. **本地验证集**：30–50 个固定 Commit 的任务，含隐藏测试，不用于 Prompt 调参。
3. **外部基准子集**：SWE-bench Lite/Verified 中资源可承受的固定子集，记录实例 ID、Harness 版本与镜像。
4. **压力/安全集**：恶意 README、提示注入、超大输出、无限循环、网络访问、路径逃逸、Secret 诱导。

必须防止：读取 Gold Patch、使用测试补丁作为上下文、在验证集上反复调 Prompt、不同模型使用不同工具或预算却直接比较。

## G.7 Agent 指标

| 类别 | 指标 |
|---|---|
| 任务成功 | Resolve Rate、FAIL_TO_PASS、PASS_TO_PASS、First-attempt Success |
| 修改质量 | Invalid Edit、Unrelated File Change、Patch Size、Regression Rate |
| 过程效率 | Iterations、Tool Calls、Context Tokens、Generated Tokens、Wall Time |
| 恢复能力 | Recovery Rate、重复错误率、超时率、取消成功率 |
| 检索质量 | File/Symbol Recall@k、MRR、Oracle Gap |
| 安全 | Policy Block Rate、逃逸测试通过率、Secret 泄漏率、违规网络请求率 |

成功门槛优先使用“相对基线”而非拍脑袋绝对值：例如在相同模型和预算下，相比 Bash-only/ReAct 基线 Resolve Rate 提高至少 5 个百分点；或在 Resolve Rate 近似相当时，将 Token/延迟降低至少 20%。门槛需在测试前登记。

---

# 附录 H：安全威胁模型与供应链

## H.1 威胁资产与攻击面

RepoPilot 会执行不可信仓库代码，其风险高于普通 RAG。至少考虑：

- 仓库 README、Issue、代码注释中的 Prompt Injection；
- 测试或安装脚本窃取环境变量、SSH Key、云凭据和 API Key；
- 命令注入、路径穿越、符号链接逃逸、挂载逃逸；
- Fork Bomb、磁盘填满、内存耗尽、超长日志、子进程泄漏；
- 恶意依赖、Typosquatting、安装阶段任意代码执行；
- 外网回连、数据外传、下载第二阶段 Payload；
- Agent 修改 CI、依赖锁、发布脚本或安全策略以扩大权限；
- 测试投机：删除测试、弱化断言、硬编码预期输出。

## H.2 P0 沙箱最低标准

- 每个任务使用全新临时工作区或 Git Worktree，任务后销毁；
- 容器内非 Root 用户，Root Filesystem 只读，仅工作目录可写；
- 默认断网，必要网络使用域名/端口 Allowlist；
- 丢弃 Linux Capabilities，启用 seccomp/AppArmor/SELinux 中可用的控制；
- 禁止挂载宿主 Docker Socket、用户主目录、SSH 目录和真实 Secret；
- 限制 CPU、内存、PIDs、磁盘、文件大小、执行时长与输出字节数；
- 对真实路径做 Canonicalize 后再校验，并拒绝越界符号链接；
- 安装依赖与执行测试分阶段授权，记录网络与文件访问；
- 所有变更通过 Patch 应用并检查文件数、路径与 Diff 大小；
- 对高风险文件设置人工批准：CI、部署、权限、依赖、密钥、基础镜像。

容器只是隔离层之一。若要执行真正不可信代码，P1 应评估 gVisor 或微虚拟机，并保留逃逸测试记录。

## H.3 模型与数据安全

- 训练数据扫描 PII、凭据、私钥、访问 Token 与高风险许可证；
- Prompt、Retrieved Context 和 Trace 默认脱敏；
- 不在日志中保存 Authorization Header、完整用户代码或 Secret；
- 评测数据、偏好数据和训练数据建立哈希级污染检查；
- 为数据删除、错误来源修正和模型版本撤回保留流程；
- Agent 输出在执行前经过策略层，模型本身不是安全边界。

## H.4 软件供应链

- 锁定依赖版本与哈希；
- 容器基础镜像固定 Digest；
- 生成 SBOM；
- CI 执行依赖漏洞、Secret 和许可证扫描；
- Release 记录 Commit、构建环境、模型哈希和配置哈希；
- 训练或评测镜像与生产镜像分离；
- 第三方模型、数据集和镜像都记录来源、许可证与校验值。

---

# 附录 I：实验与证据管理

## I.1 最小实验记录

文件路径：`docs/experiments/YYYY-MM-DD_<stage>_<run>.md`。

```markdown
# 实验记录：<run_name>

## 目标与假设
## 环境
- 平台/GPU/驱动/CUDA/PyTorch：
- 代码 Commit 与 Dirty Diff：
- 数据/Tokenizer/模型版本：

## 变量
- 主变量：
- 固定项：
- 基线：

## 精确命令
## 预算与停止条件
## 输出路径
## 结果表
## 失败、异常与处理
## 结论边界
## 下一步决策
```

## I.2 Artifact 命名

```text
<project>/<stage>/<run_id>/
├── config.resolved.yaml
├── environment.txt
├── git.json
├── data_manifest.json
├── metrics.jsonl
├── samples.jsonl
├── checkpoint/
├── profile/
├── report.md
└── decision.md
```

`run_id` 建议包含：平台、模块、变体、模型规模、序列长度、Batch 和日期。任何表格数字都能反向定位到一个 `run_id`。

## I.3 四类卡片

- **Data Card**：来源、许可证、采样、过滤、PII、污染、局限；
- **Model Card**：架构、训练数据、训练预算、评测、安全、适用与禁用场景；
- **Service Card**：模型/后端/硬件、SLO、限额、故障与回滚；
- **Agent Card**：模型、Scaffold、工具、Prompt 版本、沙箱、Benchmark、预算和已知失败模式。

## I.4 决策记录

对以下选择写 ADR：

- 自研训练循环还是 Trainer；
- DDP 还是 FSDP2；
- MLflow Tracing、Langfuse、Phoenix 三选一；
- KServe 还是 Ray Serve；
- BM25/AST/Embedding/Reranker 的保留与删除；
- gVisor/微虚拟机是否值得引入。

ADR 至少包含背景、候选方案、决策、理由、代价和复审条件。

## I.5 Claim 边界

| 证据 | 可以写 | 不应写 |
|---|---|---|
| 单次 Smoke Test | “跑通/验证接口” | “稳定提升/可扩展” |
| 单卡吞吐提升 | “在该硬件与配置下提升” | “普遍加速” |
| 2 卡效率 | “2 卡设置下扩展效率为…” | “大规模分布式训练能力已验证” |
| 自建 30 个 Agent 任务 | “在该固定任务集解决 X/Y” | “达到工业级代码 Agent 水平” |
| 调用 FlashAttention/vLLM | “集成并评测” | “实现其底层 Kernel/引擎” |

---

# 附录 J：24 周执行路线

以下路线按每周约 25–35 小时估算；若每周 10–15 小时，应扩展为约 36–40 周，而不是压缩验收步骤。

| 周期 | 核心任务 | 可发布结果 | 退出条件 |
|---|---|---|---|
| W1–2 | 工程骨架、资源卡、CI、容器、实验模板 | `v0.1-engineering` | 新环境可复现 CPU Smoke |
| W3–5 | 单一数据源、Data Card、BPE、正式 Tokenizer | `v0.2-data-tokenizer` | G1 通过 |
| W6–8 | 5M–20M Transformer、数值/因果/梯度测试 | `v0.3-model-core` | G2 通过 |
| W9–10 | 单卡训练、Checkpoint/Resume、基础评测 | `v0.4-pretrain-mvp` | G3 通过 |
| W11–12 | SDPA/Flash/AC Profile；有多卡则 DDP/FSDP2 | `v0.5-training-systems` | 性能表与根因报告完成 |
| W13–15 | 0.5B–1.5B 开源模型 SFT/LoRA；CPT 视资源 | `v0.6-post-train` | 前后评测与 Model Card 完成 |
| W16 | DPO 决策周：满足条件才做，否则补质量/评测 | `v0.7-alignment` 或负结果报告 | 有明确保留/停止结论 |
| W17–18 | vLLM、FastAPI、Compose、负载测试、量化可选 | `v0.8-serving` | API Contract 与报告完成 |
| W19 | RepoPilot Task Spec、沙箱与简单 Bash/ReAct 基线 | `repopilot-v0.1` | 可安全执行开发集 |
| W20–21 | ripgrep/BM25/Symbol 检索、单独检索评测 | `repopilot-v0.2` | 检索消融完成 |
| W22–23 | Planner–Executor–Verifier、隐藏测试、固定验证集 | `repopilot-v0.3` | Agent Card 与成功率报告完成 |
| W24 | 两项目 API 集成、Demo、Release、简历与技术文章 | `portfolio-v1.0` | 陌生环境复现和证据审计通过 |

## J.1 每周工作分配

```text
60%：实现与调试
20%：测试、评测和实验
10%：阅读官方实现/文档
10%：README、报告、图表和复现
```

文档不能全部留到最后；每个 Release 必须同时提交代码、测试、结果和说明。

## J.2 求职启动点

- W10 后：可投递初级 LLM 训练/算法工程岗位，但只能描述已完成的小模型闭环。
- W15 后：可重点投递 LLM Fine-tuning、训练平台、MLE 岗位。
- W18 后：可重点投递推理服务、LLM Systems、AI Backend 岗位。
- W23 后：可重点投递 Agent、Applied AI、Coding Agent 岗位。

---

# 附录 K：求职交付与简历证据

## K.1 两个仓库而非一个巨型仓库

推荐保持两个独立仓库：

```text
ForgeLLM
  模型、训练、评测、推理服务、模型卡

RepoPilot
  索引、工具、沙箱、Agent、Benchmark、Agent Card
```

通过版本化 OpenAI-compatible Contract 连接，不共享内部 Python 模块。集成 Demo 可放在 RepoPilot 的 `deploy/compose`，并固定 ForgeLLM 服务镜像版本。这样既能独立展示，也能证明系统集成。

## K.2 每个仓库首页必须回答

1. 解决什么问题；
2. 与常见 Tutorial 的差异；
3. 一张最小架构图；
4. 三条可量化结果；
5. 五分钟 Quickstart；
6. 完整复现命令；
7. 当前限制和未完成项；
8. 安全说明与许可证；
9. Release 与演示链接；
10. 结果对应的 Commit、配置和硬件。

## K.3 简历句式模板

先填数字再写句子，禁止提前写未来成绩：

> 在 `<GPU/精度/序列长度/并发>` 条件下，对 `<基线>` 与 `<方法>` 进行受控评测，使 `<指标>` 从 `<A>` 变为 `<B>`，同时将 `<代价指标>` 控制在 `<C>`；结果可由 `<run_id/release>` 复现。

RepoPilot 示例模板：

> 在固定 `<N>` 个隐藏测试任务和相同模型/工具预算下，相比 `<基线 Agent>` 将 Resolve Rate 从 `<A>` 提升至 `<B>`，并报告 `<Token/延迟/迭代/回归>`；通过默认断网的受限沙箱执行测试。

## K.4 面试证据包

- 5 分钟 Demo；
- 10 分钟架构说明；
- 一次失败实验及如何定位；
- 一张训练/推理性能表；
- 一张 Agent 消融表；
- 一个安全案例；
- 一个可现场运行的 CPU/小 GPU Smoke Test；
- 一份明确说明“哪些是自研、哪些是集成”的清单。

---

# 附录 L：最终 Definition of Done

两个项目只有同时满足以下标准，才算“求职版完成”；P2 项目不影响完成判定。

## L.1 ForgeLLM

- [ ] 数据来源、许可证、清洗、污染和版本可追踪；
- [ ] BPE、Tokenizer、Transformer、Causal Mask、KV Cache 有测试；
- [ ] 5M–20M 模型完成训练、评测、Checkpoint 和 Resume；
- [ ] 至少一份受控性能对比含硬件、精度、Batch、长度和重复次数；
- [ ] 至少一种开源 Base 模型完成 SFT/LoRA 前后对比；
- [ ] 关键结论有基线、Run ID 和失败记录；
- [ ] vLLM/FastAPI 服务通过 Contract、流式、超时、限流和负载测试；
- [ ] Docker Compose 可一条命令启动最小系统；
- [ ] README、Data Card、Model Card、Release 和 Demo 完整；
- [ ] 简历中的每个数字均可追溯。

## L.2 RepoPilot

- [ ] 在固定 Commit 的 Python 仓库任务上完成修改—测试—反馈闭环；
- [ ] 有简单 Agent 基线，且模型、工具、预算和 Benchmark 固定；
- [ ] 检索单独评测并完成至少三组受控消融；
- [ ] 验证集含 Agent 不可见的隐藏测试；
- [ ] 报告成功数/总数、区间、Token、延迟、迭代和回归；
- [ ] 状态机支持超时、取消、预算耗尽和断点恢复；
- [ ] 沙箱默认断网、非 Root、资源受限、无真实 Secret；
- [ ] 路径穿越、符号链接、恶意依赖、Fork Bomb 和 Prompt Injection 测试有记录；
- [ ] 能切换强模型、本地开源模型和 ForgeLLM 模型而不修改 Agent 核心；
- [ ] README、Agent Card、Benchmark Manifest、Release 和 Demo 完整。

## L.3 2026 年官方参考校验

以下链接用于校验增强版中的关键工程选择，执行时应固定实际使用版本：

- PyTorch Distributed Checkpoint：<https://docs.pytorch.org/docs/stable/distributed.checkpoint.html>
- SWE-bench 数据集与版本：<https://www.swebench.com/SWE-bench/guides/datasets/>
- SWE-bench Verified：<https://www.swebench.com/verified.html>
- OpenTelemetry Semantic Conventions：<https://opentelemetry.io/docs/specs/semconv/>
- gVisor 不可信代码隔离：<https://gvisor.dev/>

最终目标不是把清单全部打勾，而是让每个已打勾的能力都有可复现证据，并让任何未做的高级能力都有清晰、诚实的边界说明。
