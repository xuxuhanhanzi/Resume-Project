# ForgeMM 候选方向与推荐

**决策状态**：未确认。本文只用于方向比较，不是实施计划、系统设计或 Stage 规划。  
**材料依据**：配套清单中的 80 篇论文/报告与 40 个核心 GitHub 项目。

## 1. 加权评分

评分采用用户指定权重，并直接换算为 100 分：学习者可理解性 20、研究问题清晰度 20、可复现与可评测性 20、GitHub 完整度 15、简历价值 15、资源可行性 10。总分用于辅助判断，不替代证据边界和风险分析。

| 候选方向 | 可理解性/20 | 问题清晰/20 | 复现评测/20 | GitHub/15 | 简历/15 | 资源/10 | 总分 | 结论 |
|---|---|---|---|---|---|---|---|---|
| ForgeMM-ChartEvidence | 18 | 19 | 19 | 13 | 15 | 8 | 92 | 推荐。问题最清晰、可证伪、成本指标明确，同时能自然教学 VLM 架构、PEFT、grounding 与评测。 |
| ForgeMM-DocSelective | 19 | 17 | 18 | 15 | 12 | 9 | 90 | 保留为最稳妥备选，但“选择哪个模块做 LoRA”本身的研究新颖性弱于候选一。 |
| ForgeMM-TokenBudget | 16 | 18 | 18 | 13 | 15 | 8 | 88 | 保留为第二研究型候选；风险是工程测量较重且容易与候选一重叠，必须独立选题。 |

## 候选 1：ForgeMM-ChartEvidence

1. **一句话目标**：在固定视觉 Token 预算下，研究问题条件的图表区域选择是否同时提升答案正确率与证据定位质量。
2. **具体问题**：小型 VLM 在图表题中会把 OCR、数值运算和证据选择混为一体；答案偶然正确时普通 relaxed accuracy 无法判断其是否读取了正确区域。
3. **为什么适合 ForgeLLM**：可复用 ForgeLLM 的 SFT、LoRA/QLoRA、评测和实验记录；新增视觉编码、region selector 与证据指标，学习闭环清楚。
4. **主要文献依据**：ChartQAPro、LongChart VQA、Where Vision Becomes Text、Ferret、VScan、SparseVLM、Matryoshka Multimodal Models。
5. **可复用开源项目**：Qwen3-VL/Transformers、TinyLLaVA Factory、ChartQAPro、lmms-eval/VLMEvalKit、PEFT。
6. **最小模型和数据集**：主模型族：Qwen2.5/3-VL 的 3B 或更小版本；主数据集：ChartQA；外部验证：ChartQAPro。
7. **唯一主创新变量**：唯一变量：query-conditioned region/token selection；其余模型、训练 token、视觉 token 预算和解码固定。
8. **官方或公认 Baseline**：同一模型的 full-image / 无 selector 官方输入流程；另设随机/均匀裁剪同预算对照。
9. **关键指标**：ChartQA relaxed accuracy；证据区域 IoU/F1；视觉 token 数；TTFT/延迟；峰值 VRAM。
10. **新增核心模块**：ChartQA adapter、区域/patch 映射、预算约束 selector、证据标注与评测、资源 profiler。
11. **学习难度**：中高
12. **工程难度**：中高
13. **资源等级**：8GB：仅推理/极小 QLoRA；24GB：主实验；48/80GB：更高分辨率；多卡：非必要。
14. **支持假设的结果**：同预算下答案与证据指标稳定提升，且随机/遮挡/错题干预显示模型更依赖正确区域。
15. **否定假设的结果**：答案不升或证据指标下降；收益只来自更多有效像素、不同分辨率或额外训练量。
16. **与 RepoPilot 重叠**：低。只做静态图表视觉建模与评测；不做检索、工具调用和 Agent loop。
17. **选择或淘汰理由**：推荐。问题最清晰、可证伪、成本指标明确，同时能自然教学 VLM 架构、PEFT、grounding 与评测。


## 候选 2：ForgeMM-DocSelective

1. **一句话目标**：在单页文档字段抽取中，比较 projector-only、LLM LoRA 与 vision last-k 的等预算选择性 PEFT。
2. **具体问题**：低资源微调时并不清楚性能提升来自视觉适配、语言格式学习还是 projector 对齐；全量 QLoRA 容易掩盖模块贡献。
3. **为什么适合 ForgeLLM**：直接复用 ForgeLLM 的 LoRA/QLoRA/SFT 与消融框架，是最稳妥的多模态入门主线。
4. **主要文献依据**：DoRA、Multimodal OCR、TextMonkey、DocLLM、UReader、Modality-Inconsistent Continual Learning。
5. **可复用开源项目**：PEFT、Transformers、LlamaFactory/ms-swift 作为外部 baseline、PaddleOCR/dots.ocr 作为预处理或对照。
6. **最小模型和数据集**：主模型族：小型 Qwen-VL/MiniCPM-V/TinyLLaVA；主数据集：DocVQA 单页子集；外部验证：CORD 或 FUNSD。
7. **唯一主创新变量**：唯一变量：可训练组件集合；所有方案固定可训练参数量或优化步数。
8. **官方或公认 Baseline**：官方模型零样本与统一 full-LLM LoRA baseline。
9. **关键指标**：ANLS/字段 F1；目标外 VQA 保持率；可训练参数；峰值 VRAM；训练时间。
10. **新增核心模块**：文档数据 adapter、冻结/解冻策略、等预算 LoRA 配置、目标外遗忘评测。
11. **学习难度**：中
12. **工程难度**：中
13. **资源等级**：8GB：小模型 4-bit；24GB：完整等预算消融；48/80GB：非必要；多卡：非必要。
14. **支持假设的结果**：某一组件策略在相同参数/计算预算下显著优于全局 LoRA，并减少目标外能力下降。
15. **否定假设的结果**：策略差异被 seed 覆盖，或提升只来自更多参数/更长训练。
16. **与 RepoPilot 重叠**：低—中。只做模型适配；不做文档检索/RAG 编排。
17. **选择或淘汰理由**：保留为最稳妥备选，但“选择哪个模块做 LoRA”本身的研究新颖性弱于候选一。


## 候选 3：ForgeMM-TokenBudget

1. **一句话目标**：在固定视觉 Token 预算下比较全局压缩、问题无关选择与问题条件选择的 accuracy–cost Pareto。
2. **具体问题**：高分辨率 VLM 的视觉 token 成本高，但训练自由剪枝可能删除小字、坐标或图表关键区域。
3. **为什么适合 ForgeLLM**：可复用 ForgeLLM 的注意力、mask、KV cache 和评测知识，并增加明确的系统成本测量。
4. **主要文献依据**：VScan、SparseVLM、VisionZip、FastV、VisionTrim、Why and When Visual Token Pruning Fails、Scaling Capability in Token Space。
5. **可复用开源项目**：VisionTrim、LLaVA-NeXT/Qwen3-VL、vLLM/SGLang、lmms-eval。
6. **最小模型和数据集**：主模型族：LLaVA-NeXT 或 Qwen-VL 小型号；主数据集：TextVQA；外部验证：ChartQA。
7. **唯一主创新变量**：唯一变量：token selection policy；预算、编码器、LLM 与分辨率固定。
8. **官方或公认 Baseline**：官方 full-token 推理与均匀池化/随机剪枝同预算 baseline。
9. **关键指标**：任务准确率；视觉 token 数；TTFT；峰值 VRAM；FLOPs/吞吐。
10. **新增核心模块**：视觉 token hook、selector、预算控制、缓存/延迟 profiler、Pareto 报告。
11. **学习难度**：中高
12. **工程难度**：高
13. **资源等级**：8GB：训练自由推理；24GB：轻量 selector 训练；48/80GB：高分辨率；多卡：非必要。
14. **支持假设的结果**：同预算下 query-conditioned 选择在至少两个任务上形成稳定 Pareto 优势。
15. **否定假设的结果**：收益仅在单一模型/数据集出现，或实际 GPU 延迟无改善。
16. **与 RepoPilot 重叠**：低。属于模型推理效率，不涉及 RepoPilot 编排。
17. **选择或淘汰理由**：保留为第二研究型候选；风险是工程测量较重且容易与候选一重叠，必须独立选题。


## 推荐方向（等待用户确认）

- **项目暂定名称**：ForgeMM-ChartEvidence
- **一句话目标**：在固定视觉 Token 预算下，研究问题条件的图表区域选择是否提升 ChartQA 的答案与证据定位表现。
- **唯一主任务**：带证据区域定位的图表问答。
- **唯一主数据集**：ChartQA。
- **一个外部验证集**：ChartQAPro。
- **一个主模型族**：Qwen2.5/3-VL 小模型族；具体参数版本在用户确认方向后再按硬件决定。
- **一个主创新变量**：query-conditioned region/token selection。
- **一个官方/公认 Baseline**：同一模型的官方 full-image 输入流程，不加入 selector；附加同预算随机/均匀裁剪对照。
- **核心指标（不超过五个）**：ChartQA relaxed accuracy、证据区域 IoU/F1、视觉 token 数、TTFT/端到端延迟、峰值 VRAM。
- **P0 仅包含**：数据与评测 adapter、full-image baseline、固定预算 selector、证据定位指标、资源 profiler、必要的单变量消融。
- **明确不做**：多模态 GRPO/RLVR、DPO、OCR 预训练、多页文档、视频、GUI Agent、RAG、MoE、多模型族竞赛、生产部署。
- **为什么推荐**：它把“多模态基础架构—PEFT—视觉 token—grounding—可信评测”连接成一个可理解且可证伪的问题；与 RepoPilot 重叠低；即使 selector 无效，也能通过答案、证据和成本三类结果明确否定假设，而不会退化成只展示 loss 曲线的项目。


## 仍未解决的关键问题

1. 用户实际可长期使用的 GPU 是约 8GB、24GB，还是更高；这会影响模型参数版本，但不改变推荐问题定义。
2. ChartQA 是否提供足以直接构造 region/cell evidence 的原始结构；若不足，需要只补充最小证据标注，而不能借机扩展成大型合成数据工程。
3. Qwen2.5-VL 与 Qwen3-VL 在选定小参数版本上的训练许可、Transformers 支持和视觉 token hook 是否稳定。
4. “证据定位提升”应采用 patch/region IoU、pointing 命中率还是由图表源数据映射出的 cell/series F1；需在方向确认后只选一种主证据定义。
5. 固定视觉 token 预算应取单一主预算还是 2–3 个 Pareto 点；为避免项目膨胀，P0 默认只设一个主预算和 full-token 上界。

> **强制停止门**：当前推荐方向尚未得到用户确认，因此没有继续生成项目实施计划。
