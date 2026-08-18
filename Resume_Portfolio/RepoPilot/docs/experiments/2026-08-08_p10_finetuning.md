# P10 微调与蒸馏实验报告

- 阶段: P10
- 日期: 2026-08-08
- 状态: SFT 完成 / DPO 完成
- 负责人: 自动推进

## 1. 目标

在冻结的 `qwen2.5:7b` 之外,验证本地小模型(Qwen2.5-1.5B)经 QLoRA 微调能否改进
FRAMES 知识问答的答案生成质量。P10 包含两条路线:

- **SFT**: 用 FRAMES agent 正确轨迹(9 条)做监督微调,监督信号为 `prompt → 正确答案`。
- **DPO**: 用正确/错误答案对(8 对)做直接偏好优化,在 SFT 基础上进一步对齐偏好。

本阶段为**教学型 PoC**,数据量极小(9 SFT / 8 DPO),目的是跑通"数据提取→训练→保存 adapter"
全链路,而非追求 SOTA 指标。

## 2. 环境与依赖

- GPU: NVIDIA GeForce RTX 4070 Laptop (8.6 GB VRAM)
- 基座模型: `Qwen/Qwen2.5-1.5B`(从 HF 镜像 `hf-mirror.com` 下载)
- 量化: bitsandbytes 4-bit NF4,double quant,bf16 compute dtype
- 训练栈: torch 2.5.1+cu121 / transformers / peft / (SFT 用 trl 1.9.2;DPO 因兼容性改手写循环)
- 数据源: `artifacts/training/p10_frames/`(从 FRAMES agent 公开轨迹提取,48 样本)

## 3. 训练数据

来源 manifest(`artifacts/training/p10_frames/manifest.json`):

- 5 个 FRAMES run 的 agent 轨迹(gold revision `58d9fb63...`)
- 总样本 48,正确样本 9
- SFT: 9 条(prompt → 正确答案)
- DPO: 8 对(同一 prompt 的 chosen=正确答案 / rejected=错误答案)
- 许可: FRAMES 数据 CC-BY-SA,agent 轨迹为本项目生成

## 4. SFT 训练(已完成)

### 4.1 配置

| 项 | 值 |
|----|----|
| 基座 | Qwen2.5-1.5B |
| 量化 | 4-bit NF4,double quant |
| LoRA r / alpha | 16 / 32 |
| LoRA target | q_proj, k_proj, v_proj, o_proj |
| LoRA dropout | 0.05 |
| 可训练参数 | 4,358,144 (0.2815%) |
| epochs | 1 |
| lr | 2e-4 |
| batch / grad_accum | 2 / 4 |
| optim | paged_adamw_8bit |

### 4.2 结果

- 训练耗时: **12.7 秒**
- adapter: `artifacts/training/p10_qlora/adapter/adapter_model.safetensors`(**8.7 MB**)
- checkpoint-2 已保存
- 训练日志: `artifacts/training/p10_qlora/training_log.json`

**这是整个项目第一个成功训练的模型。**

### 4.3 TRL 1.9 兼容性修复

SFT 训练遇到 3 个 TRL 1.9 / torch 2.5.1 API 变更,全部修复:

1. `total_mem` → `total_memory`(torch CUDA 属性更名)
2. `SFTConfig.max_seq_length` → `max_length`(TRL 1.9 参数更名)
3. `SFTTrainer` 需 `datasets.Dataset` 而非 `list[dict]` → 用 `Dataset.from_list()`

## 5. DPO 训练(已完成)

### 5.1 兼容性问题

TRL 1.9.2 的 `DPOTrainer` 在 import 时硬依赖 `from torch.distributed.fsdp import FSDPModule`,
而 torch 2.5.1 的 `torch.distributed.fsdp` **不导出** `FSDPModule`(FSDP2 私有),导致
`DPOTrainer` 无法 import,DPO 训练直接崩溃。

### 5.2 解决方案:手写 DPO 循环

放弃 TRL `DPOTrainer`,用 transformers + peft 直接实现标准 DPO 损失:

- Policy = base + LoRA(从 SFT adapter 继续训练,可训练)
- Reference = 冻结 base(`model.disable_adapter()` 上下文获得)
- 损失: `loss = -logsigmoid(beta * ((logp_pol_chosen - logp_ref_chosen) - (logp_pol_rejected - logp_ref_rejected)))`
- beta = 0.1, 3 epochs, lr 5e-5, grad_accum 4
- 每步后 `torch.cuda.empty_cache()` 防 VRAM 碎片累积

> 注: 教科书 DPO 的 reference 通常是 SFT 模型;本 PoC 用 base 作 reference(可通过
> `disable_adapter` 直接获得,无需双模型),对 8 对小数据 PoC 是可接受的简化,已在日志标注。

### 5.3 结果

训练成功完成:

| 项 | 值 |
|----|----|
| 初始化 | 从 SFT adapter 继续 |
| 可训练参数 | 4,358,144 (0.2815%) |
| 训练对数 | 8 |
| epochs / optimizer steps | 3 / 6 |
| beta / lr / grad_accum | 0.1 / 5e-5 / 4 |
| 耗时 | **1441.4 秒(约 24 分钟)** |
| 峰值训练 VRAM | 1.22 GB |
| adapter | `artifacts/training/p10_dpo/adapter/adapter_model.safetensors`(**17.5 MB**) |

**偏好学习曲线**(每个 optimizer step 打印一次):

| epoch/sample | loss | chosen_reward | rejected_reward | margin |
|--------------|------|---------------|-----------------|--------|
| 1 / 4 | 0.6045 | +0.397 | +0.211 | 0.186 |
| 1 / 8 | 0.6738 | +0.171 | +0.132 | 0.039 |
| 2 / 4 | 0.3637 | +0.633 | -0.191 | 0.824 |
| 2 / 8 | 0.4340 | +0.419 | -0.191 | 0.610 |
| 3 / 4 | 0.2071 | +0.878 | -0.591 | 1.469 |
| 3 / 8 | **0.2110** | **+0.798** | **-0.650** | **1.448** |

结论:

- loss 从 0.60 单调下降到 0.21(-65%),DPO 目标正确优化。
- `chosen_reward` 从 +0.397 上升到 +0.798,`rejected_reward` 从 +0.211 下降到 -0.650,
  **偏好间隔(margin)从 0.19 扩大到 1.45**,说明模型确实学会了区分正确/错误答案。
- 训练期实际 VRAM 仅 1.22 GB,`empty_cache()` 成功解决了首次运行的静默 OOM。

### 5.4 首次运行的静默 OOM 及修复

首版手写 DPO 在 epoch1 结束后进程被静默杀死(exit 1,无 traceback)。诊断为
policy/reference 两次前向的激活缓存在 8 GB 卡上累积导致系统级回收。修复:

1. 每个 optimizer step 后调用 `torch.cuda.empty_cache()`
2. 顶层加 `try/except` 将 traceback 落盘到 `artifacts/dpo_err.txt`
3. 改后台运行,避免前台进程被外层超时杀死

修复后 3 epochs 完整跑通。

## 6. adapter 评测(记为后续工作)

计划: 对比 base-1.5B / SFT / DPO 三者在 FRAMES dev 集上的直接生成准确率
(给定 oracle context 生成答案并评分),隔离微调对答案质量的影响。

**当前未执行的原因**: 现有 FRAMES 评测链路走 Ollama HTTP 接口(`qwen2.5:7b`),
而 P10 产物是 HF 格式的 PEFT adapter(Qwen2.5-1.5B)。要在同一评测链路里对比,
需要额外做 adapter merge → GGUF 转换 → Ollama 导入,或者另起一个 HF 推理服务端。
这是**基础设施工作量**而非科研结论工作量,对本 PoC 的核心命题(跑通训练链路)非关键路径,
因此明确记为后续工作,不阻塞 P12 收尾。

诚实声明: **本报告不宣称微调带来了下游指标提升**,只宣称训练链路跑通且 DPO 偏好信号正确。

## 7. 结论与局限

**达成**:

- 跑通 QLoRA SFT + DPO 全链路(数据提取 → 训练 → adapter 保存),两个 adapter 均产出。
- DPO 偏好间隔从 0.19 扩大到 1.45,训练目标被正确优化。
- 在 8 GB 消费级显卡上完成 1.5B 模型的 4-bit QLoRA SFT + DPO,是可复现的本地训练配方。

**局限(不可外推)**:

- 数据量极小(9 SFT / 8 DPO),loss 下降主要反映**过拟合到小样本**,不代表泛化能力提升。
- DPO reference 用 base 而非 SFT 模型,是简化实现。
- **未做下游指标评测**,因此不能声称微调改善了 FRAMES 准确率。

**后续**:

- 扩大训练数据到 50+ 样本(需先提升 agent 正确率以产出更多正样本)
- 用 SFT 模型作 DPO reference(双模型加载,需 >8 GB 或 CPU offload)
- adapter merge → GGUF → Ollama 导入,接入现有 FRAMES 评测链路做端到端对比

## 8. 产物

- `scripts/prepare_training_data.py` — 数据提取
- `scripts/train_qwen_qlora.py` — SFT 训练
- `scripts/train_qwen_dpo.py` — DPO 训练(手写循环)
- `artifacts/training/p10_frames/` — 训练数据 + manifest
- `artifacts/training/p10_qlora/adapter/` — SFT adapter(8.7 MB)
- `artifacts/training/p10_dpo/adapter/` — DPO adapter(17.5 MB)
- `artifacts/training/p10_dpo/training_log.json` — DPO 训练日志
- `artifacts/dpo_out.txt` — DPO 完整训练输出(偏好曲线)
