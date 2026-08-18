# Stage 4 完整讲义四：QLoRA、显存账本与实验结论

## 1. 量化先区分 storage dtype 与 compute dtype

4-bit 表示的主要目标是压缩 Base 权重存储。矩阵乘时通常按 block 读取量化码和 scale，反量化到 BF16/FP16 等计算精度参与运算。不能把“权重存 4-bit”理解为“所有激活、梯度、优化器都只占 4-bit”。

```text
Base storage:  NF4 codes + quantization metadata，冻结
compute:       反量化后以 BF16 参与矩阵乘
activations:   BF16/部分 FP32，反向需要保存或重算
LoRA weights:  可训练较高精度参数
LoRA grads:    较高精度
optimizer:     Adapter 的 Adam 状态，不是整个 Base 的状态
```

**思考题：QLoRA 的 Q 修饰谁？**

答案：主要修饰冻结 Base 的权重表示；LoRA Adapter 仍用较高精度训练。若把 Adapter 也直接做不可微 4-bit 离散更新，那是另一个算法问题。

## 2. 线性 int4 与 NF4 的区别

普通对称线性量化把一个 block 的权重除以 scale，再映射到等间隔整数码。NF4 的 codebook 不是等间隔，它针对预训练权重近似正态分布设计，让概率质量更合理地分配到 16 个值。

block-wise 意味着每个小块有自己的 scale，能适应不同局部范围，但需要额外 metadata。double quantization 再压缩这些 scale，减少 scale 本身的平均存储成本。它不让每个权重真的只占严格 4.000 bit；还要计算 codebook、scale、量化状态和对齐开销。

**思考题：NF4 一定比线性 int4 准确吗？**

答案：不保证。它依赖权重分布、block size、异常值和任务。QLoRA 论文经验支持该设计，但具体模型仍应测量误差和下游行为。

## 3. 冻结量化 Base 为什么仍能反传

考虑：

```text
y = dequant(W_q) x + B(Ax)
```

`W_q` 不需要梯度，但为了算 `∂L/∂A`、`∂L/∂B` 以及向更早层传播 `∂L/∂x`，autograd 仍可穿过使用冻结权重的线性运算。常量可以参与可微函数；“参数不求梯度”不等于“计算图在此截断”。

`prepare_model_for_kbit_training` 会冻结 Base，并处理部分精度和 gradient checkpointing 边界。本项目随后用 PEFT 注入 all-linear LoRA，每一步验证所有非 `lora_` 参数没有 `.grad`。

**思考题：Base 没有梯度，前面层的 LoRA 怎么获得梯度？**

答案：反向需要冻结层对输入的 Jacobian，而不是需要对冻结权重求导。常量权重仍把上游梯度乘回输入方向，使更早层 Adapter 获得梯度。

## 4. Gradient checkpointing 的真实交换

普通反向保存中间激活；checkpointing 只保存部分边界，反向时重做前向，降低激活显存、增加计算时间。它与 4-bit 权重压缩解决不同内存项：

- 量化主要降 Base weight storage；
- checkpointing 主要降 activation storage；
- LoRA 主要降 trainable grads/optimizer state。

本项目对 QLoRA 显式固定 `use_reentrant=False`，避免依赖 PyTorch 将变化的默认行为。BF16 LoRA 也启用模型 gradient checkpointing，因此显存与吞吐比较必须阅读实际配置，而不能只看“LoRA/QLoRA”名称。

**思考题：显存足够时为什么可能关闭 checkpointing？**

答案：可以减少重计算并提高吞吐。显存与速度是交换，不存在无条件免费优化。

## 5. 为什么本机 QLoRA 只省了少量 measured peak

2-step 同协议烟雾结果：

| 路径 | load peak allocated | train peak allocated | 吞吐 |
|---|---:|---:|---:|
| BF16 LoRA（20-step） | 1,232,928,256 B | 1,939,151,360 B | 272.40 assistant tok/s |
| NF4 QLoRA（20-step） | 1,161,793,536 B | 1,929,152,000 B | 203.78 assistant tok/s |

QLoRA 的 20-step measured allocated peak 只小 9,999,360 bytes（约 9.5 MiB），吞吐慢约 25%。这说明峰值还受激活、临时 buffer、FP32 模块和 allocator 影响。报告中的 `quantized_parameters=220,200,960` 是被识别为 `Params4bit` 的参数元素数，不是字节数。

这不推翻 4-bit 权重压缩；它说明在 0.6B、8GB GPU、特定 Transformers/bitsandbytes、包含 embeddings/head 与 Adapter/activation 的进程里，Base 权重不是唯一主导项，而且 `max_memory_allocated` 未覆盖所有外部分配。规模越小，固定开销占比越高。

**思考题：是否满足计划中“QLoRA peak 必须低于 BF16”的门？**

答案：20-step 同协议 measured peak 确实略低，所以兼容性门通过；但差值远小于理论 Base 权重差，不能宣传显著节省。

## 6. 三层 Qwen 实验证据

### 6.1 BF16 LoRA smoke

2 optimizer steps、699 assistant tokens，验证：模型加载、ChatML labels、PEFT no-op、forward/backward、Base freeze、Adapter save 和前后评估全部能跑。

### 6.2 QLoRA smoke/memory lab

真实 Windows bitsandbytes NF4 加载并完成 backward。合规 v2 20 步处理 14,864 tokens，validation loss `1.6801→1.2970`，但 4 条 generation constraint 仍 0%。它验证机制和资源，不是质量实验。

### 6.3 BF16 LoRA bounded run

唯一正式配置：`r16/alpha32/dropout0.05/all-linear`、LR `2e-4`、max length 512、micro batch 1、accumulation 8；100k tokens/500 steps/45 min 先到即停。实际在 129 步、100,293 tokens 时停止。

**思考题：为什么正式 run 选 BF16 LoRA 而不是更“先进”的 QLoRA？**

答案：0.6B 在 8GB GPU 上 BF16 已可稳定运行，语义更直接、吞吐更高。QLoRA 作为资源方法单独验证即可；方法更新不自动等于当前资源下更好的 baseline。

## 7. 如何读正式结果而不越界

正向证据：

- held-out assistant loss 降 27.5%；
- correctness teacher-forced loss 降 52.0%；
- Adapter-only 训练稳定，Base 无梯度；
- 固定 token 上限和 provenance 完整执行。

负向证据：

- 16/16 严格生成全部失败；
- retention loss 上升约 3.1%；
- 输出常继续解释而不停止，并出现异常重复；
- 字符级重复率审计显著恶化；
- 单 seed、小数据、短训练，不能外推通用能力。

因此阶段完成的依据是“算法、实现、实验和失败分析闭环”，不是“得到完美 tokenizer/model”。这与用户强调的学习目标一致。

**思考题：0% exact match 是否说明 SFT 完全没用？**

答案：也不能。loss 与答案开头显示概率方向改变，但停止/完整行为不足。准确说法是严格任务门未通过，而非所有学习信号为零。

## 8. 当前开源模型技术如何映射到 Stage 4

现代开源模型常使用多阶段后训练：高质量 SFT 或冷启动数据建立格式/可读性，再用偏好优化或可验证 RL 改进行为。DeepSeek-R1 报告强调纯 RL 的 R1-Zero 会遇到可读性和语言混合问题，R1 因此加入 cold-start 与多阶段训练；这说明 SFT 不是“过时步骤”，而是为后续 RL 提供稳定行为起点。Stage 4 只实现这一层，不把后续 RL 成果归功于 SFT。

LoRA/QLoRA 是更新与资源策略，不是新的训练目标。模型仍在优化 assistant-only CE；它们不会自动修复数据质量、reward hacking、停止行为或评测设计。PEFT 能让实验在消费级 GPU 上成立，却不能替代目标函数和数据分析。

截至冻结日，DeepSeek V4 和 Kimi K3 若没有可核验官方技术报告，只能登记为待观察名称；不能为满足“最新”而虚构训练路线。Stage 5 会以已公开的 DPO、DeepSeek-R1/GRPO、DAPO、Dr. GRPO、Qwen GSPO 等一手材料为主。

**思考题：看到新模型使用 QLoRA，能否说明 QLoRA 提升了模型能力？**

答案：不能。可能只是使同一训练在较少显存下可执行。要证明能力增益，需固定数据、目标、预算和评测，与非量化 LoRA 做公平对照。

## 9. Stage 4 的完成与未完成

自动化 G4-A 至 G4-E 已完成：数据、模板、loss、手写 LoRA、PEFT、QLoRA、正式前后评测和失败审计都有证据。Stage 4 仍不包含：

- Reward Model、DPO、PPO、GRPO/DAPO/GSPO；
- 生产级数据治理或安全对齐；
- 大规模聊天质量 benchmark；
- Adapter 训练中断恢复；
- serving/merge 后部署性能。

这些边界不是缺陷掩盖，而是防止阶段无限扩张。偏好与在线 RL 进入 Stage 5，综合验收进入 Stage 6。

**思考题：什么时候 Stage 4 才算全部完成？**

答案：自动化实现现已完成；还需要学习者按 16 站完成讲义、代码追踪、最小练习和口述验收，由用户确认 G4-L。不能因为实验报告存在就代替学习完成。

## 10. 本讲代码与 Artifact 追踪

1. `BitsAndBytesConfig` 的 NF4/double-quant/BF16 设置
2. `prepare_model_for_kbit_training`
3. `get_peft_model`
4. `adapter_base_gradients_are_absent`
5. `configs/post_training/stage4_qwen_qlora_memory.toml`
6. `artifacts/stage04/qwen_qlora_memory_v1/report.json`
7. `configs/post_training/stage4_qwen_lora_bounded.toml`
8. `artifacts/stage04/qwen_lora_bounded_v2/report.json`
9. `artifacts/stage04/qwen_lora_bounded_v2/repetition_audit.json`

完成后，你应能把“量化存储、低秩更新、计算精度、训练目标、行为评测”五个概念完全分开，并对正式结果给出不夸大的结论。
