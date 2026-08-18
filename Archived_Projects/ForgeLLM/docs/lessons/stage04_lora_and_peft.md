# Stage 4 完整讲义三：手写 LoRA、Adapter Artifact 与 PEFT

## 1. 全量更新为什么昂贵

普通 Linear 权重 `W ∈ R^{d_out×d_in}` 有 `d_out*d_in` 个可训练参数。全量微调还为每个参数保存 gradient，以及 Adam 的一阶、二阶状态。对大模型，优化器状态和梯度往往比权重本身更占内存。

LoRA 冻结 `W`，只学习：

```text
A ∈ R^{r×d_in}
B ∈ R^{d_out×r}
ΔW = (alpha/r) * B @ A
y = x W^T + dropout(x) A^T B^T * (alpha/r)
```

可训练参数从 `d_out*d_in` 变成 `r*(d_in+d_out)`。只有当 `r` 远小于输入/输出维度时才是真正低秩节省。

**思考题：`B @ A` 的 rank 一定等于 r 吗？**

答案：最多为 r，可能更低；如果 A/B 行列相关或部分为零，实际 rank 会下降。r 是秩上界和中间维度，不是保证的有效秩。

## 2. `alpha/r` 为什么存在

scale 把 rank 与更新幅度部分解耦。若只增加 r 而不归一化，低秩分支的总幅度可能随分量数量增加。经典 LoRA 使用 `alpha/r`；这不是唯一理论选择，也不能保证不同 r 的优化完全等价。

本阶段固定 `r=16, alpha=32, dropout=0.05`，目的不是宣称最优，而是消除搜索。tiny 手写实验用更小 r 以突出参数量差异。

**思考题：alpha 是学习率吗？**

答案：不是。学习率控制优化器每步如何改变 A/B；alpha 在 forward 中缩放低秩分支。两者都会影响有效更新幅度，但位置和动力学不同。

## 3. 为什么 A 随机、B 为零

初始化时 `B=0`，所以 `B @ A=0`，LoRA 分支是 exact no-op，模型输出与 Base 完全相同。第一次反向时：

```text
∂L/∂B ∝ upstream @ A^T   → A 随机非零，所以 B 可获得梯度
∂L/∂A ∝ B^T @ upstream   → B 为零，所以首次 A 梯度为零
```

更新 B 后，后续 A 才开始获得梯度。这是预期次序，不是梯度断裂。

若 A/B 都为零，二者首次梯度都为零，分支永远学不动；若二者都随机，初始化就改变 Base 输出，失去稳定起点。

**思考题：B 零初始化是否意味着 LoRA 第一步完全没有参数更新？**

答案：不是。B 的梯度通常非零并在第一步更新；只有 A 的首次梯度通常为零。

## 4. 注入必须显式，冻结必须可检查

`inject_lora` 遍历命名子模块，把被允许的 `nn.Linear` 替换成 `LoRALinear`。它返回模块名和参数计数，拒绝：

- rank 非正或超过矩阵可表示范围；
- 没有匹配任何模块；
- 不明确的排除策略；
- Base 参数仍 `requires_grad=True`。

框架轨使用 PEFT 的 `target_modules="all-linear"`，但仍记录匹配后的 trainable 参数。Qwen3-0.6B 实际总参数 606,142,464，Adapter 参数 10,092,544，占 1.665%。字符串配置只是意图，实际计数才是证据。

**思考题：为什么只检查 optimizer 参数列表还不够？**

答案：漏进计算图的 Base 仍可能分配 gradient 和消耗内存；未来代码也可能换 optimizer 构造。`requires_grad`、实际 `.grad` 和 optimizer 列表是三个不同边界，都应检查。

## 5. Adapter-only Artifact 保存什么

手写实现只保存含 `lora_a/lora_b` 的 state dict，并附 metadata；加载时要求键集合和 shape 完全一致。PEFT 保存 `adapter_config.json` 和 safetensors Adapter 权重，同时保存 tokenizer 文件。

Adapter 不包含完整 Base，因此可复现加载必须同时绑定：

- Base model ID；
- 精确 Base revision；
- target modules；
- r/alpha/dropout/bias；
- 框架版本；
- Chat Template/Tokenizer revision。

只有一个 Adapter 文件、却不知道 Base revision，不是完整模型产物。

**思考题：Adapter 小是否意味着训练 checkpoint 也一定小？**

答案：不一定。可恢复训练还需 optimizer、scheduler、RNG 和进度状态；Adam 状态可能大于 Adapter 权重。Stage 4 只保存最终 Adapter，不宣称实现了完整中断恢复；这与 Stage 3 checkpoint 目标不同。

## 6. Merge / Unmerge 的数学和风险

推理时可以把增量写回 Base：

```text
W_merged = W + scaling * B @ A
```

这样不再额外执行两次小矩阵乘，但失去运行时切换 Adapter 的便利。`unmerge` 再减去同一增量。手写测试检查 merge 前、merge 后和 unmerge 后输出在登记容差内一致。

风险包括：

- 重复 merge 会把增量加两次；
- 低精度加减不保证 byte-exact 可逆；
- merge 后继续训练 A/B，Base 中已有旧增量，语义变复杂；
- 量化 Base 不能把 BF16 增量随意原地写入 4-bit 存储而不重新量化。

**思考题：部署时永远应该 merge 吗？**

答案：不是。单 Adapter、追求简单推理图时可 merge；多租户切换、组合 Adapter、审计或量化场景通常保留分离形式。必须基于部署约束选择。

## 7. 手写 LoRA 与 PEFT 的分工

手写实现用于理解不变量：shape、no-op、梯度、参数量、merge、round-trip。PEFT 用于真实 Qwen 模块命名、模型加载、Adapter Artifact 和 4-bit 集成。

本项目没有把“使用 PEFT 成功”当原理证明，也没有重新实现 PEFT 的所有生产特性。正确路线是：

```text
小 Linear 手算
→ 手写模块单元测试
→ 小 Decoder 注入实验
→ PEFT 小模型 no-op/freeze 测试
→ Qwen 固定 revision 实际加载
```

PEFT 初始化后检查所有 `lora_B` exact zero；训练每步检查非 LoRA 参数 `.grad is None`；保存后得到 Adapter-only 文件。三者分别覆盖初始化、训练和产物边界。

**思考题：手写实现与 PEFT 输出不同是否一定是 PEFT 错？**

答案：不是。需先对齐 fan-in/fan-out、transpose、scale、dropout、bias、初始化、target modules、dtype 和 train/eval 模式。只有固定所有语义后才做数值归因。

## 8. 如何解释 tiny LoRA 的负结果

手写 LoRA 在 80 步下比全量更新差，可能来自：

- trainable 参数少约 12 倍；
- Base 是随机初始化 tiny model，低秩 Adapter 不能像在强预训练 Base 上那样复用已有能力；
- 固定步数和数据极小；
- LoRA 只注入 Linear，不等于拥有全部全量自由度。

这里最关键的概念是 LoRA 的典型前提：Base 已经有广泛能力，后训练只需在相对低维的方向调整行为。对随机小模型从零学习，低秩更新天然更吃亏。

**思考题：因此能否得出 LoRA 不适合本项目？**

答案：不能。tiny 结果只说明该固定随机 Base/预算下的优化差异；真实 Qwen Base 的 PEFT 路径已经稳定下降 validation loss。两条轨道回答不同问题。

## 9. 本讲代码追踪

1. `LoRALinear.__init__`
2. `LoRALinear.forward`
3. `merge` / `unmerge`
4. `inject_lora`
5. `adapter_state_dict`
6. `save_adapter` / `load_adapter`
7. `lora_is_initial_noop`
8. `adapter_base_gradients_are_absent`
9. `load_stage4_hf_stack` 的 PEFT 构造

完成后，你应能对任意 Linear 手算全量与 LoRA 参数量，并解释从 no-op 初始化到 Adapter Artifact 的完整证据链。
