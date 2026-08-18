# Stage 2 完整讲义（二）：现代开源模型架构方法实验室

> 2026-07-28 证据更正：本讲义中的 “DeepSeek V4” 归因来自早期注册表，当前未找到可核验的 DeepSeek 官方 V4 技术报告，相关 CSA/HCA/mHC/Hash-MoE 内容只保留为教学候选，不再归因 V4。Kimi K3 现已有官方报告，其后训练内容已转入 Stage 5。以 `docs/frontier_model_technology_registry.md` 最新状态为准；历史代码和实验不删除。

> 本讲义按“旧问题→新方法→数学/状态→代码→测试→边界”组织。  
> 所有实现都是缩小的 PyTorch reference，不宣称复现官方模型质量、训练规模或生产 kernel。  
> 最近核验日期：2026-07-27。

## 1. 模型名不是学习单元，方法才是

当前参考映射：

| 官方模型/项目 | 本阶段学习的方法 |
|---|---|
| 独立教学候选（早期曾误归因 DeepSeek V4） | CSA/HCA-lite、mHC-lite、Muon、Hash-MoE |
| Kimi K2.5 | MLA、MoE、SwiGLU |
| Kimi Linear | KDA/线性注意力与全局 MLA 混合 |
| Moonshot MoBA | 可训练 block sparse attention |
| Attention Residuals | 对历史 residual state 做选择 |
| Qwen3.6 | Gated DeltaNet/full attention 混合、Sparse MoE、MTP |
| OLMo Hybrid | 3:1 DeltaNet/全注意力开放复现参考 |
| Gemma 4 | Dense/MoE、MTP draft 与推测解码 |

官方来源：

- DeepSeek-R1：<https://github.com/deepseek-ai/DeepSeek-R1>
- Moonshot AI：<https://github.com/MoonshotAI>
- Kimi K2.5：<https://github.com/MoonshotAI/Kimi-K2.5>
- Kimi Linear：<https://github.com/MoonshotAI/Kimi-Linear>
- MoBA：<https://github.com/MoonshotAI/MoBA>
- Qwen3.6：<https://github.com/QwenLM/Qwen3.6>
- Gemma 4：<https://ai.google.dev/gemma/docs/core>
- OLMo Hybrid：<https://huggingface.co/allenai/Olmo-Hybrid-7B>

Kimi K3 已在 2026-07-28 增量审计中由官方仓库/技术报告确认；其后训练方法见 Stage 5 讲义。本讲义不追溯改写 Stage 2 的历史实验结果。

### 思考题 1：为什么不为每个模型复制一个完整 Decoder？

<details>
<summary>答案</summary>

很多模型共享 RMSNorm、RoPE、SwiGLU、MoE 等模块。复制完整 Decoder 会制造重复代码，让差异难以定位。课程以稳定 baseline 为主干，只把有研究意义的变量实现为独立 reference，更适合单变量实验。

</details>

## 2. 长上下文 RoPE scaling

### 2.1 旧问题

模型只在长度 `L_train` 内见过位置。如果直接把位置扩到远大于训练长度，旋转相位进入模型不熟悉的区域，注意力可能退化。

### 2.2 Linear scaling

最直接方法是把位置压缩：

\[
p'=\frac{p}{s}
\]

扩展 2 倍时，位置 2 使用原位置 1 的角度。优点是简单，缺点是训练范围内的位置分辨率也被压缩。

### 2.3 Dynamic NTK-style scaling

在训练长度以内保持原频率；超过后根据实际 sequence length 调整 RoPE base，使长距离频率变化更平滑。项目实现的是教学公式，不把它等同于所有框架或模型的具体版本。

### 2.4 代码与测试

`ScaledRotaryEmbedding` 支持：

```text
none
linear
dynamic_ntk
```

测试验证：

- linear factor=2 时，位置 2 与原位置 1 的角度相同；
- dynamic NTK 在训练长度内与原 RoPE 完全一致。

### 思考题 2：RoPE scaling 能否自动让模型具备百万上下文能力？

<details>
<summary>答案</summary>

不能。位置编码只是条件之一。模型还需要合适的长上下文数据、训练/继续训练、注意力或状态机制、缓存与 kernel 支持，以及长上下文评测。这里只验证位置变换公式。

</details>

## 3. 从 MHA/GQA 到 MLA

### 3.1 旧问题：KV Cache 随层数、长度和 head 增长

普通 MHA 每层每 token 大约保存：

\[
2H D_h
\]

个元素，2 来自 K 和 V。GQA 减少 `Hkv`，但仍保存 head-specific K/V。

### 3.2 MLA 的核心想法

Multi-head Latent Attention（多头潜在注意力）先把 hidden state 压到低秩 latent：

\[
c^{KV}=W_{down}^{KV}h
\]

需要 attention 时再展开：

\[
K^{nope}=W_{up}^{K}c^{KV},\quad V=W_{up}^{V}c^{KV}
\]

缓存不再保存所有 head 的 K/V，而保存 `cKV`。

### 3.3 RoPE 为什么带来困难

如果 K 已经经过依赖位置的旋转，某些权重吸收/低秩重构不再直接成立。因此教学实现把 key 分成：

- `K_nope`：从 latent 重构；
- `K_rope`：较小、head 共享、单独缓存的 rotary 分量。

Query 同样分成 `Q_nope` 和 `Q_rope`，最后拼接计算 score。

### 3.4 项目缓存

```text
latent_kv [B,T,Rkv]
rotary_key [B,1,T,Rrope]
```

教学配置：

```text
Rkv=8, Rrope=2
MLA cache/token = 10 elements
MHA cache/token = 2*4*8 = 64 elements
```

测试逐 token 构造 cache，并与全序列前向对齐。

### 思考题 3：缓存减少 84.375% 是否说明模型显存必然减少同样比例？

<details>
<summary>答案</summary>

不能。这个比例只计算该教学 attention 的 KV 元素。真实显存还包括权重、激活、临时展开、allocator、其他层、dtype 和 kernel workspace。它也没有衡量质量变化。

</details>

## 4. MoBA-style Block Sparse Attention

### 4.1 旧问题

全注意力 score 矩阵是 `[T,T]`，时间和内存随长度近似二次增长。固定窗口便宜，但可能永远看不到很远的重要 token。

### 4.2 MoBA 思路

把 KV 分块；每个 query 根据与 block summary 的相似度选择少数历史块，同时保留当前因果块。

教学 mask：

1. 计算每个 KV block 的 key mean；
2. query 与历史 block summary 点积；
3. 选 top-k block；
4. 展开为 token mask；
5. 再与 causal 下三角相交。

官方 MoBA 说明其不是可无训练直接替换预训练 attention 的万能插件；它通常需要继续训练。本项目只验证 mask、因果性和退化关系。

### 4.3 退化测试

当 `top_k_blocks >= 已有 block 数` 时，MoBA mask 应等于完整 causal mask，输出必须对齐普通 attention。

### 思考题 4：为什么当前 block 必须被保留？

<details>
<summary>答案</summary>

否则 query 可能看不到最近 token 或自己，局部连续信息会被 top-k 波动破坏。保留当前因果块提供稳定局部路径，历史远程块再由路由选择。

</details>

## 5. CSA/HCA-lite

这一教学候选把长上下文效率问题拆成更细粒度检索和压缩全局路径。本项目只实现概念级 lite 版本，不再作 DeepSeek V4 归因。

### 5.1 CSA-style 分支

Compressed Sparse Attention 教学分支：

- Q/K/V 仍用于最终 attention；
- 额外低维 `index_query/index_key` 只用于选位置；
- 永远保留 local causal window；
- 从更老的 token 中选择 compressed-index top-k。

这样把“选择谁”和“用完整表示聚合内容”分开。

### 5.2 HCA-style 分支

Heavily Compressed Attention 教学分支把历史 block 压成 mean K/V summary；当前 block 只汇总到当前 position，避免未来泄漏。Query 对少量 block summary 做 attention。

### 5.3 Hybrid gate

\[
O=\sigma(g)O_{CSA}+(1-\sigma(g))O_{HCA}
\]

测试给 prefix 后追加 future，要求原 prefix 输出保持不变。

### 思考题 5：为什么 HCA 当前 block 不能直接取完整 block mean？

<details>
<summary>答案</summary>

完整 block 包含当前位置之后的 future token，会把未来信息压进 summary，造成隐蔽的数据泄漏。必须只汇总当前 prefix。

</details>

## 6. Gated DeltaNet

### 6.1 旧问题

全注意力保留所有历史 K/V，长序列成本高。线性/递归注意力尝试把历史压进固定大小 state：

```text
S_t [B,H,Dh,Dh]
```

解码时状态大小不随 T 增长。

### 6.2 Delta rule

普通累加 `S += v kᵀ` 容易反复写入冲突记忆。Delta rule 先询问当前 state 对 key 的预测：

\[
\hat v_t=S_{t-1}k_t
\]

只写入误差：

\[
e_t=v_t-\hat v_t
\]

带 decay 与写入强度：

\[
S_t=\gamma_tS_{t-1}+\beta_t e_tk_t^\top
\]

输出：

\[
o_t=S_tq_t
\]

项目对 Q/K 做 L2 normalize，并用 sigmoid 产生 `β` 和 `γ`。

### 6.3 recurrent 与 chunked

教学代码逐 token loop，是单步解码 oracle。生产 prefill 通常需要 chunked scan/fused kernel，否则 Python 循环很慢。

测试把整段 forward 与逐 token state 传递逐项对齐。

### 思考题 6：固定大小 state 是否意味着它能无损记住任意长上下文？

<details>
<summary>答案</summary>

不意味着。固定容量 state 必须压缩历史，可能遗忘或覆盖细节。它换取 O(1) 解码状态成本，但表达能力与精确检索是核心权衡。

</details>

## 7. Hybrid Attention

线性状态高效，但精确任意位置检索可能弱；全 attention 强但昂贵。Qwen3.6、OLMo Hybrid、Kimi Linear 等采用不同形式的混合调度。

项目 `HybridMixerStack` 使用：

```text
Delta → Delta → Delta → Full Attention
```

循环，即 3:1。每层仍有 residual 和 norm。

实验只验证 layer schedule、shape 与因果性。没有训练不能比较 hybrid 与纯 attention 的语言能力。

### 思考题 7：为什么不把所有层都换成线性 attention？

<details>
<summary>答案</summary>

固定 state 对某些精确、长距离、内容寻址任务可能受限。周期性全 attention 提供不经固定状态压缩的全局混合路径。

</details>

## 8. Sparse MoE

### 8.1 旧问题

Dense FFN 的所有参数对每个 token 都参与计算。增大参数容量会同步增大每 token FLOPs。

Mixture-of-Experts（专家混合）为每层放置多个 FFN，但每 token 只激活 top-k：

\[
r=\operatorname{softmax}(W_{router}h)
\]

\[
E=\operatorname{topk}(r,k)
\]

\[
y=\sum_{e\in E}\tilde r_e Expert_e(h)+SharedExpert(h)
\]

### 8.2 总参数与激活参数

- 总参数：所有专家权重；
- 激活参数：一个 token 实际选择的专家及共享专家。

大容量不等于每 token 全部计算，但路由、dispatch、通信和负载不均产生新问题。

### 8.3 负载辅助项

项目记录：

- router 平均概率；
- 每个 expert assignment fraction；
- `num_experts * sum(probability * fraction)`。

这是教学辅助项，不代表某一官方模型的完整无辅助损失路由方案。

### 思考题 8：`expert_counts.sum()` 应等于什么？

<details>
<summary>答案</summary>

等于 `batch × sequence × top_k`。共享专家不计入 routed assignment，因为它对每个 token 始终执行。

</details>

## 9. Hash-MoE Bootstrap

训练早期 router 权重尚未形成有意义分工，可能集中到少数专家。Hash bootstrap 用 token ID 的确定性函数暂时指定专家：

\[
expert=(a\cdot token\_id+b)\bmod N
\]

项目实现 `hash_expert_indices()`，相同 token、相同专家数永远得到相同路由。它用于教学“固定早期分工 vs 学习路由”的主变量。

限制：token ID 不是语义标签；Hash 平衡不自动产生专家能力。真实方案还要决定何时从固定路由切换到学习路由。

### 思考题 9：为什么 Hash 路由的确定性既是优点也是风险？

<details>
<summary>答案</summary>

优点是训练早期负载与复现更可控；风险是相同 token 永远被绑定，可能形成不合理分工，并把 Tokenizer ID 偏差传入专家结构。

</details>

## 10. mHC-lite

### 10.1 旧残差

普通 residual 只有一条状态：

\[
x_{l+1}=x_l+F_l(x_l)
\]

更深模型可能希望保留多个 residual stream，并控制它们如何混合。

### 10.2 教学约束

项目维护 `N` 条 stream，学习一个非负、近似双随机 mixing matrix：

```text
每行和 ≈ 1
每列和 ≈ 1
元素 >= 0
```

通过 Sinkhorn 交替行/列归一化获得。Branch output 再按 softmax 权重分配到各 stream。

这只实现“受约束多流残差”的教学不变量，不等同于任何未核验模型的完整高效实现。

### 思考题 10：为什么要约束 mixing matrix，而不是任意线性矩阵？

<details>
<summary>答案</summary>

非负和归一化约束让混合更接近稳定的凸组合，减少任意放大、缩小或符号翻转造成的信号失控；代价是表达自由度受限。

</details>

## 11. Attention Residuals

普通 residual 默认把最近状态直接传到下一层。Attention Residuals 把多个历史层状态看成候选：

```text
history [B,T,L,D]
query   [B,T,D]
weights [B,T,L]
```

对 layer 维做 softmax，再加权聚合历史 residual。

它解决的问题是：不同 token 可能需要不同深度的信息，不必固定只使用上一层。但代价是保存历史层状态和额外 attention 计算。

### 思考题 11：这里的 attention 与 token self-attention 有什么不同？

<details>
<summary>答案</summary>

token self-attention 在序列位置维选择历史 token；Attention Residuals 在层/残差历史维选择不同深度的状态。项目中的 weights shape 是 `[B,T,L]` 而不是 `[B,H,T,T]`。

</details>

## 12. Multi-Token Prediction（MTP）

普通语言模型每个位置预测 `t+1`。MTP 增加多个 future head：

```text
logits [B,T,K,V]
offset 0 → target t+1
offset 1 → target t+2
...
```

每个 offset 的有效长度不同：预测距离 `d` 时，最后 `d` 个位置没有标签，必须裁掉。

项目逐 offset 计算 CE，再平均；同时返回 `per_offset_loss`，防止远期 head 被平均值掩盖。

MTP 可以增加训练监督，也可为 draft/speculative decoding 提供候选，但真正无损加速还需要验收/拒绝协议和高效推理系统。

### 思考题 12：为什么不能把所有 offset 都与 `input_ids[:,1:]` 比较？

<details>
<summary>答案</summary>

因为第 2 个 head 应预测 `t+2`，第 3 个预测 `t+3`。所有 head 使用相同 shift 会把多个 head 训练成相同任务，失去 MTP 含义。

</details>

## 13. Muon

### 13.1 旧问题

AdamW 对每个参数元素维护一阶/二阶统计，稳定但会增加 optimizer state。Muon 关注二维矩阵参数，把 momentum update 经过近似正交化，使不同奇异方向的更新尺度更均匀。

### 13.2 Newton–Schulz

项目先归一化矩阵，然后迭代多项式：

\[
X_{k+1}=aX_k+(bA_k+cA_k^2)X_k,\quad A_k=X_kX_k^\top
\]

系数来自常见五步近似实现。学习重点不是背系数，而是理解它在逼近矩阵 polar factor。

### 13.3 参数分组

真实 LLM 中通常不会把所有参数都交给 Muon：

- 2D hidden matrices：Muon 候选；
- embedding、norm、bias、标量：常用 AdamW 或其他规则。

教学 `Muon` 对矩阵正交化，对 vector/scalar 只用 momentum，以便独立运行。

### 思考题 13：一个 6×6 二次函数 loss 下降，能否证明 Muon 比 AdamW 更好？

<details>
<summary>答案</summary>

不能。它只证明实现方向基本合理。比较优化器需要固定模型、数据、token budget，分别调合理超参数，报告多 seed、训练/验证曲线和计算成本。

</details>

## 14. 如何评价这些方法

| 方法 | 首要正确性证据 | 后续效果证据 |
|---|---|---|
| RoPE scaling | 角度映射、范围内不变 | 长上下文任务 |
| MLA | cache/full logits 对齐、元素数 | 质量/显存/吞吐 |
| MoBA/CSA | causal mask、退化到 full | 长程检索/吞吐 |
| HCA | prefix summary 无未来泄漏 | 全局信息保真 |
| DeltaNet | full/recurrent state 对齐 | 长程任务/速度 |
| Hybrid | schedule 与 causality | 质量-成本曲线 |
| MoE | assignment、梯度、负载 | 容量/质量/通信 |
| mHC/AttnRes | 约束和权重归一化 | 深层训练稳定性 |
| MTP | offset 标签 | loss、draft 接受率 |
| Muon | 矩阵更新/数值有限 | 多 seed 训练效率 |

### 最终思考题 14：为什么所有前沿方法都不能只用“程序能运行”验收？

<details>
<summary>答案</summary>

运行只能排除部分语法/shape 错误。前沿模块最容易出现静默错误：未来泄漏、offset 错位、cache 位置错误、路由 assignment 丢失、状态递推不等价、约束失效。必须针对方法的数学不变量设计测试。

</details>
