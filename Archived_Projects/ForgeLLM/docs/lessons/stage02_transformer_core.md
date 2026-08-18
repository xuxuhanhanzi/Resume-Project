# Stage 2 完整讲义（一）：从 Tensor 到 Decoder-only Transformer

> 学习目标：不是记住类名，而是能把数学公式、Tensor shape、PyTorch 代码和正确性证据一一对应。  
> 配套源码：`src/forgellm/model/config.py`、`layers.py`、`attention.py`、`decoder.py`、`generation.py`。  
> 配套测试：所有 `tests/unit/test_model_*.py` 和 `tests/integration/test_tiny_decoder_overfit.py`。

## 1. 模型到底在做什么

语言模型收到 token ID 序列：

```text
[BOS, 今, 天, 天, 气]
```

它不是一次只输出一个答案，而是在每个位置输出一个长度为词表大小 `V` 的 logits 向量：

```text
input_ids  [B,T]
logits     [B,T,V]
```

第 `t` 个 logits 用来预测第 `t+1` 个 token。因此训练标签需要向左错开一位：

```text
prediction = logits[:, :-1]
target     = input_ids[:, 1:]
```

### 思考题 1：为什么不能用 `logits[:, t]` 预测 `input_ids[:, t]`？

<details>
<summary>答案</summary>

因为输入 embedding 已经直接包含当前位置 token。如果让当前位置预测自身，模型可以学习复制输入，而不是根据历史预测下一个 token。自回归语言模型要求位置 `t` 只能用于预测未来的 `t+1`。

</details>

## 2. Tensor 的四个基本属性

每次读模型代码都要先问：

1. shape 是什么；
2. dtype 是什么；
3. device 在哪里；
4. storage/stride 是否允许当前 view。

常用记号：

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `T` | sequence length |
| `D` | model hidden dimension |
| `Hq` | query head 数 |
| `Hkv` | key/value head 数 |
| `Dh` | head dimension，`D/Hq` |
| `V` | vocabulary size |

例如：

```python
projected.view(B, T, H, Dh).transpose(1, 2)
```

shape 从 `[B,T,H*Dh]` 变成 `[B,H,T,Dh]`。`transpose` 通常只改变 stride，不复制 storage；再合并 head 前调用 `.contiguous()`，是为了按新顺序生成连续存储。

### 思考题 2：`reshape()` 是否永远复制数据？

<details>
<summary>答案</summary>

不是。若当前 stride 允许，它可以返回 view；不允许时会创建副本。`view()` 要求兼容的连续布局，`reshape()` 更方便但可能隐藏复制成本。模型学习阶段要同时关注 shape 正确和是否意外复制。

</details>

## 3. `nn.Module`、Parameter 与 Buffer

- `nn.Parameter`：会出现在 `model.parameters()`，默认参与优化；
- Buffer：属于模型状态、随 `.to(device)` 移动，但不参与优化；
- 普通属性：Python 配置或常量，不自动进入 `state_dict`。

本项目的例子：

- RMSNorm 的 `weight` 是 Parameter；
- RoPE 的 `inv_freq` 是 `persistent=False` Buffer；
- `max_seq_len` 是普通属性。

### 思考题 3：为什么 RoPE 频率不是 Parameter？

<details>
<summary>答案</summary>

它由 `head_dim` 和 `theta` 按固定公式生成，不需要梯度更新。注册为 Buffer 可以让它自动移动到 CPU/GPU，并参与 Module 的设备管理。

</details>

## 4. RMSNorm

### 4.1 旧问题

深层网络中激活尺度可能不断放大或缩小，造成梯度不稳定。LayerNorm 会减均值并除标准差；RMSNorm 只根据均方根缩放：

\[
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
\]

\[
y_i=w_i\frac{x_i}{\operatorname{RMS}(x)}
\]

### 4.2 实现

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)
normalized = x * torch.rsqrt(variance + eps)
output = normalized * weight
```

项目实现对 FP16/BF16 使用 FP32 累积，但让 FP64 保留 FP64。这是第一次 `gradcheck` 失败后修正的真实数值问题。

### 思考题 4：RMSNorm 会让输出均值变成 0 吗？

<details>
<summary>答案</summary>

不会。它没有减去均值，只控制均方根尺度。输出均值仍取决于输入和可学习权重。

</details>

## 5. RoPE：把相对位置信息写入 Q/K

### 5.1 为什么需要位置

如果没有位置编码，attention 只看到 token 集合，不知道排列顺序。RoPE（Rotary Position Embedding）将相邻维度看成二维向量，根据位置旋转：

\[
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
\]

不同维度对使用不同频率。位置差会进入旋转后 Q/K 的点积，因此 attention score 能表达相对位置。

### 5.2 为什么只旋转 Q/K

Attention 权重由 `QKᵀ` 决定，位置应影响“看向谁”。V 是被聚合的内容，通常不需要同样旋转。

### 5.3 代码契约

```text
Q [B,Hq,T,Dh]
K [B,Hkv,T,Dh]
positions [T]
cos/sin [1,1,T,Dh]
```

旋转是正交变换，因此理论上保持每个向量的 L2 范数。测试同时验证：位置 0 不变、旋转前后平方和一致。

### 思考题 5：为什么 `head_dim` 必须为偶数？

<details>
<summary>答案</summary>

RoPE 每两个相邻分量组成一个二维旋转平面。奇数维会留下无法配对的分量；本项目在配置创建阶段直接拒绝。

</details>

## 6. SwiGLU

普通 FFN 常写为：

\[
\operatorname{FFN}(x)=W_2\sigma(W_1x)
\]

SwiGLU 使用两个上投影，一个产生 gate，一个产生 value：

\[
\operatorname{SwiGLU}(x)=W_{down}
\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
\]

代码：

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

三个矩阵的 shape：

```text
gate/up [hidden_dim, d_model]
down    [d_model, hidden_dim]
```

### 思考题 6：为什么 Stage 2 的 CUDA 自定义算子选择 `silu(gate) * value`？

<details>
<summary>答案</summary>

这两个逐元素操作相邻、shape 相同，适合演示 kernel fusion：Python reference 很简单，forward/backward 公式明确，又确实位于现代 LLM 的 MLP 热路径中。

</details>

## 7. Attention 公式

单个 head 的 scaled dot-product attention：

\[
S=\frac{QK^\top}{\sqrt{D_h}}
\]

\[
P=\operatorname{softmax}(S+M)
\]

\[
O=PV
\]

其中 causal mask `M` 让未来位置分数变成负无穷。

项目约定布尔 mask：

```text
True  = 允许参与 attention
False = 屏蔽
```

这与部分 PyTorch API 的 mask 语义不同，所以必须在讲义和测试中固定。

### 7.1 为什么除以 `sqrt(Dh)`

若 Q/K 分量方差相近，点积方差随 `Dh` 增大。分数过大时 softmax 接近 one-hot，梯度变小。缩放用来控制 score 的典型尺度。

### 思考题 7：为什么 mask 要在 softmax 前应用？

<details>
<summary>答案</summary>

softmax 会把所有输入归一化成概率。先把非法位置变为负无穷，它们的指数才是 0；若 softmax 后再清零，剩余概率和不再为 1，还可能让未来 token 参与分母。

</details>

## 8. MHA、MQA 与 GQA

### 8.1 MHA

每个 query head 有独立 K/V head：

```text
Hq = Hkv
```

### 8.2 MQA

所有 query head 共享一组 K/V：

```text
Hkv = 1
```

### 8.3 GQA

若干 query head 共享一组 K/V：

```text
1 < Hkv < Hq
Hq % Hkv == 0
```

项目默认：`Hq=4`、`Hkv=2`，所以每个 KV head 被两个 query head 使用。

训练/完整前向中，`repeat_kv()` 将 KV 扩成 query head 数；缓存中仍保存未扩展的 `[B,Hkv,T,Dh]`，否则失去 GQA 的缓存收益。

### 思考题 8：GQA 的主要缓存收益来自减少 T 还是减少 head 数？

<details>
<summary>答案</summary>

来自减少 K/V head 数。序列长度 `T` 没变，每个历史 token 仍需保存；但每个 token 保存的 K/V head 从 `Hq` 降到 `Hkv`。

</details>

## 9. 因果性

最强的结构测试之一：

1. 对 prefix 单独前向；
2. 在后面拼接随机 future；
3. 再次前向；
4. 比较两个结果的 prefix 部分。

正确模型必须满足：

```python
model(prefix) == model(concat(prefix, future))[:, :prefix_len]
```

这比“mask 看起来是下三角”更强，因为它测试了完整实现，包括 shape、广播、SDPA mask 语义和残差路径。

### 思考题 9：如果只比较最后一个位置，能证明整体因果性吗？

<details>
<summary>答案</summary>

不能。最后位置本来就可以看到全部前文；应比较所有过去位置在添加 future 前后的输出。

</details>

## 10. Pre-Norm Transformer Block

本项目使用：

\[
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
\]

\[
y=x'+\operatorname{SwiGLU}(\operatorname{RMSNorm}(x'))
\]

称为 Pre-Norm，因为归一化发生在子层之前。残差提供较直接的梯度路径。

```text
x
├─ RMSNorm → Attention ─┐
└────────────────────── + → x'
                         ├─ RMSNorm → SwiGLU ─┐
                         └──────────────────── + → y
```

### 思考题 10：第二个 RMSNorm 应该接原始 `x` 还是 attention 残差后的 `x'`？

<details>
<summary>答案</summary>

接 `x'`。MLP 需要处理已经融合 attention 信息的当前状态。项目 `TransformerBlock.forward()` 先更新 `hidden_states`，再送入 `mlp_norm`。

</details>

## 11. Embedding、LM Head 与权重共享

Embedding 矩阵 shape：

```text
[V,D]
```

LM Head 权重 shape 也是 `[V,D]`，因为线性层计算 `[B,T,D] → [B,T,V]`。权重共享让两者指向同一个 Parameter：

```python
self.lm_head.weight = self.token_embedding.weight
```

参数计数必须避免把同一 Parameter 重复计算。PyTorch 的 `model.parameters()` 会去重同一对象。

### 思考题 11：权重共享是否意味着 embedding 输出直接等于 logits？

<details>
<summary>答案</summary>

不是。它们共享矩阵，但计算方向不同：Embedding 按 token ID 查行；LM Head 将隐藏向量与所有词表行做内积。中间还有多层 Decoder 和 final norm。

</details>

## 12. Shifted Cross-Entropy

项目实现：

```python
predictions = logits[:, :-1, :]
targets = input_ids[:, 1:]
cross_entropy(predictions.reshape(-1,V), targets.reshape(-1))
```

人工测试故意把正确类别只放在 `t→t+1` 的位置。如果错误地使用同位置标签，测试会立即失败。

### 思考题 12：长度为 1 的输入为什么不能计算 next-token loss？

<details>
<summary>答案</summary>

错位后没有任何 prediction-target 对。可以前向生成 logits，但不能从单 token 序列构造训练标签。

</details>

## 13. KV Cache：prefill 与 decode

### 13.1 无缓存

生成第 `t` 个 token 时重新处理全部 `t` 个 token，历史 K/V 被重复计算。

### 13.2 有缓存

- prefill：一次处理完整 prompt，保存每层 K/V；
- decode：每次只处理新 token，将新 K/V 追加到 cache。

```text
Q current [B,Hq,1,Dh]
K cache   [B,Hkv,t,Dh]
V cache   [B,Hkv,t,Dh]
score     [B,Hq,1,t]
```

RoPE 的新位置从 `past_length` 开始。若每次又从 0 开始，新 token 的位置相位会错误。

### 思考题 13：为什么 `is_causal=True` 对非方形 cached attention 需要格外小心？

<details>
<summary>答案</summary>

query length 为 1、key length 为 `t` 时，不同 API 对非方形 causal mask 的对齐方式可能不是“这个 query 可以看全部历史”。项目显式构造绝对位置 mask，避免把第 `t` 个 query 错误地只对齐到最左侧 key。

</details>

## 14. 生成策略

### Greedy

选择最大 logit：

```python
argmax(logits)
```

确定但容易重复。

### Temperature

\[
p_i=\operatorname{softmax}(z_i/\tau)
\]

- `τ<1` 更尖锐；
- `τ>1` 更平坦；
- 项目用 `temperature=0` 明确表示 greedy，避免除 0。

### Top-k

只保留 logit 最大的 k 个 token。

### Top-p

按概率降序，保留累计概率达到阈值的最小集合。项目通过 `cumulative - current_probability >= p` 确保越过阈值的首个 token 仍被保留。

### 思考题 14：Top-k 与 Top-p 的候选数量是否固定？

<details>
<summary>答案</summary>

Top-k 固定为 k（除非词表更小）；Top-p 的候选数量随概率分布变化。分布很尖时可能只保留少数 token，分布平坦时会保留更多。

</details>

## 15. Tiny Overfit

项目用一个极小重复序列训练 120 step，并要求 loss 低于 `0.02`。它能验证：

- forward 连通；
- shifted loss 正确；
- backward 有梯度；
- optimizer 能更新；
- 模型容量足够记忆该序列。

它不能验证：

- 通用语言能力；
- 真实验证集泛化；
- Tokenizer 质量；
- 大规模稳定性；
- 现代模块带来收益。

### 思考题 15：Tiny Overfit 失败时应该先扩大模型吗？

<details>
<summary>答案</summary>

通常不应该。先检查 loss shift、mask、梯度、学习率、参数是否进入 optimizer、数据是否重复和模型是否处于训练模式。极小任务本来就不需要大模型。

</details>

## 16. 当前正确性证据

核心自动测试覆盖：

- 配置跨字段约束；
- RMSNorm 公式与 FP64 gradcheck；
- RoPE 位置 0 和范数保持；
- SwiGLU 显式公式；
- manual attention 与 SDPA forward/backward；
- MHA/GQA head 映射；
- QK-Norm；
- prefix causality；
- attention 和完整 Decoder cache 对齐；
- shifted loss；
- state dict round-trip；
- greedy/top-k/top-p；
- Tiny Overfit；
- 5.36M 参数 GPU forward/backward。

### 最终自检题 16：为什么“53 个 Stage 2 测试通过”仍不等于 Stage 2 学习完成？

<details>
<summary>答案</summary>

测试证明已编码的行为，不证明学习者能独立解释、修改和排错，也不覆盖尚未完成的 C++/CUDA 本机构建门。Stage 2 还要求代码追踪、口述验收和系统门。

</details>
