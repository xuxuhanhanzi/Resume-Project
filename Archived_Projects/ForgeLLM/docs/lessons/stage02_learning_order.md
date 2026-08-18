# Stage 2 唯一学习入口：先主干，再前沿，再系统

> 面向第一次系统学习 Transformer 的学习者。  
> 不要按文件夹字母顺序阅读，也不要先读 DeepSeek/Kimi 复杂模块。  
> 每一站只有在“能解释 + 能跑测试 + 能回答题目”后才进入下一站。

## 0. 先知道三个词

- **Tensor（张量）**：带 shape、dtype、device 的多维数组；模型代码的基本数据载体。
- **Module（模块）**：保存参数并定义 `forward()` 的 PyTorch 对象。
- **Reference implementation（参考实现）**：优先可读和可验证，用来作为数值 oracle；不等于高性能生产实现。

### 思考题：为什么 Stage 2 不从 DeepSeek V4 的 CSA/HCA 开始？

<details>
<summary>答案</summary>

CSA/HCA 仍然依赖 Q、K、V、softmax、causal mask、residual 和 KV Cache 等基本概念。如果没有普通 attention 作为对照，就无法判断新方法删掉、压缩或替换了什么，也无法建立数值 oracle。

</details>

## 第 1 站：环境与模型配置

阅读：

1. `requirements-model.lock`；
2. `src/forgellm/model/config.py`；
3. `tests/unit/test_model_config.py`。

运行：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pytest tests\unit\test_model_config.py -q
```

你必须能解释：

- `d_model / n_heads = head_dim`；
- `n_heads / n_kv_heads = queries_per_kv`；
- 为什么 `head_dim` 必须是偶数；
- 为什么 correctness 配置默认 `dropout=0`。

通过标准：不看代码写出默认配置中 Q、K、V 的 shape。

## 第 2 站：RMSNorm、RoPE、SwiGLU

先读讲义 `stage02_transformer_core.md` 的第 1–4 章，再按顺序阅读：

1. `RMSNorm.forward()`；
2. `rotate_half()`；
3. `RotaryEmbedding.angles()` 和 `forward()`；
4. `SwiGLU.forward()`；
5. `tests/unit/test_model_layers.py`。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_model_layers.py -q
```

通过标准：能从公式手算一个四维 RMSNorm；能解释 RoPE 为什么旋转 Q/K 而不是 V；能指出 SwiGLU 的三个矩阵。

## 第 3 站：从单头公式到 MHA/GQA

阅读顺序：

1. `manual_scaled_dot_product_attention()`；
2. `causal_attention_mask()`；
3. `repeat_kv()`；
4. `CausalSelfAttention.__init__()`；
5. `CausalSelfAttention.forward()`；
6. `tests/unit/test_model_attention.py`。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_model_attention.py -q
```

暂停点：先画出下面的 shape，再看答案。

```text
hidden [B,T,D]
Q ?
K/V ?
attention score ?
output ?
```

<details>
<summary>答案</summary>

```text
Q       [B,Hq,T,Dh]
K/V     [B,Hkv,T,Dh]
repeat  [B,Hq,T,Dh]
score   [B,Hq,T,T]
output  [B,T,D]
```

</details>

## 第 4 站：KV Cache

只重读：

- `KVCache`；
- `CausalSelfAttention.forward()` 中 `past_length`、拼接与 cache-aware mask；
- `test_incremental_kv_cache_matches_full_attention`。

然后手动回答：当已有 7 个 token，只输入第 8 个 token 时，query length、key length 和 mask shape 分别是什么？

<details>
<summary>答案</summary>

`query_length=1`，拼接后 `key_length=8`，mask shape 为 `[1,8]`。这个 query 可以看到缓存中的 7 个 token 和当前 token，不能看到未来。

</details>

## 第 5 站：完整 Decoder 与 shifted loss

阅读顺序：

1. `TransformerBlock.forward()`；
2. `DecoderLM.__init__()`；
3. `DecoderLM.forward()`；
4. `next_token_loss()`；
5. `tests/unit/test_model_decoder.py`；
6. `tests/integration/test_tiny_decoder_overfit.py`。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_model_decoder.py tests\integration\test_tiny_decoder_overfit.py -q
```

通过标准：能完整口述 `input_ids → logits → shifted CE`；能解释 Tiny Overfit 证明什么、不能证明什么。

## 第 6 站：生成

阅读：

1. `GenerationConfig`；
2. `filter_logits()`；
3. `sample_next_token()`；
4. `generate()`。

重点追踪：第一次 forward 为什么输入完整 prompt，后续为什么每次只输入一个 token。

## 第 7 站：长上下文位置与 MLA

先读 `stage02_frontier_architectures.md` 的 RoPE/MLA 章节，再阅读：

1. `ScaledRotaryEmbedding`；
2. `MLACache`；
3. `MultiHeadLatentAttention`；
4. 对应测试。

必须计算：教学配置中 MHA 每 token 缓存 64 个元素，MLA 缓存 10 个元素；比例为 `10/64`。这只是结构存储量，不是质量结论。

## 第 8 站：稀疏、压缩与线性注意力

按下面顺序，不能跳：

```text
MoBA mask
→ CSA top-k token mask
→ HCA block summary
→ Gated DeltaNet recurrent state
→ HybridMixerStack 3:1 调度
```

阅读 `tests/unit/test_frontier_attention.py` 时重点看每个 causality test。

## 第 9 站：MoE、残差与 MTP

阅读顺序：

1. `hash_expert_indices()`；
2. `SparseMoE.forward()`；
3. `sinkhorn_doubly_stochastic()`；
4. `ManifoldHyperConnectionLite`；
5. `AttentionResiduals`；
6. `MultiTokenPredictionHead` 和 loss；
7. 对应测试。

通过标准：区分“总参数”和“每 token 激活参数”；能指出 route collapse、共享专家与 MTP label offset。

## 第 10 站：Muon

阅读：

1. `zeropower_via_newton_schulz5()`；
2. `Muon.step()`；
3. 两个 Muon 测试。

不要先背系数。先理解：普通 momentum 给出矩阵更新方向，Newton–Schulz 尝试让奇异值更均匀，再按学习率更新。

## 第 11 站：FlashAttention 数学与 SDPA

先读 `stage02_model_systems.md`，再阅读：

1. `online_softmax_blockwise_attention()`；
2. `test_online_softmax_blockwise_attention_matches_sdpa_and_gradients`；
3. `scripts/stage2_model_lab.py` 中固定 shape benchmark。

通过标准：能解释 running max、running sum、accumulator 为什么足够；不能声称 Python 循环实现了 FlashAttention 加速。

## 第 12 站：compile、量化、DDP

依次学习：

1. `tests/unit/test_model_compile.py`；
2. `Int8WeightOnlyLinear`；
3. `average_gradient_tensors()`；
4. `scripts/stage2_ddp_smoke.py`。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_model_compile.py tests\unit\test_model_systems.py tests\unit\test_model_distributed.py -q
.\.venv\Scripts\python.exe scripts\stage2_ddp_smoke.py
```

## 第 13 站：C++/CUDA 自定义算子

阅读顺序：

1. `silu_mul_reference()`；
2. `extensions/silu_mul/silu_mul.cpp`；
3. `extensions/silu_mul/silu_mul_cuda.cu`；
4. `_register_python_dispatch()`；
5. `scripts/build_silu_mul.py`；
6. `tests/integration/test_silu_mul_extension.py`。

当前机器已经安装“使用 C++ 的桌面开发”，并通过 MSVC/CUDA 真实构建。学习时运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_silu_mul.py
```

随后阅读 `artifacts/stage02/silu_mul_extension_report.json`，解释 CPU/CUDA 的误差、gradcheck、opcheck 和固定 shape benchmark；不要把一次本机加速比外推到其他 shape 或硬件。

## 第 14 站：最终验收

自动化证据：

```powershell
.\.venv\Scripts\python.exe scripts\stage2_model_lab.py
.\.venv\Scripts\python.exe scripts\dev.py check
```

学习者验收：

- 不看代码画出普通 Decoder；
- 从 MHA 解释到 GQA、MLA、稀疏/线性/混合注意力；
- 解释 MoE、mHC、MTP、Muon；
- 解释 `torch.compile`、量化、DDP、自定义算子各自解决什么；
- 对每种方法说出至少一个失败模式和证据边界。

只有自动化证据和学习者验收都通过，Stage 2 才能标记完成。
