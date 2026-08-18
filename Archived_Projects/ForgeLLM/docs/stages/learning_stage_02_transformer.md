# Stage 2：Transformer、现代架构与模型系统实验室

> 状态：G2-Core、G2-Arch、G2-Systems 与学习者验收全部完成  
> 启动日期：2026-07-27（Asia/Singapore）  
> 唯一学习入口：`docs/lessons/stage02_learning_order.md`  
> 本阶段不执行正式语料预训练；正式训练循环属于 Stage 3

## 1. 阶段目标

Stage 2 的目标不是训练一个效果优秀的大模型，而是建立以下能力：

1. 从公式独立实现 Decoder-only Transformer；
2. 对每个 Tensor 说出语义、shape、dtype 和 device；
3. 用数值对照、梯度、因果性与缓存等价实验判断代码是否正确；
4. 将现代开源模型中的新方法缩小为可运行的 PyTorch reference；
5. 理解 Python 组合算子、PyTorch SDPA、`torch.compile` 与 C++/CUDA 自定义算子的边界；
6. 不把小型 reference 的通过误写成对官方大模型效果或性能的复现。

## 2. 三条必修线

### G2-Core：稳定模型主干

- Tensor/stride/view/transpose/contiguous；
- Autograd、Module、Parameter、Buffer、state dict；
- RMSNorm、RoPE、SwiGLU；
- MHA、GQA、QK-Norm、Causal Mask；
- Pre-Norm Block、LM Head、权重共享、shifted CE；
- greedy、temperature、top-k、top-p；
- KV Cache、保存加载、Tiny Overfit。

### G2-Arch：现代架构方法

- Linear/Dynamic-NTK RoPE scaling；
- MLA 压缩 KV Cache；
- MoBA-style block sparse attention；
- CSA/HCA-lite 混合压缩注意力；
- Gated DeltaNet 与 3:1 hybrid mixer；
- Sparse MoE、共享专家、Hash 路由与负载辅助项；
- mHC-lite、Attention Residuals；
- MTP；
- Muon 的 Newton–Schulz 矩阵更新。

### G2-Systems：模型系统

- SDPA 与手写 attention 对照；
- FlashAttention 的 online-softmax 分块 reference；
- `torch.compile(fullgraph=True)` 和 graph break；
- INT8 weight-only 量化与误差/存储实验；
- 两进程 CPU/Gloo DDP 梯度同步；
- `silu_mul` Python→C++ CPU→CUDA→FakeTensor→Autograd→opcheck→benchmark。

## 3. 实验假设

### H-Core

若实现正确，则手写 attention 和 SDPA 在 FP32 容差内对齐；未来 token 不影响过去 logits；逐 token KV Cache 与全序列前向对齐；Tiny Set 能被过拟合。

### H-Cache

在相同 head/value 维度下，GQA 与 MLA reference 存储的每 token KV 元素应少于 MHA；这里只验证结构与元素数，不宣称质量不下降。

### H-Long

分块 online softmax 在不同 tile 大小下应与普通 causal attention 对齐；Python 循环可能更慢，不能用来评价 fused kernel 的真实性能。

### H-MoE

Top-k 路由的总 assignment 数应严格等于 `token_count × top_k`；Hash 路由应确定性可复现；微型随机输入的负载不能证明大规模训练平衡。

### H-Systems

完整 Decoder 应能由 `torch.compile(..., fullgraph=True, backend="eager")` 捕获为单图；Inductor、C++/CUDA 和分布式能力分别受 Triton、MSVC/CUDA 与进程后端约束，失败必须独立记录。

## 4. 固定项

- 核心测试默认 `torch.manual_seed()` 固定；
- 数值 oracle 优先 FP64/FP32；
- correctness 默认 `dropout=0`；
- causal mask 语义固定为 `True = 可以参与 attention`；
- 性能实验固定硬件、shape、dtype、预热和重复次数；
- 每次实验只改变一个主变量；
- 本地 GPU：NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB；
- PyTorch：`2.6.0+cu124`；CUDA Toolkit：12.4。

## 5. 当前实现文件

```text
src/forgellm/model/
├── config.py
├── layers.py
├── attention.py
├── decoder.py
├── generation.py
├── frontier_attention.py
├── frontier_layers.py
├── optim.py
├── systems.py
└── distributed.py

src/forgellm/custom_ops/
└── silu_mul.py

extensions/silu_mul/
├── silu_mul.cpp
└── silu_mul_cuda.cu
```

## 6. 分阶段门

### G2-Core 通过条件

- 公式/shape/梯度测试通过；
- manual/SDPA 对齐；
- prefix causality 通过；
- cache/no-cache 对齐；
- shifted loss 人工样例通过；
- Tiny Overfit 通过；
- 5M 级配置完成 GPU forward/backward。

### G2-Arch 通过条件

- 每个方法存在来源、旧问题、公式、reference 代码和测试；
- MLA 增量缓存与全序列结果对齐；
- MoBA/CSA/HCA/DeltaNet/Hybrid 保持因果性；
- MoE assignment、Hash 路由、mHC 约束和 MTP offset 可检查；
- 不使用短跑 loss 支撑官方大模型效果结论。

### G2-Systems 通过条件

- online-softmax 的 forward/backward 对齐 SDPA；
- `torch.compile` 单图捕获通过并保留 graph-break 修复记录；
- INT8 输出误差和存储量有固定实验；
- 两进程 DDP 梯度一致；
- C++/CUDA operator 完成 CPU/CUDA forward、gradcheck、opcheck；
- 若编译器缺失，Stage 2 只能标记“系统门未完成”。

## 7. 已知资源边界

- 不下载 DeepSeek V4、Kimi K2.5、Qwen3.6 等超大权重；
- 不训练百万上下文或超大 MoE；
- CSA/HCA、mHC、KDA/DeltaNet 均为缩小教学 reference；
- Python online-softmax 不代表 FlashAttention kernel 性能；
- Windows 官方环境缺少可工作的 Triton，当前只证明 Dynamo full-graph 捕获；
- Visual Studio C++ 工作负载、MSVC 19.42、Windows SDK 与 CUDA 12.4 已通过真实构建验证。
- 设置 `FORGELLM_TEST_EXTENSION=1` 后，全仓质量门为 156 passed、0 skipped。
- `silu_mul` CPU/CUDA forward、gradcheck、opcheck、空 Tensor 与 benchmark 已通过。

## 8. 完成定义

Stage 2 只有在以下三项均满足时才能标记完成：

```text
G2-Core 自动化证据通过
AND G2-Arch 自动化证据通过
AND G2-Systems 自定义算子编译/梯度证据通过
AND 学习者按导航完成代码追踪与口述验收
```

代码存在或测试数量增加都不能替代学习者验收。
