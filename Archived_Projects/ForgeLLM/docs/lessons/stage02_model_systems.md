# Stage 2 完整讲义（三）：SDPA、FlashAttention、compile、量化、DDP 与 C++/CUDA

> 本讲义回答一个核心问题：数学正确的 Python Module 如何走向更低内存、更高吞吐和底层算子？  
> 系统优化必须先有 reference oracle；没有固定 shape、dtype、硬件、预热和同步的速度数字没有意义。

## 1. 三层实现

```text
第一层：清晰 PyTorch 公式
        用于教学、梯度和数值 oracle
第二层：PyTorch 官方融合算子/编译器
        SDPA、torch.compile
第三层：自定义 C++/CUDA
        只有接口缺失或 profile 证明瓶颈时使用
```

不是所有 Python 代码都需要重写成 CUDA。底层代码增加编译、dtype、设备、Autograd、FakeTensor、ABI 和测试成本。

### 思考题 1：为什么自定义 kernel 前必须保留 Python reference？

<details>
<summary>答案</summary>

优化实现往往更难读，且可能只在特定 shape/dtype 正确。Python reference 提供独立数值 oracle，可比较 forward、backward 和边界输入。没有 oracle，kernel 输出“看起来正常”不能证明正确。

</details>

## 2. PyTorch SDPA

`torch.nn.functional.scaled_dot_product_attention` 接收：

```text
Q [B,Hq,L,D]
K [B,H,S,D]
V [B,H,S,Dv]
```

根据设备和输入，它可能选择 math、memory-efficient 或 FlashAttention 类后端。项目不把 SDPA 当黑盒真理：

1. 手写 `QKᵀ/sqrt(D)`；
2. 应用相同 mask；
3. softmax；
4. 乘 V；
5. 比较 forward 和 input gradient。

### Mask 陷阱

SDPA 的布尔 `attn_mask=True` 表示允许参与；某些其他 API 中 `True` 表示屏蔽。cached decode 的非方形 mask 还涉及左右对齐问题，因此项目显式生成绝对位置 mask。

### 思考题 2：SDPA 输出有 1e-6 级差异是否一定是 bug？

<details>
<summary>答案</summary>

不一定。不同融合顺序、累积精度和 kernel 会导致浮点舍入差异。应根据 dtype 和规模设置容差，同时比较误差是否有限、是否随输入异常放大、梯度是否对齐。

</details>

## 3. FlashAttention 的核心不是“少算 softmax”

普通 attention 常显式生成 `[T,T]` score/probability，反复在 GPU 高带宽内存读写。FlashAttention 是 IO-aware 方法：分块读取 Q/K/V，在片上存储中计算，并避免完整 materialize score matrix。

### 3.1 Online softmax

对一行 score 分块。已经处理的块维护：

- `m`：当前最大值；
- `l`：以 `m` 为基准的指数和；
- `acc`：以同一尺度加权的 V 累加。

新块最大值为 `m_block`：

\[
m_{new}=\max(m,m_{block})
\]

旧统计换基准：

\[
\alpha=\exp(m-m_{new})
\]

新块概率分子：

\[
P_{block}=\exp(S_{block}-m_{new})
\]

更新：

\[
l_{new}=\alpha l+\sum P_{block}
\]

\[
acc_{new}=\alpha acc+P_{block}V_{block}
\]

最终：

\[
O=acc/l
\]

项目 `online_softmax_blockwise_attention()` 同时对 Q 和 KV 分块，低精度输入用 FP32 accumulator，FP64 测试保留 FP64。

### 3.2 正确性实验

- 不同 query/key tile size 输出一致；
- forward 对齐 SDPA；
- Q/K/V gradient 对齐 SDPA；
- causal mask 在每个 tile 内根据绝对位置生成。

### 3.3 性能边界

本机固定 `[1,4,128,32]` FP32 测得 Python tiled reference 明显慢于 SDPA。这不反驳 FlashAttention，因为 Python loop 没有 kernel fusion，也没有实现片上内存调度。

### 思考题 3：online softmax 是否是近似 softmax？

<details>
<summary>答案</summary>

不是。忽略浮点舍入，它通过重标定维护与一次性 softmax 相同的最大值、分母和加权和，是精确重排，而不是截断或核近似。

</details>

## 4. `torch.compile`

`torch.compile` 主要阶段可以粗略理解为：

```text
Python execution
→ Dynamo 捕获 FX graph
→ AOTAutograd 处理 forward/backward
→ backend 生成/选择实现
```

### 4.1 真实 graph break

项目第一次 `fullgraph=True` 失败位置：

```python
if positions.numel() and (positions.min() < 0 or positions.max() >= max_seq_len):
```

这是用 Python `if` 判断 Tensor 数据。编译器不能把任意数据依赖 Python 控制流捕获为静态图。

修正：

- sequence length 的范围由上层 model/attention shape 契约验证；
- RoPE 内部保留 rank/shape 检查；
- 不在热路径把 Tensor value 转成 Python bool。

修正后 `backend="eager", fullgraph=True` 输出误差为 0。

### 4.2 为什么 Inductor 没有性能结论

当前 Windows 环境运行 Inductor 时明确失败：没有可工作的 Triton 安装。课程不安装非官方 Windows Triton fork 来伪造完成，因此只声明 graph capture 通过，不声明 compiled speedup。

### 思考题 4：把 `torch._dynamo.config.suppress_errors=True` 是否等于修复 graph break？

<details>
<summary>答案</summary>

不等于。它可能静默回退到 eager，让程序运行但失去编译收益，也掩盖未捕获原因。教学阶段应使用 `fullgraph=True` 暴露 break，并修复或明确记录边界。

</details>

## 5. INT8 Weight-only 量化

### 5.1 对称量化

FP32 weight `w` 映射到 int8：

\[
s=\frac{\max|w|}{127}
\]

\[
q=\operatorname{clamp}(\operatorname{round}(w/s),-127,127)
\]

反量化：

\[
\hat w=qs
\]

### 5.2 Per-tensor 与 per-channel

- per-tensor：整个矩阵一个 scale；
- per-channel：每个输出 channel 一组 scale。

若各行动态范围差异大，一个全局 scale 会让小范围行使用很少的量化等级。测试构造不同行尺度，要求 per-channel MSE 更小。

### 5.3 项目 reference

`Int8WeightOnlyLinear` 保存：

```text
qweight int8 [out,in]
scale float32 [out,1]
bias optional
```

forward 时重新反量化，再调用 `F.linear`。因此它能验证存储和误差，不是高性能 INT8 kernel。

本机固定实验：

- FP32 weight：131072 bytes；
- INT8 values + scale：33280 bytes；
- 比例约 0.2539；
- 最大输出绝对误差约 0.00985。

这些数字绑定当前随机矩阵，不能外推为模型精度。

### 思考题 5：为什么实际比例不是精确 25%？

<details>
<summary>答案</summary>

int8 values 是 FP32 的四分之一，但还需要保存每个输出 channel 的 FP32 scale，因此略高于 25%。真实格式还可能有 zero point、group metadata、padding 和 packing。

</details>

## 6. DDP：数据并行到底同步什么

DistributedDataParallel 为每个 process 放置一份模型：

1. 每个 rank 使用不同 mini-batch；
2. 各自 forward/backward；
3. 对梯度做 all-reduce；
4. 每个 rank 得到相同平均梯度；
5. 各自执行相同 optimizer step。

它不是把一层模型自动切到多卡；那属于 tensor/pipeline/FSDP 等不同并行策略。

### 6.1 项目实验

`scripts/stage2_ddp_smoke.py` 启动两个 CPU/Gloo process，输入不同，backward 后 all-gather 每个 replica 的 Linear weight gradient，要求逐元素完全相同且有限。

首次失败：Windows PyTorch wheel 未编译 libuv，但默认 TCPStore 请求 libuv。固定项不变，仅设置：

```python
os.environ["USE_LIBUV"] = "0"
```

重跑后通过。

### 思考题 6：两个 rank 的 loss 是否必须相同？

<details>
<summary>答案</summary>

不必。它们使用不同数据，local loss 和同步前 local gradient 可以不同。DDP 保证 all-reduce 后用于更新的 gradient 一致。

</details>

## 7. 为什么需要 C++/CUDA 自定义算子

选择顺序：

```text
PyTorch 现有算子能否表达？
→ 能：先用 PyTorch
性能 profile 是否证明瓶颈，或是否必须接外部代码？
→ 是：再创建 custom op
```

课程仍要求完成一个自定义算子闭环，因为目标包括理解 PyTorch dispatcher 和底层 kernel，而不是因为 `silu_mul` 必然比官方融合器快。

## 8. `silu_mul` 的前向与反向

前向：

\[
y=\operatorname{SiLU}(g)\odot v
\]

其中：

\[
\operatorname{SiLU}(g)=g\sigma(g)
\]

导数：

\[
\frac{d\operatorname{SiLU}}{dg}
=\sigma(g)\left[1+g(1-\sigma(g))\right]
\]

因此：

\[
\frac{\partial L}{\partial g}
=\frac{\partial L}{\partial y}\odot v\odot
\sigma(g)[1+g(1-\sigma(g))]
\]

\[
\frac{\partial L}{\partial v}
=\frac{\partial L}{\partial y}\odot\operatorname{SiLU}(g)
\]

项目 backward 在 Python `torch.library.register_autograd()` 中注册，CPU/CUDA 只实现 forward。这种分层便于先验证公式。

### 思考题 7：为什么不直接依赖 C++ forward 自动生成 backward？

<details>
<summary>答案</summary>

自定义 dispatcher op 默认没有 Autograd 规则。必须显式注册 backward，或把 op 定义为可由 Autograd 追踪的 composite。课程选择显式公式，便于学习和 gradcheck。

</details>

## 9. Dispatcher

`silu_mul.cpp` 中：

```cpp
TORCH_LIBRARY(forgellm_ops, library) {
  library.def("silu_mul(Tensor gate, Tensor value) -> Tensor");
}
```

定义 schema。然后：

```cpp
TORCH_LIBRARY_IMPL(forgellm_ops, CPU, library) { ... }
TORCH_LIBRARY_IMPL(forgellm_ops, CUDA, library) { ... }
```

为相同 op 名注册不同 dispatch key。Python 调用：

```python
torch.ops.forgellm_ops.silu_mul(gate, value)
```

dispatcher 根据 Tensor device 选择 CPU 或 CUDA。

输入契约明确检查：

- shape 相同；
- dtype 相同且浮点；
- device 相同；
- contiguous。

### 思考题 8：为什么不能在 CPU 实现中偷偷把 CUDA Tensor `.cpu()`？

<details>
<summary>答案</summary>

这会引入隐式设备拷贝、同步和巨大性能损失，也破坏 dispatcher 的设备契约。CPU/CUDA 应分别实现，错误设备应立即拒绝。

</details>

## 10. CUDA kernel

项目 kernel 采用一维映射：

```text
index = blockIdx.x * blockDim.x + threadIdx.x
```

每个 thread 处理一个元素。block size 固定 256，grid 覆盖 `numel`。

使用 `AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16, ...)` 支持 FP16/BF16/FP32/FP64；计算类型通过 `at::acc_type` 提升，最后写回原 dtype。

这只是正确性优先的 elementwise kernel。后续 benchmark 若不比 PyTorch 快，应如实记录：官方 PyTorch/编译器可能已经融合或对小 shape 更高效。

### 思考题 9：kernel launch 后为什么要检查 `C10_CUDA_KERNEL_LAUNCH_CHECK()`？

<details>
<summary>答案</summary>

CUDA launch 是异步的。显式检查能尽早暴露非法配置或 kernel 错误，否则异常可能延迟到无关的后续操作，增加定位难度。

</details>

## 11. FakeTensor 与 `opcheck`

编译/导出系统常在没有真实数据的 FakeTensor 上推断 shape、dtype 和 device。项目 fake rule：

```python
return torch.empty_like(gate)
```

同时检查输入元数据契约。

`torch.library.opcheck()` 检查 schema、autograd registration、FakeTensor 等注册一致性；`gradcheck()` 用有限差分检查解析梯度。两者互补，不能互相替代。

### 思考题 10：forward 与 reference 对齐后为什么还需要 gradcheck？

<details>
<summary>答案</summary>

forward 正确不保证 backward 公式正确。训练中错误梯度可能不报错，只会让优化方向偏离。gradcheck 用数值有限差分提供独立证据。

</details>

## 12. Windows 构建现状

已经具备：

- CUDA Toolkit 12.4；
- PyTorch `2.6.0+cu124`；
- Ninja；
- CUDA-capable RTX 4070；
- C++/CUDA source、loader、FakeTensor、Autograd 与构建验证脚本。

初次验收时缺少 Visual Studio C++ 编译器 `cl.exe`；自动 passive 安装因没有 UAC elevation 以 installer code `5007` 退出。之后学习者已通过 Visual Studio Installer 安装：

```text
使用 C++ 的桌面开发
```

至少包括：

- MSVC v143 x64/x86 build tools；
- Windows SDK；
- C++ CMake tools（推荐）。

当前验证命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_silu_mul.py
```

脚本会进入 `VsDevCmd.bat` 环境，编译 CPU/CUDA，并运行 forward、gradcheck、opcheck、空 Tensor 与 benchmark。2026-07-27 的真实构建已经通过。

### 思考题 11：源码已经写完，为什么 G2-Systems 仍不能标记完成？

<details>
<summary>答案</summary>

自定义算子最关键的风险就在编译、链接、ABI、dispatcher 和 device kernel。初版源码完成时尚不能关闭门禁；只有真实构建以及 forward、gradcheck、opcheck、边界和 benchmark 全部通过后才可关闭。本项目现在已经留下这些证据。

</details>

## 13. 公平 Benchmark 协议

必须固定：

- 同一硬件；
- 同一 shape；
- 同一 dtype/device；
- 同一 contiguous 状态；
- 同一 forward 或 forward+backward 范围；
- warmup；
- 重复次数；
- CUDA 前后同步；
- 峰值显存测量范围。

项目 `benchmark_callable()` 在 CUDA 测量前后调用 synchronize，并记录 repetitions 与 peak allocated memory。

### 思考题 12：为什么直接用 `time.time()` 包住一次 CUDA op 通常不可信？

<details>
<summary>答案</summary>

CUDA 默认异步，CPU 计时可能只测到 kernel launch；第一次运行还包含初始化/JIT/cache。必须预热、同步并重复，才能得到可解释的设备执行时间。

</details>

## 14. 当前系统证据边界

已证明：

- SDPA 与手写 attention 对齐；
- online-softmax forward/backward 与 SDPA 对齐；
- Decoder 可被 Dynamo fullgraph 捕获；
- INT8 reference 的误差/存储可测；
- 两进程 Gloo DDP 梯度一致；
- `silu_mul` Python reference forward/backward 正确；
- `silu_mul` C++ CPU 与 CUDA kernel 已真实编译、链接和加载；
- CPU/CUDA forward 最大绝对误差为 0，gradcheck 与四项 opcheck 通过；
- 空 Tensor 通过；固定 1,048,576 个 FP32 元素时，当前最终记录测得 CPU 约 1.022×、CUDA 约 1.761×。

尚未证明：

- Windows Inductor/Triton 加速；
- 多 GPU NCCL；
- 生产量化 kernel 或真实模型精度。

这些未完成项仍必须保留，不能用已通过的单机算子证据替代。自定义算子的速度结论只绑定当前 GPU、shape、dtype、预热与重复次数。
