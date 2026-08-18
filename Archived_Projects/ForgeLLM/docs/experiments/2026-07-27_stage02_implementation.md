# Stage 2：Transformer、现代架构与模型系统实现记录

> 后续证据更正（2026-07-28）：实施时登记的 DeepSeek V4/Kimi K3 新鲜度结论已被下一次 F0 审计更新。当前 Kimi K3 已有官方仓库/报告；DeepSeek V4 未找到可核验官方技术报告。Stage 2 的 lite 模块仍是有效教学代码，但不再归因 DeepSeek V4；本历史记录不删除。

> 日期：2026-07-27（Asia/Singapore）  
> 状态：G2-Core、G2-Arch、G2-Systems 自动化实现与学习者验收全部完成  
> 主要产物：`artifacts/stage02/stage2_model_lab_report.json`  
> 结论边界：这是微型 reference、正确性测试与本地系统 Smoke，不是大模型训练结果，也不是官方模型复现

## 1. 目标与问题

本轮不以训练出高质量模型为目标，而是回答以下问题：

1. 能否从公式实现一个可前向、反向、生成和缓存的 Decoder-only Transformer？
2. 手写 attention、PyTorch SDPA 与分块 online-softmax 是否数值一致？
3. MLA、MoBA、混合线性注意力、MoE、MTP 等现代方法能否缩小为可测试的教学 reference？
4. `torch.compile`、INT8、DDP 与自定义 C++/CUDA 算子的工程边界分别是什么？

## 2. 环境

| 项目 | 实际值 |
|---|---|
| 操作系统 | Windows |
| Python | 3.12.3，项目 `.venv` |
| PyTorch | `2.6.0+cu124` |
| PyTorch 编译 CUDA | 12.4 |
| 本机 CUDA Toolkit / nvcc | 12.4 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| 显存 | 8188 MiB |
| Ninja | 1.11.1.4 |
| MSVC `cl.exe` | 19.42.34433，Visual Studio 2022 Community |

模型依赖安装入口：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2.1 F0 前沿新鲜度复核

2026-07-27 的原 F0 记录后来被证据复审推翻，不再作为当前事实依据：当时登记的 DeepSeek V4 链接/技术归因未能在 2026-07-28 找到可核验的 DeepSeek 官方技术报告；同日 Moonshot AI 已发布 Kimi K3 官方仓库/报告。Stage 2 代码只保留为独立教学 reference，当前事实快照见 `docs/frontier_model_technology_registry.md`。

## 3. 实现矩阵

### 3.1 G2-Core

| 能力 | 实现 | 主要证据 |
|---|---|---|
| 配置与不变量 | `ModelConfig` | 非法维度、head 整除和后端校验测试 |
| 原子层 | RMSNorm、RoPE、SwiGLU | shape、FP64 gradcheck、旋转性质 |
| Attention | MHA、GQA、QK-Norm、manual/SDPA | 数值/梯度对照、因果性、KV Cache |
| Decoder | Pre-Norm Block、权重共享、shifted CE | forward/backward、人工 loss、保存加载 |
| 生成 | greedy、temperature、top-k、top-p | cache/no-cache 和采样边界测试 |
| 学习性 | 微型 Decoder | Tiny Set Overfit 集成测试 |

### 3.2 G2-Arch

| 方法 | 教学 reference 验证内容 | 不宣称的内容 |
|---|---|---|
| Linear / Dynamic-NTK RoPE | 频率缩放与 shape | 百万上下文质量 |
| MLA | 压缩 latent KV、增量缓存 | DeepSeek 官方实现或效果复现 |
| MoBA | block 级 top-k 因果稀疏选择 | 高性能稀疏 kernel |
| CSA/HCA-lite | 压缩分支与窗口分支融合 | Kimi 官方完整实现 |
| Gated DeltaNet / Hybrid | 递归状态更新、3:1 组合 | 大规模训练稳定性 |
| Sparse MoE | top-k、共享专家、负载辅助项 | 专家并行与生产吞吐 |
| Hash / Sinkhorn | 确定性路由、近似平衡 | 官方路由器复现 |
| mHC-lite / Attention Residuals | 残差混合约束与多流读写 | 官方训练收益 |
| MTP | 多 offset target 与辅助 loss | 推测解码端到端加速 |
| Muon | Newton–Schulz 正交化更新 | 大规模优化器结论 |

### 3.3 G2-Systems

| 能力 | 状态 | 证据/边界 |
|---|---|---|
| SDPA 对照 | 通过 | manual 与 SDPA 前向/梯度对齐 |
| online-softmax | 通过 | 不同 tile 的前向/梯度对齐；Python reference 不代表 Flash kernel |
| `torch.compile` | 部分通过 | `backend="eager", fullgraph=True` 单图通过；Inductor 因 Windows 无可工作 Triton 未验证 |
| INT8 weight-only | 通过 | 固定层的误差与参考存储量报告 |
| 双进程 DDP | 通过 | CPU/Gloo 两进程真实梯度同步；Windows 设置 `USE_LIBUV=0` |
| C++/CUDA `silu_mul` | 通过 | CPU/CUDA forward、gradcheck、四项 opcheck、空 Tensor 与固定 benchmark 通过 |

## 4. 固定实验与结果

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\stage2_model_lab.py
```

报告位置：`artifacts/stage02/stage2_model_lab_report.json`。

### 4.1 5M 级 Decoder GPU Smoke

固定配置：词表 320、`d_model=256`、7 层、8 个 query head、2 个 KV head、FFN 768、序列上限 128、QK-Norm、SDPA。

| 指标 | 结果 |
|---|---:|
| 参数量 | 5,361,856 |
| 输入/输出 | 输入 `[2, 64]`，logits `[2, 64, 320]` |
| 随机初始化 loss | 5.754850 |
| 梯度有限 | 是 |
| 峰值已分配显存 | 60.534 MiB |

随机初始化 loss 只证明计算链可执行，不能评价语言建模质量。

### 4.2 Attention 数值与性能 Smoke

固定 shape 为 `[batch=1, heads=4, sequence=128, head_dim=32]`，FP32、同一 GPU 输入。

| 项目 | 结果 |
|---|---:|
| manual vs SDPA 最大绝对误差 | `1.490116e-6` |
| Python tiled vs SDPA 最大绝对误差 | `9.536743e-7` |
| manual 平均时间 | 0.09067 ms |
| SDPA 平均时间 | 0.01118 ms |
| Python tiled 平均时间 | 2.45523 ms |

这里唯一可靠的性能结论是“在该固定输入上，PyTorch SDPA 比两个 Python reference 快”。不能据此估计真正 FlashAttention kernel 的加速比例。

### 4.3 架构结构量

| 项目 | 结果 |
|---|---:|
| MLA reference 每 token 缓存元素 | 16 |
| 同配置 MHA 每 token KV 元素 | 128 |
| 元素数比例 | 0.125 |
| MoE token 数 | 64 |
| `top_k` | 2 |
| 总 assignment | 128，严格等于 `64 × 2` |
| 各专家 assignment | `[11, 17, 14, 20, 19, 13, 17, 17]` |
| 负载辅助项 | 1.010174 |

缓存元素减少不等价于端到端显存按同比例减少，也不证明模型质量保持不变。

### 4.4 INT8 reference

固定 Linear 权重为 `[128, 256]`，采用对称 per-output-channel INT8 weight-only reference。

| 指标 | 结果 |
|---|---:|
| FP32 权重字节 | 131,072 |
| INT8 权重加 scale 字节 | 33,280 |
| 参考存储比例 | 0.25390625 |
| 最大绝对误差 | 0.0098524 |
| 平均绝对误差 | 0.00177088 |

该实现会在前向反量化，目标是理解量化误差与存储账本，不代表高性能推理 kernel。

### 4.5 `torch.compile`

一层 Decoder 在 `torch.compile(model, backend="eager", fullgraph=True)` 下与 eager 最大误差为 0。初版 RoPE 在图内对 Tensor 做 Python `if`，导致 data-dependent graph break；删除该冗余判断并把长度校验保留在 attention 边界后通过。

Inductor 未通过：官方 Windows 环境没有可工作的 Triton。当前只声明 Dynamo 单图捕获，不声明 Inductor 加速。

### 4.6 DDP

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\stage2_ddp_smoke.py
```

两个真实 CPU/Gloo 进程完成相同模型的反向和 all-reduce，rank 间梯度一致。首次运行因当前 Windows PyTorch 不带 libuv 支持失败；设置 `USE_LIBUV=0` 后通过。该实验不代表 NCCL 或多 GPU 吞吐。

## 5. 失败与修复记录

| 失败 | 原因 | 处理 | 当前状态 |
|---|---|---|---|
| PyTorch 下载依赖解析失败 | 仅使用 PyTorch wheel index，缺少通用 PyPI 依赖 | PyPI 为主 index，PyTorch CUDA 为 extra index | 已解决 |
| RMSNorm FP64 gradcheck 失败 | 无条件转 FP32 丢失 FP64 精度 | 只对 FP16/BF16 提升到 FP32 | 已解决 |
| online-softmax FP64 对照误差过大 | 同样无条件降为 FP32 | 保留 FP64 accumulation | 已解决 |
| `torch.compile` graph break | 图内 Tensor 数据触发 Python 分支 | 移除冗余数据依赖判断 | 已解决 |
| DDP rendezvous libuv 错误 | Windows PyTorch 构建不支持 libuv | `USE_LIBUV=0` | 已解决 |
| Inductor 无法编译 | Windows 官方环境无可工作的 Triton | 不安装非官方替代包；保留边界 | 未解决、非 G2 原生算子门 |
| C++/CUDA extension 无法编译 | Visual Studio 最初未安装 MSVC C++ workload | 用户安装“使用 C++ 的桌面开发” | 已解决 |
| `VsDevCmd` 后 Python 仍找不到 `cl.exe` | 当前 Codex 进程保留了原始 PATH | 构建脚本根据 `VCTOOLSINSTALLDIR` 补入编译器目录 | 已解决 |
| Autograd `setup_context` 报未知关键字 `ctx` | PyTorch 2.6 以 `ctx=` 调用，函数参数名曾写为 `context` | 参数契约改为 `ctx` | 已解决 |
| 独立 extension pytest 找不到 Ninja | 普通 PowerShell 未暴露虚拟环境 Scripts | loader 自动补入 venv 与 MSVC executable path | 已解决 |
| 空 CUDA Tensor 设备断言失败 | 测试把逻辑 `cuda` 与具体 `cuda:0` 直接比较 | 按 device type 验证并保留 shape 检查 | 已解决 |
| 固定 `sm_89` 后普通 pytest 触发 CUDA 重编失败 | 普通 pytest 不是完整 `VsDevCmd` 编译环境 | 先由正式构建脚本重编，再独立运行 opt-in 全仓门 | 已解决 |

## 6. 自定义算子阻塞解除与验收

复现命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_silu_mul.py
```

Visual Studio Installer 已安装“使用 C++ 的桌面开发”、MSVC x64/x86 Build Tools 与 Windows SDK。构建脚本能够进入 `VsDevCmd.bat`，完成 C++、CUDA 编译和 `.pyd` 链接。独立测试命令：

```powershell
$env:FORGELLM_TEST_EXTENSION = "1"
.\.venv\Scripts\python.exe -m pytest tests\integration\test_silu_mul_extension.py -q
```

结果为 `3 passed`。构建报告保存在 `artifacts/stage02/silu_mul_extension_report.json`。

| 设备 | 最大绝对误差 | gradcheck | 四项 opcheck | Reference | Extension | 加速比 |
|---|---:|---|---|---:|---:|---:|
| CPU | 0.0 | 通过 | 全部 SUCCESS | 2.17217 ms | 2.12449 ms | 1.022× |
| CUDA | 0.0 | 通过 | 全部 SUCCESS | 0.16477 ms | 0.09355 ms | 1.761× |

Benchmark 固定为 1,048,576 个 FP32 元素、5 次预热、20 次重复；CUDA 测量前后同步。该数字只适用于当前硬件、shape、dtype 与实现，不能外推为通用加速比例。

## 7. 当前结论

- 设置 `FORGELLM_TEST_EXTENSION=1` 后，全仓 `scripts/dev.py check` 通过：Ruff format/lint、mypy strict 无错误，156 项测试全部通过；
- G2-Core 的代码与自动化正确性证据完成；
- G2-Arch 的缩小算法 reference、测试与讲义完成；
- G2-Systems 的 SDPA/online-softmax、compile 单图、INT8、DDP 和原生算子门全部通过；
- Inductor 是平台能力边界，不作为手写算子学习的替代品；
- Stage 2 的全部自动化工程任务已经完成；
- 用户已于 2026-07-27 确认完成 Stage 2 学习与测试，完整 Stage 2 已关闭。

## 8. 下一步

1. 学习者从 `docs/lessons/stage02_learning_order.md` 第 1 站开始，不并行阅读所有源码；
2. 完成讲义中紧跟答案的思考题、代码追踪和最小改动实验；
3. 完成第 13 站 C++/CUDA 构建复现与结果解释；
4. 学习者验收通过后关闭完整 G2，再进入 Stage 3 预训练循环。
