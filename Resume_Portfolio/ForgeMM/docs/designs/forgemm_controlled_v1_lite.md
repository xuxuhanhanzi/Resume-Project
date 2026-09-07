# ForgeMM-Controlled v1-Lite：本地 8GB 视觉证据实验注册

> 注册日期：2026-08-24  
> 状态：已授权，待本地 GPU smoke  
> 与旧 ChartQA Stage-4 数据门禁关系：新数据/评测版本，不修改旧门禁。

## 目的与主张边界

旧 ChartQA 严格标签只有 val 106、test 123，不能承载“视觉证据有效性”的正式结论。
本注册改用一个完全由固定程序生成的图表环境：模型只接收图像和问题；评测器读取冻结的隐藏 manifest，检查证据值、来源 mark、预测 bbox、操作和答案。

因此，本实验可以主张“在受控生成图表上，ForgeMM 的 visual full-pass 改变”。它不能把
ChartQA 中表格推导出的标签称为视觉证据，也不能将该受控结果外推为真实世界图表 SOTA。
ChartQA/ChartQAPro 仅保留为自然图表的答案泛化评测。

## 数据合同

- 生成器：`forgemm-controlled-pillow-1.0.0`；Pillow-only，无 LLM 或人工标注器。
- 图像类型：垂直柱状图、折线图；每个 mark 带可见数值、类别标签和精确 raster bbox。
- 操作：`lookup`、`difference`、`sum`、`average`、`argmax`、`argmin`、`compare`。
- 切分：train 5,000、val 600、test 1,000；以 split-specific SHA-256 seed 生成。
- 每条 oracle 保存：图像 SHA-256、生成 seed、类别和值、source id、bbox、gold operation、答案。
- 测试 prompt 文件没有答案、gold operation、evidence 或 oracle 字段；训练前运行审计器。

## 视觉 full-pass

一个输出同时满足以下条件才计为 full-pass：

1. XML 和 bbox 语法合法；
2. 引用的 row、column、value 与 gold mark 完全一致；
3. 每个预测 bbox 对匹配 gold bbox 的 IoU 至少为 0.50；
4. 操作名、操作数顺序和白名单执行结果都与 oracle 一致；
5. 最终答案与可执行结果和 reference answer 一致。

只答对数值、引用了正确值但 bbox 错、或使用另一种碰巧得到相同答案的操作，都不是 full-pass。
错误代码不得包含 gold 数值或 bbox。

## 本地 Lite 资源合同

运行环境为本机 WSL2 Ubuntu，RTX 4070 Laptop（8,188 MiB）、32GB RAM。开始训练前 GPU
可用显存必须至少 6,500 MiB；关闭 Ollama 和占显存桌面程序由设备持有人完成。所有方法固定
Qwen2.5-VL-3B、NF4 QLoRA、冻结视觉编码器、batch=1、gradient checkpointing、同一
`max_pixels`。

第一次 smoke 使用 `max_pixels=200704`、文本长度 1024、输出长度 160。若发生 OOM，唯一
允许的回退是将所有方法共同降到 `112896`，创建新的运行 manifest；不能只对某个方法降分辨率。
GRPO 固定 `num_generations=2`、generation batch=1、禁用 vLLM。

## 运行顺序与停止规则

1. 生成数据，运行 `audit_controlled_dataset.py`，得到 `decision=pass`；否则停止。
2. 建立 WSL 环境，保存 PyTorch/CUDA/ms-swift 的版本和空闲显存；显存不足或 import 失败则停止。
3. 一步 SFT smoke；只有显存峰值与日志完整时才运行 E0。
4. E0：answer-only SFT 与 structured visual SFT；两者所有非主变量完全相同。
5. Structured SFT 的格式率和 val full-pass 达到预注册门槛后，才运行 GRPO。
6. 每个正式方法使用 seed 17、42、2026；test 在 dev/val 选择一次后只运行一次。

若本地 8GB 不能通过一阶 SFT smoke，则本次注册只交付可审计数据和 verifier，不产生模型效果结论。
