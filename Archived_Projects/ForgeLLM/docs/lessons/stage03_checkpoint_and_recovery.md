# Stage 3 完整讲义三：Checkpoint、精确恢复与故障诊断

## 1. Checkpoint 的定义

checkpoint 不是“模型权重备份”，而是训练状态机的快照。合格恢复必须让程序回答：

> 如果原进程没有中断，下一批数据、下一次随机数和下一次参数更新会是什么？

ForgeLLM 保存十类信息：模型、MTP head、optimizer、scheduler、scaler、trainer state、data cursor、Python RNG、Torch CPU/CUDA RNG、配置与 Tokenizer 指纹。

**思考题：只保存 `model.state_dict()` 能继续训练吗？**

答案：代码能跑，但不叫连续恢复。Adam 动量、LR 位置、随机数和数据位置都会重置，loss 曲线可能跳变，数据可能重复或跳过。

## 2. Trainer state 与数据 state

Trainer state 至少包含：

- `step`：完成了多少次 optimizer update；
- `tokens_seen`：完成了多少 target token；
- `elapsed_seconds`：已消耗墙钟预算。

数据 state 包含 dataset fingerprint、batch size、seed、epoch、position。dataset fingerprint 防止把旧 cursor 套到新数据上。

**思考题：为什么训练预算要保存 tokens seen，而不能只保存 step？**

答案：不同 batch、sequence length 和 accumulation 会让每步 token 数不同。若预算定义为 1M tokens，恢复后只看 step 可能超用或少用预算。

## 3. 指纹是兼容性门，不是装饰

加载前比较：

- resolved experiment config fingerprint；
- Tokenizer fingerprint；
- dataset fingerprint；
- checkpoint schema version。

只要不一致就拒绝恢复。强行加载“形状碰巧相同”的状态会生成无法解释的混合实验。

**思考题：只是把 `run_name` 改了，为什么配置指纹也会不匹配？**

答案：v1 选择让完整 resolved config 都参与指纹，以最大化可审计性。以后可以把“语义配置”和“展示元数据”分层，但不能在没有明确 schema 的情况下随意忽略字段。

## 4. 为什么需要原子保存

如果进程在 `torch.save` 写到一半时断电，目标文件可能存在但内容不完整。原子流程是：

```text
在同一目录创建唯一临时文件
→ 写入
→ flush
→ fsync
→ os.replace(temp, final)
```

同一文件系统中的 replace 要么留下旧完整文件，要么出现新完整文件，不暴露半写入目标。

**思考题：为什么临时文件要在同一目录？**

答案：跨文件系统移动不一定具备原子替换语义，也可能退化成复制再删除。

## 5. 恢复顺序

加载过程应先验证 envelope/schema/指纹，再修改内存状态。验证通过后依次恢复：

1. model 与 MTP head；
2. optimizer；
3. scheduler；
4. scaler；
5. trainer state；
6. data cursor；
7. RNG，最后恢复。

RNG 放最后，是因为创建模型、optimizer 或读取状态可能执行消耗随机数的操作。最后恢复才能保证下一次随机调用接上中断点。

**思考题：为什么本次真实 CUDA 恢复最初失败？**

答案：`torch.load(map_location=cuda)` 把保存的 CUDA RNG state ByteTensor 也映射到了 GPU，但 `torch.cuda.set_rng_state_all` 接口要求 CPU ByteTensor。修复为逐个 `.cpu()` 后再设置。这说明 checkpoint 正确性必须经过真实中断恢复，不能只靠“文件能保存”。

## 6. CPU exact-resume 实验

确定性 oracle 使用 CPU、FP32、dropout=0：

```text
A: 初始化 → 连续训练 N 步

B: 同种子初始化 → 训练 K 步 → 保存
   → 新建所有对象 → 加载 → 训练 N-K 步
```

比较：

- 每一步 Python float loss exact；
- 每个模型 Tensor `torch.equal`；
- optimizer nested state exact；
- scheduler/trainer state exact；
- 下一 batch indices 和 tokens exact。

CUDA kernel 可能存在非确定性，因此 GPU 通常使用容差和轨迹连续性；但数据 cursor 与 step 序号仍必须完全一致。

**思考题：为什么不能只比较恢复后的最终 loss？**

答案：不同参数和数据轨迹可能偶然得到相近 loss。逐状态和下一批比较才能定位恢复契约是否真正连续。

## 7. 故障分类

### 数据故障

- token ID 超出词表；
- cache 指纹不匹配；
- next batch 重复或跳过；
- train/validation 文档交叉。

### 数值故障

- loss NaN/Inf；
- 单个参数 gradient 非有限；
- gradient norm 突然异常；
- FP16 scaler 连续下降。

### 状态故障

- scheduler 与 step 错位；
- optimizer group 数不匹配；
- CUDA RNG device 错误；
- checkpoint schema 不兼容或文件损坏。

### 资源故障

- OOM；
- 磁盘不足；
- checkpoint 写入异常；
- wall-clock/token 预算超过上限。

每次失败都要保留：发生条件、错误、是否写入 optimizer、数据 cursor 已推进多少、修复与回归测试。

**思考题：loss 非有限时当前 batch cursor 已推进，但 optimizer 未 step，恢复应该怎么办？**

答案：默认直接终止并从最近一个完整 checkpoint 重新启动，不保存失败后的游标状态。否则会跳过坏 batch，改变实验轨迹。

## 8. 正式中断恢复证据

正式 run 的配置未改变：

- 第 1–125 步运行后使用 `manual_step_limit` 停止；
- checkpoint 保存 step=125、tokens=508,000；
- 新 Python 进程从该 checkpoint 加载；
- 日志从 step=126 继续；
- 第 247 步达到 1,003,808 target tokens；
- JSONL 中所有 train step 为连续的 1–247。

这个实验同时证明“能恢复”和“停止预算在恢复后仍延续”。

**思考题：为什么最终是 1,003,808，而不是精确 1,000,000 token？**

答案：每个 optimizer step 固定消费 4,064 target tokens，不能在不改变 batch 语义的情况下只执行半步。停止条件在完整更新后检查，因此最多超出一个 step 的 token 数。

## 9. 代码追踪顺序

1. `capture_rng_state` / `restore_rng_state`
2. `save_checkpoint_atomic` / `load_checkpoint`
3. `TrainerState.state_dict`
4. `DeterministicBatchStream.state_dict`
5. `Trainer._checkpoint_payload`
6. `Trainer.load_checkpoint`
7. `stage3_resume_equivalence.py`

学习完成后，应能在不看代码的情况下列出完整 checkpoint，并说明每项缺失会导致什么后果。
