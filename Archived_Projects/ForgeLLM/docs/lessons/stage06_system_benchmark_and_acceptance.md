# Stage 6 讲义四：系统指标、多维门禁与最终结论

## 1. 模型“答得对”之外还要验收什么

一个可用模型至少有四个独立维度：

- correctness：是否满足冻结任务；
- retention：后训练是否严重损害基础语言建模；
- stability：是否重复、截断或异常；
- efficiency：在目标硬件上的时延、吞吐与显存是否可接受。

Stage 6 故意不把四项加权成总分。因为权重会掩盖硬失败：一个几乎总输出错误的模型，不应靠更快推理补成“及格”。

<details>
<summary>思考题：为什么总分 80 不能告诉我们模型是否可部署？</summary>

80 可能来自正确性 50、速度 100 的平均，也可能来自正确性 90、速度 70；两者风险完全不同。部署约束通常是门禁而非可交换效用，所以应逐项报告和逐项通过。
</details>

## 2. 冻结系统矩阵

Q0/Q1/Q2 使用同一 GPU、BF16、greedy、`use_cache=True`，矩阵为：

- batch：1、4；
- prompt tokens：32、128、256；
- 每序列生成：32 token；
- 每格 1 次 warmup、5 次正式重复；
- 报告 p10、median、p90。

共 `3 models × 2 batch × 3 prompt = 18` 个格。Q3 只训练一步，不作为服务候选。

<details>
<summary>思考题：为什么系统矩阵不能只测 batch=1、prompt=32？</summary>

注意力与 KV-cache 成本会随上下文和 batch 改变。短 prompt、单用户的最快点不能代表并发或长上下文。小矩阵虽不完备，但能暴露扩展趋势。
</details>

## 3. TTFT 与端到端时延

Time to first token（TTFT）包括 prompt prefill 和产生首 token 的时间，直接影响交互等待。端到端时延包括完整 32-token 生成。二者必须分别测量，因为长 prompt 主要增加 prefill，而更多生成 token 主要增加 decode。

Stage 6 分别运行 1-token 和 32-token generation，避免用一个复杂计时钩子修改模型路径。代价是两次运行不是同一条请求，因此 decode rate 是近似值。

<details>
<summary>思考题：为什么需要 `torch.cuda.synchronize()`？</summary>

CUDA kernel 默认异步提交。如果 CPU 的计时器在 kernel 完成前停止，测到的只是排队时间。计时前后同步，才能把对应 GPU 工作包含进墙钟时延。
</details>

## 4. Decode tokens/s 怎样计算

近似稳态解码吞吐：

```text
decode_tok_s ≈ batch × (generated_tokens - 1)
               / (E2E_median - TTFT_median)
```

减去首 token，是为了把 prefill/首步与后续 decode 粗略分开。端到端吞吐另算：

```text
e2e_tok_s = batch × generated_tokens / E2E_median
```

两个数回答不同问题，都必须带 batch、prompt、生成长度和硬件环境。

<details>
<summary>思考题：若 TTFT 中位数大于 E2E 中位数怎么办？</summary>

这说明独立测量噪声、warmup 或计时异常使近似无效，不能得到负 decode 时间。实现会拒绝该输入；需要增加重复、检查后台负载或改用同一请求内的 token 级计时。
</details>

## 5. 为什么报告 p10/p50/p90

单次时延会被首次 kernel 编译、缓存、后台进程和温度影响。Warmup 处理最明显的冷启动，5 次正式重复再报告分位数：

- p50 代表典型值；
- p90 暴露慢尾部；
- p10 展示最好附近的稳定区间。

五次只能做轻量本地验收，不足以估计生产服务的 p99。

<details>
<summary>思考题：为什么不只报告最快一次？</summary>

最快值选择性忽略噪声和尾延迟，通常不可复现。用户体验更接近分布而非纪录值；生产 SLA 尤其关心慢尾部。
</details>

## 6. GQA 的 KV-cache 手算

每层缓存 K 和 V，所以有系数 2：

```text
KV bytes = 2 × layers × batch × sequence_length
           × num_kv_heads × head_dim × bytes_per_element
```

其中 `head_dim = hidden_size / num_attention_heads`。GQA 使用 `num_kv_heads` 而不是 query heads；这正是 GQA 降低 cache 的关键。Stage 6 计算的是 BF16 理论 K/V，不含 allocator、临时张量、权重、logits 和框架开销，并同时记录 CUDA peak allocated memory。

<details>
<summary>思考题：为什么理论 KV bytes 与显存峰值差很多并不矛盾？</summary>

峰值还包含 0.6B 权重、Adapter、激活、注意力 workspace、生成输出、PyTorch allocator 保留和其他张量。理论式只隔离 KV 本体，用于理解随 batch/context 的缩放关系。
</details>

## 7. Stage 6 的模型门禁

每个 Q0/Q1/Q2 独立判断：

- correctness：原题 strict success ≥ 0.5；
- retention：loss/token 不超过 Q0 的 1.1 倍；
- stability：平均字符 8-gram 重复 ≤ 0.25；
- efficiency：系统矩阵完整。

`model_acceptance()` 返回四个布尔 gate 和状态字符串，不返回 total score。阈值是本项目的工程教学阈值，不是行业通用标准。

<details>
<summary>思考题：Q1 通过三项、只因正确率失败，最终状态应是什么？</summary>

“training pipeline accepted, model behavior not accepted”。这承认数据、训练和评测链可工作，同时拒绝把不满足核心任务的模型包装成可接受行为模型。
</details>

## 8. 自动化门与学习者门

自动化 G6-A～G6-E 检查：

- 评测身份是否冻结；
- 是否有 192 条原始生成；
- 是否无精确污染；
- 18 个系统格是否完整；
- 每个声明是否有存在的证据文件；
- Q3 是否保持 pipeline-only 边界。

G6-L 要求学习者完成盲评、统计解释、代码追踪和口述。自动门通过不自动关闭学习者门。

<details>
<summary>思考题：为什么 near-match 不直接使自动门失败？</summary>

本数据的 near-match 主要来自预先设计的同模板、不同槽位案例。把所有候选自动视为泄漏会错误删除目标任务分布。正确做法是完整披露、人工复核并限制外推；精确答案/文本重合仍是硬门。
</details>

## 9. 声明—证据—边界

一个合格结论包含三部分：

```text
声明：在冻结的 64 例协议下，Q1 的 strict rate 为 X。
证据：quality_v2/report.json + raw_generations.jsonl + manifest.json。
边界：小型项目内数据；不外推为通用指令能力。
```

`EvidenceClaim` 要求每条声明至少一个证据路径和非空边界；最终构建器检查文件存在。它不能自动判断论证是否充分，所以学习者仍需审查声明是否超出证据。

<details>
<summary>思考题：有证据路径是否就等于声明正确？</summary>

不等于。路径存在只证明可追踪；证据可能不相关、统计不足或被错误解释。自动审计解决“无来源”，人工推理解决“来源能否支持这句话”。
</details>

## 10. 如何阅读冲突结果

常见组合及正确表述：

- Pair margin 改善、strict 不变：偏好代理改善，生成能力改善未证实；
- NLL 改善、重复恶化：teacher-forced 拟合增强，但自由生成稳定性退化；
- 原题成功、robustness 下降：任务记忆或措辞敏感；
- 质量相同、系统更慢：Adapter 没有证明行为收益，却增加运行负担；
- 所有区间跨 0：当前样本不足以稳定区分，不等于模型相同。

<details>
<summary>思考题：最诚实的负结果标题是什么？</summary>

例如“在冻结本地约束集上，DPO 提升 teacher-forced pair ranking，但未通过生成与保留门禁”。它同时保留成功的机制证据和失败的行为证据，不把任何一方隐藏。
</details>

## 11. 本讲义代码路线

1. `systems.py` 的三个纯函数；
2. `system_runner.py` 的矩阵循环与 CUDA 同步；
3. `report.py::model_acceptance`；
4. `finalize.py::build_final_report`；
5. `artifacts/stage06/final_v6/evaluation_card.md`。
