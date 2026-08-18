# 创新模块映射

> 此处“创新”指项目中的个人增量模块，不代表论文级原创或 SOTA。

| 增量模块 | 主来源 | 补充来源 | 项目问题 | 计划改动 | 主 baseline | 工程 baseline | 预期收益 | 指标 | 风险 | 可行性 |
|---|---|---|---|---|---|---|---|---|---|---:|
| 多候选 Action Token 生成 | AutoVLA | OpenVLA/通用解码背景 | 单次解码可能选到偶然高风险轨迹 | 从相同模型分布生成 K 条候选并保存置信度 | 官方 AutoVLA | B0 fast greedy | 增加可选择的安全轨迹 | 有效候选数、invalid rate、PDMS、延迟 | 轨迹多样性不足或延迟过高 | 4 |
| 确定性风险重排序 | NAVSIM | UniAD/SparseDrive规划约束背景 | token 概率不等于驾驶安全 | 用 collision/DAC/TTC/progress/comfort 对候选逐项评分 | B0 | E1 K候选无重排 | 减少碰撞/越界失败 | NC、DAC、TTC、EP、PDMS | 规则分数与官方指标不一致 | 5 |
| 风险触发快慢路由 | AutoVLA | 选择性计算/不确定性背景 | 全部使用 slow mode 延迟高，全部 fast mode可能不安全 | 仅高风险或高不确定性样本进入 slow mode | B0/B1 | E2 | 更好的安全—延迟折中 | slow rate、P95 latency、PDMS、安全失败 | slow mode 不改变规划或解释幻觉 | 4 |
| 逐场景证据链评测 | NAVSIM | ForgeMM 工程模式 | 总分掩盖失败与配置漂移 | manifest、JSONL、断点恢复、配对分析 | 官方汇总结果 | B0 | 可复现和可解释的工程结论 | 完成率、配对变化、失败类型 | 上游 loader 跳样本 | 5 |
| DriveLM风格结构化解释 | DriveLM | Senna/OmniDrive | slow reasoning 难以审计 | 输出对象、风险、意图、动作的结构化记录 | E4 | 无 | 提高演示与失败分析质量 | 格式成功率、一致性 | 解释正确但轨迹错误 | 3，v0.2 |
| TensorRT部署 | OmniDrive | TensorRT-LLM | Python推理缺少车端工程信号 | 导出/适配并测量精度与延迟 | PyTorch E4 | 无 | 展示部署能力 | 延迟、显存、精度落差 | VLM算子/多模态接口不兼容 | 2，v0.2 |

## 核心假设

### H1：候选重排序

在模型和数据固定时，K 候选中存在比 greedy 轨迹更安全且不明显损害进度的候选。

- 支持证据：E2 相对 E1/B0 减少安全失败，EP 和 PDMS 未显著退化。
- 反证：有效候选高度同质，或安全改善完全来自静止/低进度策略。

### H2：选择性慢推理

风险/不确定性能够识别一部分 fast mode 易失败场景，且 slow mode 在这些场景上提供有用的替代轨迹。

- 支持证据：被触发场景中，slow mode 的安全失败率低于 fast mode，并且总体平均延迟受控。
- 反证：触发与失败无关，或 slow mode 只增加解释长度和延迟。

### H3：组合收益

重排序与路由解决不同问题，组合 E4 的收益不只是其中一个模块的重复效果。

- 支持证据：E4 相对 E2 和 E3 都在预注册指标上提供增量收益。
- 反证：E4 相对最佳单模块没有收益或延迟不可接受。

## 固定项

每个假设验证期间固定 checkpoint、数据清单、processor、codebook、生成长度、图像设置、NAVSIM 版本、GPU 和指标脚本。阈值只在 dev 清单上选择一次，formal 运行后不再更改。

