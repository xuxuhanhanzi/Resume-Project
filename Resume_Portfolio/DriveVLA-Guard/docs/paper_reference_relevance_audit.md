# 参考项目与论文相关性审计

> 审计日期：2026-08-11  
> 选择原则：优先近三年顶会/高质量官方实现，要求代码、权重、数据或评测至少有一项能直接支持本项目。

| 项目/论文 | 年份/出处 | 代码 | 主要贡献 | 本项目相关性 | 角色 | 决策与风险 |
|---|---|---|---|---|---|---|
| AutoVLA | NeurIPS 2025 | [GitHub](https://github.com/ucla-mobility/AutoVLA) | Action Token、快慢推理、SFT、GRPO、驾驶轨迹生成 | 强 | 主模型来源 | 采用；完整 reasoning data 可能不可得，不能声称训练复现 |
| NAVSIM / Pseudo-Simulation | NeurIPS 2024 / CoRL 2025 | [GitHub](https://github.com/autonomousvision/navsim) | 规划 benchmark、PDMS/EPDMS、伪闭环评测 | 强 | 主评测来源 | 采用；结论只适用于其数据与模拟协议 |
| DriveLM | ECCV 2024 Oral | [GitHub](https://github.com/OpenDriveLab/DriveLM) | Graph VQA、感知—预测—规划推理链 | 中 | 解释与数据背景 | v0.1 只读；结构化解释作为后续扩展 |
| Senna | 2024 technical report | [GitHub](https://github.com/hustvl/Senna) | 多视角驾驶 VLM 与端到端规划结合 | 中 | 多视角背景 | 只读；不引入其 LLaVA/Vicuna 依赖 |
| UniAD | CVPR 2023 Best Paper | [GitHub](https://github.com/OpenDriveLab/UniAD) | 规划导向的感知、预测、规划统一建模 | 中/背景 | 自动驾驶基础 | 阅读并运行可视化可选；不作为 v0.1 运行依赖 |
| SparseDrive | ICRA 2025 | [GitHub](https://github.com/swc-17/SparseDrive) | 稀疏场景表示与端到端驾驶 | 中/背景 | 3D空间背景 | 只读；其稀疏表示不在 v0.1 复现 |
| UniDriveVLA | 2026 technical report | [GitHub](https://github.com/xiaomi-research/unidrivevla) | 理解、感知、规划专家解耦；Qwen3-VL；开/闭环评测 | 中/前沿 | 未来架构参考 | 代码新且复杂；不作为短周期主线 |
| OmniDrive | CVPR 2025 | [GitHub](https://github.com/NVlabs/OmniDrive) | 3D grounding、反事实推理、规划、TensorRT | 中 | 部署和推理参考 | TensorRT 延后到 v0.2 |
| Bench2Drive | NeurIPS 2024 D&B | [GitHub](https://github.com/Thinklab-SJTU/Bench2Drive) | CARLA 闭环、多能力评测 | 中 | 后续闭环参考 | 数据和环境成本较高，v0.1 暂缓 |
| ReSim | NeurIPS 2025 Spotlight | [GitHub](https://github.com/OpenDriveLab/ReSim) | 动作条件驾驶世界模型 | 背景 | 世界模型方向 | 训练成本高，不纳入 v0.1 |
| WorldEngine | 2026 technical report | [GitHub](https://github.com/OpenDriveLab/WorldEngine) | 长尾挖掘、3DGS仿真、RL post-training | 背景 | 数据闭环方向 | 体系过大，作为面试前沿阅读 |

## 主来源与补充来源边界

- 多候选 Action Token 与快慢推理：主来源是 AutoVLA。
- 轨迹安全和正式指标：主来源是 NAVSIM；内部 risk score 只是推理选择器。
- 结构化驾驶解释：DriveLM 是补充来源，v0.1 不将其描述为已实现。
- 多视角 VLM—规划协同：Senna 是背景和接口参考。
- 3D空间表示：UniAD、SparseDrive 和 UniDriveVLA 是背景来源，不是 v0.1 直接实现来源。
- TensorRT：OmniDrive 是部署参考，只有实际完成兼容和实测后才能写入项目能力。

## 暂不采用的原因

### LMDrive

学术价值高，但依赖较旧的 LAVIS/Vicuna 和 CARLA 0.9.10.1，完整训练成本高。只用于理解早期语言引导闭环驾驶，不作为当前底座。

### DrivingWorld

项目公开信息中训练、数据预处理或完整评测资产曾不完整。世界模型训练也不符合短周期和不重训原则，因此不纳入。

