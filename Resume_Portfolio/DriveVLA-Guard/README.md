# DriveVLA-Guard

实验 PC 的自动安装与顺序运行入口为 `scripts/setup_experiment_pc.sh` 和
`scripts/run_experiment_pc.sh`；通常由作品集根目录的统一编排器调用。正式档会固定下载
AutoVLA、Qwen2.5-VL-3B 与公开 NAVSIM/OpenScene 资产，并在数据条款门槛通过后依次执行
B0/E2/E3/E4。

DriveVLA-Guard 是一个面向中国车企 VLM/VLA 算法岗位的短周期作品集项目。项目以
[AutoVLA](https://github.com/ucla-mobility/AutoVLA) 的官方预训练权重为模型基线，
以 [NAVSIM](https://github.com/autonomousvision/navsim) 为主要评测协议，研究无需重新训练基础模型的
风险感知推理增强：多候选轨迹生成、确定性安全重排序和快慢推理路由。

## 一句话目标

在固定 AutoVLA checkpoint、固定数据清单和固定 NAVSIM 评测协议下，验证推理时风险约束能否在可控延迟开销内减少碰撞或越界失败，并形成可复现的逐场景证据链。

## 项目边界

本项目：

- 使用官方已经训练完成的 AutoVLA checkpoint；
- 阅读并解释 Action Token、SFT、GRPO、快慢推理和 NAVSIM 指标源码；
- 自主实现候选轨迹生成、轨迹校验、风险评分、重排序、路由、报告和可视化；
- 运行官方 baseline 与逐模块消融；
- 报告 PDMS、安全子指标、延迟、显存和失败案例。

本项目不：

- 从零训练 VLM/VLA；
- 完整复现 AutoVLA SFT 或 GRPO；
- 第一阶段替换 Qwen2.5-VL 底座；
- 第一阶段同时接入 CARLA、Bench2Drive、Senna 和 UniDriveVLA；
- 将 NAVSIM 结果描述为真实车辆道路验证；
- 在没有固定评测证据前声称性能提升或 SOTA。

## 已实现系统

```text
NAVSIM/OpenScene 输入
        |
        v
官方 AutoVLA checkpoint
        |
        +--> fast mode 生成 K 条候选轨迹
                  |
                  v
        合法性检查 + 风险评分 + 重排序
                  |
                  +--> 通过门槛：输出轨迹
                  |
                  +--> 未通过门槛：触发 slow mode
                                      |
                                      v
                             重新生成与重排序
                                      |
                                      v
                         NAVSIM 评测与失败分析
```

## 文档入口

- [项目上下文](docs/project_context_summary.md)
- [完整实施计划](docs/implementation_plan.md)
- [参考项目与论文相关性审计](docs/paper_reference_relevance_audit.md)
- [创新模块映射](docs/innovation_mapping.md)
- [实验记录规范](docs/experiments/README.md)
- [Stage 0 计划](docs/experiments/2026-08-11_stage00_baseline_plan.md)
- [架构与数据契约](docs/architecture.md)
- [AutoVLA 源码阅读笔记](docs/autovla_source_notes_cn.md)
- [冻结证据清单](artifacts/evidence_manifest.json)
- [本地敏感性与压力实验](docs/experiments/2026-08-14_local_diagnostics_v1.md)
- [合成消融结果报告](docs/result_report.md)
- [官方 AutoDL/NAVSIM 运行手册](docs/official_runbook.md)
- [简历描述与面试问答](docs/resume_interview.md)
- [第三方许可证说明](THIRD_PARTY_NOTICES.md)

## 快速运行

项目核心只依赖 NumPy 与 PyYAML，不需要 GPU：

```powershell
python -m pip install -e ".[dev]"
drivevla-guard make-synthetic --output data/synthetic/smoke_manifest.jsonl --scenes 12
drivevla-guard evaluate-synthetic `
  --config configs/e4_guard.yaml `
  --manifest data/synthetic/smoke_manifest.jsonl `
  --output artifacts/runs/my_e4/predictions.jsonl `
  --summary artifacts/runs/my_e4/summary.json
python -m pytest
```

发布复现可先安装 [requirements-dev.lock](requirements-dev.lock) 中的 Windows/CPython 3.12 固定版本，再以 `--no-deps` 安装本项目；Linux CI 仍以 `pyproject.toml` 的兼容范围为准。

也可以直接运行 [run_synthetic_suite.ps1](scripts/run_synthetic_suite.ps1)，得到 B0/B1/E1–E4 全部消融。

发布前的一键验收会执行格式、Lint、21 项测试、字节码编译、冻结证据校验，并在新目录重跑完整合成矩阵：

```powershell
python scripts/check.py
```

## 冻结合成结果

12 个场景的 `formal_v5` 是工程验证集，不是 NAVSIM，也不代表真实驾驶性能：

| Run | 碰撞代理失败 | 慢路由率 | 平均延迟 | 平均舒适度代理 |
|---|---:|---:|---:|---:|
| B0 | 6/12 | 0% | 4.0 ms | 1.000 |
| E2：K=4 风险重排序 | 0/12 | 0% | 16.0 ms | 0.718 |
| E3：选择性 slow | 3/12 | 25% | 7.5 ms | 0.892 |
| E4：组合 | 0/12 | 25% | 19.5 ms | 0.727 |

结果说明：重排序和路由链路按预期工作，但安全选择会产生舒适度—延迟权衡；E4 没有在合成集上证明优于 E2 的安全收益。所有结论仅限当前合成代理指标。

补充的 1,200 场景本地诊断在 K=1/2/4/8、3 个种子和 5 档风险阈值上完成
20,400 个 scene-run，0 执行失败。K=4 是最小零碰撞代理配置；K=8 没有额外安全收益且
声明延迟翻倍。阈值 13.8 相比 4/8 保持相同代理结果，同时把慢路由率从 50% 降至 25%。
这些仍是合成诊断，不是 NAVSIM 或真实模型性能。

## 代码结构

```text
src/drivevla_guard/
├── adapters/autovla.py       # 官方模型多候选生成适配
├── integrations/navsim_agent.py
├── codec.py                  # Action Token 纯 NumPy 解码
├── risk.py                   # 可解释风险分解
├── pipeline.py               # 重排序与快慢路由
├── evaluation.py             # JSONL、断点恢复、配对比较
└── visualization.py          # 轨迹证据图
```

## 当前状态

已完成核心实现、21 项测试、B0/B1/E1–E4 合成消融、配对比较、可视化、AutoVLA 真实接口适配和 NAVSIM agent。`formal_v5` 的 12 场景、6 组配置、72 条逐场景结果、摘要、配对结果与决策关键源码均由 SHA-256 清单保护；上游源码锁定为 `ba34eed74ce6729e7986592d0e66cbaca397b4fa`。

本地 `v1.0.0` 已在独立干净克隆中按锁文件重建，并再次通过全部质量门、证据校验和新生成的完整合成矩阵；`v1.0.1` 固化该验证记录，`v1.0.2` 增加完整本地敏感性与压力实验。

正式 NAVSIM PDMS 尚未运行：当前机器只有 8GB 显存，且没有 16.3GB AutoVLA checkpoint、Qwen2.5-VL-3B 权重、NAVSIM 数据和 metric cache。严格 preflight 会校验路径类型、依赖版本、GPU/显存容量、上游文件与 commit；只有 `ready=true` 才允许进入正式运行。在这些外部资产到位前，不把合成结果冒充官方结果。
