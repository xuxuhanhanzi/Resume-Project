# Third-party notices

DriveVLA-Guard 的核心实现为独立代码，采用 MIT License。以下资产不属于本仓库，也不随本仓库分发。

| 资产 | 来源 | 状态 |
|---|---|---|
| AutoVLA 源码 | https://github.com/ucla-mobility/AutoVLA | Academic Software License，仅教育/学术研究；禁止转移源码及其衍生物；商业主体需联系 UCLA Mobility Lab |
| AutoVLA checkpoint | https://huggingface.co/Zewei-Zhou/AutoVLA | 模型卡未声明标准 license metadata，使用前需向权利方确认 |
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | 遵循模型卡及其许可证 |
| NAVSIM | https://github.com/autonomousvision/navsim | 遵循 NAVSIM 仓库许可证 |
| nuPlan/OpenScene 数据与地图 | NAVSIM 官方下载入口 | 下载和使用前必须阅读并接受数据条款 |

`third_party/`、`checkpoints/` 和 `data/navsim/` 已加入 `.gitignore`。`scripts/run_navsim_guard.sh` 在用户自己的环境中连接这些资产，但不会重新分发它们。

本说明不是法律意见。求职公开仓库只提交本项目代码、合成数据、配置和报告。

