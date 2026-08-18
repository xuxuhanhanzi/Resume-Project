# 实验记录：autodl_official_preflight

## 1. 目标

在实际 AutoDL RTX 4090 容器中验证 DriveVLA-Guard 云端包与官方 AutoVLA/NAVSIM
前置条件，只在严格 preflight `ready=true` 后启动 B0/E2。

## 2. 环境

- 平台：AutoDL Ubuntu 22.04；
- GPU：NVIDIA GeForce RTX 4090，24,564 MiB，driver 560.35.03；
- Python：3.12.3；PyTorch：2.6.0+cu124；CUDA runtime：12.4；
- 项目：`/root/autodl-tmp/DriveVLA-Guard`；
- 上传包：`DriveVLA-Guard_cloud_v1.0.2.zip`，SHA-256 校验通过；
- 包内 manifest：93 个文件，0 failures。

## 3. 命令

```bash
sha256sum -c DriveVLA-Guard_cloud_v1.0.2.zip.sha256
python DriveVLA-Guard/scripts/verify_cloud_bundle.py DriveVLA-Guard

PYTHONPATH=src /root/autodl-tmp/envs/forgemm/bin/python \
  -m drivevla_guard.cli official-preflight \
  --autovla-root /root/autodl-tmp/AutoVLA \
  --checkpoint /root/autodl-tmp/models/AutoVLA/autovla.ckpt \
  --qwen-model /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model-config /root/autodl-tmp/AutoVLA/configs/model.yaml \
  --json-data /root/autodl-tmp/navsim_data/preprocessed_json \
  --sensor-data /root/autodl-tmp/navsim_data/sensor_blobs \
  --metric-cache /root/autodl-tmp/navsim_data/metric_cache \
  --expected-upstream-commit ba34eed74ce6729e7986592d0e66cbaca397b4fa \
  --min-vram-gb 24 \
  --output artifacts/official/preflight_autodl_20260815.json
```

## 4. 结果

| 检查 | 结果 |
|---|---|
| 云端包 SHA-256 | 通过 |
| 包内 93 文件 manifest | 通过 |
| Qwen2.5-VL-3B 路径 | 存在 |
| AutoVLA checkout/checkpoint/config | 缺失 |
| NAVSIM JSON/sensor/metric cache | 缺失 |
| hydra/nuplan/navsim | 缺失 |
| 显存门 | 24,564 MiB < 24,576 MiB |
| `ready` | `false` |

原始输出：`artifacts/official/preflight_autodl_20260815.json`。

## 5. 结论

按照预登记停止规则，没有启动 B0/E2/E3/E4。该容器不仅缺少必须由用户接受条款后从
官方入口获取的 NAVSIM/nuPlan 资产，50 GB 数据盘也不足以承载完整数据链；此外该 RTX
4090 的报告显存比当前严格 24 GiB 门少 12 MiB。继续实验需要更大数据盘、完成受限资产
授权/下载，并建议使用 40/48 GB GPU 或经单独评审后调整显存门。
