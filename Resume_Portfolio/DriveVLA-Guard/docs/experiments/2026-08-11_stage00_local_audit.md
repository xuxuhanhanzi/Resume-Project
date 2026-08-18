# 实验记录：stage00_local_audit

## 1. 目标

冻结上游源码，验证许可证隔离、真实 codebook 契约和本地官方运行条件。

## 2. 环境

- 平台：Windows 11
- GPU：NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB
- Python：3.12.3
- AutoVLA commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`
- AutoVLA checkout：本地 `third_party/AutoVLA`，已由 `.gitignore` 排除

## 3. 实验变量

- 主变量：无算法变量；从未验证转为接口/资源审计；
- 固定项：官方 main 分支浅克隆、官方 codebook。

## 4. 命令

```powershell
python -m drivevla_guard.cli audit-upstream `
  --root .\third_party\AutoVLA `
  --output .\artifacts\stage00\upstream_lock.json

python -m drivevla_guard.cli official-preflight `
  --autovla-root .\third_party\AutoVLA `
  --output .\artifacts\stage00\official_preflight_local.json
```

另用 `ActionCodebookCodec.from_pickle()` 加载真实 `agent_vocab.pkl`，解码 10 个合法 token。

## 5. 输出

- 上游 lock：本地 `artifacts/stage00/upstream_lock.json`；
- preflight：本地 `artifacts/stage00/official_preflight_local.json`；
- 两者包含本地绝对路径，因此不提交公开仓库。

## 6. 结果

| 指标 | 结果 |
|---|---:|
| 必需上游文件 | 全部存在 |
| codebook SHA256 | `e6bf8eff32497eaefca66f488dda1edefe97ed2696d1d23c01839dc27422408e` |
| codebook vocabulary | 2048 |
| 10-token decode shape | `[10,3]` |
| decode finite | true |
| 本地官方 preflight | false |

## 7. 失败与异常

- 缺少 checkpoint、Qwen 模型、NAVSIM JSON/sensor/metric cache；
- 缺少 torch、transformers、qwen-vl-utils、hydra、nuplan、navsim；
- 本地 8GB GPU 不适合作为 16.3GB checkpoint 仓库的正式多候选评测设备；
- Hugging Face checkpoint 模型卡没有标准 license metadata。

## 8. 结论

源码与 Action Token 接口审计通过；官方模型/NAVSIM 端到端 smoke 在本地明确受外部资产、依赖和显存阻塞。不得把合成 smoke 标成官方复现。

## 9. 下一步

用户准备满足许可证要求的 AutoDL 环境和数据路径后，按 `docs/official_runbook.md` 执行。

