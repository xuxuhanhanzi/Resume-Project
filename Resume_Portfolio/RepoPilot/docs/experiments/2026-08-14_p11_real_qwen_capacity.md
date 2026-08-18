# 实验记录：P11 真实 Qwen 1/4/8 容量

## 1. 目标

关闭 ScriptedProvider 负载无法代表模型推理吞吐的证据缺口，测量 RepoPilot provider →
Ollama → Qwen2.5-7B 的真实本地短请求容量。

## 2. 环境

- 平台：Windows 11，RTX 4070 Laptop 8GB；
- Python：3.12.3；Ollama：0.32.9；
- 模型：`qwen2.5:7b`，7.6B，GGUF Q4_K_M；
- digest：`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`；
- 代码基线：`9e676b2` + `scripts/benchmark_local_model_capacity.py`。

## 3. 实验变量

- 主变量：并发 `1/4/8`；
- 固定项：每档 24 请求、temperature 0、最大输出 16 tokens、同一短代码判断 prompt、
  loopback OpenAI-compatible API；
- 对比：既有 ScriptedProvider 真实 Runtime/Verifier 负载只作边界参照，不混合数值。

## 4. 命令

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_local_model_capacity.py `
  --output artifacts\runs\p11_real_qwen_capacity_v2\summary.json `
  --model qwen2.5:7b `
  --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e `
  --tasks 24 --max-output-tokens 16
```

## 5. 输出路径

- 原始 JSON：`artifacts/runs/p11_real_qwen_capacity_v2/summary.json`（本地忽略）；
- 可提交摘要：`docs/evidence/system/p11_real_qwen_capacity_v2.json`；
- 原始 SHA-256：`5c10b5c9ace34364938ce8e3fb0760c5e4ceda62aefce61de0932d6e1558097c`。

## 6. 结果

| 并发 | 成功 | 请求/s | 输出 token/s | P50 | P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 24/24 | 5.52 | 11.04 | 2.17s | 4.16s |
| 4 | 24/24 | 16.80 | 33.59 | 0.83s | 1.38s |
| 8 | 24/24 | 18.62 | 37.24 | 0.78s | 1.24s |

72/72 请求返回非空响应；1→4 吞吐提高约 204%，4→8 只提高约 10.9%，说明该固定短请求
在当前机器上 4 并发后边际收益明显下降。

## 7. 失败与异常

- `v1` pilot 使用 1/2/4、每档 12 请求并成功，原始结果保留；
- 首次命令因脚本未把 `src` 加入 import path 而失败，已修复并由 Ruff/strict mypy 覆盖；
- 本实验 prompt 很短，P50 包含同批任务的信号量排队时间；不能当作复杂 coding-agent 延迟。

## 8. 结论

真实本地 Qwen 服务容量缺口已经关闭。对当前 8GB 机器与固定短请求，4 并发是吞吐/延迟的
合理拐点；端到端 Runtime/Verifier 数值和模型服务数值继续分开报告。

## 9. 下一步

RepoPilot 没有剩余本地实验。更大上下文、长输出和跨 GPU 扩展属于云端容量实验，不是当前
本地发布门。
