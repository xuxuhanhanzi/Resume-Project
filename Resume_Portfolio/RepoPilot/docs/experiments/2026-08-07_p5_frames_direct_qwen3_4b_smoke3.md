# 实验记录：FRAMES Direct Qwen3-4B Smoke-3

## 1. 目标

验证本地模型服务、固定模型 revision、统一 runner、真实 FRAMES scorer 和 artifact 写入链路。该运行是无工具、无文档的成本探针，不是 Agent 主结果。

## 2. 固定设置

- 模型：`qwen3:4b`，Q4_K_M
- 模型 manifest SHA-256：`359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 任务：`frames-test-0000` 至 `frames-test-0002`
- temperature：0
- thinking：尝试通过 `/no_think` 关闭，但服务端仍返回思考 token
- 工具/文档：无

## 3. 命令

```powershell
python scripts\run_frames_direct_smoke.py --run-id 20260807_frames_direct_qwen3_4b_smoke3 --count 3 --model qwen3:4b --model-revision 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
```

## 4. 结果

- Accuracy：`0/3 = 0.0`
- 三个任务的最终答案均为空；输出 token 均达到上限 `128`
- 单任务耗时：`1.719s–2.156s`
- 失败原因：Qwen3 将输出预算全部用于 thinking，`/no_think` 在当前 Ollama 模板中未生效
- 处置：保留该失败结果；后续基线切换为非思考型 `qwen2.5:3b`
