# 实验记录：FRAMES Oracle-Document Qwen2.5-3B Smoke

## 1. 目标

比较固定模型在无文档与离线 oracle 文档条件下的表现，验证语料冻结、BM25 片段检索、证据记录和评分链路。

## 2. 固定设置

- 模型：`qwen2.5:3b`，manifest SHA-256 `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`
- 数据 revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- 语料：首 10 题涉及的 39 个 Wikipedia 页面，逐页记录 URL、抓取时间及内容哈希
- 每个 oracle 页面检索 2 个 BM25 片段
- temperature：0；最大输出：128 tokens；网络策略：运行时断网

## 3. 对照

- `b0`：无工具、无文档直接回答
- `b1-oracle`：同一模型、同一任务，增加冻结文档检索

## 4. 结果

| 运行 | 变量 | Accuracy | 诊断 |
|---|---|---:|---|
| fixed-short | 固定 BM25 + 短答案 | 0/3 | 证据 Recall=1.0，但关键片段排序不足 |
| planned-short v1 | 查询规划；整份计划失败则回退 | 0/3 | 规划遗漏一个来源，触发整题回退 |
| planned-short v2 | 允许局部计划与逐来源回退 | 0/3 | 已检索到关键实体，但 3B 关系组合错误 |
| planned-reasoned | 增加显式多跳推理输出 | 0/3 | 检索正确，最终关系仍推断错误 |
| Qwen3-4B reasoned | 只替换模型 | 0/3 | 512 tokens 全部被 thinking 消耗，最终答案为空 |
| Qwen2.5-7B per-source | 逐页面规划 + 7B 推理 | 0/3 | 第 1 题 F1 提升至 0.5；误判遇刺顺序 |
| Qwen2.5-7B + reviewer | 增加独立复核 | 0/3 | Reviewer 未纠正缺失证据导致的实体错误 |
| 1600-char window | 扩大定向证据窗口 | 0/3 | 第 1 题保持 F1=0.5；小模型仍错误组合关系 |
| Qwen3-4B 4096 | 固定检索 + 4096 thinking | 0/1 | 找到 Garfield/Ballou，但误判 Jane；单题 49.64s |
| 7B planner → Qwen3 | 自适应模型路由 | 0/1 | 4096 tokens 仍全部用于 thinking，70.70s 后答案为空 |

阶段判断：离线语料、检索、证据、规划、Reviewer 和模型路由链路均已真实运行；3 题正确率门禁未通过，因此不扩展到 10 题。下一轮需要更强且可控的推理模型，或先训练/蒸馏专门的多跳规划与验证模块。

后续更新：Docker 已恢复；使用 Qwen2.5-7B 的固定 10 题 Agent smoke 已完成并达到 `2/10`。详见 `2026-08-07_p5_frames_agent_qwen2_5_7b_smoke10.md`。本段保留为当时的阶段判断。

## 5. Docker 门禁

Docker Desktop 客户端可用，但 WSL2 挂载 `ext4.vhdx` 返回 `Wsl/Service/AttachDisk/MountDisk/HCS/E_ACCESSDENIED`。已完成无损的 Docker/WSL 重启并复现；DABench 不可信代码执行与 SWE-bench-Live 官方评测继续保持阻塞，不降级为宿主机直接执行。
