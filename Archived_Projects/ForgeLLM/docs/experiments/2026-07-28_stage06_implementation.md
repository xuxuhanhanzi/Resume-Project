# 2026-07-28 Stage 6 综合评测与最终验收实施记录

## 1. 结论先行

Stage 6 的自动化实现门通过，学习者门 G6-L 待完成。通过的是评测工程与证据链，不是模型行为：Q0 Base、Q1 SFT、Q2 DPO 在 32 个冻结原题上全部 strict 0/32、95% Wilson 上界 0.1072，且 64/64 输出都跑满 32-token 上限。Q2 虽把 64 对 held-out preference accuracy 提到 1.0、mean margin 提到 93.9996，却没有得到任何 strict success，并把字符 8-gram 重复率升至 0.4426。

最终状态对三个模型均为 `training pipeline accepted, model behavior not accepted`。Q2 额外未通过 stability 门。Q3 只验收一步 on-policy 管线；M3 只在自身 Tokenizer/语言建模套件内报告指标。

## 2. 冻结协议

| 项目 | 冻结值 |
|---|---|
| Base | `Qwen/Qwen3-0.6B-Base` |
| revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Q1 | `artifacts/stage04/qwen_lora_bounded_v2/adapter` |
| Q2 | `artifacts/stage05/qwen_dpo_bounded_v2/adapter` |
| Q3 | `artifacts/stage05/qwen_grpo_one_step_v2/adapter`，pipeline-only |
| M3 | `artifacts/stage03/bounded_1m/latest.pt` |
| 评测案例 | 16 correctness + 16 preference + 32 robustness = 64 |
| case SHA | `8cb746429bb576bf45a36a2f5ddcc159c57a7215139e918e973e1505a4fe8312` |
| 主生成 | greedy、temperature 0、top-p 1、max new tokens 32 |
| Bootstrap | paired、2,000 resamples、seed `20260728` |
| 盲评 | Q1/Q2、30 对、A/B 方向由 SHA-256 决定 |
| 系统矩阵 | Q0/Q1/Q2 × batch 1/4 × prompt 32/128/256，decode 32 |
| E10 | Q1、16 个 correctness case、32/64 token budget |
| 外部费用 | 0 USD |

质量/系统运行绑定：

- run fingerprint：`d2a26e41768f751e23dd8138ac2279744611b4a4f6b1c1ae67d8a0eb6ce013a1`；
- Git：`983a64ab15ce160e232210d30b63912b082b8a22+dirty`；
- 运行时 Stage 6 源码/脚本/配置聚合 SHA：`0754b0c62cb7fcb43857e569a2b7b17c2a29a76fc18c813112d4380c22bf3516`；
- Python 3.12.3、Torch 2.6.0+cu124、Transformers 5.14.1、PEFT 0.19.1；
- NVIDIA GeForce RTX 4070 Laptop GPU、BF16。

Stage 6 文件统一经过 Ruff format 后，质量、系统和最终报告全部重新运行；三者现均绑定同一个源码聚合 SHA `0754b0c62cb7fcb43857e569a2b7b17c2a29a76fc18c813112d4380c22bf3516`。旧 Artifact 保留为历史证据，没有被覆盖或跨版本拼接。

## 3. 数据与污染

v2 案例保留多轮对话中的历史 assistant turn，只隐藏最后一个待预测答案。污染规则为 NFKC + casefold + 空白合并的精确哈希，以及字符 13-gram Jaccard ≥0.8 的近重复候选。

- 结构化训练记录：2,880；
- 精确重合：0；
- near-match：16；
- 16 个候选集中在 correctness-087/095 与同模板、不同槽位的训练记录，相似度约 0.851–0.860；
- Stage 3 train 的规范化精确哈希也未命中。

这只能支持“列出的本地来源未发现精确污染”。模板 near-match 限制了开放域外推，不能写“绝对无污染”。

## 4. 正式质量结果

| 指标 | Q0 Base | Q1 SFT | Q2 DPO |
|---|---:|---:|---:|
| 原题 strict pass | 0/32 | 0/32 | 0/32 |
| Wilson 95% interval | [0, 0.1072] | [0, 0.1072] | [0, 0.1072] |
| 全集 prefix-correct-but-extra | 0.0000 | 0.4531 | 0.6250 |
| 全集 character 8-gram repetition | 0.1156 | 0.1716 | 0.4426 |
| 全集 token 3-gram repetition | 0.2406 | 0.1646 | 0.4047 |
| truncation | 1.0000 | 1.0000 | 1.0000 |
| expected-response loss/token | 1.9255 | 0.5414 | 0.5420 |
| expected-response PPL | 6.8585 | 1.7184 | 1.7194 |
| expected-response BPB | 1.5975 | 0.4492 | 0.4496 |
| retention loss/token | 1.8411 | 1.8829 | 2.0124 |
| retention/Q0 | 1.0000 | 1.0227 | 1.0930 |
| preference accuracy（64） | 0.7656 | 0.9844 | 1.0000 |
| mean chosen−rejected log-prob | 11.8437 | 17.2949 | 93.9996 |

对 Q0→Q1 和 Q1→Q2，original strict pass 的 paired difference 都是 0，2,000 次 Bootstrap 95% interval 也是 `[0,0]`。这不是“模型完全相同”，而是该 strict 指标上三者同为全失败。

Q1/Q2 的 teacher-forced 正确答案概率明显高于 Q0，Q1 也有 45.31% 输出以正确答案为前缀后继续生成；但 exact/约束协议要求及时停止，因此不能把这些样本改判成功。Q2 的极大 preference margin 与 100% pair accuracy 同时伴随 strict 0 和严重重复，构成代理目标过优化的直接反例。

## 5. M3 与 Q3 边界

M3 的固定 8 个 validation batch：loss/token 1.6245、PPL 5.0759、BPB 2.0543、8,128 target tokens、9,273 target bytes。其 Tokenizer SHA 与 Qwen 不同，且套件不同；报告器明确禁止用该 PPL 对 Q0–Q2 排名。

Q3 读取 Stage 5 报告确认 optimizer steps=1、4×4 rollout 和 on-policy 数据链。它没有进入质量排名，也没有“GRPO 提升能力”的声明。

## 6. Judge 与盲评

- 规则 Judge 对 64 对 Q1/Q2 的位置交换一致率 1.0；
- tie rate 0.109375；
- 30 对盲评包同时保存 public JSONL、含 Prompt 的 HTML 与独立 private key；
- 人工评分状态为 pending；未产生单评分者伪 kappa。

学习者必须在打开 `private_key.jsonl` 前完成评分。G6-L 不由自动测试关闭。

## 7. 系统结果

每格 1 次 warmup、5 次正式重复，报告 p10/median/p90。下表摘录 median；完整 18 格见正式 JSON。

| 模型 | batch | prompt | TTFT s | E2E 32-token s | E2E tok/s | 近似 decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Q0 | 1 | 32 | 0.0422 | 1.2607 | 25.38 | 25.44 |
| Q0 | 1 | 256 | 0.0418 | 1.2048 | 26.56 | 26.66 |
| Q0 | 4 | 32 | 0.0476 | 1.2665 | 101.06 | 101.73 |
| Q0 | 4 | 256 | 0.0830 | 1.2210 | 104.84 | 108.97 |
| Q1 | 1 | 32 | 0.0793 | 2.2237 | 14.39 | 14.46 |
| Q1 | 1 | 256 | 0.0701 | 2.2618 | 14.15 | 14.14 |
| Q1 | 4 | 32 | 0.0738 | 2.0872 | 61.33 | 61.59 |
| Q1 | 4 | 256 | 0.1155 | 2.0764 | 61.64 | 63.24 |
| Q2 | 1 | 32 | 0.0773 | 2.3408 | 13.67 | 13.70 |
| Q2 | 1 | 256 | 0.0731 | 2.3687 | 13.51 | 13.50 |
| Q2 | 4 | 32 | 0.0803 | 2.1730 | 58.90 | 59.25 |
| Q2 | 4 | 256 | 0.1431 | 2.1785 | 58.76 | 60.92 |

Q0/Q1/Q2 cold-load peak allocated 分别约 1.192/1.281/1.281 GB。batch=1、prompt 32→256 的理论 BF16 KV 从 3,670,016 增至 16,515,072 bytes；batch=4 对应 14,680,064→66,060,288 bytes。实际 peak 还包含权重、激活、workspace 与 allocator，因此明显更大。

系统 v5 重新运行后，Q0 不再出现 v4 首轮 batch=1 的异常慢值；这反过来证明短本地 microbenchmark 会受 GPU 电源/频率、形状和执行顺序影响。结果只描述本次固定顺序环境，不能据单轮结果声称某个 Adapter 必然让模型更快或更慢。

E10 的 Q1：32-token median 2.2136 秒、strict 0；64-token median 4.4365 秒、strict 0。增加一倍输出预算近似增加一倍耗时，却没有得到正确停止或能力改善。

## 8. E0–E15 对应关系

| ID | 实施证据 | 结果 |
|---|---|---|
| E0 | Manifest round-trip/篡改测试 | 通过 |
| E1 | exact + char-13 Jaccard 污染审计 | 0 exact；16 near 已披露 |
| E2 | exact/constraint/NLL/BPB 手算测试 | 通过 |
| E3 | 跨 Tokenizer PPL fail-fast | 通过 |
| E4 | 固定 seed paired Bootstrap | exact reproducible |
| E5 | Q0→Q1 同 64 题 | strict 差 0；概率指标改善 |
| E6 | Q1→Q2 同 64 题 | strict 差 0；pair/repetition 恶化分离 |
| E7 | Q3 scope 审计 | 仅一步管线 |
| E8 | 32 对 parent-linked 等义重述 | 已完成；双失败不能冒充鲁棒 |
| E9 | JSON/prefix/extra/repetition 规则攻击 | 测试通过；不改变 test rubric |
| E10 | Q1 32/64 budget | 成本翻倍，strict 不变 |
| E11 | A/B 与 B/A | 位置一致率 1.0 |
| E12 | Q0/Q1/Q2 固定系统矩阵 | 18 格完成 |
| E13 | batch 1/4 | 时延与吞吐同时报告 |
| E14 | context 32/128/256 KV | 理论值单调增长，边界已写明 |
| E15 | claim→evidence→boundary | 全部证据路径存在 |

## 9. 失败与修正记录

1. `quality_v1`：构造器删除全部历史 assistant turn，两个多轮案例形成 user/user；严格 Schema fail-fast。修复为仅隐藏最后答案，cases 升级 v2。
2. `quality_v2`：完整运行，但 expected-response NLL 含 ChatML `<|im_end|>`，byte 分母不含该特殊 token；结果被 v4 取代，没有覆盖。
3. `quality_v3`：启动后发现 `commit+dirty` 不唯一标识未提交源码，尽早终止；v4 加源码树聚合 SHA。
4. `systems_v4`：18 格与报告先成功写入，随后完整 JSON 含波斯字符，打印到 Windows GBK 终端时退出码失败。报告经 Schema、格数、E10 行数和 SHA 校验完整；后续 CLI 改打印 ASCII 简短摘要，不重复烧卡。
5. `final_v4`：默认相对路径直接对绝对根调用 `relative_to()`，在写报告前失败；构建器改为先 resolve，最终输出为 `final_v5`。
6. 最终全仓格式门要求统一 19 个 Stage 6 Python 文件；格式化会改变源码指纹，因此没有继续引用旧指纹，而是完整重跑为 `quality_v5`、`systems_v5` 和 `final_v6`。

这些失败目录均保留；没有跨版本拼接原始输出。

## 10. Artifact 与哈希

| Artifact | SHA-256/说明 |
|---|---|
| `data/processed/stage6_evaluation_v2/cases.jsonl` | `8cb74642...fe8312` |
| `artifacts/stage06/quality_v5/manifest.json` | 内含 run/source/model/data hash |
| `artifacts/stage06/quality_v5/report.json` | `177efbbb418e98ba7a731aed20e442f3dea3d7ee60eb298a1148fe3786271990` |
| `artifacts/stage06/quality_v5/raw_generations.jsonl` | 192 行，可重算行为指标 |
| `artifacts/stage06/quality_v5/blind_review/` | 30 对 public/HTML/private key |
| `artifacts/stage06/systems_v5/report.json` | `7f978cc66434685cac94dfca3cd254e0a671fae96d7151adfb2330b9b6a859d8` |
| `artifacts/stage06/final_v6/report.json` | `1e22c86761ff614926053693bf5cd8efb3a447bf2e1e0b0d380cb286d600ae1a` |
| `artifacts/stage06/final_v6/claim_evidence.json` | 声明—证据矩阵 |

最终质量门：173 个 Python 文件通过 Ruff format/check；mypy 检查 113 个源文件无问题；全仓 `244 passed, 3 skipped`，3 个跳过项均为需要显式启用编译工具链的 C++ 扩展集成测试；Stage 6 专项 `21 passed`。

## 11. 复现与学习命令

现有正式目录不可覆盖。学习者默认只运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_evaluation_schema_config.py tests\unit\test_evaluation_metrics.py tests\unit\test_evaluation_judge_systems_report.py
```

若确需复现，使用 `artifacts/stage06_student/` 下的新输出路径，并在 UTF-8 终端运行；不得调整 test 后的生成预算，也不得覆盖 v1–v6 证据。

## 12. 验收状态

- 自动化 G6-A/B/C/D/E：通过；
- 模型行为：Q0/Q1/Q2 均未接受；
- Q3：pipeline-only 接受；
- G6-L：pending，等待学习者按 18 站完成盲评、手算、代码追踪与口述。
