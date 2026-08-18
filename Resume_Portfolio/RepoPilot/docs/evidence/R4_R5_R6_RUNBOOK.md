# R4 / R5 / R6 Runbook（RepoPilot 审计整改剩余闸门）

本文件记录三件事的可复现流程、已修复的坑、以及当前真实结果。所有命令均在本机
（Windows + Git Bash + 项目 `.venv`）验证过。

> 2026-08-11 收口状态：R4、R5、R6 均已执行完成。R4 的 1.5B base 为 9/35，
> SFT+DPO Adapter 为 0/35；R5 干净 SWE holdout 为 0/5；R6 FRAMES 为
> 11/60，三次重复一致。下文保留运行时的过程状态、失败和命令作为历史记录；
> 最新结论以 `docs/STATUS.md` 与 `artifact_index.json` 为准。

> **关于 `network=deny` 的关键澄清**：评测 manifest 里的 `network_policy: deny`
> 只是给**被测 agent** 套的离线策略（防作弊），**不阻止我们走宿主机网络拉数据/
> 镜像后本地化喂评测**。沙箱宿主机网络实测可用（huggingface / pypi / docker pull
> 均通）。因此 R5/R6 不是网络硬阻塞，R4 也可拉 1.5B 权重。

---

## R6 — FRAMES 扩量到 ≥50 题（✅ 已完成，可引用）

**目标**：把 FRAMES 从 10 题（~20%，仅供参考）扩到 60 题，3 次重复，升级为可引用指标。

**步骤**：
1. 物化语料（本地 `test.tsv` 有 824 题，无需联网拉新数据）：
   ```bash
   ./.venv/Scripts/python.exe scripts/materialize_frames_corpus.py --count 60
   # 产物：evaluation/benchmarks/corpora/frames/<REV>/smoke60/{manifest.json,documents/*.txt}
   ```
2. 跑 3 次重复（qwen2.5:7b，oracle-document 协议，bm25 检索）：
   ```bash
   for i in 1 2 3; do
     ./.venv/Scripts/python.exe scripts/run_frames_oracle_smoke.py \
       --run-id r6_20260809_frames60b_rep$i --count 60 \
       --model qwen2.5:7b \
       --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e \
       --retriever-type bm25
   done
   ```

**踩过的坑（已修复，commit `6be3187`）**：
- `materialize_frames_corpus.py` 物化 URL 未 `.strip()`，而 `FramesAdapter._parse_links`
  会 strip → manifest 与 adapter 的 asset URI 差 1 个（带尾随空格的 `Calton,_Glasgow`），
  评测端 `KeyError: oracle document is missing`。修复：物化端对每链接 `.strip()`。
- 部分维基链接含非 ASCII（en dash），`urlopen` 抛 `UnicodeEncodeError` 且不在兜底 catch
  内 → 崩。修复：URL 百分号编码 + 把 `UnicodeEncodeError/ValueError` 纳入 curl 兜底。
- 两脚本 `--count` 上限原为 1–10，已放宽到 1–200；`run_frames_oracle_smoke.py` 语料路径
  由硬编码 `smoke10` 改为 `smoke{count}`。

**结果**：60 题 ×3 次重复均 **11/60 = 0.183**（temperature=0 确定性一致）。
FRAMES 现为**可引用**指标。已登记进 `docs/evidence/artifact_index.json`（verify 16/16 一致）。

---

## R5 — SWE-bench-Live 干净 holdout（🟡 管线已验证，镜像拉取中）

**目标**：用未污染任务产出可信的 resolved rate（替代 smoke3 的 0/3 开发污染结果）。

**脚本**（配置驱动，不碰原 smoke3）：
- `prepare_swebench_fresh_holdout.py`：从 `lite.parquet` 选未污染任务（seed 7）、克隆仓库
  @base_commit、生成 `fresh_public.jsonl` + `fresh_task_config.json`。
- `run_swebench_live_fresh_holdout.py`：在官方 eval 镜像（networkless、read-only root）里跑
  agent，产出 `predictions.jsonl`。
- `eval_swebench_resolved.py`：对每题应用 model_patch + 黄金 test_patch（**绝不进 agent**），
  用镜像跑 FAIL_TO_PASS/PASS_TO_PASS，自包含算出 resolved rate。

**踩过的坑（已修复，commit `b000025`）**：
- 镜像命名错误：原写 `swebench/sweb.eval.x86_64.<iid>:latest`，正确为
  `starryzhang/sweb.eval.x86_64.<iid 转义 __→_1776_>:latest`（对齐原 smoke 脚本）。
- `allowed_paths` 过窄：原只取 test 文件顶层目录会锁死 agent 改不了源码 → 改 `["."]`
  （对齐 SWE-bench 标准，agent 可改任意文件，零评估数据泄漏）。
- parquet `Timestamp`/numpy 数组不可 JSON 序列化 → 加 `_to_jsonable` / `_as_list` 转换。
- **Windows 命令行超限**：linkding 的 PASS_TO_PASS 有 **736 个 node id（80KB）**，单次
  `pytest` 命令行超过 Windows CreateProcess 限制（~32KiB）→ `WinError 206`。修复：
  `eval_swebench_resolved.py` 把 test id **分块**多次调用 pytest，全部通过才算通过。

**当前结果（1/5 任务，其余 4 个 eval 镜像拉取中）**：
- `sissbruecker__linkding-984`：agent 跑满 12 轮、10 次工具调用，**apply_patch 因猜错相对
  路径（`templates/public/bookmarks.html` 不存在）失败** → 空 patch → resolved=**False**。
  这是 7B 模型能力问题（非管线故障），管线正确捕获为未解决。
- 其余 4 任务（pytorch__torchtune-1806 / kozea__weasyprint-2387 / theoehrly__fast-f1-699 /
  deepset-ai__haystack-8609）待镜像拉完即跑。

**跑剩余 4 任务的命令**：
```bash
# 镜像拉完后（linkding 已拉好，其余 4 个后台拉取中）
./.venv/Scripts/python.exe scripts/run_swebench_live_fresh_holdout.py \
  --run-id r5_20260809_fresh_all \
  --public-jsonl evaluation/benchmarks/data/swebench_live/a637bd46829f3132e12938c8a0ca93173a977b8e/fresh5/fresh_public.jsonl \
  --task-config evaluation/benchmarks/data/swebench_live/a637bd46829f3132e12938c8a0ca93173a977b8e/fresh5/fresh_task_config.json \
  --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e
./.venv/Scripts/python.exe scripts/eval_swebench_resolved.py \
  --predictions artifacts/benchmarks/r5_20260809_fresh_all/predictions.jsonl \
  --parquet evaluation/benchmarks/data/swebench_live/a637bd46829f3132e12938c8a0ca93173a977b8e/lite.parquet \
  --workspace-root evaluation/benchmarks/workspaces/swebench_live
```

---

## R4 — P10 adapter 下游评测（🟡 已预备考，待执行/或文档闭环）

**架构事实（决定性）**：P10 adapter 的 `adapter_config.json` 写死
`base_model_name_or_path: Qwen/Qwen2.5-1.5B`，而 DABench 基线是 **qwen2.5:7b**。
**1.5B LoRA 无法合并/改善 7B 基线** —— 因此 R4 只能评"独立的 1.5B 配置"
（1.5B base vs 1.5B+adapter 在 DABench），**不会提升 7B 的 71.4% 可引用数字**。

**R4-A（可落地路径，仅证 1.5B 效果）**：
1. `ollama pull qwen2.5:1.5b`（✅ 已拉好）。
2. 装合并栈（本机未装 torch/transformers/peft；8GB GPU）：
   ```bash
   .venv/Scripts/pip install torch transformers peft
   ```
3. 合并 adapter → 1.5B base → 产出可服务模型：
   ```bash
   ./.venv/Scripts/python.exe scripts/merge_adapter_and_prep_eval.py \
     --base-model Qwen/Qwen2.5-1.5B --apply-dpo
   # 产出 artifacts/training/r4_merged_1p5b/ + Modelfile
   ollama create repopilot-qwen2.5-1p5b-adapter -f artifacts/training/Modelfile.repopilot-qwen2.5-1p5b-adapter
   ```
4. 在 DABench 上跑 1.5B base 与 1.5B+adapter 各 3 次，比较：
   ```bash
   for m in qwen2.5:1.5b repopilot-qwen2.5-1p5b-adapter; do
     for i in 1 2 3; do
       ./.venv/Scripts/python.exe scripts/run_dabench_agent_smoke.py \
         --run-id r4_1p5b_${m##*:}_rep$i --model $m --model-revision <digest> \
         --manifest evaluation/benchmarks/manifests/dabench_validation35_v2.json
     done
   done
   ```

**R4-B（若要真正影响 7B 数字）**：需在 **Qwen2.5-7B** 上重训 P10（SFT+DPO），再合并评测。
8GB GPU 跑 7B QLoRA 风险高，但这是唯一能让 adapter 改变 7B 71.4% 的路径。

**结论**：R4-A 是诚实可执行的闭环，但产出的是"1.5B 独立配置"的对比，不改变三领域主指标
（DABench 7B 71.4% 仍是最 authoritative 的 citable 数字）。R4-B 是重训，价值更高但成本高。
脚本与 runbook 已就绪；是否执行 R4-A 取决于是否要这份 1.5B 对比数据。
