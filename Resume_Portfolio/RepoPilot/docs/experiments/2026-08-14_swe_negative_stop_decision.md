# 决策记录：SWE-bench-Live 负结果停止规则

- 日期：2026-08-14
- 决策：停止在未改变代码修复策略的情况下继续消耗 5 个新任务；保留 0/5 正式负结果。

## 已有证据

- 冻结的 5 个 fresh、未污染 SWE-bench-Live task 使用官方 evaluator 与固定 qwen2.5:7b；
- resolved = 0/5：4 个任务没有产生 patch，1 个只有不足以通过测试的 620 字符文档 patch；
- 失败不是评测镜像或 Windows 命令长度故障：harness 会始终产出 predictions，官方
  FAIL_TO_PASS/PASS_TO_PASS 已在各自镜像内运行；
- 此后新增的是 Multi-Agent service、HTTP/SSE、负载和安全边界，没有改变 executor 的
  patch 生成策略或模型能力。

## 停止理由

在没有新干预变量时追加 5 个任务只能扩大同一失败样本，不检验新的可证伪假设；把任务数
跑到 10 也不能让 0/5 变成能力改进。继续运行还会消耗模型时间与大型官方镜像资源，并
增加从 validation 反复调参造成污染的风险。

因此按项目路线图允许的分支，选择“明确负结果停止决定”，而不是宣称通过 SWE 能力门。

## 重新开启条件

只有以下任一条件满足才重新打开 SWE 实验：

1. 引入经过独立开发夹具验证的新 patch-localization / tool-use 干预，使非空、可应用 patch
   率达到至少 4/5；或
2. 更换模型/训练配方，并先在不与正式 holdout 重叠的开发任务上证明 verifier 通过率改善。

重新开启时必须重新冻结未污染 task manifest，先写成功阈值和预算，再运行；不得回头在
当前 5 个正式 task 上做提示词调参。

## 对外口径

可引用：`qwen2.5:7b, 5 fresh tasks, 0/5 resolved`，作为当前本地模型代码修复能力的负结果。
不可表述为“完成 SWE-bench”或“具备代码修复成功率”；也不把停止决定算作成功结果。
