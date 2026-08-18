# 实验计划与记录规范

每个有决策价值的运行都创建独立记录：

```text
docs/experiments/<date>_<stage>_<run>.md
```

## 记录模板

```markdown
# 实验记录：<run_name>

## 1. 目标

## 2. 环境

- 平台：
- GPU：
- Python/PyTorch/CUDA：
- Conda 环境：
- 代码根目录：
- 上游 commit：
- 本项目 commit 或 changed-file list：
- 数据目录与 manifest hash：
- checkpoint revision/hash：

## 3. 实验变量

- 唯一主变量：
- 固定项：
- 对比对象：

## 4. 精确命令

## 5. 输出

- 配置快照：
- 日志：
- predictions JSONL：
- metrics JSON：
- 可视化：

## 6. 结果

| 指标 | 数值 |
|---|---:|

## 7. 失败与异常

## 8. 结论

## 9. 下一步
```

## 运行命名

```text
<platform>_<stage>_<variant>_<split>_<seed>
```

示例：

```text
autodl_s2_fast_greedy_dev_s42
autodl_s4_k4_rerank_dev_s42
autodl_s6_router_k4_formal_s42
```

## 强制规则

1. smoke 通过前不运行正式清单；
2. formal 前冻结 checkpoint、manifest、阈值和配置；
3. 失败命令和异常不删除；
4. 逐场景输出必须包含输入标识、候选、选择原因、最终轨迹和分项耗时；
5. 断点恢复必须验证已有结果是当前 manifest 的严格前缀；
6. 后续实验不得覆盖前一实验的结论，只能追加更新；
7. 内部 risk score 与 NAVSIM 官方指标分开报告；
8. 只有正式评测结果才能进入简历数字。

## 统一主结果表

| Run | 主变量 | PDMS | NC | DAC | TTC | EP | Comfort | P50延迟 | P95延迟 | 峰值显存 | 决策 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

决策只能是：`Select`、`Keep as ablation`、`Retest`、`Reject`。

