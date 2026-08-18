# 官方 AutoVLA + NAVSIM 运行手册

## 当前本地 preflight

本地 Windows 环境检查结果位于 `artifacts/stage00/official_preflight_local.json`：RTX 4070 Laptop 8GB；缺少完整 AutoVLA/NAVSIM 运行依赖、AutoVLA checkpoint、Qwen 权重、预处理 JSON、sensor blobs 和 metric cache。preflight 同时验证路径类型、GPU/显存、依赖版本、上游文件契约和 commit；当前明确返回 `ready=false`，因此本地不启动形式上会失败或 OOM 的正式实验。

## AutoDL 建议

- Linux；
- 24GB 显存仅作为最低 smoke 起点，K=4 顺序生成仍需实测；如 OOM 使用 40/48GB；
- 足够磁盘保存 16.3GB checkpoint、Qwen2.5-VL-3B 和 NAVSIM 数据；
- 用户自行阅读并接受 nuPlan/OpenScene 数据条款；
- 在权利方澄清 checkpoint license 后再公开衍生结果或文件。

## 环境

```bash
export DRIVEVLA_GUARD_ROOT=/root/autodl-tmp/DriveVLA-Guard
export AUTOVLA_ROOT=/root/autodl-tmp/AutoVLA
bash "$DRIVEVLA_GUARD_ROOT/scripts/bootstrap_autodl.sh"
```

记录：

```bash
pwd
git -C "$AUTOVLA_ROOT" rev-parse HEAD
python -m drivevla_guard.cli audit-upstream \
  --root "$AUTOVLA_ROOT" \
  --output "$DRIVEVLA_GUARD_ROOT/artifacts/official/upstream_lock.json"
```

## Preflight

```bash
python -m drivevla_guard.cli official-preflight \
  --autovla-root "$AUTOVLA_ROOT" \
  --checkpoint "$AUTOVLA_CHECKPOINT" \
  --qwen-model "$QWEN_MODEL_ROOT" \
  --model-config "$AUTOVLA_MODEL_CONFIG" \
  --json-data "$NAVSIM_JSON_DATA" \
  --sensor-data "$NAVSIM_SENSOR_DATA" \
  --metric-cache "$NAVSIM_METRIC_CACHE" \
  --expected-upstream-commit ba34eed74ce6729e7986592d0e66cbaca397b4fa \
  --min-vram-gb 24 \
  --output "$DRIVEVLA_GUARD_ROOT/artifacts/official/preflight.json"
```

只有 `ready=true` 才进入模型加载。

## 最小运行顺序

1. 上游官方 `AutoVLAAgent` B0 单场景；
2. 相同 seed/config 重复一次；
3. DriveVLA-Guard B0 adapter 对齐；
4. 10-scene E2；
5. 冻结阈值后运行 B0/E2/E3/E4 同一 manifest；
6. 汇总 PDMS、NC、DAC、TTC、EP、Comfort、P50/P95、显存、invalid rate、slow rate。

Guard adapter 的 B0/E2/E3/E4 命令由 [run_navsim_guard.sh](../scripts/run_navsim_guard.sh) 提供，通过 `RUN_VARIANT` 选择唯一配置；脚本会在修改上游 agent 配置前强制执行 preflight。任何失败必须保留精确命令和 traceback，不同时升级多个依赖。

```bash
for variant in b0 e2 e3 e4; do
  RUN_VARIANT="$variant" bash "$DRIVEVLA_GUARD_ROOT/scripts/run_navsim_guard.sh"
done
```

只有上游 B0 重复性和 Guard B0 对齐通过后，才能进入 E2；只有 E2 满足预注册安全/进度门槛后，才能继续 E3/E4。

## 结果回填

官方运行后创建 `docs/experiments/<date>_official_<run>.md`，并把 CSV/JSON 路径、环境、commit 和每项指标填入记录。不得用当前合成报告替换该步骤。
