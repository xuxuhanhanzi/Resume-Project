# 实验 PC 一键运行手册

这套入口面向 Windows 11 + WSL2 Ubuntu + RTX 5090 24GB + 64GB RAM + 2TB SSD。
四个项目分别安装依赖，统一编排器按固定顺序执行，并把每个阶段的完整日志和 JSON
receipt 写入 `artifacts/experiment-suite/<run-id>/`。失败后修复原因并重跑同一命令即可续跑。

## 1. 主机一次性准备

在 Windows 安装以下组件，然后确认 Docker Desktop 已为 Ubuntu 开启 WSL integration：

- 最新 NVIDIA Windows 驱动（WSL 内不单独安装 Linux 显卡驱动）；
- WSL2 与 Ubuntu 24.04；
- Docker Desktop；
- Git。

建议在 `%UserProfile%\.wslconfig` 给 WSL2 分配至少 60GB 内存，然后执行 `wsl --shutdown`
使配置生效。仓库必须克隆到 Ubuntu 文件系统（例如 `~/src`），不要放在 `/mnt/c` 或
`/mnt/d`，否则数万小文件、Docker bind mount 与模型数据的 I/O 会显著变慢。

## 2. 克隆与烟雾验证

在 Ubuntu 终端执行：

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/xuxuhanhanzi/Resume-Project.git
cd Resume-Project/Resume_Portfolio
bash scripts/bootstrap_experiment_pc.sh smoke
bash scripts/run_all_experiments.sh smoke
python3 scripts/experiment_suite.py status --profile smoke
```

`bootstrap_experiment_pc.sh` 会安装当前用户的 `uv`，建立三个互相隔离的 Python 环境，
下载模型/公开数据并构建 Docker 镜像。烟雾档运行四个项目的质量门槛和缩小实验，不代表
正式指标。

## 3. 正式实验

DriveVLA-Guard 使用的 OpenScene 数据为 CC BY-NC-SA 4.0。确认用途满足数据条款后，显式
记录接受决定并启动：

```bash
cd ~/src/Resume-Project/Resume_Portfolio
export DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1
export EXPERIMENT_RUN_ID=full-5090-v1
bash scripts/bootstrap_experiment_pc.sh full
bash scripts/run_all_experiments.sh full
python3 scripts/experiment_suite.py status --profile full --run-id "$EXPERIMENT_RUN_ID"
```

正式顺序如下：

1. DriveVLA-Guard：代码/合成门槛 → AutoVLA B0 → E2 → E3 → E4；
2. ForgeMM：数据审计 → E1/E2 SFT → E2 门槛 → E3/E4/E5 与消融 → 冻结验证集选型 →
   ChartQA/ChartQAPro 测试；
3. Hospital Workforce Platform：Compose 全链路验收 → 权限/幂等/冲突/审计验证 → k6 压测；
4. RepoPilot：质量门槛 → 本地模型容量 → FRAMES 三次 → DABench 三配置三次 →
   SWE-bench-Live fresh-5 生成与隔离评测。

同一个 `EXPERIMENT_RUN_ID`、Git commit 与 manifest 下，已经通过的阶段会根据 receipt 跳过。
改过代码或 manifest 后会自动视为新证据并重新执行。查看计划而不运行：

```bash
python3 scripts/experiment_suite.py list --profile full
```

只重跑一个项目：

```bash
python3 scripts/experiment_suite.py run --profile full --run-id full-5090-v1 --project ForgeMM --rerun
```

## 4. 数据、环境与产物位置

- 虚拟环境：`${EXPERIMENT_ENVS_ROOT:-~/.venvs}`；
- 模型和下载缓存：`${EXPERIMENT_ASSETS_ROOT:-~/resume-project-assets}`；
- 统一日志/receipt：`Resume_Portfolio/artifacts/experiment-suite/<run-id>`；
- 项目实验产物：各项目自己的 `artifacts` 或 `docs/experiments/artifacts`；
- 以上派生产物均由 `.gitignore` 排除，不会误上传模型、数据、令牌或大体积日志。

ForgeMM-Controlled 数据由固定 seed 的生成器在目标机重建；RepoPilot 的第三方参考源码不参与
运行，也不进入交付仓库。这样 GitHub 只保存自有代码、配置、数据身份与可引用的小型材料。

可以在首次启动前设置 `HF_TOKEN` 提高 Hugging Face 下载限额。脚本不会把令牌写入 receipt。
默认演示账号只用于本机隔离实验；若机器对外开放，请通过 `.env` 更换所有默认口令。

## 5. 停止与恢复规则

- 宿主预检、数据哈希、质量门槛或一个实验阶段失败时立即停止，不把后续阶段标记为成功；
- 先检查 `artifacts/experiment-suite/<run-id>/logs/<stage>.log`，修复后重跑原命令；
- 不手工删除成功 receipt；要有意重新运行时使用 `--rerun` 或新的 run id；
- 正式数据与模型身份必须保持固定；改变 revision、seed、配置或评测集时使用新的 run id；
- DriveVLA checkpoint 模型卡目前缺少标准许可证元数据，公开 checkpoint 衍生物或正式结果前
  仍需完成权利边界核验。
