# 四项目实验室新 PC 完整操作手册

> 文档版本：1.0  
> 核验日期：2026-09-08  
> 适用仓库：<https://github.com/xuxuhanhanzi/Resume-Project>  
> 最低自动化基线：`4b9e587d822f120866a39194cf59cc1e4517ee3e`  
> 目标设备：Windows 11、NVIDIA RTX 5090 24GB、64GB RAM、2TB SSD、Intel Core Ultra 9 级 CPU  
> 面向人员：负责初始化实验机、执行实验、监控失败和交付实验记录的工作人员

## 1. 手册目标

本手册用于把一台全新的 Windows 实验室 PC 配置成可复现的四项目实验节点，并完成：

1. Windows、WSL2、Ubuntu、NVIDIA GPU 和 Docker 环境配置；
2. 从 GitHub 拉取固定版本代码；
3. 自动建立隔离的 Python 环境；
4. 自动下载并核验模型、数据集和上游代码；
5. 先运行 smoke 验证，再顺序运行所有正式实验；
6. 保存日志、阶段 receipt、结果摘要和代码版本；
7. 在网络中断、进程失败或机器重启后继续执行；
8. 将可审计的实验结果交还项目负责人。

除本手册明确标为“PowerShell”的操作外，所有项目命令都必须在 **Ubuntu 24.04 的 WSL2 终端**中执行。

## 2. 自动化边界

仓库已经自动化以下工作：

- 安装 `uv`；
- 为 DriveVLA-Guard、ForgeMM、RepoPilot 建立互相隔离的 Python 环境；
- 安装固定的 PyTorch CUDA 12.8 依赖；
- 拉取固定 revision 的模型、数据和上游代码；
- 构建 Hospital 与 RepoPilot 所需 Docker 镜像；
- 按固定顺序运行四个项目；
- 写入逐阶段日志和 JSON receipt；
- 失败立即停止；
- 使用相同 run id 重跑时跳过已经成功且指纹一致的阶段。

以下工作不能由仓库替代，必须先由工作人员完成一次：

- Windows Update 和 BIOS/UEFI 虚拟化；
- NVIDIA Windows 驱动安装；
- WSL2 和 Ubuntu 24.04 安装；
- Docker Desktop 安装及 WSL integration；
- OpenScene 数据许可确认；
- 实验期间的供电、散热、网络和磁盘保障。

## 3. 执行流程总览

```text
Windows/BIOS 验收
  -> NVIDIA Windows 驱动验收
  -> WSL2 + Ubuntu 24.04
  -> WSL 内存/磁盘配置
  -> Docker Desktop + Ubuntu integration
  -> Ubuntu 基础工具
  -> 克隆并锁定 Git commit
  -> 主机预检
  -> smoke 环境准备
  -> smoke 四项目实验
  -> smoke 验收
  -> full 数据/模型准备
  -> full 四项目顺序实验
  -> 结果与证据交接
```

任何阶段未通过时，不得绕过检查继续后续阶段。

## 4. 执行前信息登记

实验执行人员应先填写：

| 项目 | 记录值 |
|---|---|
| 执行人员 |  |
| 开始日期 |  |
| 机器资产编号 |  |
| Windows 版本/build |  |
| GPU 型号 |  |
| NVIDIA 驱动版本 |  |
| 物理内存 |  |
| 2TB SSD 盘符 |  |
| Ubuntu 发行版名称 |  |
| WSL 版本 |  |
| Docker Desktop 版本 |  |
| Git commit |  |
| smoke run id |  |
| full run id |  |

## 5. 资源规划

### 5.1 硬性门槛

仓库的 `scripts/preflight_experiment_pc.py` 会检查：

| 资源 | smoke/full 门槛 |
|---|---:|
| WSL 内可见 RAM | 至少 58 GiB |
| RTX 显存 | 至少 23,500 MiB |
| smoke 可用磁盘 | 至少 40 GiB |
| full 可用磁盘 | 至少 300 GiB |
| 操作环境 | Linux on WSL2 |
| Docker | CLI 与 daemon 均可用 |

### 5.2 实际磁盘建议

截至 2026-09-08，DriveVLA full 下载的 32 个 OpenScene 测试相机压缩包约为 119.1 GiB；元数据和地图再增加约 1.35 GiB。脚本会保留下载包并同时生成解压数据、metric cache 和 no-CoT 数据，因此：

- 300 GiB 是自动预检的最低门槛；
- 开始 full 前建议 Ubuntu 根文件系统至少有 **500 GiB 可用空间**；
- Ubuntu 的虚拟磁盘和 Docker Desktop 的 disk image 都应放在 2TB SSD；
- 不要把仓库放在 `/mnt/c` 或 `/mnt/d` 下；
- 不要在同一 GPU 上并行运行两个项目。

OpenScene 仓库整体约 3.04TB，但脚本只选择 NAVSIM 测试所需的 metadata 与 camera archives，不会下载整个仓库。

## 6. 第一阶段：Windows 与 BIOS 准备

### 6.1 更新 Windows

1. 打开“设置 → Windows 更新”。
2. 安装所有稳定更新和硬件更新。
3. 重启 Windows。
4. 运行 `winver`，登记 Windows 版本和 build。

建议使用仍受微软和 Docker 支持的 Windows 11 版本。

### 6.2 检查硬件虚拟化

打开“任务管理器 → 性能 → CPU”，确认：

```text
虚拟化：已启用
```

如果显示未启用，进入 BIOS/UEFI，打开 Intel Virtualization Technology/VT-x，然后重启 Windows。

### 6.3 电源和运行条件

正式实验前：

- 接通稳定电源；
- 在 Windows 电源设置中暂时禁止自动睡眠和休眠；
- 确保机箱/工作站散热正常；
- 不运行游戏、额外模型服务或其他 GPU 作业；
- 不在实验期间自动安装 Windows 更新并重启；
- 不让 Docker Desktop 自动退出。

## 7. 第二阶段：NVIDIA Windows 驱动

从 NVIDIA 官方驱动页面安装支持 RTX 5090 的最新稳定 Windows 驱动：

<https://www.nvidia.com/Download/index.aspx>

安装并重启后，在 **PowerShell** 运行：

```powershell
nvidia-smi
```

验收条件：

- 能看到 `NVIDIA GeForce RTX 5090`；
- 总显存约为 24GB；
- 命令没有驱动通信错误。

**禁止在 Ubuntu 中安装 `nvidia-driver-*`、`cuda-drivers` 或会携带 Linux 显卡驱动的 CUDA 元包。** WSL2 通过 Windows 驱动向 Linux 暴露 CUDA。仓库安装的 PyTorch wheel 已包含所需 CUDA runtime，当前流程不要求单独安装完整 CUDA Toolkit。

官方说明：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>

## 8. 第三阶段：安装 WSL2 和 Ubuntu 24.04

### 8.1 启用 WSL2

以管理员身份打开 **PowerShell**：

```powershell
wsl --install --no-distribution
```

如果提示重启，立即重启。重启后继续：

```powershell
wsl --update
wsl --set-default-version 2
wsl --version
wsl --list --online
```

如果 `wsl --install` 下载一直停在 0%，可增加 `--web-download`。

官方说明：

- <https://learn.microsoft.com/en-us/windows/wsl/install>
- <https://learn.microsoft.com/en-us/windows/wsl/basic-commands>

### 8.2 把 Ubuntu 安装到 2TB SSD

以下示例假定 2TB SSD 是 `D:`。如果实际盘符不同，必须替换成实际盘符。

```powershell
wsl --install --distribution Ubuntu-24.04 --location "D:\WSL\Ubuntu-24.04"
```

若 Microsoft Store 通道不可用：

```powershell
wsl --install --web-download --distribution Ubuntu-24.04 --location "D:\WSL\Ubuntu-24.04"
```

如果 2TB SSD 就是系统盘，或者当前 WSL 不支持 `--location`，可使用：

```powershell
wsl --install --distribution Ubuntu-24.04
```

第一次打开 Ubuntu 时，根据提示创建普通 Linux 用户名和密码。不要把日常实验用户设置为 `root`。

### 8.3 验证 WSL2

在 PowerShell 运行：

```powershell
wsl --list --verbose
```

预期：

```text
NAME            STATE     VERSION
Ubuntu-24.04    Running   2
```

如 VERSION 为 1：

```powershell
wsl --set-version Ubuntu-24.04 2
```

## 9. 第四阶段：配置 WSL 内存和交换空间

WSL 默认内存上限通常是 Windows 总内存的 50%，即这台机器约 32GB，无法通过项目的 58GiB 预检。

在 PowerShell 中打开配置文件：

```powershell
notepad $env:USERPROFILE\.wslconfig
```

写入：

```ini
[wsl2]
memory=60GB
swap=32GB
localhostForwarding=true
```

如果希望交换文件也明确放在 2TB 的 `D:` 盘，可使用：

```ini
[wsl2]
memory=60GB
swap=32GB
swapfile=D:\\WSL\\wsl-swap.vhdx
localhostForwarding=true
```

保存后运行：

```powershell
wsl --shutdown
```

等待约 10 秒，再启动 Ubuntu。在 Ubuntu 中检查：

```bash
free -h
```

验收条件：`Mem` 总量接近 60GiB。`memory=60GB` 是最大上限，不代表 WSL 启动后立即占用 60GB；但正式实验期间仍应关闭 Windows 上不必要的大型软件。

官方说明：<https://learn.microsoft.com/en-us/windows/wsl/wsl-config>

### 9.1 必要时扩展 WSL 虚拟磁盘

先在 Ubuntu 查看：

```bash
df -h /
```

如果总容量或可用空间不足，在 PowerShell 中关闭 WSL：

```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --resize 1TB
```

`wsl --manage --resize` 需要较新的 WSL。不要直接用 Windows 文件工具移动或编辑 `ext4.vhdx`。

官方说明：<https://learn.microsoft.com/en-us/windows/wsl/disk-space>

## 10. 第五阶段：安装和配置 Docker Desktop

下载并安装 Docker Desktop for Windows：

<https://docs.docker.com/desktop/setup/install/windows-install/>

打开 Docker Desktop 并配置：

1. `Settings → General`：启用 `Use the WSL 2 based engine`；
2. `Settings → Resources → WSL Integration`：启用 `Ubuntu-24.04`；
3. `Settings → Resources → Advanced`：将 Docker disk image 移到 2TB SSD；
4. 应用设置并等待 Docker Desktop 完全启动；
5. 使用 Linux containers，不要切换到 Windows containers。

不要同时在 Ubuntu 中另装一套 `docker.io` daemon，否则容易出现 Docker context、端口和镜像存储冲突。

官方说明：

- <https://docs.docker.com/desktop/features/wsl/>
- <https://docs.docker.com/desktop/settings-and-maintenance/settings/>

### 10.1 在 Ubuntu 验证 Docker

```bash
docker version
docker compose version
docker info
```

以上三条都必须成功。

### 10.2 验证容器 GPU

```bash
docker run --rm -it --gpus=all \
  nvcr.io/nvidia/k8s/cuda-sample:nbody \
  nbody -gpu -benchmark
```

验收条件：容器能够识别 GPU 并完成 benchmark。

官方说明：<https://docs.docker.com/desktop/features/gpu/>

## 11. 第六阶段：Ubuntu 基础环境

打开 Ubuntu 24.04，执行：

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  git \
  curl \
  ca-certificates \
  build-essential \
  python3 \
  python3-venv \
  unzip \
  jq \
  tmux
```

检查：

```bash
git --version
curl --version
python3 --version
nvidia-smi
docker info
```

禁止使用 Windows Python、Windows Anaconda 或 Git Bash 运行项目脚本。

## 12. 网络与账户准备

首次安装需要访问：

- `github.com`、`raw.githubusercontent.com`；
- `huggingface.co` 及其大文件/CDN 域名；
- `download.pytorch.org`；
- `pypi.org`、`files.pythonhosted.org`；
- Docker Hub、`quay.io`、`nvcr.io`；
- `ollama.com`、Ollama 模型 registry；
- `repo.maven.apache.org`、`registry.npmjs.org`；
- Motional 的 nuPlan S3 下载地址。

若实验室使用代理，应在开始下载前由网络管理员为 WSL2 和 Docker Desktop 同时配置代理。不要在下载过程中临时切换代理或镜像源。

Hugging Face token 不是必需项，但可以降低匿名下载限额风险。不要把 token 写进仓库或共享文档。需要时在 Ubuntu 中使用隐藏输入：

```bash
read -rsp "HF token（可留空）: " HF_TOKEN
echo
if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
fi
```

如 Docker Hub 出现匿名拉取限额，可执行：

```bash
docker login
```

## 13. 第七阶段：克隆并锁定代码

仓库必须克隆到 Ubuntu 的 Linux 文件系统：

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/xuxuhanhanzi/Resume-Project.git
cd Resume-Project
```

确认远端和代码状态：

```bash
git remote -v
git status --short
git log -1 --oneline
git rev-parse HEAD
```

首次移交的最低自动化基线为：

```text
4b9e587d822f120866a39194cf59cc1e4517ee3e
```

如果项目负责人没有指定必须使用的实验 commit，保留刚克隆的最新 `main`，并确认它包含最低自动化基线：

```bash
git merge-base --is-ancestor \
  4b9e587d822f120866a39194cf59cc1e4517ee3e \
  HEAD && echo "automation baseline present"
```

如果项目负责人提供了一个明确 commit，则按负责人给出的 commit 运行并记录，不要自行选择其他版本。

再次确认：

```bash
git status --short
git rev-parse HEAD
```

`git status --short` 应无输出。实验开始后不要执行 `git pull`、切换分支或修改脚本；否则阶段指纹会改变，应使用新的 run id 重新建立证据。

进入项目总目录：

```bash
cd ~/src/Resume-Project/Resume_Portfolio
```

## 14. 第八阶段：主机预检

先定义本次 smoke 的唯一 run id。建议包含机器和日期，但只使用英文字母、数字、点、下划线和短横线：

```bash
export EXPERIMENT_RUN_ID=smoke-5090-20260908-v1
```

先用 full 门槛运行一次人工预检，并把结果保存在当前 smoke run id 的 operator 目录中：

```bash
python3 scripts/preflight_experiment_pc.py \
  --profile full \
  --output "artifacts/experiment-suite/${EXPERIMENT_RUN_ID}/operator/preflight-full.json"
```

检查命令输出中每一项都是 `[PASS]`。后续 smoke/full bootstrap 还会在各自 run id 下再次自动执行相应预检。

同时保存人工环境快照：

```bash
mkdir -p "artifacts/experiment-suite/${EXPERIMENT_RUN_ID}/operator"
{
  date -u
  pwd
  uname -a
  cat /etc/os-release
  python3 --version
  git rev-parse HEAD
  git status --short
  free -h
  df -h /
  nvidia-smi
  docker version
  docker compose version
} |& tee "artifacts/experiment-suite/${EXPERIMENT_RUN_ID}/operator/environment.txt"
```

若 full 预检提示磁盘不足，不得开始 OpenScene 下载。

## 15. 自动下载内容与位置

工作人员通常不需要单独运行数据下载脚本；`bootstrap_experiment_pc.sh` 会按 profile 调用它们。

### 15.1 默认目录

| 类型 | 默认位置 |
|---|---|
| Python 虚拟环境 | `~/.venvs/` |
| 共用模型和大数据资产 | `~/resume-project-assets/` |
| 总调度日志和 receipt | `Resume_Portfolio/artifacts/experiment-suite/<run-id>/` |
| 项目产物 | 各项目 `artifacts/` 或 `docs/experiments/artifacts/` |

需要自定义时，必须在首次 setup 之前设置，并在后续所有终端保持一致：

```bash
export EXPERIMENT_ENVS_ROOT="$HOME/.venvs"
export EXPERIMENT_ASSETS_ROOT="$HOME/resume-project-assets"
```

### 15.2 DriveVLA-Guard

full setup 自动处理：

- AutoVLA commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`；
- Qwen2.5-VL-3B-Instruct revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`；
- AutoVLA checkpoint revision：`a7d7ba3ed7529b248d2694c2defa31b35208340f`；
- checkpoint SHA-256：`58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473`；
- OpenScene revision：`a76f840b65e972bc45e56c2adced897498e9a026`；
- nuPlan maps v1.1；
- 数据安全解压、路径布局、metric cache 和 no-CoT 预处理。

主要位置：

```text
~/resume-project-assets/drivevla-guard/
├── AutoVLA/
├── models/
├── downloads/
├── dataset/nuplan/
└── navsim-exp/
```

OpenScene 许可：CC BY-NC-SA 4.0。数据页面：

<https://huggingface.co/datasets/OpenDriveLab/OpenScene>

只有在项目负责人和执行人员确认本次用途符合许可后，才可设置：

```bash
export DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1
```

不得重新分发 OpenScene 数据、下载归档或 AutoVLA checkpoint。AutoVLA checkpoint 模型卡当前缺少标准许可证元数据，公开其衍生物或正式结果前必须再次核验权利边界。

### 15.3 ForgeMM

setup 自动处理：

- Qwen2.5-VL-3B-Instruct 固定 revision；
- 使用 seed `20260824` 重建 ForgeMM-Controlled-v1-Lite；
- 审计受控数据；
- full 时下载固定 revision 的 ChartQA 和 ChartQAPro；
- 对关键标注文件和 ChartQAPro parquet 执行 SHA-256 校验。

主要位置：

```text
~/resume-project-assets/forgemm/
Resume_Portfolio/ForgeMM/datasets/ForgeMM-Controlled-v1-Lite/
Resume_Portfolio/ForgeMM/datasets/ChartQA/
Resume_Portfolio/ForgeMM/datasets/ChartQAPro/
```

### 15.4 Hospital Workforce Platform

没有外部训练数据集。setup 会：

- 拉取固定版本的 MySQL、Redis、Keycloak、Prometheus、Grafana 和 k6 镜像；
- 构建后端和前端镜像；
- 验证 Compose 配置。

默认端口：

| 服务 | 端口 |
|---|---:|
| MySQL | 3306 |
| Redis | 6379 |
| Backend | 8080 |
| Keycloak | 8081 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Frontend | 5173 |

检查冲突：

```bash
ss -ltn | grep -E ':(3306|6379|8080|8081|9090|3000|5173)\b' || true
```

如果端口被占用，可在运行前设置替代端口，例如：

```bash
export MYSQL_PORT=13306
export REDIS_PORT=16379
export BACKEND_PORT=18080
export KEYCLOAK_PORT=18081
export PROMETHEUS_PORT=19090
export GRAFANA_PORT=13000
export FRONTEND_PORT=15173
```

默认账号和口令只允许用于本机隔离实验，不允许将这些服务暴露到不受信任的网络。

### 15.5 RepoPilot

setup 自动处理：

- 安装并启动 Ollama；
- 拉取 `qwen2.5:7b`，运行时记录实际模型 digest；
- 下载固定 revision 的 FRAMES 和 DABench；
- full 时下载固定 revision 的 SWE-bench-Live；
- 校验数据 SHA-256；
- 构建 DABench sandbox；
- full 时准备 fresh-5 的隔离 Git workspace 和 Docker 镜像。

主要位置：

```text
~/resume-project-assets/repopilot/
Resume_Portfolio/RepoPilot/evaluation/benchmarks/data/
Resume_Portfolio/RepoPilot/evaluation/benchmarks/workspaces/
Resume_Portfolio/RepoPilot/artifacts/
```

## 16. 第九阶段：运行 smoke

smoke 不是纯代码检查。它会下载 Qwen、Ollama 模型、FRAMES/DABench 和 Docker 镜像，因此需要稳定网络和一定时间。

推荐在 `tmux` 中运行：

```bash
cd ~/src/Resume-Project/Resume_Portfolio
tmux new -s resume-smoke
```

在 tmux 会话中：

```bash
export EXPERIMENT_RUN_ID=smoke-5090-20260908-v1
bash scripts/bootstrap_experiment_pc.sh smoke
bash scripts/run_all_experiments.sh smoke
python3 scripts/experiment_suite.py status \
  --profile smoke \
  --run-id "$EXPERIMENT_RUN_ID"
```

如果 Ollama 安装过程中出现 `sudo` 提示，输入当前 Ubuntu 用户密码。不要用 `sudo` 启动整个实验脚本。

从 tmux 分离：按 `Ctrl+B`，松开后按 `D`。重新连接：

```bash
tmux attach -t resume-smoke
```

### 16.1 smoke 自动顺序

```text
host-preflight
drivevla-setup
forgemm-setup
hospital-setup
repopilot-setup
drivevla-experiments
forgemm-experiments
hospital-experiments
repopilot-experiments
```

### 16.2 smoke 完成判据

运行：

```bash
python3 scripts/experiment_suite.py status \
  --profile smoke \
  --run-id smoke-5090-20260908-v1
```

九个阶段都应显示 `passed`。同时检查：

```bash
cat artifacts/experiment-suite/smoke-5090-20260908-v1/summary.json
```

只有 smoke 全部通过，才允许开始 full。

## 17. 第十阶段：运行正式实验

### 17.1 开始前检查

- [ ] smoke 九阶段全部通过；
- [ ] OpenScene 许可已经确认；
- [ ] Ubuntu 根文件系统可用空间至少 300GiB，建议 500GiB 以上；
- [ ] Docker Desktop 正在运行；
- [ ] `nvidia-smi` 正常；
- [ ] 无其他 GPU 作业；
- [ ] Windows 已禁止自动睡眠；
- [ ] Git 工作区干净；
- [ ] 网络可以访问全部下载源；
- [ ] 已确定唯一 full run id。

### 17.2 正式启动命令

```bash
cd ~/src/Resume-Project/Resume_Portfolio
tmux new -s resume-full
```

在 tmux 会话中：

```bash
export DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1
export EXPERIMENT_RUN_ID=full-5090-20260908-v1

bash scripts/bootstrap_experiment_pc.sh full && \
bash scripts/run_all_experiments.sh full
```

`&&` 表示只有所有 setup 阶段成功后，才会自动进入实验阶段。

### 17.3 正式实验顺序

1. **DriveVLA-Guard**
   - 代码、测试与合成证据门槛；
   - AutoVLA 官方 B0 重复两次；
   - DriveVLA adapter B0；
   - E2；
   - B0 重复性、adapter parity 和 E2 安全门控；
   - 通过后继续 E3、E4；
   - 生成官方 NAVSIM 指标摘要。

2. **ForgeMM**
   - 代码质量门；
   - 数据准备与审计；
   - E1/E2 SFT baseline；
   - 冻结验证集 E2 门控；
   - E3/E4/E5 与消融矩阵；
   - 冻结验证集选型；
   - ChartQA/ChartQAPro 正式测试；
   - 生成训练和最终摘要。

3. **Hospital Workforce Platform**
   - 启动完整 Compose 服务；
   - 健康检查；
   - Keycloak 认证与 401/403/200 权限检查；
   - 员工、排班、幂等、冲突和审计验证；
   - Prometheus 检查；
   - k6 正式压测；
   - 保存 Compose 日志并关闭服务。

4. **RepoPilot**
   - 代码质量门；
   - 本地 Ollama 模型容量测试；
   - FRAMES 60 条，重复三次；
   - DABench base/guardrail/robust，分别重复三次；
   - SWE-bench-Live fresh-5 生成；
   - 隔离 workspace 中执行评测。

项目按顺序运行，避免 24GB GPU 被多个任务同时占用。

## 18. 运行状态与日志

### 18.1 查看总状态

另开一个 Ubuntu 终端：

```bash
cd ~/src/Resume-Project/Resume_Portfolio
python3 scripts/experiment_suite.py status \
  --profile full \
  --run-id full-5090-20260908-v1
```

### 18.2 查看当前阶段日志

```bash
ls -lah artifacts/experiment-suite/full-5090-20260908-v1/logs
tail -n 200 artifacts/experiment-suite/full-5090-20260908-v1/logs/*.log
```

如已知阶段名称，例如 `forgemm-experiments`：

```bash
tail -f artifacts/experiment-suite/full-5090-20260908-v1/logs/forgemm-experiments.log
```

### 18.3 查看执行计划但不运行

```bash
python3 scripts/experiment_suite.py list --profile full
```

### 18.4 总调度产物

```text
artifacts/experiment-suite/<run-id>/
├── host-preflight.json
├── summary.json
├── logs/
│   └── <stage>.log
├── receipts/
│   └── <stage>.json
└── operator/
    └── environment.txt
```

receipt 保存阶段命令、Git commit、是否 dirty、开始/结束时间、退出码、日志路径和执行指纹。

## 19. 中断与恢复

### 19.1 普通失败

调度器遇到非零退出码会立即停止，不会执行后续项目。处理步骤：

1. 打开失败阶段日志；
2. 记录准确错误、失败阶段和时间；
3. 修复网络、磁盘、Docker 或环境问题；
4. 保持相同 Git commit、环境路径和 `EXPERIMENT_RUN_ID`；
5. 重新运行原命令。

setup 阶段失败：

```bash
export DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1
export EXPERIMENT_RUN_ID=full-5090-20260908-v1
bash scripts/bootstrap_experiment_pc.sh full
```

实验阶段失败：

```bash
export DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1
export EXPERIMENT_RUN_ID=full-5090-20260908-v1
bash scripts/run_all_experiments.sh full
```

成功且指纹一致的阶段会根据 receipt 自动跳过。DriveVLA、ForgeMM 和 RepoPilot 内部还保存更细粒度的步骤 marker。

### 19.2 机器重启或断电

1. 启动 Windows；
2. 启动 Docker Desktop并等待 ready；
3. 打开 Ubuntu；
4. 检查 `nvidia-smi` 和 `docker info`；
5. 重新导出相同环境变量；
6. 重新运行原命令。

不要删除 receipt、marker、数据目录或缓存目录。

### 19.3 有意重新运行

推荐使用新的 run id：

```bash
export EXPERIMENT_RUN_ID=full-5090-20260908-v2
```

只有明确需要覆盖同一 run id 的成功阶段时才使用：

```bash
python3 scripts/experiment_suite.py run \
  --profile full \
  --run-id full-5090-20260908-v1 \
  --rerun
```

## 20. 单项目与指定阶段运行

四个合法项目名称：

```text
DriveVLA-Guard
ForgeMM
Hospital_Workforce_Platform
RepoPilot
```

只运行 ForgeMM：

```bash
python3 scripts/experiment_suite.py run \
  --profile full \
  --run-id full-5090-20260908-v1 \
  --project ForgeMM
```

从 Hospital 实验阶段继续：

```bash
python3 scripts/experiment_suite.py run \
  --profile full \
  --run-id full-5090-20260908-v1 \
  --from-stage hospital-experiments
```

这些命令只应用于故障定位或负责人明确要求的局部重跑。正式首次实验必须使用统一入口保持标准顺序。

## 21. 项目结果位置

### 21.1 DriveVLA-Guard

```text
DriveVLA-Guard/artifacts/official/<run-id>_b0_e2_gate.json
DriveVLA-Guard/artifacts/official/<run-id>_summary.json
~/resume-project-assets/drivevla-guard/navsim-exp/
```

### 21.2 ForgeMM

```text
ForgeMM/artifacts/runs/experiment_pc/<run-id>/stage04/
ForgeMM/artifacts/runs/experiment_pc/<run-id>/stage04/e2_gate.json
ForgeMM/artifacts/runs/experiment_pc/<run-id>/stage04/frozen_val_summary.json
ForgeMM/artifacts/runs/experiment_pc/<run-id>/stage04/formal_training_summary.json
ForgeMM/artifacts/runs/experiment_pc/<run-id>/stage04/cloud_final_summary.json
```

### 21.3 Hospital Workforce Platform

```text
Hospital_Workforce_Platform/docs/experiments/artifacts/experiment-pc/<run-id>/acceptance.json
Hospital_Workforce_Platform/docs/experiments/artifacts/experiment-pc/<run-id>/compose.log
Hospital_Workforce_Platform/docs/experiments/artifacts/experiment-pc/<run-id>/compose-ps.json
Hospital_Workforce_Platform/docs/experiments/artifacts/experiment-pc/<run-id>/
```

### 21.4 RepoPilot

```text
RepoPilot/artifacts/experiment_pc/<run-id>/steps/
RepoPilot/artifacts/benchmarks/<run-id>_capacity.json
RepoPilot/artifacts/benchmarks/<run-id>_frames60_rep*/
RepoPilot/artifacts/benchmarks/<run-id>_dabench_*/
RepoPilot/artifacts/benchmarks/<run-id>_swebench_fresh5/
```

不要把模型权重、数据集、Docker volume 或完整下载缓存提交到 GitHub。

## 22. 常见故障处理

### 22.1 `wsl --install` 停在 0%

```powershell
wsl --install --web-download --distribution Ubuntu-24.04
```

同时检查 Windows Update、代理和 Microsoft Store/下载策略。

### 22.2 Ubuntu 中没有 `nvidia-smi`

在 PowerShell 执行：

```powershell
wsl --update
wsl --shutdown
```

更新或重装 NVIDIA **Windows** 驱动后重启。不要在 Ubuntu 安装 Linux NVIDIA 驱动。

### 22.3 WSL 内存检查失败

检查 `%UserProfile%\.wslconfig` 是否为：

```ini
[wsl2]
memory=60GB
swap=32GB
```

然后：

```powershell
wsl --shutdown
```

重新启动 Ubuntu 并用 `free -h` 验证。

### 22.4 `docker` 命令不存在

检查 Docker Desktop：

- 使用 WSL2 backend；
- 已启用 Ubuntu-24.04 integration；
- Docker Desktop 已启动；
- Ubuntu 终端是在设置生效后重新打开的。

### 22.5 `Cannot connect to the Docker daemon`

启动 Docker Desktop，等待状态变成 running，再在 Ubuntu 执行：

```bash
docker info
```

### 22.6 Hugging Face 429、超时或下载中断

- 配置 `HF_TOKEN`；
- 检查代理和磁盘；
- 使用相同 run id 重新执行 setup；
- 不要修改 revision 或禁用哈希验证；
- 不要清空整个下载目录。

### 22.7 Docker Hub/NVCR 拉取失败

- 检查 Docker Desktop 代理；
- 必要时执行 `docker login`；
- 检查 `quay.io`、`nvcr.io` 和 Docker registry 是否被防火墙阻止；
- 使用相同 run id 重跑 setup。

### 22.8 磁盘不足

```bash
df -h /
docker system df
du -sh ~/resume-project-assets
```

不要在没有确认内容的情况下清空 Docker、数据或缓存。记录输出并联系项目负责人决定扩容或迁移。

### 22.9 Hospital 端口冲突

使用第 15.4 节的端口环境变量，不要直接修改 `docker-compose.yml`。改变端口后应使用新的 run id。

### 22.10 CUDA out of memory

1. 使用 `nvidia-smi` 检查是否有其他进程；
2. 关闭额外模型服务和 GPU 软件；
3. 确认项目正在顺序而不是并行运行；
4. 保存失败日志；
5. 不要自行修改 batch size、精度或模型版本；
6. 将失败信息交给项目负责人决定是否建立新实验配置。

### 22.11 门控实验失败

如果 DriveVLA B0/E2 gate、ForgeMM E2 gate、数据哈希或测试门失败，这属于实验或证据问题，而不是可以忽略的提示。不得强制跳过后继续产生正式结果。

## 23. 完成验收

full 完成后运行：

```bash
cd ~/src/Resume-Project/Resume_Portfolio
export EXPERIMENT_RUN_ID=full-5090-20260908-v1

python3 scripts/experiment_suite.py status \
  --profile full \
  --run-id "$EXPERIMENT_RUN_ID"

cat "artifacts/experiment-suite/${EXPERIMENT_RUN_ID}/summary.json"
git status --short
git rev-parse HEAD
```

完成标准：

- [ ] 九个统一阶段 receipt 全部为 `passed`；
- [ ] 总 summary 状态为 `passed`；
- [ ] Git commit 与登记值一致；
- [ ] Git 工作区没有人为修改的源码；
- [ ] DriveVLA gate 与正式 summary 存在；
- [ ] ForgeMM E2 gate、训练摘要和最终摘要存在；
- [ ] Hospital acceptance、k6 结果和 Compose 日志存在；
- [ ] RepoPilot FRAMES、DABench 和 SWE-bench-Live 结果存在；
- [ ] 所有失败、恢复和环境异常都有记录；
- [ ] 没有把 smoke 或 synthetic proxy 指标表述为正式指标。

## 24. 结果交接清单

执行人员应向项目负责人提供：

1. 填写完成的第 4 节信息表；
2. `EXPERIMENT_RUN_ID`；
3. `git rev-parse HEAD` 输出；
4. `operator/environment.txt`；
5. 总 `summary.json`；
6. `receipts/` 和 `logs/`；
7. 四个项目的最终 summary/gate/acceptance 文件；
8. 失败与恢复过程；
9. 任何人工更改、端口覆盖、代理设置或磁盘迁移；
10. 实际总耗时和峰值资源情况。

正式结论必须直接对应正式数据、固定 commit、固定模型/数据 revision、具体命令和保存的结果文件。smoke、synthetic 和内部 proxy 只能证明链路可运行，不能替代正式 benchmark 结论。

## 25. 官方参考资料

- Microsoft WSL 安装：<https://learn.microsoft.com/en-us/windows/wsl/install>
- Microsoft WSL 命令：<https://learn.microsoft.com/en-us/windows/wsl/basic-commands>
- Microsoft WSL 配置：<https://learn.microsoft.com/en-us/windows/wsl/wsl-config>
- Microsoft WSL 磁盘管理：<https://learn.microsoft.com/en-us/windows/wsl/disk-space>
- Microsoft WSL 文件系统建议：<https://learn.microsoft.com/en-us/windows/wsl/filesystems>
- NVIDIA CUDA on WSL：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- Docker Desktop Windows 安装：<https://docs.docker.com/desktop/setup/install/windows-install/>
- Docker Desktop WSL2：<https://docs.docker.com/desktop/features/wsl/>
- Docker Desktop GPU：<https://docs.docker.com/desktop/features/gpu/>
- Docker Desktop 设置：<https://docs.docker.com/desktop/settings-and-maintenance/settings/>
- OpenScene 数据集：<https://huggingface.co/datasets/OpenDriveLab/OpenScene>
- ChartQA 数据集：<https://huggingface.co/datasets/ahmed-masry/ChartQA>
- ChartQAPro 数据集：<https://huggingface.co/datasets/ahmed-masry/ChartQAPro>
- FRAMES 数据集：<https://huggingface.co/datasets/google/frames-benchmark>
- DABench 数据集：<https://huggingface.co/datasets/infiagent/DABench>
