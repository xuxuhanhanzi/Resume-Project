# 实验记录：R7 固定镜像沙箱现场安全验收

- 日期：2026-08-14
- 基础镜像：`python@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65`
- 本地镜像：`repopilot-sandbox:20260814`
- Image ID：`sha256:8388955c9a6ccce59758b0cebb396ac90819956a2f5da5450c7c02dcc3e30a5c`
- Docker Engine 29.6.1 / API 1.55，Linux containers。

## 目标

不再只检查 `docker run` 命令字符串，而是在固定 digest 镜像中读取内核/cgroup 状态，验证
不可信执行所声明的隔离控制确实生效。

## 构建与命令

```powershell
docker build `
  --build-arg BASE_IMAGE=python@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65 `
  -t repopilot-sandbox:20260814 -f deploy/docker/sandbox.Dockerfile .

$env:REPOPILOT_RUN_DOCKER_SECURITY = '1'
$env:REPOPILOT_SANDBOX_IMAGE = 'repopilot-sandbox:20260814'
.\.venv\Scripts\python.exe scripts\dev.py safety
```

## 现场结果

| 控制 | 容器内证据 | 结果 |
|---|---|---|
| 非 Root | uid=65534, gid=65534 | 通过 |
| 只读 RootFS | `statvfs('/').ST_RDONLY=true` | 通过 |
| capability drop | `CapEff=0000000000000000` | 通过 |
| no-new-privileges | `NoNewPrivs=1` | 通过 |
| 断网 | 到 1.1.1.1:53 的连接失败 | 通过 |
| 宿主 Docker socket | `/var/run/docker.sock` 不存在 | 通过 |
| 内存限制 | cgroup `memory.max=1073741824` | 通过 |
| PID 限制 | cgroup `pids.max=64` | 通过 |
| CPU 限制 | cgroup `cpu.max=100000 100000` | 通过 |
| 工作区 | bind mount 可写；root 其余位置只读 | 通过 |

完整安全测试组：6 passed，包含路径穿越、绝对路径、符号链接逃逸、本地不可信执行
fail-closed、高风险文件审批和上述真实容器检查。

## 未运行的破坏性攻击

没有执行真正的 fork bomb、磁盘填满或容器逃逸利用。PID、内存、只读 root 与 64 MiB
`/tmp` 的约束已由 cgroup/mount 直接读取验证；破坏性 payload 不因“项目完成”而合理化。

## 结论边界

此结果证明 Docker Desktop Linux VM 中的当前 profile。它不等于 gVisor/Firecracker、
内核级恶意代码隔离或多租户安全认证。Prompt injection 仍通过最小权限与确定性 verifier
降低影响，不能声称模型本身不会被诱导。
