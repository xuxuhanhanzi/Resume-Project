# RepoPilot

实验 PC 的自动入口为 `scripts/setup_experiment_pc.sh` 和 `scripts/run_experiment_pc.sh`。
它会建立独立环境、下载固定 revision 的 FRAMES/DABench/SWE-bench-Live、准备 Ollama
Qwen2.5-7B 和沙箱镜像，并依次运行容量、FRAMES、DABench 与 fresh SWE-bench 实验。

RepoPilot 是一个本地优先、反馈驱动、可治理的 Agent Runtime Lab + 评测骨架。它覆盖
知识研究、数据分析、软件工程三个领域，用最小实现验证 Agent 的规划、检索、工具、
执行反馈、验证、记忆和治理机制在不同任务形态下的效果。

它不是生产平台：没有 Web 前端、数据库或多租户。MCP 只会在显式信任后启动；除 stdio
服务外，仅支持无凭据的本机回环 HTTP 服务（不支持远程 HTTP/OAuth transport）。
项目最新状态和对外指标以 `docs/STATUS.md` 为唯一权威口径。

当前版本：`v1.19.0`；P61–P66 的可审计发布范围、Docker 实机安全门、无源码 DeepSeek 交互验收、离线交互转录、wheel SHA-256 收据及开发/holdout 证据边界见
[`docs/RELEASE_READINESS_v1.19.0.md`](docs/RELEASE_READINESS_v1.19.0.md)。P52–P60 的非破坏性发布基线、Docker 显式安全门、DeepSeek SSE 可靠性、取消后重启恢复、`/changes`、受限真实项目验收与 Windows CI 门禁见
[`docs/RELEASE_READINESS_v1.18.0.md`](docs/RELEASE_READINESS_v1.18.0.md)。P44–P51 的 P40 建议本地分级、取消/进程/Docker/Unicode 硬化、可恢复 provider 诊断、Windows 手动恢复与可选干净安装验收见
[`docs/RELEASE_NOTES_v1.17.0.md`](docs/RELEASE_NOTES_v1.17.0.md)。P38–P43 的无源码 DeepSeek 流式探针、可核验只读验收收据、交互诊断、项目固定 Windows 启动器与回归门禁见
[`docs/RELEASE_NOTES_v1.16.0.md`](docs/RELEASE_NOTES_v1.16.0.md)。P31–P36 的 receipt 完整性绑定、会话恢复提示、DeepSeek-first 模型提示与 Windows JSONL launcher 诊断见
[`docs/RELEASE_NOTES_v1.15.0.md`](docs/RELEASE_NOTES_v1.15.0.md)。P28–P30 的离线收据对比、严格 receipt 校验与回归发布门禁见
[`docs/RELEASE_NOTES_v1.14.0.md`](docs/RELEASE_NOTES_v1.14.0.md)。P21–P27 的 DeepSeek 公开合成开发基线、一次证据驱动修复、对照复跑、
Qwen/真实项目数据传输边界与发布验收见
[`docs/RELEASE_NOTES_v1.13.0.md`](docs/RELEASE_NOTES_v1.13.0.md)。P16–P20 的合成开发集运行收据、证据驱动改进队列、Windows launcher
冲突诊断、本地 `/workflow` 与离线发布门禁见
[`docs/RELEASE_NOTES_v1.12.0.md`](docs/RELEASE_NOTES_v1.12.0.md)。P11–P15 的 Windows 验收、合成开发集、证据闭环、Supervisor
配额与 MCP 审计见 [`docs/RELEASE_NOTES_v1.11.0.md`](docs/RELEASE_NOTES_v1.11.0.md)。P0–P4 的安装诊断、评测分类、只读附加目录、任务收据与 MCP
加固见 [`docs/RELEASE_NOTES_v1.9.0.md`](docs/RELEASE_NOTES_v1.9.0.md)。本地会话导出与分享前审阅见
[`docs/RELEASE_NOTES_v1.8.0.md`](docs/RELEASE_NOTES_v1.8.0.md)，交互终端的 asyncio 兼容修复见
[`docs/RELEASE_NOTES_v1.7.1.md`](docs/RELEASE_NOTES_v1.7.1.md)，交互终端体验与安全输入历史见
[`docs/RELEASE_NOTES_v1.7.0.md`](docs/RELEASE_NOTES_v1.7.0.md)，只读代码审查与安全审查见
[`docs/RELEASE_NOTES_v1.6.0.md`](docs/RELEASE_NOTES_v1.6.0.md)，可审计的 turn 差异与安全回退见
[`docs/RELEASE_NOTES_v1.5.0.md`](docs/RELEASE_NOTES_v1.5.0.md)，可选择、可持久化的验证计划见
[`docs/RELEASE_NOTES_v1.4.0.md`](docs/RELEASE_NOTES_v1.4.0.md)，工作流控制器与最终验证报告见
[`docs/RELEASE_NOTES_v1.3.0.md`](docs/RELEASE_NOTES_v1.3.0.md)，P0–P3 的可靠性与扩展硬化记录见
[`docs/RELEASE_NOTES_v1.2.0.md`](docs/RELEASE_NOTES_v1.2.0.md)，原有 CLI 硬化记录见
[`docs/RELEASE_NOTES_v1.1.0.md`](docs/RELEASE_NOTES_v1.1.0.md)。

## 三领域评测

| 场景 | 数据集 | 主指标 | 当前结果 |
|---|---|---|---|
| 知识研究 | Google FRAMES | 答案准确率 | **18.3% (11/60)**，3 次确定性重复，✅ 可引用 |
| 数据分析 | InfiAgent-DABench | 问题准确率 | **73.3%**，3 次为 25/35、26/35、26/35，✅ 可引用 |
| 软件工程 | SWE-bench-Live | 官方 resolved rate | **0.0% (0/5)**，干净 holdout，✅ 可引用负结果 |

三领域分开报告，禁止合成跨领域总分。

## 已实现闭环

```text
Task Spec → Context → Local Model → Structured Tool Call
         → Policy → Tool/Sandbox → Observation → Checkpoint
         → Deterministic Verifier → Complete/Repair/Fail
```

核心边界：

- `ModelProvider` 首选本地 Qwen 的 OpenAI-compatible 服务；
- `ScriptedProvider` 让 Runtime、恢复和安全测试完全离线；
- 模型只建议动作，只有确定性 Verifier 可以完成任务；
- 所有路径先规范化，任意 Shell 不是默认工具；
- 不可信执行必须使用 Docker 沙箱，缺少 Docker 时 fail-closed；
- 隐藏测试不属于 Public Task Spec，也不能挂载进 Agent 工作区；
- 每次状态转换、工具调用和验证都有 Checkpoint 与 JSONL Trace。

## 检索改进 (P8A，历史 10 题开发消融)

| 配置 | 平均准确率 | 说明 |
|---|---|---|
| B1 BM25 | 16.7% | 词法匹配 baseline |
| R1 Dense | 0.0% | 通用 embedding 在百科 chunk 上失效 |
| **R2 Hybrid (BM25+Dense RRF)** | **20.0%** | **最佳，+3.3pp** |
| R3 Hybrid+Rerank | 20.0% | embedding reranker 无额外收益 |
| A2 No-retrieval | 20.0% | 首段 vs 检索，差异不显著 |

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q --basetemp D:/Temp/rp_pytest
.\.venv\Scripts\python.exe -m repopilot demo scripted
```

Windows 上也可以使用保守的项目内安装脚本；它只会创建或复用 `.venv`，绝不会删除旧环境、
修改 PowerShell `$PROFILE`、保存 API Key 或更改全局 Python：

```powershell
Set-Location "D:\path\to\RepoPilot"
.\scripts\setup_windows.ps1 -WithDev

# 只读验收：检查项目内 launcher、doctor、shell-init、MCP 配置与聚焦恢复测试。
# 不会修改 $PROFILE、凭据、项目配置或调用任何模型。
.\scripts\acceptance_windows.ps1

# 发布前的离线门禁：不会调用模型、读取 Key、改 PATH 或写入 $PROFILE。
.\scripts\release_check_windows.ps1

# 可选的 clean-install packaging smoke：可能解析 Python 依赖，保留 artifacts\package_smoke 证据。
.\scripts\release_check_windows.ps1 -PackageSmoke

# 生成 Git 发布基线；只记录状态，绝不暂存、提交、重置或删除文件。
.\scripts\release_check_windows.ps1 -Baseline

# 仅在 Docker Desktop 正常后显式构建沙箱镜像并运行实时安全门；绝不退回宿主机执行。
.\scripts\release_check_windows.ps1 -DockerSecurity
```

若 `doctor` 报告 PATH 指向 Anaconda 或旧虚拟环境，请始终使用脚本输出的
`.venv\Scripts\repopilot.exe`，或在当前窗口执行 `shell-init`。不要在正在运行 RepoPilot 时
重装，也不要手动删除 `~epopilot-*.dist-info`；关闭会话后在新建的项目 venv 中重新安装即可。
可通过 `.\.venv\Scripts\repopilot.exe shell-doctor powershell` 只读地查看当前启动器冲突，并得到
仅对当前 PowerShell 窗口有效的修复命令。也可在任何窗口用项目固定启动器（不依赖 PATH）：

```powershell
& .\scripts\repopilot.ps1 repopilot
```

完整的手动、非破坏性恢复步骤见
[`docs/WINDOWS_ENVIRONMENT_RECOVERY.md`](docs/WINDOWS_ENVIRONMENT_RECOVERY.md)。

HTTP/SSE 服务使用可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
$env:REPOPILOT_API_TOKEN = '<至少 16 字符的随机令牌>'
repopilot-api --task-root evaluation\cases --model qwen2.5:7b
```

## v2 CLI（Alpha / Beta 核心实现中）

v2 提供面向真实项目的持久化交互会话。首次进入项目时需要显式信任工作区；默认
`manual` 权限模式会在编辑前请求确认，`plan` 模式只允许只读工具。

### 一次配置，以后直接进入项目

对于 DeepSeek 和 Qwen，RepoPilot 将 API Key 保存到 Windows 凭据管理器，而把不含密钥的
模型配置和项目别名保存在 `~/.repopilot/config.json`。环境变量仍可临时覆盖已保存的凭据，
但不再是每次启动所必需的长命令。以下只需要在首次配置（或切换 Key）时执行一次：

```powershell
# 这里的 $cli 仅用于首次定位当前虚拟环境里的 RepoPilot。
$cli = "D:\Users\27475\Desktop\Resume_Project\Resume_Portfolio\RepoPilot\.venv\Scripts\repopilot.exe"

# 密钥输入不会显示、不会写入 config.json 或 session 文件。
& $cli auth login deepseek
& $cli config set-profile deepseek-main deepseek deepseek-v4-flash --default

# 用一个简短名字保存项目路径与默认模型。
& $cli project add repopilot "D:\Users\27475\Desktop\Resume_Project\Resume_Portfolio\RepoPilot" --profile deepseek-main

# 为当前 PowerShell 会话注册 repopilot 命令；它不会修改 $PROFILE。
Invoke-Expression (& $cli shell-init powershell)
repopilot repopilot
```

如果希望每个新的 PowerShell 窗口都拥有 `repopilot`，执行 `& $cli shell-init powershell`，把输出的
**单行函数**手动追加到 PowerShell `$PROFILE`，然后重新打开终端。`repopilot repopilot`（或
`repopilot open repopilot`）会在 RepoPilot 子进程中打开保存的项目；PowerShell 的工作目录
本身不会被子进程改变。也可以直接在会话中输入 `/model` 列出模型、`/model deepseek-main`
切换模型，或 `/model add qwen-main qwen qwen-plus` 新建并立即启用一个 Qwen 配置。

可用的管理命令如下：

```powershell
repopilot auth status                  # 仅显示凭据是否存在，绝不输出 Key
repopilot doctor                       # 离线检查 profile、凭据可用性、信任和 MCP 配置
repopilot doctor --fix-plan            # 只生成 launcher 恢复建议；不会执行、改 PATH 或写 $PROFILE
repopilot auth test deepseek           # 明确发起一次小型付费请求，验证保存的 Key
repopilot auth probe deepseek          # 只发送固定公开标记，诊断 DeepSeek SSE；不读取项目文件
repopilot config show                  # 查看无密钥的模型配置和项目别名
repopilot config default qwen-main     # 修改以后新会话的默认模型
repopilot project list
repopilot project open repopilot
repopilot auth logout deepseek          # 只删除 DeepSeek 的系统凭据
repopilot shell-doctor powershell        # 离线诊断 Anaconda/PATH launcher 冲突；不写 PATH 或 $PROFILE
```

若 `repopilot` 意外从 Anaconda 或旧虚拟环境启动，先运行 `repopilot doctor`。它会显示实际
launcher、当前 Python、导入的包位置与 PATH 上的 `repopilot` 是否一致，但**绝不**打印 Key、
执行修复、删除文件或联网。若报告存在旧的 `~epopilot-*.dist-info`，先退出所有 RepoPilot
会话，再在目标虚拟环境中显式运行 `python -m pip install -e .`；不要在会话仍打开时升级。

```powershell
# 交互会话
repopilot --trust --model qwen2.5:7b

# 显式添加一个仅供读取/搜索的相邻目录；不能在该目录写文件、运行命令或执行 Git/LSP。
repopilot --trust --add-dir ..\shared-library

# 终端开发体验：默认仅流式显示模型文字；输入行尾的 \ 可继续下一行，Ctrl+C 取消当前 turn
repopilot --trust --model qwen2.5:7b

# --verbose 用于诊断，额外显示工具生命周期；--no-stream 恢复整段完成后再显示的旧行为
repopilot --trust --verbose --model qwen2.5:7b
repopilot --trust --no-stream --model qwen2.5:7b

# 单次请求；非交互模式必须显式信任
repopilot -p "解释这个项目" --trust --model qwen2.5:7b

# 面向 IDE 或脚本的单次请求：stdout 仅输出 JSONL 事件与最终结果
repopilot --trust --output-format jsonl -p "检查当前测试失败原因"

# 一次性的显式 DeepSeek 覆盖；环境变量会优先于凭据管理器中的 Key
$env:DEEPSEEK_API_KEY = "<你的 DeepSeek API Key>"
repopilot --provider deepseek --model deepseek-v4-flash --trust -p "检查当前测试失败原因"

# 如果 Windows 控制台在逐项输入 y 后意外中断：只自动批准 run_tests
# 不会自动批准 Shell、写文件、Git 或其他敏感工具
repopilot --provider deepseek --model deepseek-v4-flash --trust --allow-tests -p "检查当前测试失败原因"

# 若 Windows 控制台发送意外 Ctrl+C：启动连续会话前同步运行默认测试
# 测试报告会自动给到第一条自然语言问题；测试运行阶段有 120 秒超时
# 本会话不暴露 run_tests 工具，以免再次触发 Windows 异步子进程问题
repopilot --provider deepseek --model deepseek-v4-flash --trust --preflight-tests

# 也可用于单次分析后退出
repopilot --provider deepseek --model deepseek-v4-flash --trust --preflight-tests -p "检查当前测试失败原因"

# 一次性的显式 Qwen / DashScope（中国站 OpenAI 兼容接口）覆盖
$env:DASHSCOPE_API_KEY = "<你的 Qwen API Key>"
repopilot --provider qwen --model qwen-plus --trust

# 继续当前项目最近一次会话；首次信任后会在 ~/.repopilot 中保留信任记录
repopilot --continue --model qwen2.5:7b

# 从已有会话分叉，避免污染原来的探索路径
repopilot --resume <session-id> --fork --model qwen2.5:7b

# 仅配置、不启动项目 MCP 服务；修改必须显式信任当前工作区。
# allow/deny 是服务内的可见工具策略，不能降低 MCP 工具仍为高风险、逐次审批的规则。
repopilot mcp list
repopilot --trust mcp add docs-server --allow-remote-tool search python -m my_mcp_server
# 只允许本机 loopback HTTP MCP，不会把仓库内容发送到远程服务。
repopilot --trust mcp add-http local-docs http://127.0.0.1:3000/mcp
repopilot --trust mcp policy docs-server --allow-remote-tool search --deny-remote-tool write
repopilot --trust mcp remove docs-server
repopilot mcp doctor                    # 仅检查 launcher/回环边界；不会启动或请求 MCP server
# 显式探测才会启动/联系一个命名 server；结束后立即关闭连接。
repopilot --trust mcp probe docs-server --timeout-seconds 20
repopilot mcp history docs-server        # 仅显示本地脱敏 probe 收据；不会联系 server

# 长任务由单独的 supervisor 终端拥有，关闭交互式 RepoPilot 不会取消它。
# submit 只入队；serve 只处理当前项目；二者均需显式 --trust，且没有 shell。
repopilot --trust supervisor serve
repopilot --trust supervisor submit --label "full tests" --timeout-seconds 900 -- .\.venv\Scripts\python.exe -m pytest -q
repopilot supervisor list
repopilot supervisor show <job-id>       # 离线查看超时/启动失败等任务收据
repopilot supervisor log <job-id>
repopilot --trust supervisor stop <job-id>
repopilot --trust supervisor retry <job-id>

# 配置已安装在本机的语言服务器；RepoPilot 不会下载或执行未配置的 LSP。
repopilot --trust lsp add python pyright-langserver --stdio
repopilot lsp list
```

### 交互会话命令与设置

- `/memory` 显示已加载的 `REPOPILOT.md`、`REPOPILOT.local.md`、`AGENTS.md`、兼容的
  `CLAUDE.md`/`CLAUDE.local.md` 以及 `.repopilot/rules/**/*.md`；这些内容始终是受限的
  不可信上下文。`@relative/path.md` 可在工作区内导入指令，最大深度为四层。
- `/init` 先显示项目 `REPOPILOT.md` 草案，只有输入 `y` 才创建；`/undo` 只能撤销当前打开的
  会话中最新的 RepoPilot 小型编辑，且文件在此后未被其他方式修改。
- `/model` 显示无密钥的 DeepSeek/Qwen 模型配置；`/model <profile>` 在不丢失当前 session
  的前提下切换 Provider 和模型，并将该 profile 保存为后续新会话的默认值。使用
  `/model add <name> <deepseek|qwen> <model>` 创建配置，`/model default <profile>` 仅修改
  默认值而不切换当前 session。
- `/sessions` 列出本项目保存的会话；`/resume <id或标题>` 恢复同一 provider/model；`/clear`
  开始新会话而保留显式项目记忆；`/rename <标题>` 和 `/history [数量]` 用于整理恢复点。
- `/evidence` 只在本机显示当前权限模式、配置的 allow/deny、最近验证状态和最近 turn 的变更清单；
  它不运行命令、不调用模型。新保存的 `/verify` 完整日志保留在本地脱敏收据中，不会自动转发到后续
  云端模型回合；`/repair` 只在确有失败验证时可用，且正常工具审批仍然有效。
- `/workflow` 将计划、最近变更快照、验证/修复状态和下一条安全操作汇总为一个**本地只读**视图；
  它不会调用模型或命令。计划的批准仍只是用户意图记录，绝不会自动授予编辑或执行权限。
- `/export [name.md]` 会先显示消息数、字节数与本地目标路径；只有输入 `y` 才会创建当前会话的
  Markdown/TXT 导出。导出文件始终位于该 session 的本地 RepoPilot 状态目录，不写入仓库、不覆盖已有
  文件；会再次脱敏常见凭据，并省略工具 payload 与 Provider 的隐藏 reasoning。
- `/memory` 同时显示用户级 `~/.repopilot/REPOPILOT.md`、仓库根到当前子目录的规则，以及
  通过 `/remember <事实>` 保存的明确项目事实。用户级规则先加载，项目与子目录规则后加载。
- `/skills` 会显示用户级 `~/.repopilot/skills/` 与项目级 `.repopilot/skills/`；同名时项目级
  技能覆盖用户级技能。`/hooks` 仅查看可审计的声明式 hooks，不能执行项目脚本。
- `/mcp` 可查看启用的 MCP server、工具、资源和 prompts；`/mcp resource <server> <uri>` 与
  `/mcp prompt <server> <name> [参数=值]` 仅在以 `--mcp` 启动的可信会话中读取，不会自动注入模型。
- `/worktrees` 只列出当前项目内的 Git worktree。创建 worktree 仍须由 agent 提议并获得高风险审批；
  `/worktree <名称>` 可只读查看一个受管理 worktree 的 dirty 状态。RepoPilot 不提供自动删除
  worktree 的命令。
- `/plan draft <步骤>|<步骤>`、`/plan approve`、`/todo add <事项>`、`/todo start <编号>` 与
  `/todo done <编号>` 将计划和状态持久化在当前 session；批准计划只记录用户意图，**不会**放宽
  编辑、执行或高风险工具的审批。
- `/stats` 显示本地 token、工具、延迟和上下文统计；`/trace [数量]` 只显示可审计事件，默认不渲染
  消息正文或凭据。
- `/agents` 读取无工具后台分析的状态；`/agent <问题> | <证据>` 会先显示发送字节数与目标 Provider，
  只有输入 `y` 才会把该**显式提供的证据**发送给独立分析请求。`/stop-agent <id>` 可停止一个明确任务。
- `/tasks` 显示当前 session 启动的后台任务，`/task-log <id>` 仅在当前 CLI 仍拥有该子进程时读取
  有界输出，`/stop <id>` 会再次确认后终止一个明确任务。关闭 CLI 后，原子进程会被终止；保留的
  收据只显示 `unavailable`，绝不会自动重放。`/retry <id>` 仅可重试未被脱敏的原 argv，并且仍需
  再次确认。
- 编辑审批时会显示安全截断的 unified diff。回答 `s` 允许同一工具在当前 session 内继续执行；
  `y` 只允许这一次。
- 交互文本模式默认启用 `--stream`：支持 OpenAI 兼容 SSE 的本地、DeepSeek 与 Qwen Provider
  会逐段显示**模型文字**，工具生命周期保持静默。`--verbose` 额外显示诊断事件；`--no-stream`
  可恢复整轮完成后再显示最终文字的旧行为。
- 交互 TTY 自动启用持久的输入历史：可用 Up/Down 浏览此前输入、`Ctrl+R` 搜索历史，并在输入
  `/` 后使用 Tab 补全 session 命令；输入 `./`、`src/` 等工作区相对路径时，Tab 只枚举对应父目录
  中、且仍位于工作区内的至多 100 个候选项。历史保存在当前项目的 RepoPilot 状态目录中，写入前会
  按凭据脱敏规则遮蔽常见 API Key、Bearer token、私钥与密码赋值。缺少终端能力或依赖时会自动回退到
  原有输入方式。
- `--add-dir <目录>` 会提供 `@extra1` 等只读别名。模型只能使用 `list_added_files`、
  `read_added_file` 和 `search_added_text` 检查其中的文本；符号链接逃逸、路径穿越、写入、Shell、
  Git、测试和 LSP 一律不允许。`/status` 会显示此次会话已授予的目录。
- 对发生真实文件改动或运行 `run_tests` 的 coding turn，最终回答后会由运行时附加 `Workflow evidence`：
  它只列出本轮快照确认的 changed files 与记录的测试命令/结果。没有执行测试时会明确显示
  `Not run in this turn`，不会把模型的自然语言声称当成验证证据。

持久化工具规则可写入用户全局 `~/.repopilot/settings.json`、不提交的项目本地
`.repopilot/settings.local.json`，或通过 `--allow-tool PATTERN` / `--deny-tool PATTERN` 临时指定：

```json
{
  "permissions": {
    "allow": ["run_tests", "git_status"],
    "deny": ["run_shell", "mcp__*"]
  }
}
```

项目内可提交的 `.repopilot/settings.json` **只能**添加 `deny`，其 `allow` 会被忽略；拒绝规则
总是优先，计划模式与秘密路径保护也不能被规则绕过。

会话默认保存在 `~/.repopilot/projects/<project-hash>/sessions/`，其中包含
append-only transcript、checkpoint、工具日志、初始文件基线与可选上下文摘要。当前
交互命令包括 `/context`、`/compact`、`/diff [turn]`、`/rewind [list|turn [all|code|session]]`
和 `/verify [list|last|all|test|lint|typecheck|build|command]`。
`/diff` 只读取当前 session 的追加式 turn 文本差异；`/rewind` 会先显示历史 revision 并在
确认后恢复可验证的文本替换及可选会话 checkpoint，绝不自动删除文件，且若文件曾被手动修改便会拒绝覆盖。
`/verify list` 仅发现项目验证项，`/verify last` 仅查看当前会话最近一次持久化报告；其余
选择器才会执行对应类别，且均会经过权限审批。Shell 仅接受 tokenized argv，不经由 shell
解释器；验证失败后可使用
`/repair` 将结构化失败证据带入下一轮受治理修复。Git 已提供只读的
`status`、`log`、`show` 工具。会话还支持受审批的后台命令启动、状态读取和显式终止，
以及受策略约束的 Git 暂存、提交和创建 Worktree；自动修复编排仍在后续阶段。
单次请求可使用 `--output-format jsonl` 获取稳定的 `event` 与 `result` JSONL 记录；
退出交互 session 时，RepoPilot 会终止该 runtime 自己启动的后台进程，且 Ctrl+C 会将
当前 turn 持久化为可审计的 `cancelled` 状态，而不会自动重放未执行的工具调用。

`/review` 与 `/security-review` 使用当前未提交的 `git diff --no-ext-diff` 作为唯一代码证据，
并在把 diff 发送给所选模型前要求一次明确确认。它们没有文件、shell、Git 写入或提交权限；
安全审查只要求报告有具体利用路径的新增问题。空 diff 不会触发模型调用。

DeepSeek 是一个单独的、显式选择的云端 Provider：只允许官方
`https://api.deepseek.com/chat/completions` endpoint，并只接受
`deepseek-v4-flash`、`deepseek-v4-pro`。Key 优先从 `DEEPSEEK_API_KEY`（或
`--api-key-env` 指定的环境变量名）读取；没有环境变量时，已通过 `repopilot auth login
deepseek` 保存的 Windows 凭据管理器条目会被使用。缺少 Key 时会在本地失败。不要将 Key 写入
仓库、项目规则、命令历史或 session 文件；云端模型会接收到模型上下文中的代码、项目
指令和工具结果，因此请先在不含敏感信息的副本上测试。

Qwen 也可作为显式云端 Provider 使用，固定请求中国区 DashScope 的 OpenAI 兼容
`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` endpoint，并从
`DASHSCOPE_API_KEY`（或 `--api-key-env` 指定的环境变量）读取 Key；没有环境变量时会使用
`repopilot auth login qwen` 保存的 Windows 凭据管理器条目。可通过
`--provider qwen --model qwen-plus` 启动；模型名保持可配置，以适配已在 DashScope
账号中开通的 Qwen 模型。官方接口说明见
[QwenCloud OpenAI compatibility](https://docs.qwencloud.com/api-reference/toolkitframework/openai-compatible/overview)。

项目根目录的 `REPOPILOT.md`、`REPOPILOT.local.md`、`AGENTS.md`、兼容的 `CLAUDE.md`
与 `.repopilot/rules/**/*.md` 会被以大小限制加载为**不可信上下文**；它们不能解除路径保护、
权限审批或 Shell 限制。

高级扩展采用显式启用与最小权限：

- `.repopilot/skills/<name>/SKILL.md` 可用 `/skills` 查看、`/skill <name>` 持久化启用；
- `.repopilot/hooks.json` 只支持声明式提示，不能执行项目脚本；
- `repopilot --mcp` 才会启动 `.repopilot/mcp.json` 中的 stdio MCP 服务，所有远端工具默认
  以高风险工具经过审批；`mcp add-http` 也仅接受 `127.0.0.1`、`localhost` 或 `::1` 的无凭据 HTTP
  端点。资源与 prompts 可显式浏览；远程 HTTP/OAuth MCP transport 未实现；
- `lsp add` 只登记用户已经安装的本机语言服务器；`lsp_diagnostics` 和 `lsp_definition` 每次调用
  都需要高风险审批，不会下载服务器或访问网络；
- `web_fetch` 仅能访问公开 HTTPS 文本页面：目的地显示后需单次批准，拒绝本机/私网 DNS 结果、凭据、
  重定向与大于 1 MB 的响应；
- `.repopilot/plugins/<name>/plugin.json` 和 `~/.repopilot/plugins/<name>/plugin.json` 只可贡献
  `skills/` 中的本地文本流程，不可执行代码、注册 MCP 或获得新权限；
- `start_subagent`、`subagent_status` 与交互 `/agent` 都是没有工具权限的隔离只读分析；
- `git_stage`、`git_commit` 与 `create_worktree` 分别采用编辑或高风险审批，工作树仅创建于
  `.repopilot/worktrees/`，不提供自动删除。

端点、SSE 续传和安全边界见 [`docs/api.md`](docs/api.md)。

## 目录导航

- `src/repopilot/core/`：Agent Loop、状态、预算和共享契约；
- `providers/`、`tools/`、`context/`：决策和环境接口；
- `retrieval/`：BM25、Dense、Hybrid (RRF)、Reranker、FirstChunk 消融；
- `memory/`、`skills/`、`mcp/`、`orchestration/`：Agent 扩展机制 + Multi-Agent harness；
- `runtime/`、`security/`、`verification/`：可靠性、安全边界 + patch reviewer；
- `benchmarks/`：三领域 contracts、adapters、executors、runner；
- `evaluation/`：指标、报告、manifest、实验矩阵；
- `docs/lessons/`：Stage 1–6 教学讲义与术语表；
- `docs/experiments/`：实验报告（P5–P12）；
- `artifacts/`：本地 Trace、Checkpoint、实验结果，不提交版本库。

## 离线健康检查与评测汇总

`doctor` 不会调用模型，也不会显示密钥，适合在报告问题前先执行：

```powershell
repopilot doctor
repopilot --output-format jsonl doctor
```

公开 trial 记录可用本地 JSONL 汇总，避免把隐藏 evaluator 数据或 API Key 写进评测文件：

```powershell
repopilot eval schema
repopilot eval summary .\evaluation-records.jsonl

# 对已完成的公开 SWE 预测和 evaluator JSON 作离线失败归类；不会运行 Docker 或读取 gold test patch。
repopilot eval diagnose-swe .\predictions.jsonl .\resolved_results.json

# 只汇总独立开发集的“定位→补丁→应用→验证”记录；不运行模型、测试或 patch，也不输出 patch 文本。
repopilot eval coding-funnel .\coding-development.jsonl

# 显示内置的纯合成开发 fixture；这个命令离线，不调用模型。
repopilot eval coding-dev plan

# 只有明确选择云端 provider、信任和 fixture 写入授权后才会实际消耗 API 配额。
# 它只写入新的 artifacts/coding_development/... 工作区，绝不修改当前项目。
repopilot --provider deepseek --trust eval coding-dev run --allow-fixture-edits --run-id dev_20260823

# 对已完成的合成开发运行做纯本地检查：输出 redacted funnel、可比性 receipt 与按失败类别生成的改进队列。
# 此命令不会构造 provider、运行测试或读取原始 trace。
repopilot eval coding-dev report .\artifacts\coding_development\runs\dev_20260823

# 对两次已完成的合成运行作纯本地对比。只有 suite digest、case set、provider、model 和
# execution_profile 都相同，输出才会标记为 controlled；否则明确标为 directional_only。
repopilot eval coding-dev compare .\artifacts\coding_development\runs\baseline .\artifacts\coding_development\runs\candidate

# 验证一个新 receipt 是否绑定到未被改动的 redacted outcomes JSONL；旧 receipt 会明确显示 legacy_unbound。
repopilot eval coding-dev verify-receipt .\artifacts\coding_development\runs\dev_20260823

# 汇总多个本地开发 receipt；它永不声称是 holdout 或能力分数。
repopilot eval coding-dev dashboard .\artifacts\coding_development\runs\baseline .\artifacts\coding_development\runs\candidate

# 只显示外部官方 evaluator holdout 的非敏感元数据；不读取 task、gold patch 或隐藏测试。
repopilot eval coding-dev holdout-plan

# P37：一次无工具、只读的 DeepSeek 验收。必须明确列出每个外传文件；保护目录、凭据、会话与 artifacts 被拒绝。
repopilot --provider deepseek --trust acceptance --send-project-files `
  --include src/repopilot/runtime/cancellation.py `
  --include src/repopilot/runtime/runner.py `
  --prompt "审查取消可靠性；不要提出修改。"

# 离线检查验收的范围/哈希；不会再次读取源文件、调用模型或回显模型 assessment。
repopilot acceptance verify .\artifacts\acceptance\p40_streaming_readonly.receipt.json
repopilot acceptance show .\artifacts\acceptance\p40_streaming_readonly.intent.json
```

Windows 发布证据命令均为显式 opt-in：

```powershell
# 只生成审阅清单，不会 stage、commit、push 或删除文件。
.\scripts\release_scope_windows.ps1

# 写入 Docker/Git remote/CI workflow 前置条件报告；ReportOnly 不会因缺少 remote 而失败。
.\scripts\ci_preflight_windows.ps1 -ReportOnly

# Docker 已正常启动后执行真正的容器边界测试。
.\scripts\docker_security_gate_windows.ps1 -BuildImage

# 仅固定无工具标记 prompt 的 DeepSeek 流式验收，随后运行本地恢复/权限测试。
.\scripts\interactive_acceptance_windows.ps1

# 校验保留的本地 wheel 身份，不联网也不安装。
.\scripts\verify_package_smoke_receipt_windows.ps1 -Receipt .\artifacts\package_smoke\<run>\package-smoke.receipt.json
```

开发漏斗的严格 JSONL 格式和 holdout 隔离规则见
[`docs/evaluation/coding_development_protocol.md`](docs/evaluation/coding_development_protocol.md)。

## 声明边界

- MCP 支持显式启用的 stdio 与 loopback HTTP transport；`mcp probe` 只探测命名服务，并在
  `~/.repopilot/mcp-probes/` 写入脱敏本地收据；`mcp history` 从不连接服务。服务级 allow/deny
  只缩小可见工具集合，所有 MCP 工具仍是高风险。远程 HTTP、OAuth 与任意第三方协议兼容仍不在
  承诺范围；
- `supervisor` 是用户显式启动的本地、前台、项目范围 argv 服务，不是 Docker sandbox，也不托管模型
  或 API Key。其重启会把旧的 running job 标记为 `interrupted`，绝不自动重放；项目排队数上限为
  20，每个任务的 0.1–3600 秒超时、标签和失败原因都保存在本地收据中；
- 10 题检索消融只用于方法诊断，正式 FRAMES 指标使用 60 题、3 次重复；
- 旧 SWE 三题已被开发污染；正式负结果使用 5 个未污染任务，当前 resolved 为 0/5；
- P10 已完成 QLoRA SFT + DPO，并在匹配的 Qwen2.5-1.5B 上完成一次下游消融：base
  为 9/35，SFT+DPO adapter 为 0/35。该结果表明当前微调退化，不改变 7B 主指标；
- `orchestration/MultiAgentHarness` 已能以 Planner → AgentRuntime/Verifier → Reviewer
  执行真实任务，并支持持久化状态、进程中断恢复、取消、超时和有界返工；新增的
  fixed/rule/model/hybrid 路由支持只读模型决策、级联降级、运行时 binding 与 checkpoint
  审计；ASGI 服务层支持 Bearer 鉴权的 JSON 提交/状态/取消、SSE 事件流与断线续传、
  限流、请求上限、超时和硬并发限制；真实 Runtime/Verifier 1/4/8 已完成 360/360，
  其模型为 ScriptedProvider，因此只代表调度容量；另一个固定 digest 的真实
  Qwen2.5-7B/Q4_K_M 服务实验完成 1/4/8 并发各 24 请求（72/72 非空），吞吐为
  5.52/16.80/18.62 req/s，P95 为 4.16/1.38/1.24s。该短请求结果仍不代表端到端 coding-agent 吞吐；
- 三领域分开报告，不给综合总分。

## 复现实验

```powershell
# FRAMES B1 BM25
python scripts/run_frames_oracle_smoke.py --run-id NEW_ID --count 10 `
  --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e `
  --planned --reasoned --reviewed --reasoning-output-tokens 512

# FRAMES R2 Hybrid
python scripts/run_frames_oracle_smoke.py --run-id NEW_ID --count 10 `
  --model qwen2.5:7b --model-revision 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e `
  --planned --reasoned --reviewed --reasoning-output-tokens 512 --retriever-type hybrid
```

以上命令复现的是 10 题历史开发消融。正式 FRAMES 60 题证据与全部 run 哈希见
`docs/evidence/artifact_index.json`；固定条件为 temperature=0、seed=7、被测 Agent
network=deny。
