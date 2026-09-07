# RepoPilot v1.2.0

本版完成 Claude Code 风格路线图的 P0–P3：先提升 Agent 对模型输出的可验证性，再增加可持久化的
工作流、受控并行分析，以及本机优先的扩展接口。它仍是学习与评测用途的本地优先 CLI，而不是
Claude Code 的实现或替代品。

## P0：Provider 与工具证据

- Provider 声明原生工具调用、强制选工具、流式和 reasoning continuation 等适配能力；DeepSeek、
  Qwen 与本地 OpenAI-compatible Provider 均以该契约接入。
- 当用户明确要求一个已注册工具（例如“必须调用 `run_tests`”）时，运行时把要求传给支持
  `tool_choice` 的 Provider；即使 Provider 忽略它，也会追加一次可信运行时纠正。第二次仍未调用时，
  会停止而非接受“已执行”的无证据回答。
- 工具观察中超过 24 KB 的文本保存到 session 的 `tool-outputs/`，模型上下文仅保留摘要和可审计路径。
- `/stats` 与 `/trace` 显示本地会话统计和非消息事件；正常流式会话保持只输出模型文字。

## P1：计划、待办与本机 LSP

- `/plan` 和 `/todo` 将最多 20 项的计划持久化在 session，且在上下文压缩后仍保留。计划审批是记录，
  不是权限提升。
- `write_todos` 允许模型维护结构化待办，但它只有 session 写入能力，无法写项目文件或执行命令。
- `repopilot --trust lsp add <语言> <命令...>` 只能登记用户已经安装的本机语言服务器；
  `lsp_diagnostics` / `lsp_definition` 使用受审批的一次性 stdio 会话。

## P2：受控并行与 Worktree 观察

- `start_subagent` 和交互 `/agent` 最多同时运行两个只读后台分析器。它们没有 workspace、shell、
  MCP 或编辑能力；发送的证据会在控制台逐次披露并请求确认。
- 会话切换 Provider 前会检查后台分析是否结束，避免把在途任务误认为新模型的结果。
- `/worktree <名称>` 只读显示 RepoPilot 管理的 worktree 状态。仍然没有自动删除 worktree 的功能。

## P3：受限扩展与资料读取

- 支持 MCP 的 stdio 及 Streamable HTTP 基础 JSON-RPC 交互。HTTP 配置强制为无凭据的本机回环端点，
  不发送 Cookie、不处理 OAuth，也不允许远程 URL。该运输方式遵循 MCP 对 stdio / Streamable HTTP 与
  `MCP-Session-Id` 会话头的公开规范。
- 插件为 manifest-only：项目或用户插件只能追加 `skills/` 下的文本流程，不能运行钩子、代码或 MCP。
- `web_fetch` 对公开 HTTPS 文本使用单次目标审批、DNS 公网校验、固定 IP 建连、禁止重定向与大小限制。
- 新增有边界的 `read_notebook`、`read_pdf`、`read_image_metadata`，分别读取 Notebook 源码、最多 20 页
  PDF 文本和 PNG/JPEG 元数据；不会执行 Notebook，也不会把图片像素交给纯文本模型。

## 有意未实现的范围

- 任意远程 HTTP MCP、OAuth、自动插件执行和自动 Worktree 删除。
- 需要用户选择供应商、身份授权或允许仓库内容离开本机的 Web 搜索服务。
- 并行子 Agent 的自动文件修改或自动合并。

这些边界使功能可学习、可审计，同时避免将一个本地 CLI 的扩展入口变成无提示的数据外发通道。
