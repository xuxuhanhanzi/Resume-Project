# Stage 2 完整讲义：本地模型、结构化输出与 Coding Tools

> 本阶段目标不是证明某个 Qwen 模型能力强，而是学会把任意本地模型安全接入 Runtime。  
> 建议学习时间：10–14 小时。  
> 当前边界：Adapter 已实现并通过解析测试；本机尚无运行中的兼容模型服务，不能声称真实 Qwen 已评测。

## 0. 学习目标

完成本阶段后，你应该能：

1. 区分模型文件、推理引擎、模型服务和 ModelProvider；
2. 解释 HTTP、loopback、JSON 和 OpenAI-compatible；
3. 解释 Native Tool Calling 与 JSON Action fallback；
4. 读懂 ToolSpec 的 JSON Schema、权限、超时和副作用字段；
5. 说明 RepoPilot 的 8 个 Coding Tools 如何限制模型；
6. 手工追踪一次精确 Patch 和一次测试执行；
7. 解释为什么默认工具集中没有任意 Shell。

## 1. “把 Qwen 下载到本地”实际包含四层

新手经常把以下四件事都叫做“模型”：

```text
模型权重
  ↓ 由推理引擎加载
Ollama / llama.cpp / vLLM / Transformers
  ↓ 对外提供接口
本地模型服务 http://127.0.0.1:<port>
  ↓ 被 RepoPilot 调用
LocalOpenAICompatibleProvider
```

### 1.1 模型权重

训练得到的参数文件，例如 safetensors 或 GGUF。Base、Instruct、Coder 是不同用途：

- Base：主要学习续写，不一定遵守指令；
- Instruct：经过指令微调，更适合对话和结构化任务；
- Coder：代码数据和代码任务能力更强；
- 量化模型：用较低位宽保存参数，降低显存，可能损失部分能力。

8GB 显存不能只看“参数量”。还要考虑量化位宽、KV Cache、Context 长度、推理引擎开销和是否把部分
层放到 CPU。正式模型选择必须实测，不能只凭名字。

### 1.2 推理引擎

引擎负责：加载权重、分词、矩阵计算、KV Cache、采样和显存管理。RepoPilot 不自己实现这些重型
能力，而让本地服务负责。

### 1.3 模型服务

服务是一个长期运行的进程。它监听本机端口，收到 HTTP 请求后调用模型，再返回 JSON。

### 1.4 ModelProvider

Provider 把不同服务的请求和响应转换为 RepoPilot 内部统一的 `ModelRequest/ModelResponse`。
Runtime 因此不需要知道模型来自 Ollama、llama.cpp 还是 vLLM。

## 2. HTTP 和 loopback 的新手解释

假设模型服务监听：

```text
http://127.0.0.1:11434
```

- `http`：通信协议；
- `127.0.0.1`：loopback，只指当前电脑；
- `11434`：端口，可理解为同一台电脑上某个服务的“窗口编号”；
- `/v1/chat/completions`：服务中的具体 API 路径。

[`LocalProviderConfig`](../../src/repopilot/providers/local_openai.py) 会拒绝非 loopback URL。这个选择表达：
本类是“本地模型 Provider”，不是通用云 API Provider。它不是完整网络安全策略，但能阻止配置误用。

## 3. OpenAI-compatible 不等于使用 OpenAI 模型

“OpenAI-compatible”通常表示请求形状相似：

```json
{
  "model": "LOCAL_MODEL_NAME",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools": [],
  "temperature": 0.0,
  "max_tokens": 1024
}
```

本地 Qwen 权重仍在你的机器上推理。这里只是接口形状兼容，方便不同后端复用同一个 Adapter。
不同后端不一定支持全部字段，因此正式运行前要做能力探测。

## 4. Provider 怎样发送请求

打开 [`providers/local_openai.py`](../../src/repopilot/providers/local_openai.py)。

### 4.1 为什么使用 `asyncio.to_thread`

标准库 `urlopen()` 是同步阻塞函数。Provider 的公共接口是异步的，所以：

```python
return await asyncio.to_thread(self._complete_sync, request)
```

表示把同步 HTTP 工作放到线程，避免阻塞整个 async 事件循环。它不是增加模型推理速度，只改善
Runtime 的等待组合方式。

### 4.2 请求头和 API Key

本地服务通常不需要 Key；若配置 `api_key_env`，代码只从环境变量读取。Trace 不应记录 Key 值。
真正不可信的测试容器也不会挂载该环境变量。

### 4.3 错误分类

HTTP 408、409、429、500、502、503、504 被视为可能恢复：

- 408：请求超时；
- 429：请求过多；
- 5xx：服务端暂时故障。

其他协议错误或无效 JSON 通常不能靠原样重试解决。

## 5. 两种结构化动作来源

### 5.1 Native Tool Calling

支持工具调用的服务可能返回：

```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"calculator.py\"}"
      }
    }
  ]
}
```

Provider 解析函数名和参数，归一为：

```python
ToolCall("call_123", "read_file", {"path": "calculator.py"})
```

### 5.2 JSON AgentAction fallback

部分本地后端不稳定支持 Native Tool Calling，但能输出普通 JSON：

```json
{"type":"tool_call","tool":"read_file","arguments":{"path":"calculator.py"}}
```

Provider 尝试解析它。这样 Runtime 的内部契约不变。

### 结构化输出仍会错

模型可能输出：

- 少一个右括号；
- 工具名不存在；
- `arguments` 是字符串而不是对象；
- 参数字段拼错；
- JSON 外包裹 Markdown code fence。

因此“要求 JSON”不等于“永远得到合法 JSON”。Provider、Schema 和 Tool 本身必须继续校验。

## 6. ToolSpec：工具给模型看的说明书

Tool 基础契约在 [`tools/base.py`](../../src/repopilot/tools/base.py)，ToolSpec 在
[`core/contracts.py`](../../src/repopilot/core/contracts.py)。

一个 ReadFile ToolSpec 大致包含：

```text
name             read_file
description      读取一个有界 UTF-8 行范围
input_schema     path/start_line/end_line 的 JSON Schema
permission       READ
timeout_seconds  10
read_only        true
idempotent       true
```

### JSON Schema 的最小阅读方法

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string"},
    "start_line": {"type": "integer", "minimum": 1}
  },
  "required": ["path"],
  "additionalProperties": false
}
```

翻译成人话：

- 参数整体必须是对象；
- `path` 必须是字符串且必填；
- `start_line` 若提供，必须是至少 1 的整数；
- 不接受没有声明的额外字段。

当前 RepoPilot 把 Schema 发给模型，同时 Tool 内部再次手工校验关键边界。生产化可增加统一 JSON
Schema validator，避免每个 Tool 重复部分工作。

## 7. ToolRegistry：名称到真实能力的映射

Registry 负责：

1. 注册 Tool；
2. 拒绝重名；
3. 按稳定顺序提供 ToolSpec；
4. 根据模型给出的名字找到 Tool 对象；
5. 未知名字返回错误，而不是动态执行同名 Python 函数。

这是一条重要安全边界：模型只能调用 Allowlist 中已经注册的能力。

## 8. 八个 Coding Tools 逐个理解

实现位于 [`tools/coding.py`](../../src/repopilot/tools/coding.py)。

### 8.1 `list_files`

输入：安全相对路径、最大深度。输出：文件路径列表。

限制：

- 忽略 `.git`、`.venv`、缓存目录；
- 最多返回 500 个文件；
- 每个候选文件再次检查 Task 的 allowed/forbidden boundary；
- 不会因为根目录允许就泄露其中 forbidden 测试文件。

### 8.2 `read_file`

输入：path、start_line、end_line。输出：带行号文本。

限制：

- UTF-8 文本；
- 单文件最大 1MB；
- 单次最多 501 行；
- 路径必须存在并位于可见范围。

行范围读取是 Context Engineering 的第一步：模型通常不需要一次看到整个大文件。

### 8.3 `search_text`

使用正则表达式在可见文件中搜索。输出 path、line、text。

限制：pattern 长度、结果数、文件大小和路径权限。正则不是语义检索，但对函数名、错误字符串、
配置键非常有效。

### 8.4 `find_symbol`

使用 Python 标准库 `ast.parse()` 查找：

- 函数定义；
- 类定义；
- 赋值和带注解赋值。

若源码存在 SyntaxError，该文件会被跳过，而不是让整个 Tool 崩溃。

### 8.5 `apply_patch`

它没有开放任意脚本，而是精确替换：

```text
path
old_text
new_text
expected_replacements
```

流程：

1. 解析安全路径；
2. 读取原始内容；
3. 计算 old_text 出现次数；
4. 次数不等于预期则返回 Conflict，不修改；
5. 写入新内容；
6. 检查 changed-file 预算；
7. 后续异常则尝试恢复原内容；
8. 返回 path 和替换次数。

这叫“事务性思维”：不能让 Tool 报告失败，却偷偷留下未声明的副作用。

### 8.6 `run_tests`

模型不能提供命令字符串。命令来自 PublicTaskSpec：

```yaml
test_command: [python, verify.py]
```

为什么用列表而不是：

```yaml
test_command: "python verify.py && upload_secret"
```

列表会直接传给 `subprocess.run(..., shell=False)`，不会让 Shell 解释 `&&`、管道或命令替换。

### 8.7 `git_diff`

当前实现使用 baseline 文本快照和 `difflib.unified_diff`，并不依赖工作区必须是 Git 仓库。Forbidden
文件的内容不会进入模型可见 Diff。

### 8.8 `inspect_failure`

返回最近失败的 ToolResult，帮助 Agent 不要只重复同一动作。它是“把失败变成可查询状态”的例子。

## 9. 路径安全：为什么 `../` 很危险

模型可能提出：

```text
../../.ssh/id_rsa
C:\Users\name\.env
allowed_dir/link_to_outside/secret
```

[`security/paths.py`](../../src/repopilot/security/paths.py) 依次检查：

1. 不能是绝对路径；
2. 不能包含 `..`；
3. 对真实路径做 resolve，解析符号链接；
4. 必须仍位于 workspace；
5. 必须符合 Task allowed_paths；
6. 不能落入 forbidden_paths。

不能只检查字符串是否以 workspace 开头。例如：

```text
C:\workspace_evil
```

字符串以 `C:\workspace` 开头，却不是该目录的子路径。正确做法是路径规范化后用 Path 的层级关系检查。

## 10. 为什么没有默认 `run_command`

任意 Shell 可以：

- 删除或覆盖文件；
- 读取环境变量和用户目录；
- 下载第二阶段恶意程序；
- 创建无限子进程；
- 绕过每个窄 Tool 的 Schema。

Coding Agent 不是完全不需要 Shell，而是 P0 应先证明窄工具足够完成目标。以后若加入 `run_command`，
至少需要 executable Allowlist、tokenized arguments、cwd 检查、timeout、sandbox 和审批。

## 11. 本地模型正式选择流程

当前仓库只提供 [`configs/model/local_qwen.toml`](../../configs/model/local_qwen.toml) 占位配置，不锁死
具体 checkpoint。正确选择顺序：

1. 选择适合 8GB 显存的 Qwen Instruct/Coder 候选和量化；
2. 记录模型来源、revision/hash、许可证和文件 hash；
3. 选择一个本地服务后端并记录版本；
4. 检查 `/v1/chat/completions` 是否兼容；
5. 跑“合法 JSON Action”小测试；
6. 跑单次 read/search；
7. 跑一个微型 Bug Fix；
8. 再跑冻结的 10 任务，而不是边看答案边调 Prompt。

只有第 8 步完成后，才能报告 Resolve Rate。

## 12. 动手实验

### 实验 A：Provider 解析测试

```powershell
python -m pytest -q tests\unit\test_task_and_provider.py -vv
```

观察 Native ToolCall 与 JSON fallback 最后都变成同一个内部 ToolCall。

### 实验 B：工具闭环

```powershell
python -m pytest -q tests\unit\test_tools.py::test_patch_diff_and_test_closed_loop -vv
```

按顺序读测试：创建错误文件 → 快照 → Patch → Test → Diff。

### 实验 C：路径越界

```powershell
python -m pytest -q tests\safety\test_security_boundaries.py::test_path_traversal_and_absolute_paths_are_rejected -vv
```

解释为什么相对路径和绝对路径都可能危险。

### 实验 D：准备本地服务，但暂不下载模型

先阅读后端官方文档，确认它能提供 OpenAI-compatible endpoint。启动后只做健康检查，再填写：

```toml
[provider]
base_url = "http://127.0.0.1:PORT"
model = "YOUR_REGISTERED_MODEL"
```

不要把示例占位名称当作可直接下载的精确模型 ID。

## 13. 常见误解

### “本地模型一定不会泄密”

不完整。本地权重降低把 Prompt 发给第三方的风险，但 Agent Tool 仍可能读取秘密或恶意仓库仍可能
外连。模型位置与 Tool/Sandbox 安全是两件事。

### “JSON Schema 保证模型做正确的事”

错误。Schema 只保证形状，例如 path 是字符串；它不能证明这个文件与 Bug 有关，也不能证明 Patch
语义正确。

### “测试通过就允许 Agent 修改测试”

错误。Agent 若能弱化断言、删除测试或更改验证脚本，测试通过没有意义。因此 verify.py 被放入
forbidden_paths，正式 hidden tests 更不能进入 Agent 工作区。

### “量化只减少体积，不影响能力”

错误。量化会改变数值表示，可能影响格式遵循、长上下文和代码推理，必须在固定任务上实测。

## 14. 思考题与答案

<details>
<summary>问题 1：为什么 Tool 既有 JSON Schema，又在 Python 中检查参数？</summary>

发给模型的 Schema 主要帮助生成并支持边界校验，但不同本地后端可能不执行严格 Schema。Tool 是
最后执行边界，必须自己拒绝危险或无效输入。两层属于纵深防御。
</details>

<details>
<summary>问题 2：读取文件是幂等的吗？运行测试呢？</summary>

在环境不变时读取通常幂等。测试理论上只检查，但不可信测试可能创建文件、写缓存、启动进程，所以
RepoPilot 将 run_tests 视为有副作用且非幂等，不能盲目自动重试。
</details>

<details>
<summary>问题 3：为什么 Patch 要检查 expected_replacements？</summary>

若 old_text 出现 10 次而模型以为只有 1 次，全局替换会修改无关代码。明确预期次数可以把含糊编辑
转成 Conflict，让 Agent 重新读取和定位。
</details>

<details>
<summary>问题 4：OpenAI-compatible 是否意味着所有后端行为相同？</summary>

不意味着。字段支持、工具调用格式、Token 统计、错误码、模型模板和并行调用都可能不同。兼容只
降低接口差异，不能替代每个后端的资格测试。
</details>

## 15. Stage 2 验收清单

- [ ] 画出权重 → 引擎 → 服务 → Provider；
- [ ] 解释 loopback URL 每一部分；
- [ ] 把 Native ToolCall 手工转换为内部 ToolCall；
- [ ] 读懂一个 JSON Schema；
- [ ] 说出 8 个工具及其边界；
- [ ] 解释 Patch 的事务性回滚；
- [ ] 解释 `shell=False` 与 token list；
- [ ] 运行工具、Provider 和路径安全测试；
- [ ] 明确当前“Adapter 已实现”不等于“真实 Qwen 已评测”。
