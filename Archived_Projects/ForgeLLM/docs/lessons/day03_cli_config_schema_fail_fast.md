# Day 3 完整讲义：CLI、配置 Schema 与 fail-fast

> 学习模式：完整讲义
>
> 适用项目：ForgeLLM
>
> 验证日期：2026-07-22（Asia/Singapore）
>
> 学习目标：不仅知道名词，还能沿真实代码追踪参数、预测错误、解释校验顺序，并判断一次验证能证明什么。

## 0. 材料审计与当前证据

本讲义基于以下真实文件：

- [`src/forgellm/cli.py`](../../src/forgellm/cli.py)
- [`src/forgellm/config.py`](../../src/forgellm/config.py)
- [`src/forgellm/data/config.py`](../../src/forgellm/data/config.py)
- [`src/forgellm/runtime.py`](../../src/forgellm/runtime.py)
- [`src/forgellm/data/pipeline.py`](../../src/forgellm/data/pipeline.py)
- [`configs/runtime/smoke.toml`](../../configs/runtime/smoke.toml)
- [`configs/data/smoke.toml`](../../configs/data/smoke.toml)
- [`tests/unit/test_config.py`](../../tests/unit/test_config.py)
- [`tests/unit/test_data_config.py`](../../tests/unit/test_data_config.py)
- [`tests/integration/test_init_run_cli.py`](../../tests/integration/test_init_run_cli.py)
- [`tests/smoke/test_data_pipeline_cli.py`](../../tests/smoke/test_data_pipeline_cli.py)

本次实际验证结果：

- `RunConfig` 成功加载，稳定哈希为 `21c69700...a3730`；
- `DataConfig` 成功加载，稳定哈希为 `151dd20c...d0ffe`；
- 未知运行字段 `secret` 被拒绝；
- 数据切分比例总和不是 10000 时被拒绝；
- `test_bps = true` 被拒绝，不会被误当成整数 1；
- Day 3 相关配置与 CLI 测试 `15/15` 通过。

这些证据能证明当前 Windows/Python 环境中的已覆盖行为正确；不能证明所有错误输入都被覆盖，也不能替代 Linux CI、wheel 安装验证或正式数据审计。

## 1. 学习地图

Day 3 解决的核心问题是：**怎样让用户从命令行提交的配置，在昂贵任务开始前变成可信、明确、可复现的程序输入？**

```text
用户输入命令
    ↓
argparse 解析命令名和字符串参数
    ↓
Path 等基础 Python 对象
    ↓
tomllib 读取 TOML
    ↓
普通 Mapping[str, object]
    ↓
Schema 校验：结构、字段、类型、范围、跨字段约束
    ↓
frozen dataclass：RunConfig / DataConfig
    ↓
核心函数：initialize_run / run_data_pipeline
    ↓
结构化结果或明确错误码
```

需要掌握的六个核心概念：

1. CLI 是边界层，不是业务算法层；
2. TOML 是外部配置的文本表示；
3. Schema 是程序允许的配置契约；
4. resolved config 是程序实际接受并使用的配置；
5. frozen dataclass 是校验后的稳定数据对象；
6. fail-fast 是在副作用和昂贵计算前终止无效任务。

## 2. CLI 为什么是“边界层”

CLI 是 Command-Line Interface，即命令行界面。它位于“人类输入的字符串”和“程序内部 Python 对象”之间。

ForgeLLM 的 CLI 负责：

- 定义有哪些子命令；
- 解析 `--config`、`--input` 等参数；
- 把路径字符串初步转换为 `Path`；
- 选择应该执行的用例；
- 捕获预期的用户输入错误；
- 向 stdout 输出成功结果，向 stderr 输出错误；
- 返回适当的进程退出码。

它不负责：

- 在 CLI 函数中实现数据清洗算法；
- 在 CLI 函数中拼装全部运行元数据；
- 把所有业务校验都塞入 `argparse`；
- 隐藏无法处理的编程错误。

### 为什么要分层

如果全部逻辑都写在 CLI 中，测试就必须反复启动子进程、构造字符串并读取终端输出。现在核心入口 `main()` 接受 `Sequence[str]`，测试可以直接调用：

```python
result = main(["doctor"])
```

数据流水线也能直接接受已经校验的 `DataConfig`，而不依赖命令行。这使测试更快，核心函数也可以被 Notebook、服务端接口或其他 Python 模块复用。

### 类比及其边界

可以把 CLI 看成机场值机柜台：它检查输入格式、把旅客引导到正确流程，但不会亲自驾驶飞机。

这个类比不完全准确：CLI 不只是“转发”，它还承担错误展示和退出码协议；而真实系统中的权限、安全与资源调度可能还需要更独立的边界层。

## 3. TOML：外部文本如何变成 Python 数据

TOML 是一种面向配置的文本格式。ForgeLLM 的运行配置如下：

```toml
[run]
name = "cpu-smoke"
stage = "stage01"
seed = 1337
log_level = "INFO"
```

其中：

- `[run]` 是 table，可理解为一组命名字段；
- `name`、`stage` 是字符串；
- `seed` 是整数；
- `log_level` 是字符串，但后续只能取允许的枚举值。

`tomllib.load()` 只负责解析 TOML 语法，并不会自动知道 ForgeLLM 的业务规则。例如：

```toml
[run]
name = "cpu-smoke"
stage = "stage01"
seed = -999
log_level = "随便写"
```

这可能仍是合法 TOML，却不是合法的 ForgeLLM 配置。因此必须有第二层 Schema 校验。

### 语法正确不等于业务正确

这是 Day 3 最重要的区别之一：

```text
TOML parser 问：这段文本能否被解析？
Schema validator 问：这些值是否符合 ForgeLLM 的契约？
```

文件不存在或 TOML 括号错误属于读取/语法问题；`seed = -1`、未知字段或比例总和错误属于 Schema/业务约束问题。

## 4. Schema 校验的五个层次

这里的 Schema 不是某个第三方库，而是一套由代码明确实现的配置契约。

### 4.1 顶层结构

`load_run_config()` 要求整个文件恰好只有一个 `[run]` 顶层表：

```python
if set(document) != {"run"}:
    raise ConfigError("config must contain exactly one top-level [run] table")
```

这会拒绝：

- 完全没有 `[run]`；
- 同时出现 `[run]` 与 `[extra]`；
- 顶层键拼写错误。

### 4.2 未知字段与缺失字段

允许字段集合是：

```python
{"name", "stage", "seed", "log_level"}
```

代码使用集合差集：

```python
unknown = set(values) - _RUN_FIELDS
missing = _RUN_FIELDS - set(values)
```

例如用户误写 `loglevel`：

- `loglevel` 是 unknown；
- `log_level` 是 missing。

当前实现先报告 unknown，修正后如果仍缺字段，再报告 missing。这是 fail-fast 的体现：一次报告当前校验顺序遇到的第一个错误类别。

拒绝未知字段比静默忽略更安全。假设用户写了 `trian_bps = 8000`，静默忽略会让程序使用其他值继续运行，用户却误以为配置生效。明确失败能把拼写错误变成可见问题。

### 4.3 类型校验

`seed` 必须是真正的整数：

```python
if isinstance(seed, bool) or not isinstance(seed, int):
    ...
```

为什么还要先排除 `bool`？因为 Python 中：

```python
isinstance(True, int)  # True
```

`bool` 是 `int` 的子类，`True == 1`，`False == 0`。但配置中的 `seed = true` 明显不是用户想表达的随机种子。如果只写 `isinstance(value, int)`，布尔值会错误通过。

### 4.4 单字段范围与枚举

示例包括：

- `0 <= seed < 2**32`；
- `log_level` 只能是 `DEBUG/INFO/WARNING/ERROR/CRITICAL`；
- `unicode_normalization` 只能是 `NFC/NFD/NFKC/NFKD`；
- `min_chars` 不能为负；
- 数据切分比例不能为负。

正则 `_SLUG_PATTERN` 还要求 `name` 与 `stage`：

- 使用小写字母、数字、下划线或连字符；
- 第一个字符必须是小写字母或数字；
- 总长度最多 64。

它既提高 Run ID 的可读性，也降低路径中出现空格、`../` 等危险或含糊内容的风险。

### 4.5 跨字段约束

只检查每个字段的类型还不够。数据配置要求：

```python
0 <= min_chars <= max_chars
```

以及：

```python
train_bps + validation_bps + test_bps == 10000
```

`bps` 是 basis points，中文常译为“基点”。10000 个基点等于 100%，因此：

- 8000 = 80%；
- 1000 = 10%；
- 1000 = 10%。

使用整数基点而不是浮点数 `0.8 + 0.1 + 0.1`，可以避免二进制浮点表示带来的相等比较问题。

## 5. 为什么校验后要构造 frozen dataclass

`RunConfig` 与 `DataConfig` 都声明为：

```python
@dataclass(frozen=True, slots=True)
```

### dataclass

`dataclass` 自动生成初始化、比较和便于阅读的字符串表示。字段契约集中写在类定义中，比在程序各处传递无结构字典更清楚。

### frozen=True

对象构造后不能普通赋值修改：

```python
config.seed = 42  # 会失败
```

它表达的工程意图是：一旦配置通过校验并进入核心流程，同一次 Run 不应在中途悄悄改变参数。

注意：`frozen=True` 不是通用的“深度不可变”保证。如果字段里包含可变列表，列表内容仍可能改变。本项目配置字段都是字符串和整数，因此当前风险较小。

### slots=True

`slots` 限制实例只能拥有声明过的属性，并通常减少对象开销。更重要的是，它能更早暴露错误属性名，而不是静默给对象加一个新属性。

### 为什么不一直传字典

未校验字典的值类型是 `object`，任何字段都可能缺失、拼错或类型错误。校验后的 `DataConfig` 给核心函数一个更强的承诺：所需字段都存在，并已满足已实现的约束。

## 6. 原始配置与 resolved config

### 原始配置

原始配置是用户写入 TOML 的文本及其初次解析结果。它可能包含：

- 任意字段顺序；
- 用户输入的值；
- 未来版本可能支持的省略字段；
- 甚至无效或未知内容。

### resolved config

resolved config 是程序完成解析、默认值补齐、类型转换和校验后，实际交给核心逻辑的配置。

当前 ForgeLLM 要求字段全部显式提供，暂时没有很多默认值，因此原始值与 resolved 值看起来很接近。但它们仍不是同一概念：resolved config 已通过契约，并由 `as_dict()` 以稳定结构导出。

`initialize_run()` 保存的是：

```text
config.resolved.json
```

其中还加入配置 SHA-256。这样实验结果记录的是“程序实际使用了什么”，而不只是“用户最初写了什么”。以后若加入默认批大小、路径展开或别名规范化，这个区别会更明显。

### 为什么需要稳定序列化

`fingerprint()` 会：

1. 调用 `as_dict()`；
2. 使用 `sort_keys=True` 排序键；
3. 使用固定分隔符；
4. 编码为 UTF-8；
5. 计算 SHA-256。

所以只改变 TOML 字段顺序，不会改变 resolved config 的哈希；实际配置值变化则应改变哈希。

哈希只能帮助检测内容是否一致，不能证明配置设计正确、数据合法或实验结论真实。

## 7. fail-fast：失败要发生在哪里

fail-fast 指系统发现输入无效时尽快终止，不继续执行昂贵计算或产生误导性产物。

理想顺序是：

```text
读取配置
→ 校验配置
→ 检查必要输入
→ 创建正式产物或启动昂贵任务
```

在 `init-run` 中，`load_run_config()` 先执行，只有成功后才调用 `initialize_run()` 创建目录。因此未知字段不会留下一个看似正式的 Run 目录。

在 `data-pipeline` 中，`load_data_config()` 先执行，成功后才进入 `run_data_pipeline()`。比例错误会在读取数据集和写输出前失败。

### fail-fast 不等于“把所有错误都吞掉”

CLI 只捕获预期的领域错误和常见 I/O 错误：

- `ConfigError`；
- `DataConfigError`；
- `DataPipelineError`；
- `FileExistsError`；
- `OSError`；
- `init-run` 还处理时间等值错误 `ValueError`。

它们会被转成用户可理解的 stderr 信息和退出码 2。意料之外的编程错误不应无条件用 `except Exception` 隐藏，否则真正的 bug 会伪装成普通输入错误。

## 8. 两条完整调用链

### 8.1 `init-run`

```text
forgellm init-run --config configs/runtime/smoke.toml ...
→ console-script 启动器
→ forgellm.cli.entrypoint()
→ SystemExit(main())
→ build_parser()
→ argparse 把 --config/--artifacts-dir/--repo-root 转为 Path
→ args.command == "init-run"
→ load_run_config(args.config)
   → 以二进制方式打开 TOML
   → tomllib.load()
   → 检查顶层只能有 [run]
   → RunConfig.from_mapping()
   → 检查 unknown/missing/type/range/枚举
   → 返回 frozen RunConfig
→ initialize_run(config, artifacts_dir, repo_root)
   → 计算 resolved config 哈希
   → 构造 Run ID
   → 创建唯一目录
   → 写 config/environment/git/run/events 五类证据
   → 返回 RunArtifacts
→ CLI 输出 JSON
→ main() 返回 0
→ 进程退出码 0
```

关键细节：`argparse` 的 `type=Path` 只把字符串转换为 `Path`，不会保证文件存在。真正打开配置文件发生在 `load_run_config()`。

### 8.2 `data-pipeline`

```text
python -m forgellm data-pipeline --config ... --input ...
→ forgellm/__main__.py
→ cli.entrypoint()
→ main()
→ argparse 解析 5 个必填参数
→ args.command == "data-pipeline"
→ load_data_config(args.config)
   → tomllib.load()
   → 检查只能有 [data]
   → DataConfig.from_mapping()
   → 检查字段、类型、范围和跨字段比例
   → 返回 frozen DataConfig
→ run_data_pipeline(
     data_config,
     input_path,
     output_dir,
     source_name=...,
     source_license=...,
   )
→ 流水线读取、规范化、过滤、去重、切分并写审计产物
→ 返回 PipelineResult
→ CLI 从结果对象提取计数和 manifest 路径
→ stdout 输出 JSON，返回 0
```

`source-name` 和 `source-license` 由 CLI 作为字符串传入流水线，不属于当前 `DataConfig`。这是当前设计选择：处理策略由 TOML 固定，数据来源元数据由运行命令提供。

## 9. 错误如何变成退出码

需要区分三种情况：

### argparse 层错误

未知子命令或缺少必填参数时，`argparse` 会打印 usage/error，并抛出 `SystemExit(2)`。此时还没进入具体业务分支。

### 已预期的业务/配置错误

例如未知字段：

```text
load_run_config()
→ raise ConfigError(...)
→ CLI 的 except 捕获
→ stderr: forgellm: error: ...
→ main() return 2
→ entrypoint() 将其变成 SystemExit(2)
```

直接在单元测试里调用 `main([...])` 时观察到的是整数返回值；从终端通过 `entrypoint()` 执行时，整数会变成进程退出码。

### 未预期的程序错误

如果出现未捕获异常，程序会显示 traceback 并非零退出。开发阶段这通常比把错误全部隐藏更有利于定位 bug。

## 10. 动手实验

### 实验 A：加载并观察 resolved config

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from forgellm.config import load_run_config; c=load_run_config(Path('configs/runtime/smoke.toml')); print(c); print(c.as_dict()); print(c.fingerprint())"
```

稳定观察点：

- 输出是 `RunConfig(...)`；
- `as_dict()` 包含四个字段；
- 哈希是 64 个十六进制字符；
- 当前 smoke 配置哈希以 `21c69700` 开头。

该实验能证明当前配置可被当前代码加载和校验；不能证明所有配置都有效或 wheel 安装正确。

### 实验 B：验证未知字段、布尔值和比例约束

这些行为已有自动化测试，可直接运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_config.py tests\unit\test_data_config.py -q
```

当前结果为 `14 passed`。加上 `init-run` 集成测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_config.py tests\unit\test_data_config.py tests\integration\test_init_run_cli.py -q
```

当前结果为 `15 passed`。

### 实验 C：端到端数据 CLI Smoke

```powershell
.\.venv\Scripts\python.exe -m pytest tests\smoke\test_data_pipeline_cli.py -q
```

测试在 pytest 临时目录中运行，不会把正式输出写进仓库。它验证：

- 真正启动一个 Python 子进程；
- `python -m forgellm` 入口有效；
- 参数能传到数据流水线；
- 退出码为 0；
- stdout 是 JSON；
- 样例数据得到 9 条输入、4 条保留、5 条拒绝；
- manifest 文件确实生成。

### 常见失败排查顺序

1. 确认使用的是仓库 `.venv` 中的 Python；
2. 确认 `forgellm.__file__` 指向预期源码；
3. 检查 TOML 能否解析；
4. 根据错误信息检查顶层 table；
5. 检查未知和缺失字段；
6. 检查类型、范围与跨字段约束；
7. 配置通过后再检查输入路径和输出冲突；
8. 最后才进入流水线业务错误排查。

## 11. 常见误区总表

| 误区 | 正确理解 |
|---|---|
| TOML 能解析就说明配置正确 | TOML 语法与业务 Schema 是两层验证 |
| `type=Path` 会检查文件存在 | 它只构造 `Path`，打开文件时才检查 |
| 未知字段可以忽略 | 拼写错误可能被掩盖，应默认拒绝 |
| Python 的布尔值不是整数 | `bool` 是 `int` 子类，业务校验必须显式排除 |
| frozen 表示绝对深度不可变 | 它阻止字段重新赋值，不自动冻结嵌套可变对象 |
| fail-fast 就是捕获所有异常 | 只应转换预期错误，不应隐藏程序 bug |
| config 哈希相同就证明实验正确 | 哈希只证明序列化后的配置内容相同 |
| CLI 测试通过就证明核心算法正确 | CLI 测试只覆盖入口、参数流与指定断言 |

## 12. 思考题与对应答案

### Q1｜基础｜CLI 边界

为什么 `cli.py` 应负责参数解析和错误展示，却不应实现数据清洗算法？

<details>
<summary>查看参考答案与解析</summary>

**最小合格答案：** CLI 是外部字符串输入与内部对象之间的边界。算法放在独立函数中才能被测试、Notebook、服务接口和其他 Python 代码复用。

**完整答案：** CLI 负责协议层行为，包括命令、参数、stdout/stderr 和退出码；核心函数负责领域行为。分离后，算法测试不必每次启动子进程，其他入口也能共享同一实现。若清洗逻辑塞进 CLI，容易出现不同入口各写一份算法、行为逐渐不一致。

**常见误区：** “CLI 完全不能做校验。”CLI 可以做参数是否存在等边界校验，但业务 Schema 应放在可复用的配置层。

</details>

### Q2｜基础｜TOML 与 Schema

一份 TOML 文件语法正确，为什么仍可能被 ForgeLLM 拒绝？

<details>
<summary>查看参考答案与解析</summary>

**答案：** `tomllib` 只判断文本是否符合 TOML 语法；ForgeLLM 还要求正确顶层表、完整且无未知字段、正确类型、合法范围及跨字段关系。`seed = -1` 可以是合法 TOML 整数，但违反 ForgeLLM 的种子范围。

**常见误区：** 把 parser 与 validator 当成同一个组件。

</details>

### Q3｜关键｜未知字段

为什么严格拒绝未知字段通常比静默忽略更安全？请用拼写错误举例。

<details>
<summary>查看参考答案与解析</summary>

**最小合格答案：** 如果把 `train_bps` 拼成 `trian_bps`，静默忽略会让程序继续使用别的值，用户却误以为设置已生效。拒绝未知字段能在任务开始前暴露错误。

**完整答案：** 配置是实验输入。静默忽略会造成“用户认为的实验”和“程序实际运行的实验”不一致，破坏可复现性和结论可信度。向后兼容有时需要允许扩展字段，但应通过显式版本或命名空间设计，而不是无条件忽略。

</details>

### Q4｜关键｜resolved config

原始 TOML 与 `config.resolved.json` 有什么概念差异？为什么实验应保存后者？

<details>
<summary>查看参考答案与解析</summary>

**最小合格答案：** 原始 TOML 是用户输入；resolved config 是经过解析、默认值处理、类型转换和校验后程序实际使用的值。保存后者才能准确回答本次运行真正使用了什么。

**项目边界：** 当前配置要求所有字段显式提供，所以两者差异较小；未来加入默认值或规范化后差异会更明显。

**常见误区：** 只保存 resolved config 就可以丢弃原始配置。审计要求较高时，两者都可能值得保存，但其用途不同。

</details>

### Q5｜关键｜bool 陷阱

为什么整数配置检查必须显式排除 `bool`？

<details>
<summary>查看参考答案与解析</summary>

**答案：** Python 中 `bool` 是 `int` 子类，`isinstance(True, int)` 为真。如果只检查 `int`，`test_bps = true` 会被当成 1，导致含义错误。代码先检查 `isinstance(value, bool)`，再检查整数类型。

**迁移：** 任何“整数但绝不能接受开关值”的配置都应考虑这个问题，例如 seed、batch size、端口和计数。

</details>

### Q6｜应用｜basis points

为什么切分比例使用 8000/1000/1000，而不是 0.8/0.1/0.1？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 整数基点可以精确验证总和等于 10000，避免浮点表示与相等比较问题，也便于基于整数区间进行确定性切分。8000、1000、1000 分别代表 80%、10%、10%。

**常见误区：** 整数基点能解决所有统计误差。它只解决比例表达和边界的确定性，不能保证小数据集实际样本数恰好符合百分比。

</details>

### Q7｜理解｜frozen 与 slots

`@dataclass(frozen=True, slots=True)` 分别表达什么？它是否保证绝对不可变？

<details>
<summary>查看参考答案与解析</summary>

**答案：** `frozen=True` 阻止普通字段重新赋值，表达配置构造后不应改变；`slots=True` 限制实例属性集合并通常减少开销。它不保证嵌套列表等对象深度不可变，但当前配置字段都是字符串和整数。

</details>

### Q8｜关键｜退出码

未知子命令与无效配置都可能得到退出码 2，但它们分别在哪一层产生？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 未知子命令由 `argparse` 在解析阶段直接抛出 `SystemExit(2)`；无效配置由 `load_*_config()` 抛出领域错误，CLI 捕获后打印 `forgellm: error: ...` 并让 `main()` 返回 2，随后 `entrypoint()` 用 `SystemExit(main())` 将其变成进程退出码。

**评分重点：** 必须说明“直接调用 `main()` 返回整数”和“终端执行产生进程退出码”的区别。

</details>

### Q9｜代码追踪｜Path

`argparse.add_argument(..., type=Path)` 是否能保证配置文件存在？如果不能，错误在哪里出现？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 不能。`type=Path` 只把字符串构造成 `Path`。文件存在性和权限在 `path.open('rb')` 时才确定；`OSError` 被包装成 `ConfigError` 或 `DataConfigError`，再由 CLI 转成错误信息和退出码 2。

</details>

### Q10｜诊断｜字段拼写

用户把 `log_level` 写成 `loglevel`。当前实现会先报告 unknown 还是 missing？为什么？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 两个集合都会检测到问题，但代码先执行 `if unknown`，所以先报告 `Unknown run fields: loglevel`。修正或移除未知字段后，如果仍没有 `log_level`，才会报告 missing。

**开放讨论：** 也可以一次报告两类错误，但当前 fail-fast 顺序更简单。选择哪种取决于错误体验和实现复杂度，不存在普遍唯一答案。

</details>

### Q11｜机制｜稳定哈希

为什么调换 TOML 字段顺序不应改变配置哈希？当前代码如何做到？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 字段顺序不改变配置语义。代码先导出 resolved 字典，再用 `json.dumps(..., sort_keys=True, separators=(',', ':'))` 进行确定性序列化，最后对 UTF-8 字节计算 SHA-256，所以映射的输入顺序不会影响结果。

**边界：** 不同代码版本的规范化规则可能产生不同 resolved 表示，因此还必须记录 Git 和环境信息。

</details>

### Q12｜设计｜新增字段

如果要给 `RunConfig` 新增 `max_steps`，至少需要修改和测试哪些地方？

<details>
<summary>查看参考答案与解析</summary>

**评分要点：**

1. 把字段加入允许集合 `_RUN_FIELDS`；
2. 在 dataclass 中声明类型；
3. 从 mapping 读取；
4. 排除 bool 并校验整数、范围；
5. 构造 `RunConfig` 时传入；
6. 在 `as_dict()` 中输出，使 resolved config 与哈希包含它；
7. 更新 smoke TOML 和相关文档；
8. 增加有效值、缺失、错误类型、越界、哈希变化与 CLI 集成测试；
9. 明确它是必填还是有默认值，并处理向后兼容。

只改 dataclass 而不改允许字段和 `as_dict()`，会造成加载失败或证据链遗漏。

</details>

### Q13｜关键｜fail-fast 位置

数据配置比例错误时，为什么应该在读取整个数据集和创建输出目录之前失败？

<details>
<summary>查看参考答案与解析</summary>

**答案：** 比例错误仅依赖配置即可发现。提前失败能避免浪费 I/O/CPU/GPU、避免产生部分产物，也防止错误配置形成看似正式的实验记录。当前 CLI 先调用 `load_data_config()`，成功后才调用 `run_data_pipeline()`，符合这一原则。

**常见误区：** “任何失败都必须零副作用。”实际系统可能先写临时日志；重点是不要在已知无效输入下启动昂贵或不可逆工作，并要明确清理/原子写策略。

</details>

### Q14｜证据边界｜测试

Day 3 的 15 项测试通过后，最强的合理结论是什么？哪些说法属于过度声明？

<details>
<summary>查看参考答案与解析</summary>

**合理结论：** 当前 Windows/Python 环境中，已编码的 RunConfig/DataConfig 有效与无效案例、稳定哈希以及 `init-run` 集成路径按测试预期工作。

**过度声明：**

- 所有可能配置错误都已覆盖；
- Linux CI 一定通过；
- wheel 安装一定正确；
- 数据流水线算法完全无 bug；
- 训练结果可复现；
- 正式数据合法或高质量。

测试通过是具体证据，不是无限范围的质量证明。

</details>

## 13. 验收标准

总分 28 分，每题 2 分：

- 24–28：能够迁移；
- 20–23：能够应用，修正错题后通过；
- 14–19：初步掌握，需要重画调用链并重做实验；
- 0–13：尚未掌握，需要重新学习第 2–9 节。

Q3、Q4、Q5、Q8、Q13 是关键题。任一关键题为 0 分时，不建议直接进入下一天。

完成 Day 3 的最低证据：

- [ ] 不看源码画出 `init-run` 调用链；
- [ ] 不看源码画出 `data-pipeline` 配置加载链；
- [ ] 能解释 TOML 语法校验与 Schema 校验的差异；
- [ ] 能解释 unknown、missing、type、range、cross-field 五层约束；
- [ ] 能解释为什么排除 bool；
- [ ] 能解释原始配置与 resolved config；
- [ ] 能预测三个无效配置的失败位置和错误类型；
- [ ] 能说明测试通过的证据边界；
- [ ] Day 3 相关测试通过；
- [ ] 用自己的话回答关键题，而不是照抄答案。

## 14. 本日结论

Day 3 的主线不是“学会写 TOML”，而是建立一条可信输入链：

```text
外部字符串
→ 可解析文本
→ 严格 Schema
→ 稳定配置对象
→ 可审计 resolved config
→ 核心任务
```

这条链把配置错误阻挡在昂贵工作之前，也让后续 Run ID、配置哈希、日志、数据切分和训练实验有明确输入。只有当你能解释失败发生在哪一层、为什么发生，以及一次通过不能证明什么时，才算真正掌握 Day 3。
