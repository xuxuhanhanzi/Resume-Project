# 数据流水线最小闭环设计

## 1. 范围

本设计覆盖 Month 01 / Day 6–10 的工程基线：

```text
JSONL Reader
→ Schema 校验
→ Unicode/换行/尾随空白规范化
→ 可解释质量过滤
→ Document ID 去重
→ 规范化文本 SHA-256 精确去重
→ 内容哈希确定性切分
→ 稳定 JSONL Writer
→ Manifest 与统计报告
```

本阶段只在仓库内自建测试夹具上运行。正式数据源、许可证审计和 Data Card 属于 Day 11–15，不能由本阶段替代。

## 2. 输入契约

每行必须是一个且仅包含以下字段的 JSON Object：

```json
{"id": "document-id", "text": "document text"}
```

- `id`：非空字符串，最长 256 字符；
- `text`：字符串；
- 缺字段、额外字段、类型错误或无效 JSON 不终止整条流水线，而是进入拒绝统计；
- 拒绝文件不保存原始文本，只保存输入行哈希引用、可用时的文档 ID 和原因。

## 3. 规范化契约

- 默认 Unicode NFC；
- `CRLF` 和单独 `CR` 转为 `LF`；
- 删除每行尾随空白，并删除文档首尾空白；
- 不压缩文档内部连续空格；
- 规范化策略是配置的一部分，并参与配置哈希。

NFC 是首月的保守选择。NFKC 可能改变兼容字符的语义，不在没有受控评测时启用。

## 4. 过滤与去重

过滤原因固定为可统计枚举：

- `blank_line`；
- `invalid_json`；
- `invalid_schema`；
- `empty_text`；
- `too_short`；
- `too_long`；
- `disallowed_control_character`；
- `duplicate_document_id`；
- `duplicate_exact`。

处理不依赖输入顺序：相同 Document ID 保留内容哈希最小的记录；相同规范化文本保留 Document ID 字典序最小的记录。所有拒绝项和输出文档在写入前稳定排序。

## 5. 切分

使用 `SHA-256(f"{split_seed}:{content_sha256}")` 映射到 0–9999 的 bucket，再按 basis points 配置划分 train/validation/test。三者必须合计 10000。

因此：

- 输入顺序变化不会改变文档所属 split；
- 新增其他文档不会重新洗牌已有文档；
- 修改 seed 或切分比例会改变配置哈希，必须形成新数据版本。

## 6. 输出契约

输出目录必须不存在，流水线拒绝覆盖已有结果。每次运行产生：

- `train.jsonl`；
- `validation.jsonl`；
- `test.jsonl`；
- `rejects.jsonl`；
- `report.json`；
- `manifest.json`。

Manifest 至少记录：Schema 版本、来源名、许可证标签、输入文件名/哈希/字节数、解析后配置及其哈希、输出文件哈希/字节数/记录数、总计数和拒绝原因。

## 7. 安全与隐私边界

- Manifest 不记录绝对用户路径；
- Reject 不保存原始文本；
- 日志不保存 Secret；
- 当前仅检测控制字符，不宣称完成 PII、凭据、许可证或 Benchmark 污染扫描；
- 正式数据下载前必须确认可用存储并完成来源/许可证审计。

