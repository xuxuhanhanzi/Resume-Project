# ForgeLLM Byte-level BPE Tokenizer 规格 v1

## 1. 范围

该规格定义 Stage 1 手写 Tokenizer 的可观察行为。它用于学习、正确性验证和小模型实验，不声称兼容任一现有模型的词表或 Token ID。

## 2. 输入契约

- API 输入是 Python `str`，并用 UTF-8 strict 编码成字节；
- Tokenizer 不修改换行、不 strip、不做 NFC/NFKC 等 Unicode 规范化；
- 数据预处理必须在 Tokenizer 训练前完成并写入数据 Manifest；
- BPE pair 不跨文档边界；
- 空字符串可以编码，普通 Token 序列为空；若请求 BOS/EOS，则只输出相应特殊 Token。

这样设计可以保证 Tokenizer 是可逆编码器，而不是隐藏的数据清洗器。NFC 与 NFD 的视觉结果可能相同，但字节序列不同，因此会得到不同 Token 序列并分别 round-trip。

## 3. 固定 Token ID

| Token | ID | 解码默认行为 |
|---|---:|---|
| `<pad>` | 0 | 跳过 |
| `<bos>` | 1 | 跳过 |
| `<eos>` | 2 | 跳过 |
| `<unk>` | 3 | 跳过；正常 byte-level 编码不会产生 |
| byte `0x00..0xFF` | 4..259 | 还原对应字节 |
| learned merge | 260 起 | 还原为其左右 Token 字节串拼接 |

特殊 Token 不是 UTF-8 内容的一部分。`decode(..., skip_special_tokens=False)` 只用于调试，会输出特殊 Token 的可见字符串；模型正文 round-trip 使用默认跳过行为。

## 4. 训练算法

给定每篇文档的基础 byte Token ID 序列：

1. 统计每篇文档内全部相邻 pair 的总频率；
2. 找到最高频 pair；
3. 频率相同时，选择 `(left_id, right_id)` 字典序最小的 pair；
4. 若频率小于 `min_pair_frequency`，停止；
5. 为该 pair 分配下一个连续 Token ID；
6. 对每篇文档从左到右执行非重叠替换；
7. 重复，直到达到 `vocab_size` 或没有满足条件的 pair。

训练文档的遍历顺序不能影响频率总和或 tie-break，因此同一文档多重集合应产生同一模型。

## 5. 编码算法

1. 将输入转为 UTF-8 bytes；
2. 映射到 ID `byte + 4`；
3. 按训练时的 merge rank 顺序逐一做左到右非重叠替换；
4. 根据调用参数在首尾插入 BOS/EOS。

编码不是重新选择“当前最高频 pair”，而是重放冻结的 merge 顺序。否则同一个模型无法定义稳定的编码函数。

## 6. 解码算法

- 普通 Token 对应的 byte payload 按序拼接，再用 UTF-8 strict 解码；
- 默认跳过特殊 Token；
- 未知 ID、损坏词表或最终字节不是合法 UTF-8 时立即报错；
- `<unk>` 仅为下游模型接口保留，正常编码路径的 unknown rate 必须为 0。

## 7. Artifact 格式

模型使用 UTF-8 JSON：

- `schema_version`: `forgellm-byte-bpe-v1`；
- `config`: 解析后的训练配置；
- `special_tokens`: 名称到固定 ID 的映射；
- `vocabulary_hex`: 每个普通/merge Token 的 bytes 十六进制字符串，特殊 Token 使用 `null`；
- `merges`: 按 rank 排列的 `left_id`、`right_id`、`new_id`；
- `model_sha256`: 对不含该字段的 canonical JSON payload 计算 SHA-256。

加载器必须重新计算哈希并验证所有结构不变量。保存默认拒绝覆盖已有文件，避免把一次实验 Artifact 静默改写。

## 8. 可复现边界

模型 SHA-256 证明的是以下内容一致：解析后配置、特殊 Token 契约、词表 bytes 和 merge 顺序。它不证明训练语料相同；训练记录还必须保存输入文件 SHA-256、来源、许可证和数据 Manifest。

## 9. 与成熟库比较时必须固定的契约

成熟实现可能使用 GPT-2 风格 byte-to-Unicode 映射、regex pre-tokenization、自动前缀空格或不同 tie-break。比较时至少报告：

- vocab budget 和实际词表大小；
- initial alphabet；
- pre-tokenizer 与 decoder；
- special tokens；
- 是否添加 prefix space；
- 相同评测文本上的 round-trip、bytes/token、fertility 和吞吐。

Token ID 不同不等于实现错误；只有在契约相同的项目上才能逐项对齐。

