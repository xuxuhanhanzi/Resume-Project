# Stage 1：最小数据闭环与 Byte-level BPE Tokenizer

> 状态：G1-A、G1-B 自动化实现与学习者验收全部完成  
> 启动日期：2026-07-25（Asia/Singapore）  
> 预计周期：2 周  
> 前置门禁：Stage 0 / S0 已通过  
> 阶段门禁：G1 = G1-A（经典 BPE 正确性）+ G1-B（现代方法实现与学习验收）

## 1. 目标与用户价值

学习者不应直接从本阶段计划开始学习。本文件负责门禁和证据边界；唯一教学入口是 `docs/lessons/stage01_learning_order.md`。

本阶段不把 Tokenizer 当成一个库调用题，而是完成从 UTF-8 字节、BPE merge、固定特殊 Token，到可保存、可加载、可评测 Artifact 的最小闭环。完成后，学习者应当能够：

- 手算并解释一次 BPE 的 pair frequency、tie-break 和 merge；
- 用纯 Python 实现确定性的 byte-level BPE；
- 说明 Unicode 字符、UTF-8 字节、Token 和 Token ID 的区别；
- 训练、保存、加载、编码、解码并验证完全一致；
- 用固定指标比较 raw-byte、手写 BPE 和成熟库实现；
- 识别“算法正确”“工程可复现”和“正式语料质量达标”是三个不同结论。
- 解释 pre-tokenization、BPE-dropout、Unigram、Picky BPE、SuperBPE 与动态 byte patch 的核心差异。

## 2. 可证伪假设

在固定、合法、已经完成 Unicode 规范化和精确去重的 UTF-8 文档上，采用固定特殊 Token ID、确定性 tie-break 和固定配置的纯 Python byte-level BPE：

1. 对任意合法 UTF-8 输入实现 `decode(encode(text)) == text`；
2. 同一训练文档集合即使输入顺序变化，也生成相同模型 SHA-256；
3. 保存再加载后，词表、merge 顺序和编码结果完全一致；
4. 相比 raw-byte 基线，在阶段夹具上提高 bytes/token，且 unknown rate 为 0；
5. 行为可与成熟 Tokenizer 库进行同口径比较，并能解释差异来自实现契约，而不是只比较一个总分。

只要其中任一项失败，G1-A 就不通过。G1-A 通过只代表经典 BPE 闭环；完整 G1 还要求完成 G1-B 的现代方法实现与学习者验收。

## 3. 主变量与固定项

### 主变量

Tokenizer 路径：

- B0：raw UTF-8 byte，每个字节一个 Token；
- B1：ForgeLLM 纯 Python byte-level BPE；
- R1：Hugging Face `tokenizers` 的 ByteLevel+BPE 参考实现（可选依赖）。

### 固定项

- 输入单位：文档；不跨文档统计 pair，也不跨文档执行 merge；
- 编码：UTF-8 strict；Tokenizer 内不再做 Unicode 规范化；
- 特殊 Token：`<pad>=0`、`<bos>=1`、`<eos>=2`、`<unk>=3`；
- 普通字节 Token：ID `4..259`，其中 `token_id = byte + 4`；
- merge 选择：最高频优先，频率相同时选择 `(left_id, right_id)` 字典序更小者；
- merge 替换：从左到右、非重叠；
- 配置：`vocab_size=320`、`min_pair_frequency=2`；
- correctness 实验仅使用仓库自有测试夹具，不产生正式语料质量结论。

## 4. 数据边界

### G1-A correctness 夹具

- 路径：`tests/fixtures/tokenizer/`；
- 来源：ForgeLLM 项目为测试目的原创；
- 许可证标记：`project-test-fixture`；
- 用途：算法、序列化、CLI、指标和边界测试；
- 禁止结论：不得据此声称对自然语言域具有代表性、生产可用或达到正式压缩质量。

### G1-B 方法实验夹具

- 继续使用 `tests/fixtures/tokenizer/` 的项目原创中英、代码、空白、Emoji 与边界文本；
- 固定 `vocab_size=300`，不做词表规模优化；
- 不下载 Wikipedia，不把数据许可证工程设为模型学习的前置阻塞；
- 所有结果只用于解释算法行为，不代表正式自然语言质量；
- 真正预训练时可复用成熟 Tokenizer，或在模型消融需要时另行冻结正式语料。

详细实验矩阵、实现边界和学习者门禁见 `docs/stages/g1b_tokenizer_method_lab_plan.md`。

## 5. 指标与口径

| 指标 | 定义 | G1-A 门槛 |
|---|---|---:|
| round-trip rate | 完全满足 `decode(encode(text)) == text` 的文档数 / 总文档数 | 100% |
| bytes/token | UTF-8 总字节数 / 普通 Token 总数 | B1 > B0 |
| chars/token | Unicode code point 总数 / 普通 Token 总数 | 报告，不设跨语言绝对阈值 |
| fertility | 普通 Token 总数 / 非空白 segment 总数；segment 使用 Python `str.split()` | 报告并注明分母 |
| unknown rate | `<unk>` 数 / 普通 Token 总数 | 0% |
| special-token accuracy | BOS/EOS 插入位置、PAD/UNK 固定 ID 的通过断言数 / 总断言数 | 100% |
| throughput | UTF-8 字节数 / 计时秒数 | 报告环境与协议，不作为本阶段否决项 |
| determinism | 固定配置与相同文档多重集合生成同一 SHA-256 | 必须通过 |

吞吐只在同一机器、同一解释器、相同预热和重复协议下比较。微型夹具的吞吐波动很大，只作为管线连通证据。

## 6. Stop/Go 门禁

### G1-A：算法与工程正确性

- [x] 配置 schema 严格校验，未知字段和非法类型 fail-fast；
- [x] 纯 Python BPE 的训练、encode、decode、保存、加载完成；
- [x] 英文、中文、Emoji、组合字符、空串和空白边界测试通过；
- [x] 固定特殊 Token ID，raw-byte 基线和指标定义完成；
- [x] 同文档不同顺序的模型哈希相同；
- [x] 损坏 Artifact、错误 schema 和非法 Token ID 会被拒绝；
- [x] CLI train/evaluate 完成，已有工程质量门全部通过；
- [x] Hugging Face `tokenizers==0.23.1` ByteLevel+BPE 参考路径已验证。

### G1-B：现代 Tokenizer 方法

- [x] 无边界、空白边界与 Unicode-class pre-tokenization 已实现；
- [x] BPE-dropout 与手写 Unigram EM/Viterbi/sampling 已实现；
- [x] Picky BPE 的 merge/remove 教学事件流已实现；
- [x] SuperBPE 两阶段 pre-tokenization curriculum 已实现；
- [x] 特殊 Token 显式许可、byte offset 与 entropy patching 已实现；
- [x] 统一方法实验、分 subset 指标、完整讲义和实验记录已生成；
- [x] 学习者完成代码追踪、思考题与最终口述验收。

自动化实现与证据已经完成；学习者随后已确认完成代码追踪、思考题与最终口述验收，完整 G1 和 Stage 1 已通过。该结论只覆盖教学夹具和方法实验，不证明现代 Tokenizer 方法对真实语言模型有下游收益。

## 7. 失败处理

- round-trip 失败：保存最短反例，优先检查特殊 Token 过滤、UTF-8 byte 拼接和 merge 顺序；
- 非确定性：固定 pair tie-break、文档排序无关统计，并比较序列化前的 canonical payload；
- unknown rate 非零：检查 256 个基础 byte Token 是否完整，禁止用字符级 OOV 掩盖问题；
- BPE 不优于 raw byte：先检查算法不变量；本阶段不通过扩大词表或下载大语料追求指标；
- 成熟库差异：分别核对 pre-tokenizer、initial alphabet、special token、decoder 和前缀空格契约；
- 随机切分不变化：检查 dropout/temperature 与概率分布，但不能为制造漂亮数字破坏正确性；
- Picky/Super 论文收益未复现：保留“教学版”限定，不把结构实验写成模型质量结论；
- entropy patch round-trip 失败：检查 patch 边界是否严格覆盖每一个输入 byte。

## 8. 预期 Artifact

- `src/forgellm/tokenization/`：实现；
- `configs/tokenizer/bpe_v1.toml`：冻结配置；
- `docs/tokenizer/tokenizer_spec_v1.md`：格式与行为契约；
- `tests/fixtures/tokenizer/`：可审计夹具与来源声明；
- `artifacts/stage01_tokenizer_candidate/`：本地候选模型和评测报告；
- `docs/experiments/2026-07-25_stage01_tokenizer_candidate.md`：实验记录；
- `docs/lessons/stage01_byte_level_bpe_tokenizer.md`：完整讲义；
- `docs/lessons/stage01_learning_order.md`：新手唯一学习入口与严格文件顺序；
- `docs/lessons/stage01_tokenizer_method_lab.md`：G1-B 完整讲义；
- `src/forgellm/tokenization/advanced_bpe.py` 与 `unigram.py`：现代方法教学实现；
- `scripts/tokenizer_method_lab.py`：统一实验入口；
- `docs/experiments/2026-07-27_stage01_g1b_method_lab.md`：G1-B 实验证据；
- `HANDOFF_SUMMARY.md` 与进度台账：当前状态和下一步。
