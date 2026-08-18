# 实验记录：Stage 1 Byte-level BPE Tokenizer Candidate

## 1. 目标与可证伪假设

目标是在不依赖第三方 Tokenizer 的主路径中，实现一个确定性、可逆、可保存和可评测的 byte-level BPE，并与 raw UTF-8 byte 基线及 Hugging Face `tokenizers` 成熟实现做同口径对照。

假设：固定语料与配置下，ForgeLLM BPE 应达到 100% exact round-trip、0% unknown、模型哈希不受文档顺序影响，并在测试夹具上获得高于 raw-byte 的 bytes/token。任一条件失败则 G1-A 不通过。

## 2. 环境

- 日期：2026-07-25（Asia/Singapore）；
- 平台：Windows 11 `10.0.22631`；
- Python：3.12.3；
- ForgeLLM 路径：`D:\Users\27475\Desktop\Resume_Project\ForgeLLM`；
- 分支：`agent/month01-engineering-study`；
- 基线 commit：`983a64ab15ce160e232210d30b63912b082b8a22`；
- Git 状态：dirty；包含此前用户/路线文档和本次 Stage 1 增量修改；
- 核心手写实现：无运行时第三方依赖；
- 成熟参考：Hugging Face `tokenizers==0.23.1`；
- GPU/付费资源：未使用。

## 3. 数据与配置

### 训练夹具

- 文件：`tests/fixtures/tokenizer/train.jsonl`；
- 来源：ForgeLLM 项目原创；
- 许可证标记：`project-test-fixture`；
- 文档数：8；
- 文件大小：893 bytes；
- SHA-256：`dc74bb063effded401a5a1e9bc559951b0b7ccf50f791242a029176137c211ed`。

### 测试夹具

- 文件：`tests/fixtures/tokenizer/test.jsonl`；
- 文档数：5；
- 文件大小：428 bytes；
- SHA-256：`9a60938829127ce6fbf8efd3a90a9881226cbc89b6b36d10e06cb3a3f5410538`。

### 配置

- 文件：`configs/tokenizer/bpe_v1.toml`；
- `vocab_size=320`；
- `min_pair_frequency=2`；
- resolved config SHA-256：`468dada8e8fb37bda8e73731c2f2ae09b2e47c561679816aca158762f98afad2`；
- 固定特殊 Token：PAD/BOS/EOS/UNK = 0/1/2/3；
- 参考库契约：`ByteLevel(add_prefix_space=False,use_regex=False)`、显式 byte alphabet、相同特殊 Token 和预算。

这些文件仅用于 correctness 和工程候选，不代表正式自然语言分布。

## 4. 主变量与固定项

- 主变量：raw byte、ForgeLLM byte BPE、Hugging Face ByteLevel+BPE；
- 固定：相同训练夹具、相同测试夹具、词表预算 320、最小 pair 频率 2、特殊 Token ID；
- 明确差异：ForgeLLM 直接使用整数 byte ID 和 `(left_id,right_id)` tie-break；HF 使用其 ByteLevel 内部表示与 trainer 行为；
- 计时：同一进程环境，每篇测试文本 encode 100 次；微型计时只用于路径级观察。

## 5. 执行命令

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tokenizer.lock

.\.venv\Scripts\python.exe -m forgellm tokenizer-train --config configs\tokenizer\bpe_v1.toml --input tests\fixtures\tokenizer\train.jsonl --output-dir artifacts\stage01_tokenizer_candidate\model --source-name forgellm-test-fixture --source-license project-test-fixture

.\.venv\Scripts\python.exe -m forgellm tokenizer-evaluate --model artifacts\stage01_tokenizer_candidate\model\tokenizer.json --input tests\fixtures\tokenizer\test.jsonl --output artifacts\stage01_tokenizer_candidate\evaluation_test.json --encode-repeats 100

.\.venv\Scripts\python.exe -m forgellm tokenizer-compare --config configs\tokenizer\bpe_v1.toml --model artifacts\stage01_tokenizer_candidate\model\tokenizer.json --train-input tests\fixtures\tokenizer\train.jsonl --evaluation-input tests\fixtures\tokenizer\test.jsonl --output-dir artifacts\stage01_tokenizer_candidate\reference_comparison_test --encode-repeats 100

.\.venv\Scripts\python.exe scripts\dev.py check
```

## 6. Artifact

- 手写模型：`artifacts/stage01_tokenizer_candidate/model/tokenizer.json`；
- 训练 Manifest：`artifacts/stage01_tokenizer_candidate/model/manifest.json`；
- test 评测：`artifacts/stage01_tokenizer_candidate/evaluation_test.json`；
- HF 原生模型：`artifacts/stage01_tokenizer_candidate/reference_comparison_test/huggingface_tokenizer.json`；
- 三路对照：`artifacts/stage01_tokenizer_candidate/reference_comparison_test/comparison.json`。

哈希：

- ForgeLLM canonical model fingerprint：`82ccbedc1b9c2713dcf94533a7a9422a126ab281b84021b813cfa7e2e0305223`；
- ForgeLLM 模型 JSON 文件 SHA-256：`e580582e9ef1c3167dd53767f64b0b51d07ea9f7f0ad505b3a2df7cdeea9a003`；
- test 评测文件 SHA-256：`d8d05cab4bb98f7467307f5ca2368c02ba4f37dc8e4e4401cbd1cf3d87385ba5`；
- comparison 文件 SHA-256：`09a328d9c0efadca955b3b47db73530459afe8bf34c06e1bc4cc37281a92f810`；
- HF 模型文件 SHA-256：`022c04e993f677e04c14dd4f871bfb2e86084a83d4ad0b9e548511df68b9be07`。

## 7. 结果

训练得到实际词表 320，其中基础/特殊 Token 260 个，learned merge 60 个。

| 实现 | 词表 | Test Tokens | bytes/token | chars/token | fertility | round-trip | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw UTF-8 bytes | 256 | 182 | 1.0000 | 0.7637 | 9.5789 | 1.0 | 0.0 |
| ForgeLLM byte BPE | 320 | 152 | 1.1974 | 0.9145 | 8.0000 | 1.0 | 0.0 |
| HF ByteLevel+BPE 0.23.1 | 320 | 149 | 1.2215 | 0.9329 | 7.8421 | 1.0 | 0.0 |

- ForgeLLM 比 raw-byte 少 30 Token，即 `30/182 ≈ 16.5%`；
- HF 参考比 raw-byte 少 33 Token，即 `33/182 ≈ 18.1%`；
- 特殊 Token 检查 6/6，通过率 100%；
- ForgeLLM 的文档顺序不变性由反转输入集成测试验证；
- 保存/加载后 fingerprint 和编码结果完全一致。

微型 encode 吞吐：

| 实现 | bytes/s | 解释 |
|---|---:|---|
| raw UTF-8 bytes | 106,557,377 | 极短直接转换 |
| ForgeLLM byte BPE | 182,771 | 纯 Python，按 60 个 merge 反复扫描 |
| HF ByteLevel+BPE | 3,635,274 | 成熟 Rust 实现 |

夹具只有 182 个测试 bytes，吞吐对计时噪声敏感；该表不能用于生产容量规划或严谨性能结论。

## 8. 质量门结果

- Ruff format：通过；
- Ruff lint：通过；
- mypy strict：通过，41 个受检源文件；
- pytest：68/68 通过；
- unit/integration/smoke：均通过；
- 付费/GPU：0。

## 9. 异常与处理

1. 首次安装 `tokenizers` 因沙箱网络策略失败；在获得联网批准后从已配置 PyPI 镜像成功安装 0.23.1。
2. 首轮夹具文件存在额外空行，严格 JSONL Reader 按设计拒绝；去除明确空行后通过，未放宽 schema。
3. 首轮 mypy 对测试中的递归 `JsonValue` 直接数值比较报错；通过显式类型收窄修复，未关闭 strict。
4. ForgeLLM 吞吐显著低于 HF，符合学习实现的 `O(merges × sequence_length)` 重扫结构；不伪装为性能完成。

## 10. 结论与边界

G1-A（算法与工程正确性）通过：手写 byte-level BPE 的训练、持久化、可逆编码、确定性、特殊 Token、指标、CLI 和成熟库参考路径均有自动测试与候选 Artifact 证据。

完整 G1 尚未通过，因为 G1-B（正式语料质量）仍缺少用户确认的正式数据来源、许可证、目标领域、代表性分割和 Data Card。当前 16.5% Token 减少只描述项目测试夹具，不得外推为自然语言或生产质量，不得直接写入简历成果。

## 11. 下一步

1. 与用户冻结正式语料方向、许可证和 train/validation/test；
2. 运行现有数据流水线并冻结正式 Manifest；
3. 只用 train 学词表，用 validation 做有限配置选择，最终在 test 与子集报告；
4. 关闭 G1-B 后确认完整 G1；
5. 可并行开始 Stage 2 的 PyTorch 与 Decoder-only Transformer 完整教学准备，但正式预训练前不得跳过 G1-B。
