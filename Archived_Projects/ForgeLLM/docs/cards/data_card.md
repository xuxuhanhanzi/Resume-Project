# Data Card：Stage 3 TinyStories 教学子集

## 来源与用途

- 上游：`roneneldan/TinyStories/TinyStories-valid.txt`
- 许可证：CDLA-Sharing-1.0
- 原始大小：19,447,282 bytes
- 原始 SHA-256：`94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`
- 用途：小模型预训练方法、恢复和指标教学；不作 TinyStories benchmark 声明。

## 处理与切分

- 解析 21,990 篇故事，NFC/newline 规范化并做精确去重；
- 固定 seed `20260727` 按内容哈希 90%/5%/5% 切分；
- train 19,747、validation 1,128、test 1,115；0 拒绝；
- split 在 tokenize/packing 之前完成。

## Tokenization

- Stage 1 byte-BPE，320 vocab；
- fingerprint：`82ccbedc1b9c2713dcf94533a7a9422a126ab281b84021b813cfa7e2e0305223`；
- 每文档独立 encode 并追加 EOS；
- token cache 绑定 split SHA、Tokenizer fingerprint 和 sequence length。

## 局限

- 英文合成儿童故事，词汇和叙事结构狭窄；
- 上游 validation 文件被重新划分，不能与官方 validation 结果比较；
- 没有独立 PII/版权检测，只依赖合成数据来源和上游许可证；
- 不代表 Web、代码、中文、知识或安全分布；
- test split 在 Stage 3 不用于挑选方法。

## 本地删除与重建

原始/处理数据位于 Git 忽略的 `data/raw` 和 `data/processed`。如需删除，应逐个明确文件操作；禁止递归批量删除。重建入口为 `fetch_stage3_corpus.py` 和 `prepare_stage3_corpus.py`。

---

# Data Card：Stage 4 监督式后训练数据

## 项目原创 correctness 数据

- 名称：`stage4_correctness_v1`；
- 生成器：`src/forgellm/post_training/correctness_data.py`；
- 规模：64 train / 16 validation / 16 test；
- 许可证：project-original；
- 任务：精确标识符、精确两词、JSON、前缀/必含字符串四类确定性约束；部分为多轮记录；
- train SHA-256：`ca2197cd1a560022c92f9bdf2ed7f223b897f5e4244423d643c96171adddb322`。

它用于模板/loss/tiny-overfit 和固定行为测试，不代表自然聊天分布。test 只用于冻结评测，正式 SFT 训练使用公开数据 train split。

## SmolTalk constraints 子集

- 上游：`HuggingFaceTB/smoltalk` / `smol-constraints`；
- revision：`5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`；
- 上游该新增子集声明为 Apache-2.0；
- 上游规模：34,424 train / 1,812 test；
- 冻结规模：2,048 train / 256 validation / 256 test；
- 过滤：严格角色状态机，总消息字符数不超过 1,600；
- 选择：canonical messages 精确去重，按内容 SHA-256 排序后固定取前缀；
- train SHA-256：`fe86a098652848bbae08652951d1da6bc3efed69e3c2e9139218ab01f5ca34f5`。

处理拒绝 4,358 条过长 upstream train 和 222 条过长 upstream test；三个冻结 split 的内容哈希无交叉。该确定性小子集只用于方法教学和短训练，不支持总体分布或 benchmark 结论。

## 模板与 Tokenization

- 模型：Qwen3-0.6B-Base 的固定 tokenizer revision；
- 模板：分段 Qwen ChatML；
- 直接监督：所有 assistant content 与其 `<|im_end|>`；
- 不直接监督：system/user、role prefix、换行和 padding；
- 最大长度：512；截断后无 assistant target 的样本 fail-fast；
- baseline 不做 packing。

## 保留集

Stage 4 复用 Stage 3 TinyStories validation 的固定前 32 条作为同模型前后 causal-loss 哨兵。该指标只比较 Qwen Base 与同一 Qwen Adapter；不得与 Stage 3 的 320-vocab PPL 横向比较。

## 主要局限

- 规模小、英语和约束任务占主导；
- 只做精确内容去重，未做语义近重复或系统性 PII 审计；
- correctness 模板化程度高；
- SmolTalk 子集未证明代表真实用户分布；
- 字符上限会改变长度分布；
- 没有安全、偏见、事实性和多语言完整评测。

---

# Data Card：Stage 5 偏好约束数据

## 来源与用途

- 名称：`stage5_preference_constraints_v1`；
- 来源：ForgeLLM 项目原创确定性生成；
- 许可证：project-original educational data；
- 用途：Preference Schema、verifier、RM、DPO 与极小 GRPO 方法教学；
- 不用于真人偏好、开放对话、安全或通用推理结论。

## 任务与切分

- 512 个任务，覆盖精确标识符、整数加法、紧凑 JSON、整数排序、ASCII 反转；
- rejected 覆盖错误答案、格式错误、额外解释、截断、重复和长度投机；
- 在 rejected 构造前按任务语义 SHA-256 排序并切成 384 train / 64 validation / 64 test；
- train/validation/test SHA-256 分别为 `c2326f8d...1545`、`d563bdde...267d`、`1e818038...315d`；
- prompt fingerprint 跨 split 无交叉。

## Verifier 与局限

训练 total reward 只接受 byte-exact 回答；non-empty、no-extra、no-repetition、structural 作为分离审计字段。验证集的 320 个“正确前缀+垃圾/包装/重复/空格”攻击全部被 strict total 拒绝。

数据高度模板化、答案短、主要为英语指令；没有自然语言偏好噪声、多人标注、一致性统计或安全维度。DPO 在该数据上取得 pair accuracy 1.0 不能外推为开放域偏好能力。

---

# Data Card：Stage 6 冻结综合评测集 v2

## 构成与身份

- 16 条 Stage 4 correctness test 原题；
- 16 条 Stage 5 preference test 原题（按冻结文件顺序取前 16）；
- 每条原题对应一条确定性等义重述，共 32 条 robustness；
- 总计 64 条，案例 SHA-256：`8cb746429bb576bf45a36a2f5ddcc159c57a7215139e918e973e1505a4fe8312`；
- 文件：`data/processed/stage6_evaluation_v2/cases.jsonl`；
- 只允许 test split；生成预算和评分规则不能在看到答案后调整。

## 多轮修正

初版 v1 删除了所有历史 assistant turn，使两个多轮任务形成连续 user 消息并触发严格 Schema。v1 数据和失败运行作为诊断证据保留。v2 只隐藏最后一个待预测 assistant answer，保留此前完整对话历史；因此 v1/v2 哈希不同，结果禁止拼接。

## 污染审计

- 规范化：NFKC、Unicode casefold、空白合并；
- 精确：规范化文本 SHA-256；
- 近重复：字符 13-gram Jaccard，阈值 0.8；
- 结构化训练候选：2,880 条；另对 Stage 3 训练文本做精确哈希筛查；
- 精确重合：0；
- 近重复候选：16，集中在两个多轮 correctness test 与同模板、不同槽位的训练记录，相似度约 0.85；
- 候选完整记录在 `preparation_report.json`，不静默删除。

## 适用范围与局限

该数据用于检验本项目的精确格式、短答案、偏好约束和局部措辞鲁棒性。它高度模板化、规模小、主要为英语，不覆盖开放知识、长推理、代码执行、多语言、安全、偏见或工具代理。0 个精确重合只适用于已列出的本地训练文件和规范化规则；16 个模板近重复限制了向开放任务外推的能力。
