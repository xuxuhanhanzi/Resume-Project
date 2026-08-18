# ForgeMM 云端上传包说明

## 包内已有

- ForgeMM `v0.2.3` 全部可提交源码、配置、测试、文档与证据；
- ChartQA 完整本地 payload：62,654 files，1,036,820,929 bytes；
- ChartQAPro：parquet + manifest + dataset card，209,487,866 bytes；
- EvidenceStore v7：human 580 + augmented 1,183 条唯一严格记录；
- Stage 4 v2：3,526 Structured SFT rows + 1,763 GRPO rows，使用 `datasets/...` 相对图片路径；
- v2 全量审计与输出哈希。

该 ZIP 仅用于你的私有云端实验，不要公开再分发其中的数据 payload；正式使用前仍需复核
ChartQA、ChartQAPro 与模型许可证。

## 上传后仍需下载/安装

1. Qwen2.5-VL-3B-Instruct 模型：
   `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`（官方仓库约 7.52GB）；
2. GPU 训练软件栈：CUDA 兼容的 PyTorch、Transformers、qwen-vl-utils、bitsandbytes、
   vLLM 与 `ms-swift==4.2.2`；
3. 系统工具：Git/Git LFS、编译工具和 Hugging Face 或 ModelScope 下载工具；
4. 可选实验跟踪工具。ZIP 不包含 pip wheels、Conda 包、Docker 镜像或模型缓存。

不需要再次下载 ChartQA、ChartQAPro、EvidenceStore 或 v2 SFT/GRPO 数据。

## 云端目录建议

```text
/root/autodl-tmp/
├── ForgeMM/
│   ├── datasets/
│   └── artifacts/runs/stage04_training_data_v2/
├── models/Qwen2.5-VL-3B-Instruct/
└── outputs/forgemm/
```

## 上传后第一步

```bash
cd /root/autodl-tmp
sha256sum -c ForgeMM_cloud_v0.2.3_with_data.zip.sha256
unzip ForgeMM_cloud_v0.2.3_with_data.zip
cd ForgeMM

python scripts/verify_cloud_bundle.py .
python scripts/audit_environment.py --project-root . --require-datasets \
  --output artifacts/runs/cloud_stage00/environment.json
python scripts/audit_training_datasets.py \
  --sft artifacts/runs/stage04_training_data_v2/structured_sft.jsonl \
  --grpo artifacts/runs/stage04_training_data_v2/grpo.jsonl \
  --output artifacts/runs/cloud_stage00/training_data_audit.json \
  --project-root .
```

随后先固定 Qwen model revision、ms-swift 4.2.2 commit、torch/CUDA/vLLM/bitsandbytes 版本，
按 `docs/implementation_plan.md` 依次执行单图推理、32 样本 QLoRA、4×2 GRPO 和字段透传
smoke。任一步失败都不得直接进入正式 E0-E5。

## 上传前后校验

- ZIP 总体 SHA-256 位于同名 `.sha256` 文件；
- 包内 `CLOUD_BUNDLE_MANIFEST.json` 记录每个文件的大小与 SHA-256；
- 解压后运行 `python ForgeMM/scripts/verify_cloud_bundle.py ForgeMM`。
