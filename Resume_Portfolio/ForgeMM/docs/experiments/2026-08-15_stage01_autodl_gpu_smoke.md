# 实验记录：stage01_autodl_gpu_smoke

## 1. 目标

在 AutoDL RTX 4090 上关闭 ForgeMM Stage 1 的真实多模态推理、QLoRA SFT、
`4 prompts × 2 rollouts` GRPO、额外字段奖励透传与 checkpoint 保存门，并保留依赖失败证据。

## 2. 环境

- 平台：AutoDL，Ubuntu 22.04；
- GPU：NVIDIA GeForce RTX 4090，24,564 MiB，driver 560.35.03；
- Python：3.12.3；PyTorch：2.6.0+cu124；CUDA runtime：12.4；
- 训练栈：ms-swift 4.2.2、Transformers 4.51.3、TRL 0.26.2、PEFT 0.19.1、
  bitsandbytes 0.45.5、qwen-vl-utils 0.0.14；
- 环境：`/root/autodl-tmp/envs/forgemm`；
- 代码：`/root/autodl-tmp/ForgeMM`；
- 模型：`/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct`；
- 数据：`artifacts/runs/stage04_training_data_v2/`，ChartQA/ChartQAPro 冻结哈希全部通过。

最终环境见 `artifacts/runs/cloud_stage00/environment_gpu_final.json` 和
`runtime_lock_final.json`。TRL 声明要求 Transformers >=4.56.1，但该版本在当前
Qwen2.5-VL 图像生成路径产生 attention/cache 维度回归；项目使用显式环境变量
`FORGEMM_TRL_TRANSFORMERS_451=1` 提供两个不改变算法的接口桥接，并保留
`pip_check_final.txt`。

## 3. 实验变量

- 推理：原始模型与 50-step Structured SFT adapter；
- SFT：NF4 4-bit QLoRA，LoRA rank 8 / alpha 16，冻结视觉塔与 aligner；
- GRPO：每组 2 个 rollout，三路 ForgeMM reward，先验证零奖励失败样例，再验证非恒定奖励；
- 固定项：相同底座、数据协议、最大图像像素、LoRA 结构和随机种子 42。

## 4. 关键命令

```bash
swift infer \
  --model /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --infer_backend pt \
  --val_dataset artifacts/runs/stage04_training_data_v2/structured_sft.jsonl \
  --val_dataset_sample 20 --max_new_tokens 160 --temperature 0 --stream false \
  --attn_impl sdpa \
  --result_path artifacts/runs/cloud_stage01/infer_chartqa20.jsonl

python /root/autodl-tmp/envs/forgemm/lib/python3.12/site-packages/swift/cli/sft.py \
  --model /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --dataset artifacts/runs/stage04_training_data_v2/structured_sft.jsonl#128 \
  --split_dataset_ratio 0 --tuner_type lora --quant_method bnb --quant_bits 4 \
  --bnb_4bit_quant_type nf4 --torch_dtype bfloat16 --attn_impl sdpa \
  --freeze_vit true --freeze_aligner true --target_modules all-linear \
  --lora_rank 8 --lora_alpha 16 --max_pixels 262144 --max_length 2048 \
  --learning_rate 5e-5 --max_steps 50 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --gradient_checkpointing true \
  --output_dir artifacts/runs/cloud_stage01/sft_quick50

FORGEMM_TRL_TRANSFORMERS_451=1 PYTHONPATH=src python \
  /root/autodl-tmp/envs/forgemm/lib/python3.12/site-packages/swift/cli/rlhf.py \
  --rlhf_type grpo --model /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapters artifacts/runs/cloud_stage01/sft_quick50/checkpoint-50 \
  --dataset artifacts/runs/stage04_training_data_v2/grpo.jsonl#4 \
  --external_plugins src/forgemm/swift_plugin.py \
  --reward_funcs forgemm_task forgemm_evidence forgemm_operation \
  --tuner_type lora --quant_method bnb --quant_bits 4 --bnb_4bit_quant_type nf4 \
  --num_generations 2 --generation_batch_size 2 --use_vllm false \
  --max_completion_length 160 --max_steps 4 --gradient_checkpointing false \
  --output_dir artifacts/runs/cloud_stage01/grpo_smoke_v22
```

## 5. 输出路径

- SFT：`artifacts/runs/cloud_stage01/sft_quick50/checkpoint-50`；
- GRPO：`artifacts/runs/cloud_stage01/grpo_smoke_v22/checkpoint-4`；
- 日志：`artifacts/runs/cloud_stage01/*.log`；
- 本机归档：`../cloud_packages/forgemm_cloud_results_20260815.tar.gz`；
- 归档 SHA-256：`3f552e29de8ade0f8eb14b37ed9acf6e50c9597d622714e416fa719c04897d28`。

## 6. 结果

| 指标 | 结果 |
|---|---:|
| 原始模型 20 样本推理 | 20/20 完成，0.899 samples/s |
| 50-step SFT 最终 token accuracy | 0.9889 |
| 50-step SFT 最终 loss | 0.0740 |
| SFT 后 4 样本完整结构输出 | 3/4 |
| GRPO step 1 reward / std | 0.5 / 0.7071 |
| GRPO step 3 reward / std | 1.5 / 2.1213 |
| GRPO 三路 reward 非零 | task/evidence/operation 均通过 |
| GRPO 峰值显存 | 6.28 GiB |
| GRPO checkpoint | checkpoint-4，完整 optimizer/rng/trainer state |
| 本机质量门 | Ruff、strict mypy、57 tests 全通过 |

## 7. 失败与异常

- Torch 2.5.1 缺少 ms-swift 需要的 `FSDPModule`，升级至 Torch 2.6.0 后 SFT 通过；
- TRL 0.17 不满足 ms-swift 4.2.2 的 GRPO 接口，升级至 TRL 0.26.2；
- Transformers 4.56.1 在普通单图推理与 GRPO 中均复现视觉 attention/cache 长度翻倍，
  回退到已通过基线的 4.51.3 后消失；
- vLLM 0.8.5 与当前 ms-swift/TRL 的权重同步和 LoRA 映射不兼容，因此冻结
  Transformers rollout；所有失败日志均保留，未覆盖；
- 2-step SFT 的 GRPO 奖励恒为零；增加到 50-step SFT 后出现非零且非恒定奖励。

## 8. 结论

Stage 1 真实 GPU 兼容性门已关闭：推理、QLoRA、三路字段透传、GRPO 优势训练与恢复型
checkpoint 均可执行。该结果证明训练链路可运行，不等于 E0-E5/A1-A2 正式质量矩阵完成，
也不支持性能提升或 SOTA 主张。

## 9. 下一步

正式 Stage 4 前应扩大 E1/E2 的冻结训练与验证集评测；只有 E2 达到预登记的格式、可执行率
和准确率门槛，才进入 E3-E5 多种子矩阵。
