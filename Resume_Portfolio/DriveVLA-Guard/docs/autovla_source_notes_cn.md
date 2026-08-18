# AutoVLA 源码阅读笔记

> 上游：`ucla-mobility/AutoVLA`  
> 冻结 commit：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`  
> 审计日期：2026-08-11

## 1. 模型输入

`models/autovla.py::AutoVLA.get_prompt()` 读取前、前左、前右三路相机，每路四帧，并加入车辆速度、加速度和导航命令。它用 Qwen2.5-VL processor 生成视频 token 与文本 token。当前官方 nuPlan 配置使用 10 个未来 pose、0.5 秒间隔、5 秒时域。

## 2. Action Token

`models/action_tokenizer.py::ActionTokenizer` 从 `agent_vocab.pkl` 读取形状为 `[vocab, 6, 4, 2]` 的车辆轮廓 codebook。每个 action token 表示相对上一状态的局部轨迹片段；rollout 逐 token 旋转和平移轮廓，用最后一帧轮廓中心与朝向恢复连续 pose。

本项目的 `ActionCodebookCodec` 使用 NumPy 独立实现相同 rollout，并对 token 范围、输入维度和空序列做显式校验。它不复制上游 codebook，真实运行时从用户自己的 AutoVLA checkout 加载。

## 3. fast/slow thinking

官方 `AutoVLA.get_prompt()` 通过 `use_cot` 切换 system prompt：fast 直接预测 action，slow 允许场景分析、关键对象、行为预测、ego 意图和最终动作。官方模型并不是两个独立网络，而是同一个 Qwen2.5-VL backbone 的两种生成提示/训练模式。

本项目 adapter 在每次生成前临时切换 `model.use_cot`，完成后恢复原值，避免模式污染下一场景。

## 4. 官方推理接口的限制

`AutoVLA.predict()`：

- 使用配置中的采样参数；
- 一次只返回一条序列；
- 只保留 `token_id >= action_start_id`；
- 没有返回 token log-prob；
- action token 数不足时缺少稳定错误契约。

本项目因此直接调用 `vlm.generate()`，顺序生成 K 条候选以控制 KV-cache 峰值，限制 token 必须落在 codebook 的闭区间，并保存平均 transition log-prob、raw text、动作 token、错误和精确延迟。

## 5. SFT 与 GRPO

`SFTAutoVLA.training_step()` 在语言模型 loss 之外识别 action token 区域，并对 CoT 样本提高 loss 权重。`GRPOAutoVLA` 对生成轨迹计算 PDM reward，以组内 reward 标准化得到 advantage，并加入 reference policy KL；如果输出复杂场景 CoT，还会加入长度相关 penalty，目标之一是减少不必要推理。

本项目不重新训练这些模块，只审计调用链并利用官方 checkpoint。

## 6. NAVSIM 接口

官方 `AutoVLAAgent.compute_trajectory()` 从预处理 JSON 构造 feature，调用模型并返回 NAVSIM `Trajectory`。官方 `run_pdm_score_cot.py` 在模型输出后从 metric cache 计算 PDMS。

本项目 `DriveVLAGuardAgent` 设置 `requires_scene=True`，但只取当前历史末帧 annotations；不会把 future frame、human future trajectory 或 metric cache 传给 reranker。

## 7. 审计发现

- 官方 evaluation shell 中 `CONFIG_PATH="$./config/..."` 看起来包含多余 `$`，本项目运行脚本不沿用该写法；
- checkpoint Hugging Face 仓库约 16.3GB，模型卡缺少标准 license metadata；
- AutoVLA 源码许可证仅允许教育/学术研究并禁止转移衍生物，因此本仓库不提交其源码、权重或 codebook；
- 官方 reasoning data 仍未公开，不能声称复现完整训练数据。

