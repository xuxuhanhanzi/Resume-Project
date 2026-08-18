# ForgeMM v0.2.3

本地实验闭环版本。训练数据 v2 使用 `datasets/...` 可迁移路径，1,763 个图片引用与
3,526 个 gold completion 全量通过；E3/E4/E5 的三种子 CPU 算法诊断和 dual checkpoint
恢复通过。真实 Qwen2.5-VL/ms-swift 训练与质量实验仍必须在 Linux 24GB+ GPU 上执行。
