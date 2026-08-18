# R7：v1.0.0 干净克隆发布验收

## 范围

- 源 commit：`98a2f73f0caeab8ed76bc4d1696e53a04554cfcd`；
- 源 tag：`v1.0.0`；
- 克隆目录：`D:\Temp\drivevla-clean-98a2f73-20260814-b`；
- Python：3.12.3；
- 依赖：`requirements-dev.lock` 固定版本；
- 安装：锁文件后执行 `pip install --no-build-isolation --no-deps -e .`。

## 结果

- Ruff format：28 files already formatted；
- Ruff lint：通过；
- pytest：21/21 通过；
- compileall：通过；
- `verify-evidence`：`formal_v5` 全部语义检查与 SHA-256 哈希通过；
- 新鲜 B0/B1/E1–E4：72/72 场景成功，指标与冻结结果一致；
- Git：精确匹配 `v1.0.0`，检查后工作树无未跟踪/修改文件（生成目录均按设计忽略）。

## 结论

本地发布可以脱离原工作目录重建。该结论只覆盖 CPU 合成工程链路；官方 AutoVLA/NAVSIM 仍受 GPU、权重和受限数据外部门禁约束。

