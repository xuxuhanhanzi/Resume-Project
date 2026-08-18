# DriveVLA-Guard 云端上传包说明

## 包内已有

- DriveVLA-Guard `v1.0.2` 的全部可提交源码、配置、测试、文档和冻结本地证据；
- AutoDL bootstrap、官方资产 preflight、B0/E2/E3/E4 NAVSIM 启动脚本；
- 不含 `.git`、虚拟环境、缓存和本地大体积诊断输出。

## 包内刻意不含，必须在云端获取

1. AutoVLA 官方仓库：`https://github.com/ucla-mobility/AutoVLA.git`，必须 checkout
   `ba34eed74ce6729e7986592d0e66cbaca397b4fa`；
2. AutoVLA checkpoint：`https://huggingface.co/Zewei-Zhou/AutoVLA`；
3. Qwen2.5-VL-3B-Instruct：`https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`；
4. NAVSIM/nuPlan/OpenScene maps、logs、sensor blobs；
5. 与固定 split 对应的 metric cache；可按官方脚本生成，不要求从第三方下载；
6. AutoVLA 所需的预处理 JSON；可从官方数据运行 preprocessing 生成；
7. Conda/Python/CUDA 依赖及 Git LFS/Hugging Face 下载工具。

AutoVLA 上游代码和数据没有打包，是因为本地许可证审计将其标记为
`academic_noncommercial_nontransferable` / `redistribution_allowed=false`。请只从官方入口获取并
自行接受数据与模型条款，不要把本 ZIP 公开发布为包含第三方资产的镜像。

## 云端目录建议

```text
/root/autodl-tmp/
├── DriveVLA-Guard/
├── AutoVLA/
├── models/
│   ├── AutoVLA/<checkpoint.ckpt>
│   └── Qwen2.5-VL-3B-Instruct/
└── navsim_data/
    ├── maps/
    ├── navsim_logs/
    ├── sensor_blobs/
    ├── preprocessed_json/
    └── metric_cache/
```

## 上传后第一步

```bash
cd /root/autodl-tmp
sha256sum -c DriveVLA-Guard_cloud_v1.0.2.zip.sha256
unzip DriveVLA-Guard_cloud_v1.0.2.zip

git clone https://github.com/ucla-mobility/AutoVLA.git
git -C AutoVLA checkout ba34eed74ce6729e7986592d0e66cbaca397b4fa

conda env create -f AutoVLA/environment.yml
conda activate autovla
bash AutoVLA/install.sh
pip install -e AutoVLA --no-warn-conflicts
pip install -e AutoVLA/navsim --no-warn-conflicts
pip install -e DriveVLA-Guard
```

模型与数据路径准备好后，严格按 `DriveVLA-Guard/docs/official_runbook.md` 设置环境变量并
运行 `official-preflight`。只有输出 `ready=true` 才能开始正式模型加载和 NAVSIM 评测。

## 上传前后校验

- ZIP 总体 SHA-256 位于同名 `.sha256` 文件；
- 包内 `CLOUD_BUNDLE_MANIFEST.json` 记录每个文件的大小与 SHA-256；
- 解压后可运行 `python DriveVLA-Guard/scripts/verify_cloud_bundle.py DriveVLA-Guard`。
