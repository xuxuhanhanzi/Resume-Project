# 实验记录：H5 独立仓库与干净克隆复现

## 1. 冻结对象

- 独立 Git 仓库分支：`main`；
- 首个冻结提交：`b374fccec15b2c9a27c7a3efba6f557cba5af3c5`；
- 首个 annotated tag：`v1.0.0`；
- 测试日期：2026-08-14。

仓库建立前执行了已暂存内容的常见私钥/令牌模式扫描；未提交 `.env`、`.tools`、
`target`、`node_modules` 或 `dist`。

## 2. 干净克隆

```powershell
git clone --branch v1.0.0 `
  D:\Users\27475\Desktop\Resume_Project\Resume_Portfolio\Hospital_Workforce_Platform `
  $env:TEMP\hospital-clean-clone-b374fcc
```

构建与测试仅复用外部 JDK 21、Maven 3.9.9、Docker Desktop 和 npm 工具链，不复用原仓库
源码、`target`、`dist` 或 `node_modules`。

## 3. 结果

| 门槛 | 结果 |
|---|---|
| `npm ci` | 34 packages，0 vulnerabilities |
| `npm run build` | Vite 6.4.3，10 modules，2.31 秒 |
| `mvn --batch-mode test` | 7/7，0 failure/error/skipped，27.237 秒 |
| 集成数据库 | Testcontainers MySQL 8.4.11 |
| `docker compose config --quiet` | 通过 |
| `git status --short` | 空，构建后克隆仍干净 |

## 4. 发布结论

`v1.0.0` 在干净克隆中可重建并通过测试。本记录作为验证后提交进入 `v1.0.1`；不回写、
不移动旧标签。远程托管与公开视频属于外部发布动作，不在本地复现结论中伪装为已完成。
