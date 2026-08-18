# 实验记录：H0 Java 21 容器构建基线

## 1. 目标

确认当前源码能够在固定 Maven/Java 21 容器中编译、打包为 Spring Boot JAR，并生成非 root
运行镜像；同时记录本机缺少 Java/Maven 时的可复现替代入口。

## 2. 环境

- 平台：Windows + Docker Desktop Linux containers
- Docker Desktop：29.6.1
- Builder：`maven:3.9.9-eclipse-temurin-21`
- Runtime：`eclipse-temurin:21-jre`
- 代码路径：`D:\Users\27475\Desktop\Resume_Project\Resume_Portfolio\Hospital_Workforce_Platform`
- 数据集：不适用

## 3. 实验变量

- 主变量：在 Docker BuildKit 中执行 Maven/Java 21 构建；
- 固定项：`pom.xml`、55 个主源文件、3 个测试源文件；
- 排除项：本阶段使用 `-DskipTests`，不把测试编译等同于测试执行。

## 4. 命令

```powershell
docker build --network=host --progress=plain `
  -t hospital-workforce-platform:verification .
```

## 5. 输出

- 镜像：`hospital-workforce-platform:verification`
- Manifest list digest：`sha256:783cf510904e63049eed4fd8191d1df4f1eb76e83ccae1636bf6c120ffe25a1b`
- Spring Boot JAR：构建阶段 `/workspace/target/hospital-workforce-platform-0.1.0-SNAPSHOT.jar`

## 6. 结果

| 验收项 | 结果 |
|---|---|
| Java release | 21 |
| 主源文件编译 | 55/55 |
| 测试源文件编译 | 3/3 |
| Spring Boot repackage | 通过 |
| Runtime image export | 通过 |
| 测试执行 | 未执行，本阶段不计 |

## 7. 失败与异常

- 前三次构建均在 Maven Central 下载中出现 `Premature end of Content-Length`；
- Dockerfile 增加 BuildKit Maven repository cache 和有限 HTTP retry 后，成功依赖能够跨重试保留；
- 最终使用 `--network=host` 完成剩余依赖下载、Java 编译与镜像导出；
- 该问题属于网络传输，不是 Java 编译错误。失败记录不删除。

## 8. 结论

当前后端源码和测试源码能够在 Java 21 下编译，Spring Boot 可执行 JAR 与非 root runtime
镜像能够生成。H0 构建基线通过，但 Testcontainers、Keycloak、Compose 和 k6 尚未验证，
项目不能据此标记为完成。

## 9. 下一步

构建可复用的 Maven 测试运行入口，挂载 Docker socket 运行 MySQL Testcontainers；随后补齐
授权矩阵、幂等与审计测试。
