# 实验记录：H1-H2 后端、容器数据库与安全闭环

## 1. 目标

在 Java 21 与真实 MySQL 容器上执行后端测试，验证模块架构、工资单状态机、并发排班、
幂等重放、审计记录以及 HTTP 认证/角色授权，不以测试编译或 mock 数据库代替运行结果。

## 2. 环境

- Windows + Docker Desktop 4.82.0；
- Docker Engine 29.6.1，API 1.55；
- Eclipse Temurin 21.0.12+8；
- Apache Maven 3.9.9；
- Testcontainers 2.0.5；
- MySQL 8.4 Testcontainer；
- Spring Boot 3.5.3。

## 3. 兼容性修复

原项目固定 Testcontainers 1.20.6，其 docker-java 客户端无法连接最低 API 为 1.40 的
Docker Engine 29.6.1，`/info` 返回 HTTP 400。升级至 Testcontainers 2.0.5，并按 2.x
迁移模块坐标与 `MySQLContainer` 包名后，客户端成功连接并协商 Docker API 1.55。

Windows 运行时显式使用当前 Docker Desktop Linux context：

```powershell
$env:DOCKER_HOST = 'npipe:////./pipe/dockerDesktopLinuxEngine'
```

## 4. 最终命令

```powershell
$env:JAVA_HOME = (Resolve-Path '.tools\jdk\jdk-21.0.12+8').Path
$env:MAVEN_OPTS = '-Dmaven.repo.local=' + (Resolve-Path '.tools').Path + '\m2'
$env:DOCKER_HOST = 'npipe:////./pipe/dockerDesktopLinuxEngine'
& '.tools\maven\apache-maven-3.9.9\bin\mvn.cmd' --batch-mode test
```

## 5. 结果

| 测试类/验收点 | 结果 |
|---|---|
| `ArchitectureTest`：模块架构 | 1/1 通过 |
| `PayslipTest`：工资单状态机 | 1/1 通过 |
| 并发重叠排班仅一个事务成功 | 通过 |
| 相同 `Idempotency-Key` 返回原 Shift | 通过 |
| 创建资源写入可查询审计记录 | 通过 |
| Department API：匿名 401、EMPLOYEE 403、HR_ADMIN 200 | 通过 |
| Redis 默认序列化：Department 可写入并读取缓存 | 通过 |
| Flyway V1 迁移 + Hibernate validate | 通过 |
| 总计 | **7/7，0 failure，0 error，0 skipped** |

最终 Maven 结果：`BUILD SUCCESS`，总耗时 58.033 秒；集成测试使用真实 MySQL 8.4，
未使用 H2/in-memory 数据库。

## 6. 固化报告

Surefire 文本报告位于 `docs/experiments/artifacts/h1-h2-surefire/`：

| 文件 | SHA-256 |
|---|---|
| `com.hospital.workforce.ArchitectureTest.txt` | `A2D61273FA8AFF9C794A2559C8037A6C4AEFBF04BE6BA863698F77DD8C303F86` |
| `com.hospital.workforce.payroll.PayslipTest.txt` | `A36FD5353E4872D805B8335D24CAB79FF41006B97DE37A7CC1A7F7083E91CDC4` |
| `com.hospital.workforce.SchedulingConcurrencyIntegrationTest.txt` | `0CC21953B6BA748ED4B800BFDD24DF85C5B500E8F7FD9ED9D65B70D5C03BF8A1` |

## 7. 保留的异常记录

- 容器内挂载 Windows Docker socket 与旧 Testcontainers 均在环境探测阶段失败；
- 主机首次解析 Maven 依赖出现多个 `Premature end of Content-Length`，通过复用已成功
  构建镜像中的 Maven 缓存完成，不删除失败记录；
- Flyway 输出 MySQL 8.4 高于其已测试的 8.1 警告，但迁移、验证与全部业务断言均通过；
- Mockito 动态加载 agent 输出面向未来 JDK 的告警，本次 Java 21 测试不受影响。

## 8. 结论与下一步

H1 与 H2 已通过。Redis 回归测试同时覆盖了 H3 真实环境中发现的缓存序列化问题，避免
仅在无缓存的测试路径中通过。
