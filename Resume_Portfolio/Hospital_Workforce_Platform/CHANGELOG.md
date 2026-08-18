# Changelog

## 1.0.1 - 2026-08-14

- 在独立临时目录完成 `v1.0.0` 干净克隆复现；
- 前端锁定依赖生产构建、后端 7/7 MySQL Testcontainers 测试和 Compose 配置均通过；
- 增加可定位到提交和命令的 H5 发布验证记录。

## 1.0.0 - 2026-08-14

- 完成组织、人事、排班、请假/调班、薪资、招聘、培训、审计模块；
- 完成 Keycloak RBAC、MySQL/Flyway、Redis 缓存与 Spring Modulith 边界验证；
- 完成七服务 Docker Compose、Prometheus/Grafana 与 Vue/Nginx 生产构建；
- 7/7 自动化测试通过，覆盖真实 MySQL、并发冲突、幂等、审计和权限矩阵；
- 完成 50 名员工、1,000 条排班、最高 100 VU 的 5 分钟 k6 正式实验；
- 固化复现脚本、原始结果、哈希、ADR、架构图、ER 图和简历证据边界。
