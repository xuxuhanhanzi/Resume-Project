# 医院人员与排班管理平台：项目计划

## 项目定位

这是一个面向 Java 后端求职的独立重构项目。系统聚焦医院组织、人事、排班、薪资、招聘、培训、权限及审计，而不试图实现完整电子病历系统。

## 核心目标

- 用模块化单体展示 Java/Spring Boot 架构能力。
- 用排班冲突、并发控制、薪资快照和审批流程展示业务深度。
- 用 OAuth2/JWT、数据权限、审计、自动化测试、监控和压测展示工程能力。
- 以 Docker Compose、示例数据和文档让项目可复现。

## 技术栈

Java 21、Spring Boot、Spring Modulith、Spring Data JPA、Spring Security OAuth2 Resource Server、MySQL 8、Flyway、Redis、Keycloak、Vue 3、JUnit 5、Testcontainers、Micrometer、Prometheus、Grafana、k6、Docker Compose、GitHub Actions。

## 周期与里程碑

| 周次 | 目标 | 本仓库对应交付物 |
| --- | --- | --- |
| 第 1 周 | 明确边界、架构和开发环境 | `docs/`、Docker、Flyway、CI、Maven/Vue 骨架 |
| 第 2 周 | 组织与员工 | `organization`、`workforce` 模块与 API |
| 第 3 周 | 排班闭环 | `scheduling` 模块、班次创建与查询 |
| 第 4 周 | 并发与审批 | 悲观锁、幂等键、调班/请假、并发测试 |
| 第 5 周 | 薪资 | 计算快照、状态机、审批 API |
| 第 6 周 | 安全与审计 | Keycloak、角色校验、审计记录 |
| 第 7 周 | 工程化 | Redis、领域事件、监控、管理端页面 |
| 第 8 周 | 交付 | 集成/性能测试、报告、部署及简历材料 |

## P0 验收场景

1. 人事管理员创建科室与员工，并维护岗位和资质。
2. 科室主管为员工创建班次；离职员工、时间重叠或资质不足时必须拒绝。
3. 并发请求为同一员工分配重叠班次时，只允许一个请求成功。
4. 薪资模块按员工基础工资、夜班数、绩效和奖惩生成不可变工资单快照。
5. 工资单只能按照 `DRAFT → SUBMITTED → APPROVED → PAID` 流转。
6. 不同角色仅能访问授权的资源；所有写操作生成审计记录。
7. 项目可通过 Docker Compose 启动；关键流程有 MySQL Testcontainers 集成测试。

## 非目标

- 不在第一版实现完整病历、药房、住院、医保或支付。
- 不为了技术展示强行拆分十余个微服务。
- 不以 Redis 或消息队列替代数据库的事务和唯一约束。
- 不将后续个人重构工作表述为 2021 年实习期间完成。

## 完成定义

每一项功能均需具备数据库迁移、接口契约、参数校验、权限控制、错误响应、单元或集成测试、审计记录和文档更新。

## 后续可选扩展

- FHIR R4 Patient/Practitioner/Appointment 适配层。
- Transactional Outbox + RabbitMQ/Kafka。
- 多院区数据隔离。
- 排班规则求解与人员配置优化。
