# Hospital Workforce and Scheduling Platform

WSL2 实验 PC 可通过 `scripts/setup_experiment_pc.sh` 构建服务，再由
`scripts/run_experiment_pc.sh` 自动执行 Compose 验收与 k6 压测；作品集根目录的统一编排器
会负责调用并保存阶段日志。

面向 Java 后端求职的医院人员与排班管理平台。项目以模块化单体实现组织人事、排班、薪资、招聘、培训、权限和审计，并提供容器化、本地监控、自动化测试及性能测试脚本。

## 核心能力

- **组织与人员**：科室、员工、岗位、资质及员工生命周期。
- **并发安全排班**：按员工行加锁、时间窗冲突检查、幂等键和数据库约束共同避免重复或重叠班次。
- **薪资审批**：使用 `BigDecimal` 计算薪资，生成不可变快照并以状态机控制审批。
- **安全与审计**：Keycloak JWT、角色校验、操作审计和敏感信息隔离。
- **工程化**：Flyway、Redis 缓存、Spring Modulith、Testcontainers、Actuator、Prometheus/Grafana、k6、GitHub Actions。

## 快速开始

需要 Docker Desktop。复制 `.env.example` 为 `.env` 后运行：

```powershell
docker compose up --build
```

服务就绪后执行一键验收：

```powershell
.\scripts\demo.ps1
```

服务地址：

- API: `http://localhost:8080`
- Swagger UI: `http://localhost:8080/swagger-ui/index.html`
- Keycloak: `http://localhost:8081`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Web: `http://localhost:5173`

## 文档

- [项目计划](PROJECT_PLAN.md)
- [架构说明](docs/architecture.md)
- [ADR-0001：模块化单体](docs/adr/0001-modular-monolith.md)
- [领域与 API](docs/domain.md)
- [安全模型](docs/security.md)
- [测试与性能策略](docs/testing-and-performance.md)
- [演示指南](docs/demo-guide.md)
- [3 分钟演示分镜](docs/demo-storyboard.md)
- [H1-H2 后端测试记录](docs/experiments/2026-08-14_h1_h2_backend_tests.md)
- [H3 Compose 全链路记录](docs/experiments/2026-08-14_h3_compose_e2e.md)
- [H4 k6 正式压测记录](docs/experiments/2026-08-14_h4_k6_load_test.md)
- [H5 干净克隆与发布记录](docs/experiments/2026-08-14_h5_clean_clone_release.md)

## 项目真实性说明

本仓库是对早期医院人事管理实习项目的独立现代化重构。请将原实习经历与本项目的后续重构成果在简历中分开表述。

版本化变更见 [CHANGELOG](CHANGELOG.md)。当前冻结证据版本为 `v1.0.1`。
