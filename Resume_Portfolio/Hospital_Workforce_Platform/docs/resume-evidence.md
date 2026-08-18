# 简历与面试证据清单

本项目应与 2021 年实习项目分开表述：前者是现代化独立重构，后者仅保留当时真实参与内容。

## 可在验证后使用的表述模板

- Reengineered a hospital HR system into a Java 21/Spring Boot modular monolith covering workforce, scheduling, payroll, recruitment, training, audit, and role-based access control.
- Designed a concurrency-safe scheduling workflow using employee-row pessimistic locking, overlap validation, database constraints, optimistic versions, and idempotency keys.
- Implemented immutable payroll snapshots and an approval state machine with `BigDecimal`-based compensation rules.
- Established database-backed integration testing, containerized deployment, API documentation, metrics, and load-test scripts.
- 在单机 Docker Compose、50 名员工与 1,000 条排班的已认证查询场景中，以 k6 逐步升至
  100 VU，完成 16,746 次请求且错误率 0%，吞吐 55.70 req/s，p95 5.83 ms；该表述不得
  延伸为写吞吐或生产容量结论。

## 需要保留的证据

- Git 提交历史；
- 架构图、ER 图和 ADR；
- Testcontainers 测试报告；
- 并发测试结果；
- k6 原始结果与 Grafana 截图；
- Docker Compose 启动截图；
- 2–3 分钟演示视频。

已完成的正式性能数字见 `docs/experiments/2026-08-14_h4_k6_load_test.md`。CPU/RAM 为两次
瞬时采样，不应描述为峰值或稳定上限。
