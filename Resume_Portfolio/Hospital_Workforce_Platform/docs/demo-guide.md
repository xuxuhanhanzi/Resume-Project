# 演示指南

## 启动

```powershell
Copy-Item .env.example .env
docker compose up --build
```

首次启动会下载容器与 Maven/npm 依赖。等待 MySQL、Keycloak 和后端健康检查完成后再调用 API。

启动后可运行可重复的 P0 验收脚本；它验证七个服务、401/403/200 权限矩阵、幂等重放、
重叠冲突和审计轨迹，并且不会打印 access token：

```powershell
.\scripts\demo.ps1
```

## 获取开发令牌

以下账号仅用于本地演示，生产环境必须移除：

| 用户 | 角色 | 密码 |
| --- | --- | --- |
| `hr.admin` | `HR_ADMIN` | `Password1!` |
| `manager` | `DEPARTMENT_MANAGER` | `Password1!` |
| `finance` | `FINANCE` | `Password1!` |
| `auditor` | `AUDITOR` | `Password1!` |
| `employee` | `EMPLOYEE` | `Password1!` |

```powershell
curl.exe -X POST http://localhost:8081/realms/hospital/protocol/openid-connect/token `
  -H "Content-Type: application/x-www-form-urlencoded" `
  -d "client_id=hospital-api" `
  -d "grant_type=password" `
  -d "username=hr.admin" `
  -d "password=Password1!"
```

将响应中的 `access_token` 保存为环境变量后，携带 `Authorization: Bearer <token>` 调用 API。

## 演示顺序

1. 以人事管理员身份创建急诊科及一名具有 `RN` 资质的护士。
2. 为护士创建草稿班次并发布；再次创建时间重叠的班次，确认收到 `409`。
3. 并行提交相同时间段的创建请求，确认仅一条成功。
4. 创建并审批请假，尝试在请假区间创建班次，确认收到 `409`。
5. 创建工资单，依次提交、审批和支付；在错误顺序审批时确认收到 `409`。
6. 使用审计员查询 `/api/audit-logs`，展示完整操作轨迹。
7. 打开 Prometheus 与 Grafana，展示 JVM、HTTP 和数据库连接池指标。

## 性能测试

先生成固定规模的真实数据：

```powershell
$seed = .\scripts\seed-performance.ps1 -EmployeeCount 50 -ShiftsPerEmployee 20
$env:DEPARTMENT_ID = [string]$seed.DepartmentId
$env:ACCESS_TOKEN = '<HR_ADMIN access token>'
docker compose --profile perf run --rm load-test `
  run --summary-export=/results/formal-summary.json /scripts/scheduling.js
```

正式复现结果与边界见
[`docs/experiments/2026-08-14_h4_k6_load_test.md`](experiments/2026-08-14_h4_k6_load_test.md)。
