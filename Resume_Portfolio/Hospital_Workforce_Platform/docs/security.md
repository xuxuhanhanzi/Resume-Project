# 安全模型

Keycloak 负责认证和签发 JWT，Spring Boot 作为 OAuth2 Resource Server 验证令牌。角色由 `realm_access.roles` 映射为 `ROLE_*` 权限。

| 角色 | 权限范围 |
| --- | --- |
| SYSTEM_ADMIN | 系统配置；不自动拥有薪资读取权 |
| HR_ADMIN | 科室、员工、招聘、培训和工资单提交 |
| DEPARTMENT_MANAGER | 仅维护本人科室的排班 |
| FINANCE | 生成、审批和支付工资单 |
| EMPLOYEE | 仅查看本人的排班和工资单 |
| AUDITOR | 只读访问审计记录 |

数据权限不能只依赖 URL：服务层必须校验当前用户是否属于目标科室或是否为目标员工本人。第一版保留了角色级检查和接口边界，后续可在员工与 Keycloak subject 建立映射后启用完整行级检查。

密码、JWT、数据库凭据不应进入代码、日志或 Git 仓库。
