param(
    [int]$EmployeeCount = 50,
    [int]$ShiftsPerEmployee = 20,
    [string]$ApiBaseUrl = 'http://localhost:8080',
    [string]$KeycloakBaseUrl = 'http://localhost:8081'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($EmployeeCount -lt 1) { throw 'EmployeeCount must be positive' }
if ($ShiftsPerEmployee -lt 1 -or $ShiftsPerEmployee -gt 28) { throw 'ShiftsPerEmployee must be between 1 and 28' }

$demoPassword = if ($env:HOSPITAL_DEMO_PASSWORD) { $env:HOSPITAL_DEMO_PASSWORD } else { 'Password1!' }
$token = (Invoke-RestMethod -Method Post `
    -Uri "$KeycloakBaseUrl/realms/hospital/protocol/openid-connect/token" `
    -ContentType 'application/x-www-form-urlencoded' `
    -Body @{ client_id = 'hospital-api'; grant_type = 'password'; username = 'hr.admin'; password = $demoPassword }).access_token
$headers = @{ Authorization = "Bearer $token" }
$stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()

$department = Invoke-RestMethod -Method Post -Uri "$ApiBaseUrl/api/departments" -Headers $headers `
    -ContentType 'application/json' -Body (@{ code = "PERF-$stamp"; name = 'Performance Dataset' } | ConvertTo-Json -Compress)

$createdShifts = 0
for ($employeeIndex = 1; $employeeIndex -le $EmployeeCount; $employeeIndex++) {
    $employee = Invoke-RestMethod -Method Post -Uri "$ApiBaseUrl/api/employees" -Headers $headers `
        -ContentType 'application/json' -Body (@{
            employeeNo = "PERF-$stamp-$employeeIndex"
            fullName = "Performance Nurse $employeeIndex"
            departmentId = $department.id
            jobTitle = 'Nurse'
            qualification = 'RN'
            baseSalary = '5000.00'
        } | ConvertTo-Json -Compress)

    for ($day = 1; $day -le $ShiftsPerEmployee; $day++) {
        $dayText = $day.ToString('00')
        $shiftHeaders = @{
            Authorization = "Bearer $token"
            'Idempotency-Key' = "perf-$stamp-$employeeIndex-$dayText"
        }
        Invoke-RestMethod -Method Post -Uri "$ApiBaseUrl/api/shifts" -Headers $shiftHeaders `
            -ContentType 'application/json' -Body (@{
                employeeId = $employee.id
                startAt = "2026-09-${dayText}T08:00:00Z"
                endAt = "2026-09-${dayText}T16:00:00Z"
                requiredQualification = 'RN'
            } | ConvertTo-Json -Compress) | Out-Null
        $createdShifts++
    }
}

[pscustomobject]@{
    DepartmentId = $department.id
    Employees = $EmployeeCount
    ShiftsPerEmployee = $ShiftsPerEmployee
    TotalShifts = $createdShifts
    Window = '2026-09-01T00:00:00Z/2026-10-01T00:00:00Z'
    AccessTokenPrinted = $false
}
