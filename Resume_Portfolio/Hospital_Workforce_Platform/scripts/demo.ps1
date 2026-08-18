param(
    [string]$ApiBaseUrl = 'http://localhost:8080',
    [string]$KeycloakBaseUrl = 'http://localhost:8081',
    [string]$PrometheusBaseUrl = 'http://localhost:9090',
    [string]$GrafanaBaseUrl = 'http://localhost:3000'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$demoPassword = if ($env:HOSPITAL_DEMO_PASSWORD) { $env:HOSPITAL_DEMO_PASSWORD } else { 'Password1!' }

function Assert-Equal($Expected, $Actual, [string]$Label) {
    if ($Expected -ne $Actual) { throw "$Label expected $Expected but received $Actual" }
}

function Wait-Healthy([string]$Uri, [int]$Attempts = 45) {
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -SkipHttpErrorCheck -Uri $Uri
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw "Timed out waiting for $Uri"
}

function Get-DemoToken([string]$Username) {
    $token = Invoke-RestMethod -Method Post `
        -Uri "$KeycloakBaseUrl/realms/hospital/protocol/openid-connect/token" `
        -ContentType 'application/x-www-form-urlencoded' `
        -Body @{ client_id = 'hospital-api'; grant_type = 'password'; username = $Username; password = $demoPassword }
    return $token.access_token
}

function Invoke-JsonRequest(
    [string]$Method,
    [string]$Uri,
    [string]$Token = '',
    [hashtable]$Body = $null,
    [hashtable]$ExtraHeaders = @{}
) {
    $headers = @{}
    if ($Token) { $headers.Authorization = "Bearer $Token" }
    foreach ($key in $ExtraHeaders.Keys) { $headers[$key] = $ExtraHeaders[$key] }
    $parameters = @{ Method = $Method; Uri = $Uri; Headers = $headers; SkipHttpErrorCheck = $true }
    if ($null -ne $Body) {
        $parameters.ContentType = 'application/json'
        $parameters.Body = $Body | ConvertTo-Json -Compress
    }
    return Invoke-WebRequest @parameters
}

Wait-Healthy "$ApiBaseUrl/actuator/health"
Wait-Healthy "$KeycloakBaseUrl/realms/hospital"
Wait-Healthy "$PrometheusBaseUrl/-/ready"
Wait-Healthy "$GrafanaBaseUrl/api/health"

$frontend = Invoke-WebRequest -SkipHttpErrorCheck -Uri 'http://localhost:5173/'
Assert-Equal 200 $frontend.StatusCode 'frontend status'

$targets = Invoke-RestMethod -Uri "$PrometheusBaseUrl/api/v1/targets"
$backendTarget = $targets.data.activeTargets | Where-Object { $_.labels.job -eq 'hospital-backend' }
Assert-Equal 'up' $backendTarget.health 'Prometheus backend target'

$hrToken = Get-DemoToken 'hr.admin'
$employeeToken = Get-DemoToken 'employee'
$auditorToken = Get-DemoToken 'auditor'
$stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$departmentCode = "DEMO-$stamp"
$departmentPayload = @{ code = $departmentCode; name = 'Emergency Demo' }

$anonymous = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/departments" -Body $departmentPayload
Assert-Equal 401 $anonymous.StatusCode 'anonymous department creation'

$employeeDenied = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/departments" $employeeToken $departmentPayload
Assert-Equal 403 $employeeDenied.StatusCode 'EMPLOYEE department creation'

$departmentResponse = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/departments" $hrToken $departmentPayload
Assert-Equal 200 $departmentResponse.StatusCode 'HR_ADMIN department creation'
$department = $departmentResponse.Content | ConvertFrom-Json

$employeeNo = "N-$stamp"
$employeeResponse = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/employees" $hrToken @{
    employeeNo = $employeeNo
    fullName = 'Ava Demo'
    departmentId = $department.id
    jobTitle = 'Nurse'
    qualification = 'RN'
    baseSalary = '5000.00'
}
Assert-Equal 200 $employeeResponse.StatusCode 'HR_ADMIN employee creation'
$createdEmployee = $employeeResponse.Content | ConvertFrom-Json

$shiftPayload = @{
    employeeId = $createdEmployee.id
    startAt = '2028-01-10T08:00:00Z'
    endAt = '2028-01-10T16:00:00Z'
    requiredQualification = 'RN'
}
$idempotencyKey = "demo-shift-$stamp"
$firstShiftResponse = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/shifts" $hrToken $shiftPayload @{ 'Idempotency-Key' = $idempotencyKey }
$replayResponse = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/shifts" $hrToken $shiftPayload @{ 'Idempotency-Key' = $idempotencyKey }
Assert-Equal 200 $firstShiftResponse.StatusCode 'first shift creation'
Assert-Equal 200 $replayResponse.StatusCode 'idempotent shift replay'
$firstShift = $firstShiftResponse.Content | ConvertFrom-Json
$replayedShift = $replayResponse.Content | ConvertFrom-Json
Assert-Equal $firstShift.id $replayedShift.id 'idempotent shift id'

$overlap = Invoke-JsonRequest 'Post' "$ApiBaseUrl/api/shifts" $hrToken $shiftPayload
Assert-Equal 409 $overlap.StatusCode 'overlapping shift creation'

$audit = Invoke-JsonRequest 'Get' "$ApiBaseUrl/api/audit-logs?size=100" $auditorToken
Assert-Equal 200 $audit.StatusCode 'AUDITOR audit read'
if ($audit.Content -notmatch [regex]::Escape($departmentCode)) { throw 'Audit log does not contain the demo department' }
if ($audit.Content -notmatch [regex]::Escape($employeeNo)) { throw 'Audit log does not contain the demo employee' }

[pscustomobject]@{
    Backend = 'UP'
    Frontend = 'UP'
    Keycloak = 'UP'
    PrometheusTarget = 'up'
    Grafana = 'UP'
    AuthorizationMatrix = '401/403/200'
    EmployeeCreated = $createdEmployee.id
    IdempotentShift = $firstShift.id
    OverlapStatus = 409
    AuditTrail = 'verified'
    AccessTokensPrinted = $false
}
