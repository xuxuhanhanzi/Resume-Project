[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [switch]$SkipTests,
    [switch]$PackageSmoke,
    [switch]$Baseline,
    [switch]$DockerSecurity,
    [switch]$ReleaseScope,
    [switch]$CIPreflight
)

# P20 release gate for an existing project-local virtual environment.  It is
# intentionally diagnostic: it never changes PATH, a PowerShell profile,
# credentials, project configuration, or package metadata.  Test caches and
# Python bytecode are disabled so the working tree is not dirtied by this gate.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
$cli = Join-Path $repository ".venv\Scripts\repopilot.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python was not found: $python. Run .\scripts\setup_windows.ps1 -WithDev first."
}
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Project-local launcher was not found: $cli. Run .\scripts\setup_windows.ps1 -WithDev first."
}

Set-Location -LiteralPath $repository
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

Write-Host "Checking installed dependency consistency..."
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed." }

Write-Host "Checking packaged synthetic fixture discovery..."
& $cli eval coding-dev plan
if ($LASTEXITCODE -ne 0) { throw "The bundled synthetic development fixture could not be loaded." }

Write-Host "Checking project-local launcher and session-only shell integration..."
& (Join-Path $PSScriptRoot "acceptance_windows.ps1") -ProjectRoot $repository -SkipTests
if ($LASTEXITCODE -ne 0) { throw "The Windows acceptance gate failed." }

$git = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $git) {
    Write-Host "Checking whitespace errors in the current diff..."
    & git -C $repository diff --check
    if ($LASTEXITCODE -ne 0) { throw "git diff --check reported a whitespace error." }
} else {
    Write-Warning "Git was not found; skipped git diff --check."
}

if (-not $SkipTests) {
    Write-Host "Running release lint, type and test gates..."
    & $python -m ruff check . --no-cache
    if ($LASTEXITCODE -ne 0) { throw "Ruff failed." }
    & $python -m mypy --no-incremental src tests scripts
    if ($LASTEXITCODE -ne 0) { throw "Mypy failed." }
    & $python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "Pytest failed." }
}

if ($PackageSmoke) {
    Write-Host "Running opt-in clean-install package smoke check..."
    & (Join-Path $PSScriptRoot "package_smoke_windows.ps1") -ProjectRoot $repository
    if ($LASTEXITCODE -ne 0) { throw "Clean-install package smoke check failed." }
}

if ($Baseline) {
    Write-Host "Recording a non-destructive release baseline..."
    & (Join-Path $PSScriptRoot "release_baseline_windows.ps1") -ProjectRoot $repository
    if ($LASTEXITCODE -ne 0) { throw "Release-baseline recording failed." }
}

if ($ReleaseScope) {
    Write-Host "Recording a non-destructive release-scope inventory..."
    & (Join-Path $PSScriptRoot "release_scope_windows.ps1") -ProjectRoot $repository
    if ($LASTEXITCODE -ne 0) { throw "Release-scope inventory failed." }
}

if ($CIPreflight) {
    Write-Host "Checking strict Docker and remote-CI prerequisites..."
    & (Join-Path $PSScriptRoot "ci_preflight_windows.ps1") -ProjectRoot $repository
    if ($LASTEXITCODE -ne 0) { throw "CI preflight reported an unresolved prerequisite." }
}

if ($DockerSecurity) {
    Write-Host "Running the explicit Docker security gate..."
    & (Join-Path $PSScriptRoot "docker_security_gate_windows.ps1") -ProjectRoot $repository -BuildImage
    if ($LASTEXITCODE -ne 0) { throw "Docker security gate failed or Docker is unavailable." }
}

Write-Host "Release gate completed. No cloud model or credential was used."
