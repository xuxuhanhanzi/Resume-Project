[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputRoot,
    [switch]$ReportOnly
)

# P62 CI preflight.  It records the prerequisites needed to run the existing
# local Docker gate and GitHub Actions workflow, without pushing, changing a
# remote, starting containers, or altering a Git configuration.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
$workflow = Join-Path $repository ".github\workflows\ci.yml"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $workflow -PathType Leaf)) {
    throw "GitHub Actions workflow was not found: $workflow"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\ci_preflight\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing CI-preflight output path: $OutputRoot"
}

$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$blockers = [System.Collections.Generic.List[string]]::new()
$origin = ""
if ($null -eq (Get-Command git -ErrorAction SilentlyContinue)) {
    $blockers.Add("git CLI is unavailable")
} else {
    $origin = ([string]((& git -C $repository remote get-url origin 2>$null) -join "")).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($origin)) {
        $blockers.Add("Git remote 'origin' is not configured; remote CI cannot be triggered")
        $origin = ""
    }
}

& $python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text(encoding='utf-8')); print('workflow parsed')"
if ($LASTEXITCODE -ne 0) { $blockers.Add("GitHub Actions workflow YAML did not parse") }

$dockerAvailable = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)
$dockerServer = ""
if (-not $dockerAvailable) {
    $blockers.Add("Docker CLI is unavailable")
} else {
    $dockerServer = (& docker version --format "{{.Server.Version}}" 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($dockerServer)) {
        $blockers.Add("Docker daemon is unavailable; the live container security gate cannot run")
        $dockerServer = ""
    }
}

$payload = [ordered]@{
    schema_version = 1
    recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    repository = $repository
    workflow = ".github/workflows/ci.yml"
    remote_origin = $origin
    docker = [ordered]@{ cli_available = $dockerAvailable; server_version = $dockerServer }
    blockers = @($blockers)
    remote_ci_ready = ($blockers.Count -eq 0)
    note = "Preflight only. No push, remote change, container build, model call, credential read, or configuration mutation was performed."
}
$target = Join-Path $output.FullName "ci-preflight.json"
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $target -Encoding utf8
Write-Host "P62 CI preflight recorded: $target"
if ($blockers.Count -eq 0) {
    Write-Host "CI prerequisites are present. Run the Docker gate and push a human-reviewed commit to execute remote CI."
    exit 0
}
Write-Warning ("CI prerequisites remain blocked: " + ($blockers -join "; "))
if ($ReportOnly) { exit 0 }
exit 2
