[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$Image = "repopilot-sandbox:20260814",
    [switch]$BuildImage
)

# P53 opt-in Docker security gate. A missing daemon is reported as a blocker;
# RepoPilot never substitutes host execution for the container boundary.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python was not found: $python. Run .\scripts\setup_windows.ps1 -WithDev first."
}
if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI is unavailable. No container, host command, or fallback test was started."
    exit 2
}

& docker version --format "{{.Server.Version}}"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker daemon is unavailable. No container, host command, or fallback test was started."
    exit 2
}

if ($BuildImage) {
    Write-Host "Building the pinned RepoPilot sandbox image..."
    & docker build -t $Image -f (Join-Path $repository "deploy\docker\sandbox.Dockerfile") $repository
    if ($LASTEXITCODE -ne 0) { throw "Docker sandbox image build failed." }
} else {
    & docker image inspect $Image | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Sandbox image '$Image' is unavailable. Re-run with -BuildImage after reviewing the Dockerfile."
    }
}

$env:REPOPILOT_RUN_DOCKER_SECURITY = "1"
$env:REPOPILOT_SANDBOX_IMAGE = $Image
Write-Host "Running the live Docker security boundary test..."
& $python -m pytest -q tests\safety\test_security_container_live.py -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Live Docker security boundary test failed." }
Write-Host "P53 Docker security gate passed. No host-execution fallback was used."
