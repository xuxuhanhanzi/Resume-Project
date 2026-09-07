[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$WithDev,
    [switch]$WithService
)

# A deliberately conservative bootstrapper. It only creates or reuses the
# project-local .venv; it never edits $PROFILE, touches Credential Manager, or
# removes an existing environment or installation.
$ErrorActionPreference = "Stop"

if ($WithDev -and $WithService) {
    throw "Use either -WithDev or -WithService, not both."
}

$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $repository ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$venvCli = Join-Path $venv "Scripts\repopilot.exe"
$projectLauncher = Join-Path $repository "scripts\repopilot.ps1"

Set-Location -LiteralPath $repository

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating the project-local virtual environment..."
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv with '$Python'. Install Python 3.11 or 3.12, or pass -Python <path>."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed. The existing .venv was kept unchanged."
}

$installArguments = @("-m", "pip", "install", "-e")
if ($WithDev) {
    $installArguments += ".[dev]"
} elseif ($WithService) {
    $installArguments += ".[service]"
} else {
    $installArguments += "."
}
& $venvPython @installArguments
if ($LASTEXITCODE -ne 0) {
    throw "RepoPilot installation failed. The existing .venv was not removed."
}

& $venvCli --version
if ($LASTEXITCODE -ne 0) {
    throw "The RepoPilot launcher did not start after installation."
}
& $venvCli doctor

Write-Host ""
Write-Host "For this PowerShell window, register the project-local launcher:"
Write-Host "  Invoke-Expression (& '$venvCli' shell-init powershell)"
Write-Host "Then use: repopilot <saved-project-name>"
Write-Host "Or use the collision-proof project launcher in any PowerShell window:"
Write-Host "  & '$projectLauncher' <saved-project-name>"
Write-Host "The script did not edit your PowerShell profile or credentials."
