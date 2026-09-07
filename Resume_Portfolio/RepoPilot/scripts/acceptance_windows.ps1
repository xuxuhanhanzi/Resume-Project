[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [switch]$SkipTests
)

# A read-only, local acceptance check for the Windows entry path.  It does not
# change a PowerShell profile, credential store, configuration, or workspace,
# and it never constructs a model provider.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
$cli = Join-Path $repository ".venv\Scripts\repopilot.exe"
$projectLauncher = Join-Path $repository "scripts\repopilot.ps1"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python was not found: $python. Run .\scripts\setup_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Project-local launcher was not found: $cli. Run .\scripts\setup_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $projectLauncher -PathType Leaf)) {
    throw "Project-local PowerShell launcher was not found: $projectLauncher."
}

Set-Location -LiteralPath $repository

Write-Host "Checking project-local launcher..."
& $cli --version
if ($LASTEXITCODE -ne 0) { throw "The project-local RepoPilot launcher failed to start." }

Write-Host "Checking collision-proof project PowerShell launcher..."
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $projectLauncher --version
if ($LASTEXITCODE -ne 0) { throw "The project-local PowerShell launcher failed to start." }

Write-Host "Checking offline installation diagnostics..."
& $cli doctor --project $repository
if ($LASTEXITCODE -ne 0) { throw "RepoPilot doctor reported an invalid project or configuration." }

$shellInit = & $cli shell-init powershell
if ($LASTEXITCODE -ne 0 -or $shellInit -notmatch "function repopilot") {
    throw "RepoPilot did not produce a safe PowerShell shell-init function."
}
Write-Host "PowerShell shell-init is available (not applied by this script)."

# Verify the generated function in an isolated child PowerShell.  This proves
# that it shadows an Anaconda/PATH launcher only for that child session; the
# caller's aliases, PATH and $PROFILE remain untouched.
$escapedCli = $cli.Replace("'", "''")
$shellProbe = @"
`$ErrorActionPreference = 'Stop'
Invoke-Expression (& '$escapedCli' shell-init powershell)
`$functionCommand = Get-Command repopilot -CommandType Function -ErrorAction Stop
if (`$functionCommand.ScriptBlock.ToString() -notmatch [regex]::Escape('$escapedCli')) {
    throw 'The session-local repopilot function does not target the project-local launcher.'
}
& repopilot --version
if (`$LASTEXITCODE -ne 0) { throw 'The session-local repopilot function did not start.' }
"@
& powershell.exe -NoLogo -NoProfile -NonInteractive -Command $shellProbe
if ($LASTEXITCODE -ne 0) {
    throw "Project-local shell-init did not pass the isolated PowerShell launcher check."
}
Write-Host "PowerShell shell-init passes an isolated launcher check."

Write-Host "Checking offline MCP configuration..."
& $cli mcp doctor
if ($LASTEXITCODE -ne 0) {
    Write-Warning "MCP doctor reported an invalid configured server. Fix that server before using --mcp."
}

if (-not $SkipTests) {
    Write-Host "Running the focused Windows/recovery test suite..."
    & $python -m pytest -q tests/unit/test_windows_setup_script.py tests/unit/test_cli_profiles.py tests/unit/test_runtime_cancellation.py tests/unit/test_terminal.py
    if ($LASTEXITCODE -ne 0) { throw "Focused Windows/recovery tests failed." }
}

Write-Host "Windows acceptance check completed. No cloud model or API key was used."
