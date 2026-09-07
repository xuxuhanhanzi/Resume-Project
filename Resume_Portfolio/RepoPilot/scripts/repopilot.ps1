[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

# Stable project-local entry point for Windows. It deliberately does not edit
# PATH or $PROFILE, so Anaconda/global console-script collisions cannot change
# which RepoPilot installation this command starts.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cli = Join-Path $repository ".venv\Scripts\repopilot.exe"

if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Project-local RepoPilot launcher was not found: $cli. Run .\scripts\setup_windows.ps1 first."
}

Set-Location -LiteralPath $repository
& $cli @Arguments
exit $LASTEXITCODE
