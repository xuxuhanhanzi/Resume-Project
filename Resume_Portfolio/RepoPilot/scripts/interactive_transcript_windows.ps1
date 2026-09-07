[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputRoot
)

# P64 offline transcript acceptance.  Slash commands below are local-only and
# construct a local provider solely for session setup; no model request, tool,
# credential, profile, PATH, or PowerShell profile is used or changed.
$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$cli = Join-Path $repository ".venv\Scripts\repopilot.exe"
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Project-local RepoPilot launcher was not found: $cli"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\interactive_transcript\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing transcript output path: $OutputRoot"
}
$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$inputPath = Join-Path $output.FullName "commands.txt"
@("/status", "/workflow", "/changes", "/context", "/exit") | Set-Content -LiteralPath $inputPath -Encoding utf8
$transcriptPath = Join-Path $output.FullName "transcript.txt"

$transcriptLines = @(Get-Content -LiteralPath $inputPath | & $cli --provider local --model local-transcript --trust --no-stream *>&1)
if ($LASTEXITCODE -ne 0) { throw "Offline interactive transcript command failed." }
$transcriptLines | Set-Content -LiteralPath $transcriptPath -Encoding utf8
$transcript = $transcriptLines -join "`n"
foreach ($required in @("RepoPilot", "Project:", "Model:", "Workflow", "Changes", "Context", "Session saved")) {
    if ($transcript -notmatch [regex]::Escape($required)) {
        throw "Offline transcript omitted required local UX evidence: $required"
    }
}
Write-Host "P64 offline interactive transcript completed. Evidence retained at: $($output.FullName)"
