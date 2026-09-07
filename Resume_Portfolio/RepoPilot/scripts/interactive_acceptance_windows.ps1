[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$Provider = "deepseek",
    [string]$Model = "deepseek-v4-flash",
    [string]$OutputRoot
)

# P63 opt-in live-provider acceptance.  The fixed prompt requests a known
# marker and forbids tools, so no repository file, test output, or user prompt
# is intentionally sent to the provider.  Local interruption/edit/recovery
# behaviour remains covered by deterministic integration tests below.
$ErrorActionPreference = "Stop"
if ($Provider -ne "deepseek") {
    throw "This bounded acceptance currently supports only the explicitly configured DeepSeek provider."
}
$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
$cli = Join-Path $repository ".venv\Scripts\repopilot.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf) -or -not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Project-local RepoPilot launcher or Python was not found."
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\interactive_acceptance\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing interactive-acceptance output path: $OutputRoot"
}
$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$prompt = "Only output REPOPILOT_STREAM_OK. Do not call tools. Do not read, analyze, or modify project files."

Write-Host "Running the bounded DeepSeek streaming check..."
$jsonl = @(& $cli --provider $Provider --model $Model --trust --output-format jsonl -p $prompt)
if ($LASTEXITCODE -ne 0) { throw "Bounded DeepSeek streaming check failed." }
$jsonlPath = Join-Path $output.FullName "stream.jsonl"
$jsonl | Set-Content -LiteralPath $jsonlPath -Encoding utf8
$resultLine = @($jsonl | Where-Object { $_ -match '"type"\s*:\s*"result"' } | Select-Object -Last 1)
if ($resultLine.Count -ne 1) { throw "Streaming check did not emit exactly one result event." }
$result = $resultLine[0] | ConvertFrom-Json
if ($result.status -ne "completed" -or $result.tool_calls -ne 0 -or $result.answer -notmatch "REPOPILOT_STREAM_OK") {
    throw "Streaming check did not complete with the expected no-tool marker."
}

Write-Host "Running deterministic cancellation, edit-policy, and recovery coverage..."
& $python -m pytest -q `
    tests\integration\test_session_interruption_recovery.py `
    tests\unit\test_runtime_cancellation.py `
    tests\unit\test_session_revisions.py `
    tests\unit\test_session_undo.py `
    tests\unit\test_permission_rules.py -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Deterministic interactive reliability coverage failed." }

$receipt = [ordered]@{
    schema_version = 1
    recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    provider = $Provider
    model = $Model
    provider_prompt_scope = "fixed marker only; no project file selection or tools"
    provider_result = [ordered]@{ status = $result.status; tool_calls = $result.tool_calls; marker = "REPOPILOT_STREAM_OK" }
    local_coverage = @("cancellation", "cross-runtime recovery", "edit revision", "undo", "permission policy")
    note = "The saved JSONL is limited to the fixed-prompt provider transaction. It is not a real-project coding-task evaluation."
}
$receiptPath = Join-Path $output.FullName "interactive-acceptance.receipt.json"
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Host "P63 interactive acceptance completed. Evidence retained at: $($output.FullName)"
