[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputRoot
)

# P52 release-baseline evidence. This script observes Git state and writes one
# retained report; it never stages, commits, resets, deletes, or modifies a
# PowerShell profile, credentials, PATH, or package installation.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$git = Get-Command git -ErrorAction SilentlyContinue
if ($null -eq $git) {
    throw "Git was not found; a release baseline cannot be recorded."
}

& git -C $repository rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "ProjectRoot is not inside a Git work tree: $repository"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\release_baseline\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing release-baseline output path: $OutputRoot"
}

$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$rows = @(& git -C $repository status --porcelain=v1 --untracked-files=normal)
if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
$diffCheck = @(& git -C $repository diff --check)
$diffCheckExitCode = $LASTEXITCODE
$head = (& git -C $repository rev-parse HEAD).Trim()
$branch = (& git -C $repository branch --show-current).Trim()

$tracked = @($rows | Where-Object { -not $_.StartsWith("?? ") })
$untracked = @($rows | Where-Object { $_.StartsWith("?? ") })
$payload = [ordered]@{
    schema_version = 1
    recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    repository = $repository
    head = $head
    branch = $branch
    working_tree = [ordered]@{
        tracked_change_count = $tracked.Count
        untracked_entry_count = $untracked.Count
        entries = $rows
    }
    diff_check = [ordered]@{
        passed = ($diffCheckExitCode -eq 0)
        output = $diffCheck
    }
    note = "Observation only. No files were staged, committed, reset, deleted, or repaired."
}
$target = Join-Path $output.FullName "release-baseline.json"
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $target -Encoding utf8

Write-Host "P52 release baseline recorded: $target"
Write-Host "Tracked changes: $($tracked.Count); untracked top-level entries: $($untracked.Count)"
Write-Host "No Git staging, commit, reset, deletion, or environment mutation was performed."
