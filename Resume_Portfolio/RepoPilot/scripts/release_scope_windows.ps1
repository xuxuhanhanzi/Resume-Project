[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputRoot
)

# P61 release-scope inventory.  This script only classifies the current Git
# status so a maintainer can make an explicit commit decision.  It never stages,
# commits, resets, deletes, excludes, or rewrites any project file.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ($null -eq (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found; a release scope inventory cannot be recorded."
}
& git -C $repository rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "ProjectRoot is not inside a Git work tree: $repository"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\release_scope\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing release-scope output path: $OutputRoot"
}

function Get-ReleaseClassification([string]$Path, [bool]$Untracked) {
    $normalized = $Path.Replace("\", "/")
    if ($normalized -like ".repopilot/*") {
        return @{ classification = "local_state"; default_action = "exclude" }
    }
    if ($normalized -like "artifacts/*" -or $normalized -like "logs/*") {
        return @{ classification = "generated_evidence"; default_action = "exclude" }
    }
    if ($normalized -like "Reference code/*") {
        return @{ classification = "reference_material"; default_action = "exclude" }
    }
    if ($normalized -match "^(src|tests|scripts|docs|\.github|deploy|configs|examples)/") {
        return @{ classification = "product_candidate"; default_action = "review_then_include" }
    }
    if ($normalized -in @("README.md", "CHANGELOG.md", "pyproject.toml")) {
        return @{ classification = "product_candidate"; default_action = "review_then_include" }
    }
    if ($Untracked) {
        return @{ classification = "unclassified"; default_action = "review_required" }
    }
    return @{ classification = "tracked_change"; default_action = "review_then_include" }
}

$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$rows = @(& git -C $repository status --porcelain=v1 --untracked-files=normal)
if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
$head = (& git -C $repository rev-parse HEAD).Trim()
$branch = (& git -C $repository branch --show-current).Trim()
$entries = foreach ($row in $rows) {
    if ($row.Length -lt 4) { continue }
    $isUntracked = $row.StartsWith("?? ")
    $path = $row.Substring(3)
    $classification = Get-ReleaseClassification -Path $path -Untracked $isUntracked
    [ordered]@{
        status = $row.Substring(0, 2)
        path = $path
        tracked = -not $isUntracked
        classification = $classification.classification
        default_action = $classification.default_action
    }
}
$groups = [ordered]@{}
foreach ($entry in $entries) {
    $key = [string]$entry.classification
    if (-not $groups.Contains($key)) { $groups[$key] = 0 }
    $groups[$key] += 1
}
$payload = [ordered]@{
    schema_version = 1
    recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    repository = $repository
    head = $head
    branch = $branch
    entries = @($entries)
    classification_counts = $groups
    next_step = "A maintainer must explicitly review this inventory before staging or committing."
    safety_note = "Observation only. This script did not stage, commit, reset, delete, ignore, or repair any file."
}
$target = Join-Path $output.FullName "release-scope.json"
$payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $target -Encoding utf8

Write-Host "P61 release scope recorded: $target"
Write-Host "Entries requiring review: $($entries.Count)"
Write-Host "No Git staging, commit, reset, deletion, ignore-rule, or environment mutation was performed."
