[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Receipt
)

# P65 offline package-evidence verifier.  It validates only the retained local
# wheel hash and bounded receipt schema; it installs nothing and never contacts
# a provider, package index, credential store, or remote service.
$ErrorActionPreference = "Stop"
$receiptPath = (Resolve-Path -LiteralPath $Receipt).Path
$root = Split-Path -Parent $receiptPath
$raw = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($raw.schema_version -ne 1 -or $raw.package.name -ne "repopilot") {
    throw "Package-smoke receipt has an unsupported schema."
}
$wheelName = [string]$raw.package.wheel
$expected = [string]$raw.package.wheel_sha256
if ($wheelName -notmatch '^repopilot-[A-Za-z0-9.+_-]+\.whl$' -or $expected -notmatch '^[a-f0-9]{64}$') {
    throw "Package-smoke receipt has an invalid wheel identity or SHA-256."
}
$wheel = Join-Path $root (Join-Path "wheelhouse" $wheelName)
if (-not (Test-Path -LiteralPath $wheel -PathType Leaf)) {
    throw "Recorded package wheel is missing: $wheel"
}
$actual = (Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "Package wheel SHA-256 does not match the retained receipt."
}
Write-Host "Package smoke receipt verified: repopilot $($raw.package.version), SHA-256 $actual"
