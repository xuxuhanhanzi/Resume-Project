[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputRoot
)

# P51 clean-install evidence. This is intentionally opt-in because pip may
# resolve package dependencies. The known project venv supplies the already
# verified build backend, avoiding a second build-isolation download. It never
# changes PATH, $PROFILE, credentials, or an existing output directory.
# Generated evidence is retained for review.
$ErrorActionPreference = "Stop"

$repository = (Resolve-Path -LiteralPath $ProjectRoot).Path
$sourcePython = Join-Path $repository ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $sourcePython -PathType Leaf)) {
    throw "Project-local Python was not found: $sourcePython. Run .\scripts\setup_windows.ps1 -WithDev first."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $repository "artifacts\package_smoke\$stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse an existing package-smoke output path: $OutputRoot"
}

$output = New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop
$wheelhouse = Join-Path $output.FullName "wheelhouse"
$cleanVenv = Join-Path $output.FullName "clean-venv"
$cleanPython = Join-Path $cleanVenv "Scripts\python.exe"
$cleanCli = Join-Path $cleanVenv "Scripts\repopilot.exe"
$receiptPath = Join-Path $output.FullName "package-smoke.receipt.json"
$version = (& $sourcePython -c "import repopilot; print(repopilot.__version__)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
    throw "Could not read the installed RepoPilot version from the project virtual environment."
}

New-Item -ItemType Directory -Path $wheelhouse -ErrorAction Stop | Out-Null
Write-Host "Building wheelhouse for repopilot $version..."
& $sourcePython -m pip wheel --no-build-isolation --wheel-dir $wheelhouse $repository
if ($LASTEXITCODE -ne 0) { throw "Wheelhouse build failed." }
$packageWheel = @(Get-ChildItem -LiteralPath $wheelhouse -Filter "repopilot-$version-*.whl" -File)
if ($packageWheel.Count -ne 1) {
    throw "Expected exactly one RepoPilot wheel in the local wheelhouse."
}
$wheelDigest = (Get-FileHash -LiteralPath $packageWheel[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()

Write-Host "Creating an isolated package-smoke virtual environment..."
& $sourcePython -m venv $cleanVenv
if ($LASTEXITCODE -ne 0) { throw "Could not create clean package-smoke virtual environment." }

Write-Host "Installing from the local wheelhouse only..."
& $cleanPython -m pip install --no-index --find-links $wheelhouse "repopilot==$version"
if ($LASTEXITCODE -ne 0) { throw "Clean wheel installation failed." }

& $cleanPython -c "import repopilot; assert repopilot.__version__ == '$version'; print(repopilot.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Clean environment imported an unexpected RepoPilot version." }
& $cleanCli --version
if ($LASTEXITCODE -ne 0) { throw "Clean environment RepoPilot launcher failed." }
& $cleanCli doctor --project $repository
if ($LASTEXITCODE -ne 0) { throw "Clean environment RepoPilot doctor failed." }

$receipt = [ordered]@{
    schema_version = 1
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    package = [ordered]@{
        name = "repopilot"
        version = $version
        wheel = $packageWheel[0].Name
        wheel_sha256 = $wheelDigest
    }
    checks = [ordered]@{
        isolated_install = "passed"
        import_version = "passed"
        launcher = "passed"
        doctor = "passed"
    }
    note = "The wheel was built locally with no build isolation and installed from this retained local wheelhouse only."
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Host "P65 package smoke receipt recorded: $receiptPath"
Write-Host "P51 package smoke completed. Evidence retained at: $($output.FullName)"
