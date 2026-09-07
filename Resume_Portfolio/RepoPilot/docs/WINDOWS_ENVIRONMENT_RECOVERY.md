# Windows launcher and environment recovery

Use this guide when `repopilot` resolves to Anaconda, an old virtual
environment, or a missing package. The procedure is deliberately manual and
non-destructive: it does not change PATH, `$PROFILE`, Credential Manager, or
remove any old package metadata.

## 1. Diagnose the command that PowerShell found

From the repository root, run the project-local executable directly:

```powershell
$repo = "D:\path\to\RepoPilot"
Set-Location -LiteralPath $repo
$cli = Join-Path $repo ".venv\Scripts\repopilot.exe"

& $cli doctor --project $repo
& $cli shell-doctor powershell
Get-Command repopilot -All
```

`doctor` and `shell-doctor` are diagnostics: neither command contacts a model,
prints an API key, changes a profile, or repairs the environment automatically.

## 2. Start the intended project without relying on PATH

Use either command below. Both pin execution to the repository's `.venv` and
do not alter the parent PowerShell working directory or PATH.

```powershell
& $cli repopilot
& .\scripts\repopilot.ps1 repopilot
```

For just the current PowerShell window, register the safe alias function:

```powershell
Invoke-Expression (& $cli shell-init powershell)
repopilot repopilot
```

The function lasts only for that PowerShell process. Do not copy it into
`$PROFILE` unless you have independently reviewed it and want a persistent
personal customization.

## 3. Reinstall only the project-local package when required

First exit every active RepoPilot session. Then run:

```powershell
Set-Location -LiteralPath $repo
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\repopilot.exe --version
& .\.venv\Scripts\repopilot.exe doctor --project $repo
```

Do not delete `~epopilot-*.dist-info`, an Anaconda install, or a virtual
environment merely because a diagnostic reports a collision. Keep the output
of `Get-Command repopilot -All` and choose an explicit project-local command
until you have decided how to manage global Python installations.

## 4. Verify a release candidate

The ordinary release gate is offline with respect to model providers:

```powershell
.\scripts\release_check_windows.ps1
```

For a separate clean-virtual-environment packaging smoke check, opt in
explicitly. It may resolve Python package dependencies and writes retained
evidence under `artifacts\package_smoke`; it never deletes an existing path.

```powershell
.\scripts\release_check_windows.ps1 -PackageSmoke
```
