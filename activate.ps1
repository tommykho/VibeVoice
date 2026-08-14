<#
.SYNOPSIS
  Activate this project's virtual environment in the current PowerShell session.

.DESCRIPTION
  Run it as  .\activate.ps1  -- a .cmd wrapper cannot do this, because a batch
  file runs in a child cmd.exe whose environment dies with it.

  The venv is not always inside the repository: a checkout on a network drive
  wants its venv on local disk, since torch is several GB of DLLs read on every
  import. Record that location in a .venvpath file and this script will find it.
#>
$ErrorActionPreference = 'Stop'

$candidates = @()
$marker = Join-Path $PSScriptRoot '.venvpath'
if (Test-Path $marker) { $candidates += (Get-Content $marker -Raw).Trim() }
$candidates += Join-Path $PSScriptRoot '.venv'
if ($env:LOCALAPPDATA) { $candidates += Join-Path $env:LOCALAPPDATA 'VibeVoice\venv' }
if ($env:USERPROFILE)  { $candidates += Join-Path $env:USERPROFILE '.venv' }

foreach ($venv in $candidates) {
    if (-not $venv) { continue }
    $script = Join-Path $venv 'Scripts\Activate.ps1'
    if (Test-Path $script) {
        . $script
        Write-Host "activated $venv" -ForegroundColor Green
        python -c "import sys; print(sys.version.split()[0], '--', sys.executable)"
        exit 0
    }
}

Write-Host 'No virtual environment found. Looked in:' -ForegroundColor Yellow
$candidates | Where-Object { $_ } | ForEach-Object { Write-Host "  $_" }
Write-Host ''
Write-Host 'Create one with:  uv venv --python 3.11 --seed .venv'
Write-Host 'Or record an existing one:  "C:\path\to\venv" | Set-Content .venvpath'
exit 1
