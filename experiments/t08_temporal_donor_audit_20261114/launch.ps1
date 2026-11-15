$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path (Get-Location) 'Prabhat\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Prabhat environment missing' }
$base = 'results/t08_temporal_donor_audit_20261114_r1'
if (Test-Path -LiteralPath $base) { throw 'Output exists. Preserve it; inspect before any fresh suffixed attempt.' }
& $python -m compileall -q experiments/t08_temporal_donor_audit_20261114
if ($LASTEXITCODE -ne 0) { throw 'Syntax check failed' }
& $python -u -m experiments.t08_temporal_donor_audit_20261114.run --stage smoke --output "$base/smoke"
if ($LASTEXITCODE -ne 0) { throw 'Smoke audit failed; full audit not started' }
$smoke = Get-Content -Raw "$base/smoke/audit.json" | ConvertFrom-Json
if (!$smoke.passed) { throw 'Smoke gate failed' }
& $python -u -m experiments.t08_temporal_donor_audit_20261114.run --stage full --output "$base/full"
if ($LASTEXITCODE -ne 0) { throw 'Full audit failed' }
Write-Host 'Audit complete. Return results to Astra before new training.'
