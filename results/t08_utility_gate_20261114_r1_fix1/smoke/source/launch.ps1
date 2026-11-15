$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python=Join-Path (Get-Location) 'Prabhat\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Prabhat environment missing' }
$base='results/t08_utility_gate_20261114_r1_fix1'
& $python -m compileall -q experiments/t08_utility_gate_20261114
if ($LASTEXITCODE -ne 0) { throw 'Syntax check failed' }
& $python -u -m experiments.t08_utility_gate_20261114.run --stage smoke --output "$base/smoke"
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed. Full not started.' }
Write-Host 'Smoke complete. Return audit and equivalence diagnostics to Astra before full training.'
