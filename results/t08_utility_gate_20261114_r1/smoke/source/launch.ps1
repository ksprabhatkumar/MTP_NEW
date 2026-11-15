$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python=Join-Path (Get-Location) 'Prabhat\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Prabhat environment missing' }
$base='results/t08_utility_gate_20261114_r1'
& $python -m compileall -q experiments/t08_utility_gate_20261114
if ($LASTEXITCODE -ne 0) { throw 'Syntax check failed' }
& $python -u -m experiments.t08_utility_gate_20261114.run --stage smoke --output "$base/smoke"
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed. Full not started.' }
& $python -u -m experiments.t08_utility_gate_20261114.run --stage full --output "$base/full"
if ($LASTEXITCODE -ne 0) { throw 'Full failed. Preserve results and return error.' }
Write-Host 'Completed utility-gate pilot; return results to Astra. T08 remains open.'
