param([ValidateSet('Smoke','Full','All')][string]$Stage='All')
$ErrorActionPreference='Stop'
$project=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $project
$python=Join-Path $project 'Prabhat\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Prabhat Python missing; report to Astra before installing or changing environment.' }
$base='results/t08_predictive_20261114_r1'
New-Item -ItemType Directory -Force $base | Out-Null
# One owner for the entire smoke/full queue, including the transition between stages.
$lock=[System.IO.File]::Open((Join-Path $project "$base/queue.lock"),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
 Start-Transcript -Path "$base/terminal.log" -Append
 $Host.UI.RawUI.WindowTitle='Prabhat - T08 predictive repair'
 if ($Stage -in @('Smoke','All')) {
  & $python -m compileall -q experiments/t08_predictive_20261114
  if ($LASTEXITCODE -ne 0) { throw 'Syntax checks failed' }
  & $python -u -m experiments.t08_predictive_20261114.checks --output "$base/smoke"
  if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
  & $python -u -m experiments.t08_predictive_20261114.run --output "$base/smoke" --epochs 4 --subjects 2 --seeds 13 --batch-size 64
  if ($LASTEXITCODE -ne 0) { throw 'Smoke failed; no full training' }
  & $python -u -m experiments.t08_predictive_20261114.review --output "$base/smoke"
  if ($LASTEXITCODE -ne 0) { throw 'Smoke artifact audit failed' }
 }
 if ($Stage -in @('Full','All')) {
  $smoke=Get-Content -Raw "$base/smoke/status.json" | ConvertFrom-Json
  $audit=Get-Content -Raw "$base/smoke/reproducibility_audit.json" | ConvertFrom-Json
  if ($smoke.state -ne 'complete' -or !$audit.passed) { throw 'A completed audited smoke is required' }
  # Rechecks source/data hashes; a changed implementation invalidates the smoke gate.
  & $python -u -m experiments.t08_predictive_20261114.review --output "$base/smoke"
  if ($LASTEXITCODE -ne 0) { throw 'Smoke fingerprint no longer matches' }
  & $python -u -m experiments.t08_predictive_20261114.checks --output "$base/full"
  if ($LASTEXITCODE -ne 0) { throw 'Full preflight failed' }
  & $python -m pip freeze | Out-File "$base/full/requirements.lock.txt" -Encoding utf8
  if ($LASTEXITCODE -ne 0) { throw 'Environment capture failed' }
  & $python -u -m experiments.t08_predictive_20261114.run --output "$base/full" --epochs 16 --subjects 2,5 --seeds 13,29 --batch-size 64
  if ($LASTEXITCODE -ne 0) { throw 'Full pilot failed; preserve checkpoints and traceback' }
  & $python -u -m experiments.t08_predictive_20261114.review --output "$base/full"
  if ($LASTEXITCODE -ne 0) { throw 'Full artifact audit failed' }
 }
 Write-Host 'Requested stage completed. Return evidence to Astra; T08 is not automatically complete.' -ForegroundColor Green
} catch {
 $_ | Out-String | Set-Content "$base/terminal_error.txt"
 throw
} finally {
 try { Stop-Transcript } catch {}
 $lock.Dispose()
}
