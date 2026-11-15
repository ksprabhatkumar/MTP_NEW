$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'Prabhat\Scripts\python.exe'
$base = 'results/o1_t01_t08_20261028'
try {
 & $python -u check_o1_selection.py --output "$base/smoke"
 if ($LASTEXITCODE -ne 0) { throw 'New operator preflight failed' }
 & $python -u run_o1_selection.py --output "$base/smoke" --epochs 4 --subjects 2 --seeds 13 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'End-to-end smoke failed; full pilot NOT launched' }
 & $python -u check_o1_selection.py --output "$base/full"
 if ($LASTEXITCODE -ne 0) { throw 'Full configuration preflight failed' }
 & $python -m pip freeze | Out-File "$base/full/requirements.lock.txt" -Encoding utf8
 Write-Host 'SMOKE PASSED. Starting fixed 12-job source-only selection pilot.' -ForegroundColor Green
 & $python -u run_o1_selection.py --output "$base/full" --epochs 16 --subjects 2,5 --seeds 13,29 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'Pilot failed. Preserve checkpoints and inspect error.txt' }
 Write-Host 'All selection jobs and artifact audits completed.' -ForegroundColor Green
} catch {
 $_ | Out-String | Set-Content "$base/selection_error.txt"
 throw
}
