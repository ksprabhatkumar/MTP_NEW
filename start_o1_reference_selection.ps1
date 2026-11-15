$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Prabhat - O1 original message correction confirmation'
$base = 'results/o1_reference_selection_20261028'
New-Item -ItemType Directory -Force $base | Out-Null
Start-Transcript -Path "$base/terminal.log" -Append
$python = Join-Path $PSScriptRoot 'Prabhat\Scripts\python.exe'
try {
 & $python -u check_o1_reference_selection.py --output "$base/smoke"
 if ($LASTEXITCODE -ne 0) { throw 'Reference correction preflight failed' }
 & $python -u run_o1_reference_selection.py --output "$base/smoke" --epochs 4 --subjects 2 --seeds 13 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'Smoke failed; no full launch' }
 & $python -u check_o1_reference_selection.py --output "$base/full"
 if ($LASTEXITCODE -ne 0) { throw 'Full preflight failed' }
 & $python -m pip freeze | Out-File "$base/full/requirements.lock.txt" -Encoding utf8
 & $python -u run_o1_reference_selection.py --output "$base/full" --epochs 16 --subjects 2,5 --seeds 13,29 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'Full pilot failed; preserve outputs and resume' }
 Write-Host 'Reference correction completed and artifacts audited.' -ForegroundColor Green
} catch { $_ | Out-String | Set-Content "$base/terminal_error.txt"; Write-Host $_ -ForegroundColor Red }
finally { Stop-Transcript }
